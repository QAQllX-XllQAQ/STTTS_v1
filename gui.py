"""STTTS GUI — Desktop interface for Speech-to-Text → Text-to-Speech.

Features:
- STT: Whisper (local GPU) / Google Cloud STT / Xiaomi MiMo ASR
- TTS: GPT-SoVITS / Edge-TTS / Google Cloud TTS / MiMo TTS
- Modes: Continuous (VAD) / Push-to-Talk
- Audio output to any device (headphones, speakers, Voicemeeter, etc.)
"""

import atexit
import collections
import ctypes
import os
import subprocess
import sys
import threading
import traceback

import PySimpleGUI as sg

# Ensure package is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from sttts.audio import list_devices, parse_device_index
from sttts.config import load_config, save_config, get_default_ref_audio, reset_config
from sttts.tts import (
    tts_gptsovits, tts_edge, tts_google, tts_mimo, check_gptsovits,
    EDGE_VOICES, GOOGLE_VOICES, MIMO_VOICES,
)
from sttts.stt import stt_whisper, stt_google_vad, stt_google_cloud, stt_mimo, stt_ptt

sg.theme("LightBlue3")

# ── Constants ─────────────────────────────────────────────
_LOG_MAX_LINES = 500          # max lines kept in Log / Transcription panels
_SRV_CHECK_INTERVAL = 2500   # ms between GPT-SoVITS server health checks
_TTS_CACHE_NAMES = {
    'gpt': 'gptsovits_tmp.wav',
    'edge': 'edge_tts_tmp.mp3',
    'google': 'google_tts_tmp.wav',
    'mimo': 'mimo_tts_tmp.wav',
}


# ── GPT-SoVITS process management ───────────────────────

class GPTSoVITSManager:
    """Manages the GPT-SoVITS API server subprocess."""

    GPT_CMD = [
        'conda', 'run', '-n', 'GPTSoVits', 'python',
        os.path.join(os.path.dirname(__file__) or '.', 'GPT-SoVITS', 'api_v2.py'),
        '-a', '127.0.0.1', '-p', '9880',
        '-c', r'GPT_SoVITS/configs/tts_infer.yaml',
    ]

    def __init__(self):
        self.process = None
        self._stderr_log = os.path.join(
            os.path.dirname(__file__) or '.', 'gptsovits_stderr.log'
        )

    def start(self, cwd=None):
        """Start the GPT-SoVITS API server. Returns (success, message)."""
        if self.process and self.process.poll() is None:
            return False, "GPT-SoVITS already running"
        try:
            stderr_fd = open(self._stderr_log, 'a', encoding='utf-8', errors='replace')
            self.process = subprocess.Popen(
                self.GPT_CMD,
                cwd=cwd or os.path.join(os.path.dirname(__file__) or '.', 'GPT-SoVITS'),
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=stderr_fd,
            )
            self._stderr_fd = stderr_fd
            return True, f"GPT-SoVITS starting (PID {self.process.pid}), wait ~40s..."
        except Exception as e:
            return False, f"Failed to start GPT-SoVITS: {e}"

    def stop(self):
        """Stop the GPT-SoVITS API server. Returns message."""
        if not self.process or self.process.poll() is not None:
            self._close_stderr()
            self.process = None
            return "GPT-SoVITS not running"
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self._close_stderr()
            self.process = None
            return "GPT-SoVITS stopped"
        except Exception as e:
            self._close_stderr()
            self.process = None
            return f"Error stopping GPT-SoVITS: {e}"

    def is_running(self):
        """Check if the process is still alive."""
        return self.process is not None and self.process.poll() is None

    def cleanup(self):
        """Force-kill on exit."""
        if self.process and self.process.poll() is None:  # FIXED: was `is not None`
            try:
                self.process.kill()
            except Exception:
                pass
        self._close_stderr()
        self.process = None

    def _close_stderr(self):
        fd = getattr(self, '_stderr_fd', None)
        if fd:
            try:
                fd.close()
            except Exception:
                pass
            self._stderr_fd = None


# ── GPT-SoVITS Model Management ─────────────────────────

GPTSOVITS_CONFIG = os.path.join(
    os.path.dirname(__file__) or '.', 'GPT-SoVITS', 'GPT_SoVITS', 'configs', 'tts_infer.yaml'
)


def list_gptsovits_models():
    """Read available model presets from tts_infer.yaml."""
    try:
        import yaml
    except ImportError:
        return ['custom']
    try:
        with open(GPTSOVITS_CONFIG, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not cfg:
            return ['custom']
        return list(cfg.keys())
    except Exception:
        return ['custom']


def get_active_gptsovits_model():
    """Get the currently active model preset name."""
    try:
        import yaml
        with open(GPTSOVITS_CONFIG, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if cfg and 'custom' in cfg:
            custom = cfg['custom']
            for name, preset in cfg.items():
                if name == 'custom':
                    continue
                if (preset.get('t2s_weights_path') == custom.get('t2s_weights_path') and
                        preset.get('vits_weights_path') == custom.get('vits_weights_path')):
                    return name
            return 'custom'
    except Exception:
        pass
    return 'custom'


def switch_gptsovits_model(preset_name, ckpt_path=None, pth_path=None, version='v2'):
    """Switch the active GPT-SoVITS model by updating the custom section."""
    try:
        import yaml
        with open(GPTSOVITS_CONFIG, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not cfg:
            return False, "Failed to read config"
        if preset_name == 'custom' and ckpt_path and pth_path:
            if not os.path.exists(ckpt_path):
                return False, f"CKPT not found: {ckpt_path}"
            if not os.path.exists(pth_path):
                return False, f"PTH not found: {pth_path}"
            cfg['custom'] = {
                'bert_base_path': 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
                'cnhuhbert_base_path': 'GPT_SoVITS/pretrained_models/chinese-hubert-base',
                'device': cfg.get('custom', {}).get('device', 'cuda'),
                'is_half': cfg.get('custom', {}).get('is_half', True),
                't2s_weights_path': ckpt_path,
                'version': version,
                'vits_weights_path': pth_path,
            }
            with open(GPTSOVITS_CONFIG, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
            return True, f"Custom model set: {os.path.basename(pth_path)} (v={version})"
        if preset_name not in cfg:
            return False, f"Preset '{preset_name}' not found"
        if preset_name == 'custom':
            return True, "Using custom config"
        preset = cfg[preset_name].copy()
        cfg['custom'] = preset
        with open(GPTSOVITS_CONFIG, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        return True, f"Switched to {preset_name} (v{preset.get('version', '?')})"
    except Exception as e:
        return False, f"Failed to switch model: {e}"


# ── STT runner ──────────────────────────────────────────

def run_stt(cfg, stop_event, on_text, log_fn, set_status=None):
    """Run the appropriate STT engine based on config."""
    stt_type = cfg.get('stt', 'whisper')
    try:
        if stt_type == 'whisper':
            stt_whisper(cfg.get('dev_in'), stop_event, on_text, log_fn,
                        cfg.get('whisper_model', 'small'))
        elif stt_type == 'google_vad':
            stt_google_vad(cfg.get('dev_in'), cfg.get('gkey', ''), stop_event,
                           on_text, log_fn, cfg.get('google_model', 'default'))
        elif stt_type == 'google_cloud':
            stt_google_cloud(cfg.get('dev_in'), cfg.get('gkey', ''), stop_event,
                             on_text, log_fn, cfg.get('google_model', 'default'))
        elif stt_type == 'mimo':
            stt_mimo(cfg.get('dev_in'), cfg.get('mimo_key', ''),
                     cfg.get('mimo_lang', 'zh'), stop_event, on_text, log_fn)
        elif stt_type == 'ptt':
            stt_ptt(
                cfg.get('dev_in'), cfg.get('ptt_rec', 'F8'), cfg.get('ptt_play', 'F9'),
                cfg.get('ptt_engine', 'whisper'), cfg, stop_event, on_text, log_fn,
                set_status,
            )
        else:
            log_fn(f"Unknown STT type: {stt_type}")
    except Exception as ex:
        log_fn(f'STT thread crashed: {ex}')
        log_fn(traceback.format_exc())
    if set_status:
        set_status('⏹ Stopped')


# ── TTS playback ────────────────────────────────────────

_tts_lock = threading.Lock()
_last_tts_file = None


def play_tts(text, cfg, log_fn):
    """Synthesize and play TTS based on config. Thread-safe."""
    global _last_tts_file
    with _tts_lock:
        try:
            tts_type = cfg.get('tts', 'edge')
            out_device = cfg.get('dev_out')
            if tts_type == 'gpt':
                ref = cfg.get('ref', '')
                if not ref or not os.path.exists(ref):
                    log_fn("ERROR: Reference audio not found")
                    return
                result = tts_gptsovits(
                    text, ref,
                    prompt_lang=cfg.get('prompt_lang', 'zh'),
                    prompt_text=cfg.get('prompt_text', ''),
                    text_lang=cfg.get('text_lang', 'zh'),
                    out_device=out_device,
                    base_url=cfg.get('gpt_url', 'http://127.0.0.1:9880'),
                    top_k=cfg.get('top_k', 20),
                    top_p=cfg.get('top_p', 0.6),
                    temperature=cfg.get('temperature', 0.6),
                    speed_factor=cfg.get('speed_factor', 1.0),
                    repetition_penalty=cfg.get('repetition_penalty', 1.35),
                    text_split_method=cfg.get('text_split_method', 'cut1'),
                )
            elif tts_type == 'edge':
                result = tts_edge(text, voice=cfg.get('voice', ''), out_device=out_device)
            elif tts_type == 'google':
                result = tts_google(text, voice=cfg.get('voice', ''),
                                    api_key=cfg.get('gkey', ''), out_device=out_device)
            elif tts_type == 'mimo':
                result = tts_mimo(text, voice=cfg.get('voice', 'mimo_default'),
                                  api_key=cfg.get('mimo_key', ''), out_device=out_device)
            else:
                return
            # Track last cache file
            cache_name = _TTS_CACHE_NAMES.get(tts_type)
            if cache_name:
                _last_tts_file = _find_tts_cache(cache_name)
            log_fn(result)
        except Exception as e:
            log_fn(f"TTS error: {e}")


def _find_tts_cache(filename):
    import tempfile
    path = os.path.join(tempfile.gettempdir(), filename)
    return path if os.path.exists(path) else None


def replay_cached_tts(cfg, log_fn):
    if _last_tts_file and os.path.exists(_last_tts_file):
        try:
            from sttts.audio import play_wav
            out_device = parse_device_index(cfg.get('DEV_OUT', ''))
            play_wav(_last_tts_file, out_device)
            log_fn("Replayed cached audio")
        except Exception as e:
            log_fn(f"Replay error: {e}")
    else:
        log_fn("No cached audio")


# ── Config builders ─────────────────────────────────────

def _resolve_tts(values):
    """Determine TTS engine from radio values."""
    for key, name in (('TTS_GPT', 'gpt'), ('TTS_GOOGLE', 'google'),
                      ('TTS_MIMO', 'mimo'), ('TTS_EDGE', 'edge')):
        if values.get(key):
            return name
    return 'edge'


def _resolve_stt(values):
    """Determine STT engine from radio values. Returns (stt_type, ptt_engine)."""
    if values.get('MODE_PTT'):
        ptt_engine = 'whisper'
        if values.get('STT_G'):
            ptt_engine = 'google_vad'
        elif values.get('STT_MIMO'):
            ptt_engine = 'mimo'
        return 'ptt', ptt_engine
    if values.get('STT_MIMO'):
        return 'mimo', None
    if values.get('STT_G'):
        stt = 'google_vad' if values.get('GM_VAD', True) else 'google_cloud'
        return stt, None
    return 'whisper', None


def build_run_cfg(values):
    """Build a runtime config dict from current GUI values."""
    stt, ptt_engine = _resolve_stt(values)
    tts = _resolve_tts(values)
    cfg = {
        'stt': stt,
        'tts': tts,
        'dev_in': parse_device_index(values.get('DEV_IN', '')),
        'ref': values.get('REF', ''),
        'prompt_text': values.get('PROMPT_TEXT', ''),
        'prompt_lang': values.get('PROMPT_LANG', 'zh'),
        'text_lang': values.get('TEXT_LANG', 'zh'),
        'top_k': values.get('TOP_K', 20),
        'top_p': values.get('TOP_P', 0.6),
        'temperature': values.get('TEMPERATURE', 0.6),
        'speed_factor': values.get('SPEED_FACTOR', 1.0),
        'repetition_penalty': values.get('REPETITION_PENALTY', 1.35),
        'text_split_method': values.get('TEXT_SPLIT', 'cut5'),
        'gkey': values.get('GKEY', ''),
        'voice': values.get('VOICE', ''),
        'gpt_url': values.get('GPT_URL', 'http://127.0.0.1:9880'),
        'whisper_model': values.get('WHISPER_MODEL', 'small'),
        'google_model': values.get('GOOGLE_MODEL', 'default'),
        'mimo_key': values.get('MIMO_KEY', ''),
        'mimo_lang': values.get('MIMO_LANG', 'zh'),
        'ptt_rec': values.get('PTT_REC', 'F8'),
        'ptt_play': values.get('PTT_PLAY', 'F9'),
    }
    if stt == 'ptt':
        cfg['ptt_engine'] = ptt_engine
    return cfg


def build_ptt_cfg(values):
    """Build a PTT config dict from current GUI values."""
    _, ptt_engine = _resolve_stt(values)
    return {
        'stt': 'ptt',
        'tts': _resolve_tts(values),
        'dev_in': parse_device_index(values.get('DEV_IN', '')),
        'dev_out': parse_device_index(values.get('DEV_OUT', '')),
        'ref': values.get('REF', ''),
        'prompt_text': values.get('PROMPT_TEXT', ''),
        'prompt_lang': values.get('PROMPT_LANG', 'zh'),
        'text_lang': values.get('TEXT_LANG', 'zh'),
        'top_k': values.get('TOP_K', 20),
        'top_p': values.get('TOP_P', 0.6),
        'temperature': values.get('TEMPERATURE', 0.6),
        'speed_factor': values.get('SPEED_FACTOR', 1.0),
        'repetition_penalty': values.get('REPETITION_PENALTY', 1.35),
        'text_split_method': values.get('TEXT_SPLIT', 'cut5'),
        'gkey': values.get('GKEY', ''),
        'voice': values.get('VOICE', ''),
        'gpt_url': values.get('GPT_URL', 'http://127.0.0.1:9880'),
        'whisper_model': values.get('WHISPER_MODEL', 'small'),
        'google_model': values.get('GOOGLE_MODEL', 'default'),
        'mimo_key': values.get('MIMO_KEY', ''),
        'mimo_lang': values.get('MIMO_LANG', 'zh'),
        'ptt_rec': values.get('PTT_REC', 'F8'),
        'ptt_play': values.get('PTT_PLAY', 'F9'),
        'ptt_engine': ptt_engine or 'whisper',
    }


# ── GUI visibility helpers ──────────────────────────────

def switch_stt_visibility(window, is_google, is_mimo):
    """Show/hide STT-related UI columns."""
    window['COL_GM'].update(visible=is_google)
    window['COL_MIMO'].update(visible=is_mimo)
    window['COL_GKEY'].update(visible=is_google)
    window['COL_WHISPER'].update(visible=not (is_google or is_mimo))


def switch_tts_visibility(window, tts_mode, voice_memory=None):
    """Show/hide TTS-related UI columns. Preserves voice selection per engine."""
    show_gpt = tts_mode == 'gpt'
    show_voice = tts_mode in ('edge', 'google', 'mimo')
    window['COL_REF'].update(visible=show_gpt)
    window['COL_GPT'].update(visible=show_gpt)
    window['VOICE'].update(visible=show_voice)
    if not show_voice:
        return

    voice_map = {
        'edge': (EDGE_VOICES, 'zh-CN-XiaoxiaoNeural'),
        'google': (GOOGLE_VOICES, 'Auto (default)'),
        'mimo': (MIMO_VOICES, 'mimo_default'),
    }
    voices, default = voice_map.get(tts_mode, (EDGE_VOICES, 'zh-CN-XiaoxiaoNeural'))
    # Restore last-used voice for this engine, or fall back to default
    value = default
    if voice_memory and tts_mode in voice_memory:
        value = voice_memory[tts_mode]
    window['VOICE'].update(values=voices, value=value, visible=True)


# ── Layout builder ──────────────────────────────────────

def create_layout(cfg_saved, in_devs, out_devs, def_dev_in, def_dev_out, default_ref):
    """Build the full GUI layout — compact two-column design."""

    # ── Left column: STT + Mode ──
    stt_col = sg.Column([
        [sg.Frame('STT Engine', [
            [sg.Radio('Whisper (local GPU)', 'STT', key='STT_W', default=True, enable_events=True),
             sg.Radio('Google Cloud STT', 'STT', key='STT_G', enable_events=True),
             sg.Radio('Xiaomi MiMo', 'STT', key='STT_MIMO', enable_events=True)],
            [sg.pin(sg.Col([
                [sg.Radio('Local VAD + REST', 'GOOGLE_MODE', key='GM_VAD', default=True),
                 sg.Radio('Cloud process (full audio)', 'GOOGLE_MODE', key='GM_CLOUD'),
                 sg.Text('  Model:'), sg.Combo(
                     ['default', 'command_and_search', 'phone_call', 'video'],
                     default_value=cfg_saved.get('GOOGLE_MODEL', 'default'),
                     key='GOOGLE_MODEL', size=(22, 1))],
            ], key='COL_GM', visible=False))],
            [sg.pin(sg.Col([
                [sg.Text('MiMo key:'), sg.Input(
                    key='MIMO_KEY', size=(35, 1),
                    default_text=cfg_saved.get('MIMO_KEY', ''), password_char='*'),
                 sg.Text('Lang:'), sg.Combo(
                     ['zh', 'en', 'auto'],
                     default_value=cfg_saved.get('MIMO_LANG', 'zh'),
                     key='MIMO_LANG', size=(6, 1))],
            ], key='COL_MIMO', visible=False))],
            [sg.pin(sg.Col([
                [sg.Text('Whisper model:'), sg.Combo(
                    ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
                     'medium', 'medium.en', 'large-v2'],
                    default_value=cfg_saved.get('WHISPER_MODEL', 'small'),
                    key='WHISPER_MODEL', size=(15, 1))],
            ], key='COL_WHISPER', visible=True)),
             sg.Text('Input:'), sg.Combo(
                in_devs, default_value=cfg_saved.get('DEV_IN', def_dev_in),
                key='DEV_IN', size=(30, 1))],
            [sg.pin(sg.Col([
                [sg.Text('Google key:'), sg.Input(
                    key='GKEY', size=(50, 1),
                    default_text=cfg_saved.get('GKEY', ''), password_char='*')],
            ], key='COL_GKEY', visible=False))],
        ])],
        [sg.Frame('Mode', [
            [sg.Radio('Continuous (VAD)', 'MODE', key='MODE_CONT', default=True, enable_events=True),
             sg.Radio('Push-to-Talk', 'MODE', key='MODE_PTT', enable_events=True)],
            [sg.pin(sg.Col([
                [sg.Text('Record key:'), sg.Input(
                    key='PTT_REC', size=(8, 1),
                    default_text=cfg_saved.get('PTT_REC', 'F8')),
                 sg.Text('Play key:'), sg.Input(
                     key='PTT_PLAY', size=(8, 1),
                     default_text=cfg_saved.get('PTT_PLAY', 'F9'))],
            ], key='COL_PTT', visible=False))],
        ])],
    ], vertical_alignment='top')

    # ── Right column: TTS + Audio Output ──
    tts_col = sg.Column([
        [sg.Frame('TTS Engine', [
            [sg.Radio('GPT-SoVITS', 'TTS', key='TTS_GPT', default=True, enable_events=True),
             sg.Radio('Edge-TTS', 'TTS', key='TTS_EDGE', enable_events=True),
             sg.Radio('Google TTS', 'TTS', key='TTS_GOOGLE', enable_events=True),
             sg.Radio('MiMo TTS', 'TTS', key='TTS_MIMO', enable_events=True)],
            [sg.pin(sg.Col([
                [sg.Text('Ref audio:'), sg.Input(
                    key='REF', size=(35, 1), default_text=default_ref),
                 sg.FileBrowse(file_types=(("WAV", "*.wav"),))],
                [sg.Text('Prompt text:'), sg.Input(
                    key='PROMPT_TEXT', size=(35, 1),
                    default_text=cfg_saved.get('PROMPT_TEXT', '')),
                 sg.Text('Ref lang:'), sg.Combo(
                    ['zh', 'en', 'ja'],
                    default_value=cfg_saved.get('PROMPT_LANG', 'zh'),
                    key='PROMPT_LANG', size=(5, 1))],
                [sg.Text('Text lang:'), sg.Combo(
                    ['zh', 'en', 'ja'],
                    default_value=cfg_saved.get('TEXT_LANG', 'zh'),
                    key='TEXT_LANG', size=(5, 1))],
            ], key='COL_REF', visible=True))],
            [sg.pin(sg.Col([
                [sg.Text('Model:'), sg.Combo(
                    list_gptsovits_models(),
                    default_value=get_active_gptsovits_model(),
                    key='GPT_MODEL', size=(15, 1), enable_events=True),
                 sg.Text('URL:'), sg.Input(
                    cfg_saved.get('GPT_URL', 'http://127.0.0.1:9880'),
                    key='GPT_URL', size=(18, 1))],
                [sg.pin(sg.Col([
                    [sg.Text('GPT .ckpt:'), sg.Input(
                        key='GPT_CKPT', size=(35, 1),
                        default_text=cfg_saved.get('GPT_CKPT', '')),
                     sg.FileBrowse(file_types=(("CKPT", "*.ckpt"),))],
                    [sg.Text('SoVITS .pth:'), sg.Input(
                        key='GPT_PTH', size=(35, 1),
                        default_text=cfg_saved.get('GPT_PTH', '')),
                     sg.FileBrowse(file_types=(("PTH", "*.pth"),))],
                    [sg.Button('Apply Custom Model', key='GPT_APPLY_CUSTOM', size=(20, 1))],
                ], key='COL_CUSTOM_MODEL', visible=False))],
                [sg.Text('Version:'), sg.Combo(
                    ['auto', 'v1', 'v2', 'v2Pro', 'v3', 'v4'],
                    default_value=cfg_saved.get('GPT_VERSION', 'auto'),
                    key='GPT_VERSION', size=(8, 1)),
                 sg.Text('top_k:'), sg.Input(
                    str(cfg_saved.get('TOP_K', 20)), key='TOP_K', size=(5, 1)),
                 sg.Text('top_p:'), sg.Input(
                    str(cfg_saved.get('TOP_P', 0.6)), key='TOP_P', size=(5, 1)),
                 sg.Text('temp:'), sg.Input(
                    str(cfg_saved.get('TEMPERATURE', 0.6)), key='TEMPERATURE', size=(5, 1)),
                 sg.Text('speed:'), sg.Input(
                    str(cfg_saved.get('SPEED_FACTOR', 1.0)), key='SPEED_FACTOR', size=(5, 1)),
                 sg.Text('rep_pen:'), sg.Input(
                    str(cfg_saved.get('REPETITION_PENALTY', 1.35)), key='REPETITION_PENALTY', size=(5, 1))],
                [sg.Button('▶ GPT-SoVITS', key='GPT_START', size=(14, 1)),
                 sg.Button('■ GPT-SoVITS', key='GPT_STOP', size=(14, 1),
                           disabled=True, button_color='red'),
                 sg.Text(check_gptsovits(), key='SRV', size=(16, 1))],
            ], key='COL_GPT', visible=True))],
            [sg.Combo(EDGE_VOICES, default_value='zh-CN-XiaoxiaoNeural',
                      key='VOICE', size=(30, 1))],
        ])],
        [sg.Frame('Audio Output', [
            [sg.Text('Play to:'), sg.Combo(
                out_devs, default_value=cfg_saved.get('DEV_OUT', def_dev_out),
                key='DEV_OUT', size=(45, 1))],
        ])],
    ], vertical_alignment='top')

    # ── Bottom: Control + Status merged ──
    ctrl = sg.Frame('Control', [
        [sg.Button('▶ Start', key='START', size=(10, 1)),
         sg.Button('■ Stop', key='STOP', size=(10, 1), disabled=True),
         sg.Button('Reset Config', key='RESET_CFG', size=(12, 1)),
         sg.Text('', size=(4, 1)),
         sg.Text('Idle', key='STATUS', size=(55, 1), text_color='blue')],
    ])

    # ── Bottom: Transcription + Log side by side ──
    panels = sg.Column([
        [sg.Frame('Transcription', [
            [sg.Multiline(size=(50, 4), key='TEXT', disabled=True, autoscroll=True)],
        ]),
         sg.Frame('Log', [
            [sg.Multiline(size=(50, 4), key='LOG', disabled=True, autoscroll=True)],
        ])],
    ], pad=(0, 0))

    return [
        [stt_col, tts_col],
        [ctrl],
        [panels],
    ]


# ── State restore ───────────────────────────────────────

def restore_state(window, cfg_saved):
    """Restore saved STT/TTS/Mode radio states and visibility."""
    # STT engine
    if cfg_saved.get('STT_G'):
        window['STT_G'].update(value=True)
        switch_stt_visibility(window, True, False)
        if cfg_saved.get('GM_CLOUD'):
            window['GM_CLOUD'].update(value=True)
    elif cfg_saved.get('STT_MIMO'):
        window['STT_MIMO'].update(value=True)
        switch_stt_visibility(window, False, True)

    # Mode
    if cfg_saved.get('MODE_PTT'):
        window['MODE_PTT'].update(value=True)
        window['COL_PTT'].update(visible=True)

    # TTS engine
    if cfg_saved.get('TTS_EDGE'):
        window['TTS_EDGE'].update(value=True)
        switch_tts_visibility(window, 'edge')
    elif cfg_saved.get('TTS_GOOGLE'):
        window['TTS_GOOGLE'].update(value=True)
        switch_tts_visibility(window, 'google')
    elif cfg_saved.get('TTS_MIMO'):
        window['TTS_MIMO'].update(value=True)
        switch_tts_visibility(window, 'mimo')
    else:
        switch_tts_visibility(window, 'gpt')


# ── Main ────────────────────────────────────────────────

def main():
    cfg_saved = load_config()
    in_devs = list_devices('input')
    out_devs = list_devices('output')
    def_dev_in = in_devs[0] if in_devs else ''
    def_dev_out = out_devs[0] if out_devs else ''
    default_ref = cfg_saved.get('REF', '') or get_default_ref_audio()

    layout = create_layout(cfg_saved, in_devs, out_devs, def_dev_in, def_dev_out, default_ref)
    window = sg.Window('STT → TTS', layout, finalize=True)

    # ── State ────────────────────────────────────────────
    stop_event = threading.Event()
    gpt_manager = GPTSoVITSManager()
    last_values = dict(cfg_saved)  # copy — updated every event loop iteration

    # Bounded line buffers for Log / Transcription (avoid O(n²) string concat)
    log_buf = collections.deque(maxlen=_LOG_MAX_LINES)
    text_buf = collections.deque(maxlen=_LOG_MAX_LINES)
    # Dirty flags: set from any thread, flushed in the event loop
    log_dirty = threading.Event()
    text_dirty = threading.Event()

    # Voice memory: {engine_name: last_voice_value}
    voice_memory = {}
    _current_tts = 'gpt'  # track active TTS for voice memory

    # Server status check throttle
    _srv_tick = 0

    # Register cleanup
    def cleanup():
        stop_event.set()
        gpt_manager.cleanup()

    atexit.register(cleanup)

    # ── Helper functions (thread-safe) ───────────────────

    def log(msg):
        """Append message to the Log buffer. Thread-safe."""
        log_buf.append(str(msg))
        log_dirty.set()

    def handle_text(text, cfg, replay=False):
        """Handle recognized text: display and play TTS. Thread-safe."""
        text_buf.append(text)
        text_dirty.set()
        if replay:
            log(f"Replaying: {text}")
            threading.Thread(target=replay_cached_tts, args=(cfg, log), daemon=True).start()
        else:
            log(f"Recognized: {text}")
            threading.Thread(target=play_tts, args=(text, cfg, log), daemon=True).start()

    def flush_buffers():
        """Push dirty buffers to the GUI. Called from event loop only."""
        if log_dirty.is_set():
            window['LOG'].update('\n'.join(log_buf) + '\n')
            log_dirty.clear()
        if text_dirty.is_set():
            window['TEXT'].update('\n'.join(text_buf) + '\n')
            text_dirty.clear()

    def update_srv_status(gpt_url):
        """Update GPT-SoVITS server status indicator."""
        st = check_gptsovits(gpt_url)
        window['SRV'].update(st)
        return st.startswith('✅')  # FIXED: was `'✅' in st`

    def start_stt(cfg):
        """Start STT in a background thread."""
        def _run():
            run_stt(
                cfg, stop_event,
                lambda text, **kw: handle_text(text, cfg, **kw),
                log,
                lambda m: window.write_event_value('_STATUS', m),
            )
            window.write_event_value('_STATUS', '⏹ Stopped')

        threading.Thread(target=_run, daemon=True).start()

    def do_start(values):
        """Validate and start STT. Returns True on success."""
        cfg = build_run_cfg(values)
        # Validation
        if cfg['stt'] == 'mimo' and not cfg.get('mimo_key'):
            window['STATUS'].update('❌ MiMo API key required', text_color='red')
            log('ERROR: MiMo API key required')
            return False
        if cfg['stt'].startswith('google') and (not cfg.get('gkey') or 'YOUR_' in cfg.get('gkey', '')):
            window['STATUS'].update('❌ Google API key required', text_color='red')
            log('ERROR: Google API key required for Google STT')
            return False
        if cfg['tts'] == 'google' and (not cfg.get('gkey') or 'YOUR_' in cfg.get('gkey', '')):
            window['STATUS'].update('❌ Google API key required', text_color='red')
            log('ERROR: Google API key required for Google TTS')
            return False
        if cfg['tts'] == 'gpt' and not os.path.exists(cfg.get('ref', '')):
            window['STATUS'].update('❌ Ref audio not found', text_color='red')
            log(f'ERROR: Ref audio not found — {cfg.get("ref", "")}')
            log('HINT: Switch TTS to Edge-TTS (no ref audio needed) or place a .wav file')
            return False
        # Validate PTT keys
        if cfg.get('stt') == 'ptt':
            try:
                from sttts.stt import resolve_vk as _rvk
                for _k, _label in [(cfg.get('ptt_rec', ''), 'Record'),
                                   (cfg.get('ptt_play', ''), 'Play')]:
                    _rvk(_k)
            except ValueError:
                window['STATUS'].update(f'❌ Invalid PTT key: "{_k}"', text_color='red')
                log(f'ERROR: {_label} key "{_k}" is not a valid key name. '
                    f'Use names like "f8", "f9", "space", "a", etc.')
                return False

        stop_event.clear()
        window['START'].update(disabled=True)
        window['STOP'].update(disabled=False)
        text_buf.clear()
        log_buf.clear()
        text_dirty.set()
        log_dirty.set()
        window['STATUS'].update('🟢 Running...', text_color='green')
        save_config(values)
        start_stt(cfg)
        return True

    def do_stop():
        """Stop STT."""
        stop_event.set()
        window['START'].update(disabled=False)
        window['STOP'].update(disabled=True)
        window['STATUS'].update('⏹ Stopped', text_color='blue')

    # ── Restore saved settings ───────────────────────────
    restore_state(window, cfg_saved)

    # Auto-start PTT if was active last session
    # Use values from the restored GUI widgets (not raw cfg_saved keys)
    if cfg_saved.get('MODE_PTT'):
        window['START'].update(disabled=True)
        window['STOP'].update(disabled=False)
        window['STATUS'].update('🟢 PTT listening...', text_color='green')
        # Fire event so it runs in the event loop with correct values
        window.write_event_value('_AUTO_PTT', None)

    # ── Event loop ───────────────────────────────────────
    while True:
        event, values = window.read(timeout=200)

        if event == sg.WINDOW_CLOSED:
            stop_event.set()
            gpt_manager.stop()
            break

        # Always track last_values for save on exit
        if values:
            last_values = values

        # Flush line buffers to GUI
        flush_buffers()

        # ── STT engine switch ────────────────────────────
        if event in ('STT_W', 'STT_G', 'STT_MIMO'):
            try:
                switch_stt_visibility(window, values['STT_G'], values['STT_MIMO'])
                save_config(values)
            except Exception:
                pass

        # ── Mode switch ──────────────────────────────────
        if event in ('MODE_CONT', 'MODE_PTT'):
            is_ptt = values['MODE_PTT']
            window['COL_PTT'].update(visible=is_ptt)
            save_config(values)

            if is_ptt and not stop_event.is_set():
                ptt_cfg = build_ptt_cfg(values)
                window['START'].update(disabled=True)
                window['STOP'].update(disabled=False)
                text_buf.clear()
                log_buf.clear()
                text_dirty.set()
                log_dirty.set()
                window['STATUS'].update('🟢 PTT listening...', text_color='green')
                stop_event.clear()
                start_stt(ptt_cfg)
            elif not is_ptt:
                stop_event.set()
                window['START'].update(disabled=False)
                window['STOP'].update(disabled=True)
                window['STATUS'].update('⏹ Stopped', text_color='blue')

        # ── TTS engine switch ────────────────────────────
        if event in ('TTS_GPT', 'TTS_EDGE', 'TTS_GOOGLE', 'TTS_MIMO'):
            try:
                # Save current voice before switching
                if _current_tts in ('edge', 'google', 'mimo'):
                    cur_voice = values.get('VOICE', '')
                    if cur_voice:
                        voice_memory[_current_tts] = cur_voice

                if values['TTS_GPT']:
                    _current_tts = 'gpt'
                    switch_tts_visibility(window, 'gpt', voice_memory)
                elif values['TTS_EDGE']:
                    _current_tts = 'edge'
                    switch_tts_visibility(window, 'edge', voice_memory)
                elif values['TTS_GOOGLE']:
                    _current_tts = 'google'
                    switch_tts_visibility(window, 'google', voice_memory)
                elif values['TTS_MIMO']:
                    _current_tts = 'mimo'
                    switch_tts_visibility(window, 'mimo', voice_memory)
                save_config(values)
            except Exception:
                pass

        # ── Periodic server status check (throttled) ─────
        if event == '__TIMEOUT__':
            _srv_tick += 1
            if _srv_tick * 200 >= _SRV_CHECK_INTERVAL:
                _srv_tick = 0
                gpt_url = values.get('GPT_URL', 'http://127.0.0.1:9880')
                now_running = update_srv_status(gpt_url)
                window['GPT_START'].update(disabled=now_running)
                window['GPT_STOP'].update(disabled=not now_running)
            continue

        # ── GPT-SoVITS control ───────────────────────────
        if event == 'GPT_START':
            ok, msg = gpt_manager.start()
            log(msg)

        elif event == 'GPT_STOP':
            log(gpt_manager.stop())
            update_srv_status(values.get('GPT_URL', 'http://127.0.0.1:9880'))

        elif event == 'GPT_APPLY_CUSTOM':
            ckpt = values.get('GPT_CKPT', '')
            pth = values.get('GPT_PTH', '')
            ver = values.get('GPT_VERSION', 'auto')
            if ver == 'auto':
                ver = 'v1' if 'v1' not in pth.lower() and 'v2' not in pth.lower() else ('v2' if 'v2' in pth.lower() else 'v1')
            if ckpt and pth:
                ok, msg = switch_gptsovits_model('custom', ckpt_path=ckpt, pth_path=pth, version=ver)
                log(msg)
            else:
                log('ERROR: Select both .ckpt and .pth files first')

        elif event == 'GPT_MODEL':
            model_name = values.get('GPT_MODEL', '')
            if model_name:
                is_custom = (model_name == 'custom')
                window['COL_CUSTOM_MODEL'].update(visible=is_custom)
                if not is_custom:
                    ok, msg = switch_gptsovits_model(model_name)
                    log(msg)

        # ── Start/Stop ───────────────────────────────────
        if event == 'START':
            try:
                do_start(values)
            except Exception as ex:
                window['STATUS'].update(f'❌ Start failed: {ex}', text_color='red')
                log(f'ERROR on start: {ex}')
                log(traceback.format_exc())

        elif event == 'STOP':
            do_stop()

        # ── Reset config ─────────────────────────────────
        elif event == 'RESET_CFG':
            reset_config()
            log('Config reset to defaults. Restart to apply.')

        # ── Auto-start PTT ───────────────────────────────
        elif event == '_AUTO_PTT':
            # Use current GUI values (widgets already restored)
            ptt_cfg = build_ptt_cfg(values)
            stop_event.clear()
            start_stt(ptt_cfg)

        # ── Async events from threads ────────────────────
        elif event == '_STATUS':
            window['STATUS'].update(values[event], text_color='blue')

    # ── Cleanup on exit ──────────────────────────────────
    save_config(last_values)
    gpt_manager.stop()
    window.close()


if __name__ == '__main__':
    # Auto-elevate to admin if needed (required for global keyboard hooks)
    import ctypes as _ct
    try:
        _is_admin = _ct.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        _is_admin = False
    if not _is_admin:
        _script = os.path.abspath(sys.argv[0])
        _cwd = os.path.dirname(_script)
        _pythonw = sys.executable.replace('python.exe', 'pythonw.exe')
        if not os.path.exists(_pythonw):
            _pythonw = sys.executable  # fallback
        _ct.windll.shell32.ShellExecuteW(
            None, "runas", _pythonw,
            f'"{_script}"', _cwd, 0,
        )
        sys.exit(0)

    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            ctypes.windll.user32.MessageBoxW(0, f"STTTS GUI crashed:\n\n{tb}", "STTTS Error", 0x10)
        except Exception:
            print(tb, file=sys.stderr)
            sys.exit(1)
