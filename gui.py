"""STTTS GUI — Desktop interface for Speech-to-Text → Text-to-Speech.

Features:
- STT: Whisper (local GPU) / Google Cloud STT / Xiaomi MiMo ASR
- TTS: GPT-SoVITS / Edge-TTS / Google Cloud TTS
- Modes: Continuous (VAD) / Push-to-Talk
- Audio output to any device (headphones, speakers, Voicemeeter, etc.)
"""

import atexit
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
from sttts.config import load_config, save_config, get_default_ref_audio
from sttts.tts import (
    tts_gptsovits, tts_edge, tts_google, check_gptsovits,
    EDGE_VOICES, GOOGLE_VOICES,
)
from sttts.stt import stt_whisper, stt_google_vad, stt_google_cloud, stt_mimo, stt_ptt

sg.theme("LightBlue3")


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

    def start(self, cwd=None):
        """Start the GPT-SoVITS API server. Returns (success, message)."""
        if self.process and self.process.poll() is None:
            return False, "GPT-SoVITS already running"
        try:
            self.process = subprocess.Popen(
                self.GPT_CMD,
                cwd=cwd or os.path.join(os.path.dirname(__file__) or '.', 'GPT-SoVITS'),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"GPT-SoVITS starting (PID {self.process.pid}), wait ~40s..."
        except Exception as e:
            return False, f"Failed to start GPT-SoVITS: {e}"

    def stop(self):
        """Stop the GPT-SoVITS API server. Returns message."""
        if not self.process or self.process.poll() is not None:
            self.process = None
            return "GPT-SoVITS not running"
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
            return "GPT-SoVITS stopped"
        except Exception as e:
            self.process = None
            return f"Error stopping GPT-SoVITS: {e}"

    def is_running(self):
        """Check if the process is still alive."""
        return self.process is not None and self.process.poll() is None

    def cleanup(self):
        """Force-kill on exit."""
        if self.process and self.process.poll() is not None:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None


# ── STT runner ──────────────────────────────────────────

def run_stt(cfg, stop_event, on_text, log_fn, set_status=None):
    """Run the appropriate STT engine based on config.

    Args:
        cfg: Config dict with STT settings.
        stop_event: threading.Event to signal stop.
        on_text: Callback for recognized text.
        log_fn: Logging callback.
        set_status: Optional status callback.
    """
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

tts_lock = threading.Lock()


def play_tts(text, cfg, log_fn):
    """Synthesize and play TTS based on config. Thread-safe."""
    with tts_lock:
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
                    out_device=out_device,
                    base_url=cfg.get('gpt_url', 'http://127.0.0.1:9880'),
                )
            elif tts_type == 'edge':
                result = tts_edge(text, voice=cfg.get('voice', ''), out_device=out_device)
            elif tts_type == 'google':
                result = tts_google(text, voice=cfg.get('voice', ''),
                                    api_key=cfg.get('gkey', ''), out_device=out_device)
            else:
                return
            log_fn(result)
        except Exception as e:
            log_fn(f"TTS error: {e}")


# ── Build PTT config from GUI values ────────────────────

def build_ptt_cfg(values):
    """Build a PTT config dict from current GUI values."""
    ptt_engine = 'whisper'
    if values.get('STT_G'):
        ptt_engine = 'google_vad'
    elif values.get('STT_MIMO'):
        ptt_engine = 'mimo'

    return {
        'stt': 'ptt',
        'tts': 'gpt' if values.get('TTS_GPT') else ('google' if values.get('TTS_GOOGLE') else 'edge'),
        'dev_in': parse_device_index(values.get('DEV_IN', '')),
        'dev_out': parse_device_index(values.get('DEV_OUT', '')),
        'ref': values.get('REF', ''),
        'gkey': values.get('GKEY', ''),
        'voice': values.get('VOICE', ''),
        'gpt_url': values.get('GPT_URL', 'http://127.0.0.1:9880'),
        'whisper_model': values.get('WHISPER_MODEL', 'small'),
        'google_model': values.get('GOOGLE_MODEL', 'default'),
        'mimo_key': values.get('MIMO_KEY', ''),
        'mimo_lang': values.get('MIMO_LANG', 'zh'),
        'ptt_rec': values.get('PTT_REC', 'F8'),
        'ptt_play': values.get('PTT_PLAY', 'F9'),
        'ptt_engine': ptt_engine,
    }


def build_run_cfg(values):
    """Build a runtime config dict from current GUI values."""
    if values.get('MODE_PTT'):
        stt = 'ptt'
        ptt_engine = 'whisper'
        if values.get('STT_G'):
            ptt_engine = 'google_vad'
        elif values.get('STT_MIMO'):
            ptt_engine = 'mimo'
    elif values.get('STT_MIMO'):
        stt = 'mimo'
    elif values.get('STT_G'):
        stt = 'google_vad' if values.get('GM_VAD', True) else 'google_cloud'
    else:
        stt = 'whisper'

    tts = 'gpt' if values.get('TTS_GPT') else ('google' if values.get('TTS_GOOGLE') else 'edge')

    cfg = {
        'stt': stt,
        'tts': tts,
        'dev_in': parse_device_index(values.get('DEV_IN', '')),
        'dev_out': parse_device_index(values.get('DEV_OUT', '')),
        'ref': values.get('REF', ''),
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


# ── GUI visibility helpers ──────────────────────────────

def switch_stt_visibility(window, is_google, is_mimo):
    """Show/hide STT-related UI columns."""
    window['COL_GM'].update(visible=is_google)
    window['COL_MIMO'].update(visible=is_mimo)
    window['COL_GKEY'].update(visible=is_google)
    window['COL_WHISPER'].update(visible=not (is_google or is_mimo))


def switch_tts_visibility(window, tts_mode):
    """Show/hide TTS-related UI columns."""
    show_gpt = tts_mode == 'gpt'
    window['COL_REF'].update(visible=show_gpt)
    window['COL_GPT'].update(visible=show_gpt)
    if tts_mode == 'edge':
        window['VOICE'].update(values=EDGE_VOICES, value='zh-CN-XiaoxiaoNeural')
    elif tts_mode == 'google':
        window['VOICE'].update(values=GOOGLE_VOICES, value='Auto (default)')
    else:
        window['VOICE'].update(values=[], value='')


# ── Main ────────────────────────────────────────────────

def main():
    cfg_saved = load_config()
    in_devs = list_devices('input')
    out_devs = list_devices('output')
    def_dev_in = in_devs[0] if in_devs else ''
    def_dev_out = out_devs[0] if out_devs else ''
    default_ref = cfg_saved.get('REF', get_default_ref_audio())

    # ── Layout ───────────────────────────────────────────
    layout = [
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
             sg.Text('Input device:'), sg.Combo(
                 in_devs, default_value=cfg_saved.get('DEV_IN', def_dev_in),
                 key='DEV_IN', size=(40, 1))],
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
        [sg.Frame('TTS Engine', [
            [sg.Radio('GPT-SoVITS', 'TTS', key='TTS_GPT', default=True, enable_events=True),
             sg.Radio('Edge-TTS', 'TTS', key='TTS_EDGE', enable_events=True),
             sg.Radio('Google TTS', 'TTS', key='TTS_GOOGLE', enable_events=True)],
            [sg.pin(sg.Col([
                [sg.Text('Ref audio:'), sg.Input(
                    key='REF', size=(42, 1), default_text=default_ref),
                 sg.FileBrowse(file_types=(("WAV", "*.wav"),))],
            ], key='COL_REF', visible=True))],
            [sg.pin(sg.Col([
                [sg.Text('GPT URL:'), sg.Input(
                    cfg_saved.get('GPT_URL', 'http://127.0.0.1:9880'),
                    key='GPT_URL', size=(35, 1))],
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
                key='DEV_OUT', size=(50, 1))],
        ])],
        [sg.Frame('Control', [
            [sg.Button('▶ Start', key='START', size=(10, 1)),
             sg.Button('■ Stop', key='STOP', size=(10, 1), disabled=True)],
        ])],
        [sg.Frame('Status', [
            [sg.Text('Idle', key='STATUS', size=(75, 1), text_color='blue')],
        ])],
        [sg.Frame('Transcription', [
            [sg.Multiline(size=(85, 4), key='TEXT', disabled=True, autoscroll=True)],
        ])],
        [sg.Frame('Log', [
            [sg.Multiline(size=(85, 8), key='LOG', disabled=True, autoscroll=True)],
        ])],
    ]

    window = sg.Window('STT → TTS', layout, finalize=True)

    # ── State ────────────────────────────────────────────
    stop_event = threading.Event()
    gpt_manager = GPTSoVITSManager()
    last_values = cfg_saved.copy()  # Initialize with saved config to avoid NameError

    # Register cleanup
    def cleanup():
        stop_event.set()
        gpt_manager.cleanup()

    atexit.register(cleanup)

    # ── Helper functions ─────────────────────────────────

    def log(msg):
        """Append message to the Log multiline."""
        try:
            window.write_event_value('_LOG', str(msg))
        except Exception:
            pass

    def handle_text(text, cfg):
        """Handle recognized text: display and play TTS."""
        window.write_event_value('_TEXT', text)
        log(f"Recognized: {text}")
        threading.Thread(target=play_tts, args=(text, cfg, log), daemon=True).start()

    def update_srv_status(gpt_url):
        """Update GPT-SoVITS server status indicator."""
        st = check_gptsovits(gpt_url)
        window['SRV'].update(st)
        return '✅' in st

    def start_stt(cfg):
        """Start STT in a background thread."""
        def _run():
            run_stt(
                cfg, stop_event,
                lambda text: handle_text(text, cfg),
                log,
                lambda m: window.write_event_value('_STATUS', m),
            )
            window.write_event_value('_STATUS', '⏹ Stopped')

        threading.Thread(target=_run, daemon=True).start()

    # ── Restore saved settings ───────────────────────────
    if cfg_saved.get('STT_G'):
        window['STT_G'].update(value=True)
        switch_stt_visibility(window, True, False)
        if cfg_saved.get('GM_CLOUD'):
            window['GM_CLOUD'].update(value=True)
    elif cfg_saved.get('STT_MIMO'):
        window['STT_MIMO'].update(value=True)
        switch_stt_visibility(window, False, True)

    if cfg_saved.get('MODE_PTT'):
        window['MODE_PTT'].update(value=True)
        window['COL_PTT'].update(visible=True)

    if cfg_saved.get('TTS_EDGE'):
        window['TTS_EDGE'].update(value=True)
        switch_tts_visibility(window, 'edge')
    elif cfg_saved.get('TTS_GOOGLE'):
        window['TTS_GOOGLE'].update(value=True)
        switch_tts_visibility(window, 'google')
    else:
        switch_tts_visibility(window, 'gpt')

    # Auto-start PTT if was active last session
    if cfg_saved.get('MODE_PTT'):
        ptt_cfg = build_ptt_cfg(cfg_saved)
        window['START'].update(disabled=True)
        window['STOP'].update(disabled=False)
        window['STATUS'].update('🟢 PTT listening...', text_color='green')
        window.write_event_value('_AUTO_PTT', ptt_cfg)

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
                window['TEXT'].update('')
                window['LOG'].update('')
                window['STATUS'].update('🟢 PTT listening...', text_color='green')
                stop_event.clear()
                start_stt(ptt_cfg)
            elif not is_ptt:
                stop_event.set()
                window['START'].update(disabled=False)
                window['STOP'].update(disabled=True)
                window['STATUS'].update('⏹ Stopped', text_color='blue')

        # ── TTS engine switch ────────────────────────────
        if event in ('TTS_GPT', 'TTS_EDGE', 'TTS_GOOGLE'):
            try:
                if values['TTS_GPT']:
                    switch_tts_visibility(window, 'gpt')
                elif values['TTS_EDGE']:
                    switch_tts_visibility(window, 'edge')
                elif values['TTS_GOOGLE']:
                    switch_tts_visibility(window, 'google')
                save_config(values)
            except Exception:
                pass

        # ── Periodic update ──────────────────────────────
        if event == '__TIMEOUT__':
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

        # ── Start/Stop ───────────────────────────────────
        if event == 'START':
            try:
                cfg = build_run_cfg(values)
                # Validation
                if cfg['stt'] == 'mimo' and not cfg.get('mimo_key'):
                    window['STATUS'].update('❌ MiMo API key required', text_color='red')
                    log('ERROR: MiMo API key required')
                    continue
                if cfg['stt'].startswith('google') and (not cfg.get('gkey') or 'YOUR_' in cfg.get('gkey', '')):
                    window['STATUS'].update('❌ Google API key required', text_color='red')
                    log('ERROR: Google API key required for Google STT')
                    continue
                if cfg['tts'] == 'google' and (not cfg.get('gkey') or 'YOUR_' in cfg.get('gkey', '')):
                    window['STATUS'].update('❌ Google API key required', text_color='red')
                    log('ERROR: Google API key required for Google TTS')
                    continue
                if cfg['tts'] == 'gpt' and not os.path.exists(cfg.get('ref', '')):
                    window['STATUS'].update(f'❌ Ref audio not found', text_color='red')
                    log(f'ERROR: Ref audio not found — {cfg.get("ref", "")}')
                    log('HINT: Switch TTS to Edge-TTS (no ref audio needed) or place a .wav file')
                    continue
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
                        continue

                stop_event.clear()
                window['START'].update(disabled=True)
                window['STOP'].update(disabled=False)
                window['TEXT'].update('')
                window['LOG'].update('')
                window['STATUS'].update('🟢 Running...', text_color='green')
                save_config(values)
                start_stt(cfg)
            except Exception as ex:
                window['STATUS'].update(f'❌ Start failed: {ex}', text_color='red')
                log(f'ERROR on start: {ex}')
                log(traceback.format_exc())

        elif event == 'STOP':
            stop_event.set()
            window['START'].update(disabled=False)
            window['STOP'].update(disabled=True)
            window['STATUS'].update('⏹ Stopped', text_color='blue')

        # ── Async events from threads ────────────────────
        elif event == '_TEXT':
            window['TEXT'].update(window['TEXT'].get() + values[event] + '\n')
        elif event == '_LOG':
            window['LOG'].update(window['LOG'].get() + values[event] + '\n')
        elif event == '_STATUS':
            window['STATUS'].update(values[event], text_color='blue')
        elif event == '_AUTO_PTT':
            start_stt(values[event])

    # ── Cleanup on exit ──────────────────────────────────
    save_config(last_values)
    gpt_manager.stop()
    window.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            ctypes.windll.user32.MessageBoxW(0, f"STTTS GUI crashed:\n\n{tb}", "STTTS Error", 0x10)
        except Exception:
            print(tb, file=sys.stderr)
            sys.exit(1)
