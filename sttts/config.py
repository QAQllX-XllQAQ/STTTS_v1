"""Configuration management: load/save settings from config.json."""

import json
import os
import tempfile
import shutil

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)) or '.', 'config.json')

# ── All persistable keys with their defaults ─────────────────
# Format: key -> default_value
# Radio keys store True/False; others store their actual value.
CONFIG_DEFAULTS = {
    # STT engine selection (Radios)
    'STT_W': True,
    'STT_G': False,
    'STT_MIMO': False,
    # Google sub-mode
    'GM_VAD': True,
    'GM_CLOUD': False,
    'GOOGLE_MODEL': 'default',
    # MiMo STT
    'MIMO_KEY': '',
    'MIMO_LANG': 'zh',
    # Whisper
    'WHISPER_MODEL': 'small',
    # Mode
    'MODE_CONT': True,
    'MODE_PTT': False,
    'PTT_REC': 'F8',
    'PTT_PLAY': 'F9',
    # TTS engine selection (Radios)
    'TTS_GPT': True,
    'TTS_EDGE': False,
    'TTS_GOOGLE': False,
    'TTS_MIMO': False,
    # GPT-SoVITS
    'REF': '',
    'PROMPT_TEXT': '',
    'PROMPT_LANG': 'zh',
    'TEXT_LANG': 'zh',
    'GPT_CKPT': '',
    'GPT_PTH': '',
    'GPT_VERSION': 'auto',
    'TOP_K': 20,
    'TOP_P': 0.6,
    'TEMPERATURE': 0.6,
    'SPEED_FACTOR': 1.0,
    'TEXT_SPLIT': 'cut1',
    'REPETITION_PENALTY': 1.35,
    'GKEY': '',
    'MIMO_KEY_TTS': '',  # separate from STT key
    'VOICE': '',
    # Audio devices
    'DEV_IN': '',
    'DEV_OUT': '',
}

# Keys that are boolean (Radio buttons)
_BOOL_KEYS = {k for k, v in CONFIG_DEFAULTS.items() if isinstance(v, bool)}

PERSIST_KEYS = list(CONFIG_DEFAULTS.keys())


def load_config():
    """Load configuration from config.json, merged with defaults.
    Returns a complete dict — every key in CONFIG_DEFAULTS is guaranteed present.
    """
    saved = {}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                saved = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load config: {e}")
    # Merge: saved overrides defaults
    cfg = dict(CONFIG_DEFAULTS)
    cfg.update(saved)
    return cfg


def save_config(values):
    """Save relevant keys from values dict to config.json (atomic write).

    Only keys in PERSIST_KEYS are persisted. Writes to a temp file first,
    then renames — avoids data loss on crash.
    """
    cfg = {}
    for k in PERSIST_KEYS:
        v = values.get(k)
        if v is not None:
            cfg[k] = v
    try:
        dir_ = os.path.dirname(CONFIG_PATH) or '.'
        fd, tmp = tempfile.mkstemp(suffix='.json', dir=dir_)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            shutil.move(tmp, CONFIG_PATH)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as e:
        print(f"Warning: Failed to save config: {e}")


def reset_config():
    """Reset config.json to defaults."""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(dict(CONFIG_DEFAULTS), f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Warning: Failed to reset config: {e}")


def get_default_ref_audio():
    """Get default reference audio path (relative to package root)."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)) or '.', 'ref_audio.wav')
