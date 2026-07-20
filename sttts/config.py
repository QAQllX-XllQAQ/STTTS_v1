"""Configuration management: load/save settings from config.json."""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)) or '.', 'config.json')

# Keys that should be persisted
PERSIST_KEYS = [
    'GKEY', 'WHISPER_MODEL', 'GOOGLE_MODEL', 'MIMO_KEY', 'MIMO_LANG',
    'VOICE', 'REF', 'GPT_URL', 'DEV_IN', 'DEV_OUT',
    'STT_W', 'STT_G', 'STT_MIMO',
    'TTS_GPT', 'TTS_EDGE', 'TTS_GOOGLE',
    'GM_VAD', 'GM_CLOUD', 'MODE_CONT', 'MODE_PTT',
    'PTT_REC', 'PTT_PLAY',
]


def load_config():
    """Load configuration from config.json. Returns dict (empty on error)."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to load config: {e}")
    return {}


def save_config(values):
    """Save relevant keys from values dict to config.json."""
    cfg = {}
    for k in PERSIST_KEYS:
        v = values.get(k)
        if v is not None:
            cfg[k] = v
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Warning: Failed to save config: {e}")


def get_default_ref_audio():
    """Get default reference audio path (relative to package root)."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)) or '.', 'ref_audio.wav')
