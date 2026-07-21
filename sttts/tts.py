"""TTS engines: GPT-SoVITS, Edge-TTS, Google Cloud TTS, MiMo TTS."""

import base64
import os
import tempfile

import requests

from .audio import play_wav


# ── Voice presets ────────────────────────────────────────

EDGE_VOICES = [
    'zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural', 'zh-CN-YunjianNeural',
    'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural', 'zh-CN-XiaohanNeural',
]

GOOGLE_VOICES = [
    'Auto (default)',
    'zh-CN-Neural2-A', 'zh-CN-Neural2-B', 'zh-CN-Neural2-C', 'zh-CN-Neural2-D',
    'zh-CN-Standard-A', 'zh-CN-Standard-B', 'zh-CN-Standard-C', 'zh-CN-Standard-D',
    'zh-CN-Studio-A', 'zh-CN-Studio-B', 'zh-CN-Studio-C',
]

# MiMo TTS voices
MIMO_VOICES = [
    'mimo_default', '冰糖', '茉莉', '苏打', '白桦',
    'Mia', 'Chloe', 'Milo', 'Dean',
]


# ── GPT-SoVITS ───────────────────────────────────────────

def tts_gptsovits(text, ref_audio, prompt_lang='ja', prompt_text='',
                  out_device=None, base_url='http://127.0.0.1:9880'):
    """Synthesize speech via GPT-SoVITS API and play it.

    Args:
        text: Text to synthesize.
        ref_audio: Path to reference audio WAV file.
        prompt_lang: Language of the reference audio prompt.
        prompt_text: Text content of the reference audio (optional).
        out_device: Output device index (None = default).
        base_url: GPT-SoVITS API base URL.

    Returns:
        Status string describing the result.

    Raises:
        RuntimeError: If TTS request fails.
    """
    if not text:
        raise ValueError("Empty text")
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    url = f'{base_url}/tts'
    params = {
        'text': text,
        'text_lang': 'zh',
        'ref_audio_path': ref_audio,
        'prompt_lang': prompt_lang,
        'prompt_text': prompt_text,
        'text_split_method': 'cut5',
        'batch_size': 1,
        'media_type': 'wav',
        'streaming_mode': False,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"GPT-SoVITS HTTP {resp.status_code}: {resp.text[:200]}")
    if len(resp.content) < 1000:
        raise RuntimeError(f"GPT-SoVITS returned too little data ({len(resp.content)} bytes)")

    tmp = os.path.join(tempfile.gettempdir(), 'gptsovits_tmp.wav')
    with open(tmp, 'wb') as f:
        f.write(resp.content)
    play_wav(tmp, out_device)
    return f"GPT-SoVITS OK ({len(resp.content) // 1024}KB)"


# ── Edge-TTS ─────────────────────────────────────────────

def tts_edge(text, voice='zh-CN-XiaoxiaoNeural', out_device=None):
    """Synthesize speech via Microsoft Edge TTS and play it.

    Args:
        text: Text to synthesize.
        voice: Edge TTS voice name.
        out_device: Output device index (None = default).

    Returns:
        Status string.

    Raises:
        RuntimeError: If TTS fails.
    """
    import edge_tts
    import asyncio

    if not text:
        raise ValueError("Empty text")

    tmp = os.path.join(tempfile.gettempdir(), 'edge_tts_tmp.mp3')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(edge_tts.Communicate(text, voice).save(tmp))
    finally:
        loop.close()

    play_wav(tmp, out_device)
    return "Edge-TTS OK"


# ── Google Cloud TTS ─────────────────────────────────────

def tts_google(text, voice='Auto (default)', api_key='', out_device=None):
    """Synthesize speech via Google Cloud TTS and play it.

    Args:
        text: Text to synthesize.
        voice: Google TTS voice name.
        api_key: Google Cloud API key.
        out_device: Output device index (None = default).

    Returns:
        Status string.

    Raises:
        RuntimeError: If TTS fails.
    """
    if not text:
        raise ValueError("Empty text")

    url = f'https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}'
    voice_cfg = {'languageCode': 'zh-CN'}
    if voice and voice != 'Auto (default)':
        voice_cfg['name'] = voice
    body = {
        'input': {'text': text},
        'voice': voice_cfg,
        'audioConfig': {'audioEncoding': 'LINEAR16', 'speakingRate': 1.0},
    }
    resp = requests.post(url, json=body, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Google TTS HTTP {resp.status_code}: {resp.text[:200]}")

    audio_bytes = base64.b64decode(resp.json()['audioContent'])
    tmp = os.path.join(tempfile.gettempdir(), 'google_tts_tmp.wav')
    with open(tmp, 'wb') as f:
        f.write(audio_bytes)
    play_wav(tmp, out_device)
# ── MiMo TTS ─────────────────────────────────────────────

def tts_mimo(text, voice='mimo_default', api_key='', out_device=None):
    """Synthesize speech via Xiaomi MiMo TTS and play it.

    Args:
        text: Text to synthesize.
        voice: MiMo voice ID (e.g. '冰糖', 'Chloe', 'mimo_default').
        api_key: MiMo API key.
        out_device: Output device index (None = default).

    Returns:
        Status string.

    Raises:
        RuntimeError: If TTS fails.
    """
    if not text:
        raise ValueError("Empty text")

    url = 'https://api.xiaomimimo.com/v1/chat/completions'
    body = {
        'model': 'mimo-v2.5-tts',
        'messages': [
            {'role': 'assistant', 'content': text},
        ],
        'audio': {
            'format': 'wav',
            'voice': voice or 'mimo_default',
        },
    }
    headers = {
        'api-key': api_key,
        'Content-Type': 'application/json',
    }
    resp = requests.post(url, json=body, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"MiMo TTS HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    audio_b64 = data.get('choices', [{}])[0].get('message', {}).get('audio', {}).get('data', '')
    if not audio_b64:
        raise RuntimeError("MiMo TTS returned no audio data")

    audio_bytes = base64.b64decode(audio_b64)
    tmp = os.path.join(tempfile.gettempdir(), 'mimo_tts_tmp.wav')
    with open(tmp, 'wb') as f:
        f.write(audio_bytes)
    play_wav(tmp, out_device)
    return f"MiMo TTS OK ({len(audio_bytes) // 1024}KB)"


# ── Dispatcher ───────────────────────────────────────────

TTS_ENGINES = {
    'gpt': tts_gptsovits,
    'edge': tts_edge,
    'google': tts_google,
    'mimo': tts_mimo,
}

def check_gptsovits(base_url='http://127.0.0.1:9880'):
    """Check if GPT-SoVITS API is running. Returns status string."""
    try:
        requests.get(f'{base_url}/docs', timeout=2)
        return "✅ GPT-SoVITS"
    except requests.RequestException:
        return "❌ GPT-SoVITS"
