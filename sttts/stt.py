"""STT engines: Whisper, Google Cloud STT, Xiaomi MiMo ASR, Push-to-Talk."""

import base64
import collections
import ctypes
import json
import threading

import numpy as np
import pyaudio
import requests
import webrtcvad

from .audio import pcm_to_wav


# ── Whisper (RealtimeSTT) ────────────────────────────────

def stt_whisper(dev_idx, stop_event, on_text, log_fn, model='small'):
    """Continuous STT using RealtimeSTT (Whisper-based)."""
    from RealtimeSTT import AudioToTextRecorder

    log_fn(f"Loading Whisper STT (model={model})...")
    rec = AudioToTextRecorder(
        model=model, realtime_model_type=model,
        input_device_index=dev_idx,
        spinner=False,
        on_realtime_transcription_stabilized=lambda t: log_fn(f"⇒ {t}") if t else None,
        silero_sensitivity=0.5,
        webrtc_sensitivity=2,
        post_speech_silence_duration=0.4,
        pre_recording_buffer_duration=0.5,
        min_length_of_recording=0.3,
        min_gap_between_recordings=0.05,
        realtime_processing_pause=0.05,
    )
    log_fn("Whisper STT ready (real-time mode)")
    try:
        while not stop_event.is_set():
            text = rec.text()
            if text and not stop_event.is_set():
                on_text(text)
    finally:
        try:
            rec.shutdown()
        except Exception:
            pass


# ── Google Cloud STT (local VAD) ─────────────────────────

def stt_google_vad(dev_idx, api_key, stop_event, on_text, log_fn, model='default'):
    """Local VAD + Google Cloud REST API."""
    RATE, CHUNK, VAD_MODE = 16000, 480, 1

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=RATE,
            input=True, input_device_index=dev_idx,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        log_fn(f"Failed to open audio stream: {e}")
        p.terminate()
        return

    vad = webrtcvad.Vad(VAD_MODE)
    log_fn(f"Google STT ready (local VAD, model={model})")

    PRE_BUFFER_FRAMES = int(RATE / CHUNK * 0.5)
    ring_buffer = collections.deque(maxlen=PRE_BUFFER_FRAMES)
    url = f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}'

    try:
        while not stop_event.is_set():
            frames, triggered, silence = [], False, 0
            MAX_SILENCE = int(RATE / CHUNK * 0.5)

            while not stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception:
                    continue
                ring_buffer.append(data)
                if vad.is_speech(data, RATE):
                    if not triggered:
                        triggered = True
                        frames.extend(ring_buffer)
                    frames.append(data)
                    silence = 0
                elif triggered:
                    frames.append(data)
                    silence += 1
                    if silence > MAX_SILENCE:
                        break

            if not frames or stop_event.is_set():
                continue

            config = {
                'encoding': 'LINEAR16', 'sampleRateHertz': RATE,
                'languageCode': 'zh-CN', 'useEnhanced': True,
            }
            if model and model != 'default':
                config['model'] = model
            body = {
                'config': config,
                'audio': {'content': _b64encode(b''.join(frames))},
            }

            try:
                resp = requests.post(url, json=body, timeout=15)
                if (resp.status_code == 400
                        and 'not supported for language' in resp.text
                        and model != 'default'):
                    log_fn(f"Model '{model}' unsupported for zh-CN, retrying with default")
                    config.pop('model', None)
                    resp = requests.post(url, json={**body, 'config': config}, timeout=15)

                if resp.status_code == 200:
                    results = resp.json().get('results')
                    if results:
                        text = results[0]['alternatives'][0]['transcript']
                        if text and not stop_event.is_set():
                            on_text(text)
                            ring_buffer.clear()
                else:
                    log_fn(f"Google STT error {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as e:
                log_fn(f"Google STT error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Google Cloud STT (full cloud VAD) ────────────────────

def stt_google_cloud(dev_idx, api_key, stop_event, on_text, log_fn, model='default'):
    """Cloud VAD mode: sends fixed 2-second chunks, lets Google handle VAD."""
    RATE, CHUNK_DURATION = 16000, 2.0
    CHUNK_SIZE = int(RATE * CHUNK_DURATION)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=RATE,
            input=True, input_device_index=dev_idx,
            frames_per_buffer=1024,
        )
    except Exception as e:
        log_fn(f"Failed to open audio stream: {e}")
        p.terminate()
        return

    log_fn(f"Google STT ready (cloud VAD, model={model})")
    buffer, bytes_read, last_text = [], 0, ''
    url = f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}'

    try:
        while not stop_event.is_set():
            try:
                data = stream.read(1024, exception_on_overflow=False)
            except Exception:
                continue
            buffer.append(data)
            bytes_read += len(data)

            if bytes_read >= CHUNK_SIZE * 2:
                audio_bytes = b''.join(buffer)
                buffer, bytes_read = [], 0

                config = {
                    'encoding': 'LINEAR16', 'sampleRateHertz': RATE,
                    'languageCode': 'zh-CN',
                    'enableAutomaticPunctuation': True, 'useEnhanced': True,
                }
                if model and model != 'default':
                    config['model'] = model
                body = {
                    'config': config,
                    'audio': {'content': _b64encode(audio_bytes)},
                }

                try:
                    resp = requests.post(url, json=body, timeout=15)
                    if (resp.status_code == 400
                            and 'not supported for language' in resp.text
                            and model != 'default'):
                        log_fn(f"Model '{model}' unsupported for zh-CN, retrying with default")
                        config.pop('model', None)
                        resp = requests.post(url, json={**body, 'config': config}, timeout=15)

                    if resp.status_code == 200:
                        results = resp.json().get('results')
                        if results:
                            text = results[0]['alternatives'][0]['transcript']
                            if text and text != last_text and not stop_event.is_set():
                                last_text = text
                                on_text(text)
                    else:
                        log_fn(f"Google STT error {resp.status_code}: {resp.text[:200]}")
                except requests.RequestException as e:
                    log_fn(f"Google STT error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Xiaomi MiMo ASR ──────────────────────────────────────

def stt_mimo(dev_idx, api_key, language, stop_event, on_text, log_fn):
    """Xiaomi MiMo ASR: local VAD + MiMo streaming API."""
    RATE, CHUNK, VAD_MODE = 16000, 480, 1

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=RATE,
            input=True, input_device_index=dev_idx,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        log_fn(f"Failed to open audio stream: {e}")
        p.terminate()
        return

    vad = webrtcvad.Vad(VAD_MODE)
    log_fn(f"MiMo ASR ready (language={language})")

    PRE_BUFFER_FRAMES = int(RATE / CHUNK * 0.5)
    ring_buffer = collections.deque(maxlen=PRE_BUFFER_FRAMES)

    try:
        while not stop_event.is_set():
            frames, triggered, silence = [], False, 0
            MAX_SILENCE = int(RATE / CHUNK * 0.5)

            while not stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception:
                    continue
                ring_buffer.append(data)
                if vad.is_speech(data, RATE):
                    if not triggered:
                        triggered = True
                        frames.extend(ring_buffer)
                    frames.append(data)
                    silence = 0
                elif triggered:
                    frames.append(data)
                    silence += 1
                    if silence > MAX_SILENCE:
                        break

            if not frames or stop_event.is_set():
                continue

            audio_b64 = _b64encode(pcm_to_wav(b''.join(frames)))
            body = {
                "model": "mimo-v2.5-asr",
                "messages": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {
                        "data": f"data:audio/wav;base64,{audio_b64}"}}
                ]}],
                "asr_options": {"language": language},
                "stream": True,
            }

            try:
                resp = requests.post(
                    "https://api.xiaomimimo.com/v1/chat/completions",
                    json=body, timeout=30, stream=True,
                    headers={"api-key": api_key, "Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    full_text = ""
                    for line in resp.iter_lines(decode_unicode=False):
                        if stop_event.is_set():
                            break
                        if line:
                            line = line.decode('utf-8')
                        if line and line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_text += content
                                    log_fn(f"⇒ {full_text}")
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
                    if full_text and not stop_event.is_set():
                        on_text(full_text)
                        ring_buffer.clear()
                else:
                    log_fn(f"MiMo ASR error {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as e:
                log_fn(f"MiMo ASR error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Push-to-Talk ─────────────────────────────────────────

def stt_ptt(dev_idx, rec_key, play_key, engine, engine_cfg,
            stop_event, on_text, log_fn, set_status=None):
    """Push-to-Talk STT: hold rec_key to record, press play_key to replay.

    Uses Win32 GetAsyncKeyState for key detection — works globally
    regardless of window focus, no hooks or message loops needed.
    """
    import string as _string

    rec_key = rec_key.strip().lower()
    play_key = play_key.strip().lower()

    # Map key names to Win32 Virtual-Key codes
    VK_MAP = {
        'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73, 'f5': 0x74,
        'f6': 0x75, 'f7': 0x76, 'f8': 0x77, 'f9': 0x78, 'f10': 0x79,
        'f11': 0x7A, 'f12': 0x7B,
        'space': 0x20, 'enter': 0x0D, 'return': 0x0D, 'tab': 0x09,
        'shift': 0xA0, 'ctrl': 0xA2, 'alt': 0xA4, 'win': 0x5B,
        'lshift': 0xA0, 'rshift': 0xA1, 'lctrl': 0xA2, 'rctrl': 0xA3,
        'lalt': 0xA4, 'ralt': 0xA5,
        'esc': 0x1B, 'escape': 0x1B,
        'backspace': 0x08, 'delete': 0x2E, 'insert': 0x2D,
        'home': 0x24, 'end': 0x23, 'pageup': 0x21, 'pagedown': 0x22,
        'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
        'capslock': 0x14, 'numlock': 0x90, 'scrolllock': 0x91,
        'printscreen': 0x2C, 'pause': 0x13,
        'numpad0': 0x60, 'numpad1': 0x61, 'numpad2': 0x62, 'numpad3': 0x63,
        'numpad4': 0x64, 'numpad5': 0x65, 'numpad6': 0x66, 'numpad7': 0x67,
        'numpad8': 0x68, 'numpad9': 0x69,
        'multiply': 0x6A, 'add': 0x6B, 'subtract': 0x6D,
        'decimal': 0x6E, 'divide': 0x6F,
    }
    for c in _string.ascii_lowercase:
        VK_MAP[c] = ord(c.upper())
    for c in _string.digits:
        VK_MAP[c] = ord(c)

    def _resolve_vk(key_name):
        if key_name in VK_MAP:
            return VK_MAP[key_name]
        if len(key_name) == 1:
            return ord(key_name.upper())
        raise ValueError(f"Unknown key: '{key_name}'")

    try:
        rec_vk = _resolve_vk(rec_key)
        play_vk = _resolve_vk(play_key)
    except ValueError as e:
        log_fn(f"ERROR: {e}. Use names like 'f8', 'f9', 'space', 'a', etc.")
        return

    _GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState

    def is_key_down(vk):
        return (_GetAsyncKeyState(vk) & 0x8000) != 0

    REC_RATE = 16000

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=REC_RATE,
            input=True, input_device_index=dev_idx,
            frames_per_buffer=512,
        )
    except Exception as e:
        log_fn(f"Failed to open audio stream: {e}")
        p.terminate()
        return

    # Whisper model loading
    whisper_model = None
    model_ready = (engine != 'whisper')
    if engine == 'whisper':
        from faster_whisper import WhisperModel
        model_name = engine_cfg.get('whisper_model', 'small')
        log_fn(f"Loading Whisper model ({model_name})...")
        try:
            whisper_model = WhisperModel(model_name, device='cuda', compute_type='float16')
        except Exception:
            whisper_model = WhisperModel(model_name, device='cpu', compute_type='int8')
        model_ready = True
        log_fn("Whisper model ready")

    log_fn(f"PTT ready — hold [{rec_key}] to record, press [{play_key}] to replay")
    if set_status:
        set_status('🟢 PTT listening...')

    def transcribe_audio(audio_bytes):
        if engine == 'whisper':
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = whisper_model.transcribe(audio, language='zh')
            return ' '.join(s.text for s in segments).strip()
        elif engine in ('google_vad', 'google_cloud'):
            cfg = {
                'encoding': 'LINEAR16', 'sampleRateHertz': REC_RATE,
                'languageCode': 'zh-CN', 'useEnhanced': True,
            }
            m = engine_cfg.get('google_model', 'default')
            if m and m != 'default':
                cfg['model'] = m
            body = {'config': cfg, 'audio': {'content': _b64encode(audio_bytes)}}
            try:
                resp = requests.post(
                    f"https://speech.googleapis.com/v1/speech:recognize?key={engine_cfg['gkey']}",
                    json=body, timeout=15,
                )
                if resp.status_code == 200:
                    r = resp.json().get('results')
                    if r:
                        return r[0]['alternatives'][0]['transcript']
            except requests.RequestException:
                pass
        elif engine == 'mimo':
            wav = pcm_to_wav(audio_bytes)
            body = {
                "model": "mimo-v2.5-asr",
                "messages": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {
                        "data": f"data:audio/wav;base64,{_b64encode(wav)}"}}
                ]}],
                "asr_options": {"language": engine_cfg.get('mimo_lang', 'zh')},
                "stream": True,
            }
            try:
                resp = requests.post(
                    "https://api.xiaomimimo.com/v1/chat/completions",
                    json=body, timeout=30, stream=True,
                    headers={"api-key": engine_cfg['mimo_key'], "Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    t = ""
                    for line in resp.iter_lines(decode_unicode=False):
                        if line:
                            line = line.decode('utf-8')
                        if line and line.startswith("data: "):
                            d = line[6:]
                            if d == "[DONE]":
                                break
                            try:
                                c = json.loads(d).get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if c:
                                    t += c
                            except (json.JSONDecodeError, IndexError, KeyError):
                                pass
                    return t.strip()
            except requests.RequestException:
                pass
        return ""

    last_text = ""
    play_was_down = False
    recording = False
    audio_frames = []

    try:
        while not stop_event.is_set():
            rec_down = is_key_down(rec_vk)
            play_down = is_key_down(play_vk)

            if rec_down:
                if not recording:
                    recording = True
                    audio_frames = []
                    log_fn("🔴 Recording...")
                    if set_status:
                        set_status('🔴 Recording...')
                try:
                    data = stream.read(512, exception_on_overflow=False)
                    audio_frames.append(data)
                except Exception:
                    pass
            else:
                if recording:
                    recording = False
                    log_fn("⏹ Stopped, transcribing...")
                    if set_status:
                        set_status('⏳ Transcribing...')
                    if audio_frames and model_ready:
                        text = transcribe_audio(b''.join(audio_frames))
                        if text and not stop_event.is_set():
                            last_text = text
                            on_text(text)
                    audio_frames = []
                    if set_status:
                        set_status('🟢 PTT listening...')
                else:
                    import time
                    time.sleep(0.01)

            # Play key: trigger on press edge
            if play_down and not play_was_down:
                play_was_down = True
                if last_text and not stop_event.is_set():
                    log_fn(f"▶ Replaying: {last_text}")
                    on_text(last_text)
            elif not play_down:
                play_was_down = False

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Helpers ──────────────────────────────────────────────

def _b64encode(data):
    """Base64 encode bytes to string."""
    return base64.b64encode(data).decode()
