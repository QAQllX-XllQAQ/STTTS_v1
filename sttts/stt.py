"""STT engines: Whisper, Google Cloud STT, Xiaomi MiMo ASR, Push-to-Talk."""

import base64
import collections
import ctypes
import ctypes.wintypes
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

# Key name → VK code mapping
_VK_MAP = {
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
}
import string as _string
for _c in _string.ascii_lowercase:
    _VK_MAP[_c] = ord(_c.upper())
for _c in _string.digits:
    _VK_MAP[_c] = ord(_c)

# Win32 constants
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
_GetAsyncKeyState = _user32.GetAsyncKeyState

# Set proper argtypes to avoid overflow on 64-bit handles
_user32.CreateWindowExW.argtypes = [
    ctypes.wintypes.DWORD, ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR,
    ctypes.wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.wintypes.HWND, ctypes.wintypes.HMENU, ctypes.wintypes.HINSTANCE, ctypes.c_void_p,
]
_user32.CreateWindowExW.restype = ctypes.wintypes.HWND
_user32.RegisterClassW.argtypes = [ctypes.c_void_p]
_user32.RegisterClassW.restype = ctypes.wintypes.ATOM
_user32.RegisterHotKey.argtypes = [ctypes.wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
_user32.RegisterHotKey.restype = ctypes.wintypes.BOOL
_user32.UnregisterHotKey.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]
_user32.UnregisterHotKey.restype = ctypes.wintypes.BOOL
_user32.DestroyWindow.argtypes = [ctypes.wintypes.HWND]
_user32.DestroyWindow.restype = ctypes.wintypes.BOOL
_user32.PeekMessageW.argtypes = [ctypes.c_void_p, ctypes.wintypes.HWND, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
_user32.PeekMessageW.restype = ctypes.wintypes.BOOL
_user32.DefWindowProcW.argtypes = [ctypes.wintypes.HWND, ctypes.c_uint, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM]
_user32.DefWindowProcW.restype = ctypes.wintypes.LPARAM

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)

class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ('style', ctypes.c_uint),
        ('lpfnWndProc', WNDPROC),
        ('cbClsExtra', ctypes.c_int),
        ('cbWndExtra', ctypes.c_int),
        ('hInstance', ctypes.wintypes.HINSTANCE),
        ('hIcon', ctypes.wintypes.HICON),
        ('hCursor', ctypes.wintypes.HANDLE),
        ('hbrBackground', ctypes.wintypes.HBRUSH),
        ('lpszMenuName', ctypes.wintypes.LPCWSTR),
        ('lpszClassName', ctypes.wintypes.LPCWSTR),
    ]


def resolve_vk(key_name):
    """Resolve key name string to VK code. Raises ValueError."""
    key_name = key_name.strip().lower()
    if key_name in _VK_MAP:
        return _VK_MAP[key_name]
    if len(key_name) == 1:
        return ord(key_name.upper())
    raise ValueError(f"Unknown key: '{key_name}'")


def stt_ptt(dev_idx, rec_key, play_key, engine, engine_cfg,
            stop_event, on_text, log_fn, set_status=None):
    """Push-to-Talk STT: hold rec_key to record, press play_key to replay.

    Uses RegisterHotKey (same as Discord) for global hotkey detection.
    No admin required, works regardless of window focus.
    """
    rec_key = rec_key.strip().lower()
    play_key = play_key.strip().lower()

    try:
        rec_vk = resolve_vk(rec_key)
        play_vk = resolve_vk(play_key)
    except ValueError as e:
        log_fn(f"ERROR: {e}. Use names like 'f8', 'f9', 'space', 'a', etc.")
        return

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

    # ── RegisterHotKey + message loop (Discord-style) ──────────

    # Create an invisible message-only window
    _def_proc = WNDPROC(lambda h, m, w, l: _user32.DefWindowProcW(h, m, w, l))
    wc = WNDCLASS()
    wc.lpfnWndProc = _def_proc
    wc.lpszClassName = 'STTTS_PTT_Hotkey'
    wc.hInstance = _kernel32.GetModuleHandleW(None)
    atom = _user32.RegisterClassW(ctypes.byref(wc))
    reg_err = _kernel32.GetLastError()
    log_fn(f"[debug] RegisterClassW atom={atom} err={reg_err}")
    hwnd = _user32.CreateWindowExW(
        0, 'STTTS_PTT_Hotkey', 'STTTS Hotkey', 0, 0, 0, 0, 0,
        0, 0, wc.hInstance, 0,
    )
    wnd_err = _kernel32.GetLastError()
    log_fn(f"[debug] CreateWindowExW hwnd={hwnd} err={wnd_err}")
    if not hwnd:
        log_fn(f"ERROR: Failed to create hotkey window (class_err={reg_err}, wnd_err={wnd_err})")
        stream.stop_stream(); stream.close(); p.terminate()
        return

    # Register global hotkeys — with all modifier combos so they work
    # even when Shift/Ctrl/Alt are held (e.g. in-game)
    MODIFIERS = [0, 0x0001, 0x0002, 0x0004, 0x0003, 0x0005, 0x0006, 0x0007]  # none,Alt,Ctrl,Ctrl+Alt,Shift+Ctrl,Shift+Alt,Shift+Ctrl+Alt,Shift
    rec_ids = []
    play_ids = []
    next_id = 10

    for mod in MODIFIERS:
        rid = next_id; next_id += 1
        if _user32.RegisterHotKey(hwnd, rid, mod | MOD_NOREPEAT, rec_vk):
            rec_ids.append(rid)
        pid = next_id; next_id += 1
        if _user32.RegisterHotKey(hwnd, pid, mod | MOD_NOREPEAT, play_vk):
            play_ids.append(pid)

    if not rec_ids:
        log_fn(f"ERROR: Failed to register any hotkey for [{rec_key}]")
        _user32.DestroyWindow(hwnd)
        stream.stop_stream(); stream.close(); p.terminate()
        return

    log_fn(f"Hotkeys registered: [{rec_key}] record ({len(rec_ids)} combos), [{play_key}] replay ({len(play_ids)} combos)")

    # Message loop in a background thread
    msg = ctypes.wintypes.MSG()
    last_text = ""
    recording = False
    audio_frames = []
    hotkey_thread_id = _kernel32.GetCurrentThreadId()

    def _is_key_held(vk):
        return (_GetAsyncKeyState(vk) & 0x8000) != 0

    try:
        while not stop_event.is_set():
            # GetMessageW blocks until a message arrives or timeout
            has_msg = _user32.PeekMessageW(ctypes.byref(msg), hwnd, 0, 0, 1)  # PM_REMOVE
            if has_msg:
                if msg.message == WM_HOTKEY:
                    if msg.wParam in rec_ids:
                        # Rec key pressed — start recording
                        recording = True
                        audio_frames = []
                        log_fn("🔴 Recording...")
                        if set_status:
                            set_status('🔴 Recording...')

                        # Record while key is held
                        while _is_key_held(rec_vk) and not stop_event.is_set():
                            try:
                                data = stream.read(512, exception_on_overflow=False)
                                audio_frames.append(data)
                            except Exception:
                                pass

                        # Key released — transcribe
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

                    elif msg.wParam in play_ids:
                        # Play key pressed — replay last text
                        if last_text and not stop_event.is_set():
                            log_fn(f"▶ Replaying: {last_text}")
                            on_text(last_text)

                    _user32.TranslateMessage(ctypes.byref(msg))
                    _user32.DispatchMessageW(ctypes.byref(msg))
            else:
                # No message — sleep briefly to avoid busy-waiting
                import time
                time.sleep(0.02)

    finally:
        for rid in rec_ids:
            _user32.UnregisterHotKey(hwnd, rid)
        for pid in play_ids:
            _user32.UnregisterHotKey(hwnd, pid)
        _user32.DestroyWindow(hwnd)
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Helpers ──────────────────────────────────────────────

def _b64encode(data):
    """Base64 encode bytes to string."""
    return base64.b64encode(data).decode()
