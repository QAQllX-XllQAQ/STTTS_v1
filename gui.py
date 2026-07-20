import PySimpleGUI as sg
import threading, os, time, pyaudio, requests, tempfile, json, base64

sg.theme("LightBlue3")

# ── helpers ──────────────────────────────────────────────

def list_devices(kind='input'):
    p = pyaudio.PyAudio()
    items = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if kind == 'input' and info['maxInputChannels'] > 0:
            items.append(f"{i}: {info['name'].strip()}")
        elif kind == 'output' and info['maxOutputChannels'] > 0:
            items.append(f"{i}: {info['name'].strip()}")
    p.terminate()
    return items

def check_gptsovits(base_url="http://127.0.0.1:9880"):
    try:
        requests.get(f'{base_url}/docs', timeout=2)
        return "✅ GPT-SoVITS"
    except:
        return "❌ GPT-SoVITS"

def play_wav(path, device_idx=None):
    import soundfile as sf, sounddevice as sd
    data, sr = sf.read(path)
    sd.play(data, sr, device=device_idx)
    sd.wait()

EDGE_VOICES = ['zh-CN-XiaoxiaoNeural','zh-CN-XiaoyiNeural','zh-CN-YunjianNeural',
               'zh-CN-YunxiNeural','zh-CN-YunyangNeural','zh-CN-XiaohanNeural']
GOOGLE_VOICES = ['Auto (default)','zh-CN-Neural2-A','zh-CN-Neural2-B','zh-CN-Neural2-C','zh-CN-Neural2-D',
                 'zh-CN-Standard-A','zh-CN-Standard-B','zh-CN-Standard-C','zh-CN-Standard-D',
                 'zh-CN-Studio-A','zh-CN-Studio-B','zh-CN-Studio-C']

# ── TTS ──────────────────────────────────────────────────

def tts_gpt(text, ref_audio, prompt_lang, prompt_text, out_device, base_url="http://127.0.0.1:9880"):
    url = f'{base_url}/tts'
    params = {
        'text': text, 'text_lang': 'zh',
        'ref_audio_path': ref_audio,
        'prompt_lang': prompt_lang, 'prompt_text': prompt_text,
        'text_split_method': 'cut5', 'batch_size': 1,
        'media_type': 'wav', 'streaming_mode': False
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200 or len(resp.content) < 1000:
        return f"TTS failed: {resp.text[:100]}"
    tmp = os.path.join(tempfile.gettempdir(), 'gpt_tmp.wav')
    with open(tmp, 'wb') as f: f.write(resp.content)
    play_wav(tmp, out_device)
    return f"GPT-SoVITS OK ({len(resp.content)//1024}KB)"

def tts_edge(text, voice, out_device):
    import edge_tts, asyncio
    tmp = os.path.join(tempfile.gettempdir(), 'edge_tmp.mp3')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(edge_tts.Communicate(text, voice).save(tmp))
    finally:
        loop.close()
    import soundfile as sf, sounddevice as sd
    data, sr = sf.read(tmp)
    sd.play(data, sr, device=out_device)
    sd.wait()
    return f"Edge-TTS OK"

def tts_google(text, voice, api_key, out_device):
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
        return f"Google TTS failed: {resp.text[:200]}"
    audio_bytes = base64.b64decode(resp.json()['audioContent'])
    tmp = os.path.join(tempfile.gettempdir(), 'google_tts.wav')
    with open(tmp, 'wb') as f: f.write(audio_bytes)
    play_wav(tmp, out_device)
    return f"Google TTS OK"

def stt_whisper(dev_idx, stop_event, on_text, log, model='small'):
    from RealtimeSTT import AudioToTextRecorder
    log(f"Loading Whisper STT (model={model})...")
    rec = AudioToTextRecorder(
        spinner=False, model=model, input_device_index=dev_idx,
        realtime_model_type=model, language='zh',
        enable_realtime_transcription=True,
        on_realtime_transcription_stabilized=lambda t: log(f"⇒ {t}") if t else None,
        silero_sensitivity=0.5, webrtc_sensitivity=2,
        post_speech_silence_duration=0.4,
        pre_recording_buffer_duration=0.5,
        min_length_of_recording=0.3, min_gap_between_recordings=0.05,
        realtime_processing_pause=0.05,
    )
    log("Whisper STT ready (real-time mode)")
    while not stop_event.is_set():
        t = rec.text()
        if t and not stop_event.is_set():
            on_text(t)
    try: rec.shutdown()
    except: pass


def stt_google_vad(dev_idx, api_key, stop_event, on_text, log, model='default'):
    """Local VAD + REST API: webrtcvad detects speech, sends only speech segments"""
    import pyaudio as pa, webrtcvad, collections, base64 as b64
    RATE, CHUNK, VAD_MODE = 16000, 480, 1
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt16, channels=1, rate=RATE,
                    input=True, input_device_index=dev_idx,
                    frames_per_buffer=CHUNK)
    vad = webrtcvad.Vad(VAD_MODE)
    log(f"Google STT ready (local VAD, model={model})")
    PRE_BUFFER_FRAMES = int(RATE / CHUNK * 0.5)
    ring_buffer = collections.deque(maxlen=PRE_BUFFER_FRAMES)
    while not stop_event.is_set():
        frames, triggered, silence = [], False, 0
        MAX_SILENCE = int(RATE / CHUNK * 0.5)
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
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
        config = {'encoding':'LINEAR16','sampleRateHertz':RATE,'languageCode':'zh-CN',
                   'useEnhanced':True}
        if model and model != 'default':
            config['model'] = model
        body = {
            'config': config,
            'audio': {'content': b64.b64encode(b''.join(frames)).decode()}
        }
        url = f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}'
        try:
            resp = requests.post(url, json=body, timeout=15)
            if resp.status_code == 400 and 'not supported for language' in resp.text and model != 'default':
                log(f"Google STT: model '{model}' unsupported for zh-CN, retrying with default")
                config.pop('model', None)
                resp = requests.post(url, json={**body, 'config': config}, timeout=15)
            if resp.status_code == 200:
                results = resp.json().get('results')
                if results:
                    text = results[0]['alternatives'][0]['transcript']
                    if text and not stop_event.is_set():
                        on_text(text)
                        ring_buffer.clear()  # prevent re-triggering on same audio
            else:
                log(f"Google STT error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            log(f"Google STT error: {e}")
    stream.stop_stream(); stream.close(); p.terminate()
def stt_google_cloud(dev_idx, api_key, stop_event, on_text, log, model='default'):
    """Cloud VAD mode: send 2s fixed chunks, let Google handle VAD"""
    import pyaudio as pa, base64 as b64
    RATE, CHUNK_DURATION = 16000, 2.0
    CHUNK_SIZE = int(RATE * CHUNK_DURATION)
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt16, channels=1, rate=RATE,
                    input=True, input_device_index=dev_idx,
                    frames_per_buffer=1024)
    log(f"Google STT ready (cloud VAD, model={model})")
    buffer = []
    bytes_read = 0
    last_text = ''
    while not stop_event.is_set():
        data = stream.read(1024, exception_on_overflow=False)
        buffer.append(data)
        bytes_read += len(data)
        if bytes_read >= CHUNK_SIZE * 2:
            audio_bytes = b''.join(buffer)
            buffer = []; bytes_read = 0
            config = {'encoding':'LINEAR16','sampleRateHertz':RATE,'languageCode':'zh-CN',
                       'enableAutomaticPunctuation':True, 'useEnhanced':True}
            if model and model != 'default':
                config['model'] = model
            body = {
                'config': config,
                'audio': {'content': b64.b64encode(audio_bytes).decode()}
            }
            url = f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}'
            try:
                resp = requests.post(url, json=body, timeout=15)
                if resp.status_code == 400 and 'not supported for language' in resp.text and model != 'default':
                    log(f"Google STT: model '{model}' unsupported for zh-CN, retrying with default")
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
                    log(f"Google STT error {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                log(f"Google STT error: {e}")


def _pcm_to_wav(pcm_data, sample_rate=16000):
    import struct
    data_len = len(pcm_data)
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_len, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b'data', data_len)
    return header + pcm_data


def stt_mimo(dev_idx, api_key, language, stop_event, on_text, log):
    """Xiaomi MiMo ASR: local VAD + MiMo API"""
    import pyaudio as pa, webrtcvad, collections, base64 as b64
    RATE, CHUNK, VAD_MODE = 16000, 480, 1
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt16, channels=1, rate=RATE,
                    input=True, input_device_index=dev_idx,
                    frames_per_buffer=CHUNK)
    vad = webrtcvad.Vad(VAD_MODE)
    log(f"MiMo ASR ready (language={language})")
    PRE_BUFFER_FRAMES = int(RATE / CHUNK * 0.5)
    ring_buffer = collections.deque(maxlen=PRE_BUFFER_FRAMES)
    while not stop_event.is_set():
        frames, triggered, silence = [], False, 0
        MAX_SILENCE = int(RATE / CHUNK * 0.5)
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
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
        audio_b64 = b64.b64encode(_pcm_to_wav(b''.join(frames))).decode()
        body = {
            "model": "mimo-v2.5-asr",
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {
                    "data": f"data:audio/wav;base64,{audio_b64}"}}
            ]}],
            "asr_options": {"language": language},
            "stream": True
        }
        try:
            resp = requests.post("https://api.xiaomimimo.com/v1/chat/completions",
                                json=body, timeout=30, stream=True,
                                headers={"api-key": api_key, "Content-Type": "application/json"})
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
                                log(f"⇒ {full_text}")
                        except:
                            pass
                if full_text and not stop_event.is_set():
                    on_text(full_text)
                    ring_buffer.clear()
            else:
                log(f"MiMo ASR error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            log(f"MiMo ASR error: {e}")



def stt_ptt(dev_idx, rec_key, play_key, engine, engine_cfg, stop_event, on_text, log, set_status=None):
    try:
        import keyboard
    except ImportError:
        log("ERROR: keyboard library required for PTT. Run: pip install keyboard")
        return
    import pyaudio as pa
    import numpy as np

    rec_key = rec_key.strip()
    play_key = play_key.strip()

    REC_RATE = 16000
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt16, channels=1, rate=REC_RATE,
                    input=True, input_device_index=dev_idx,
                    frames_per_buffer=512)

    recording = False
    audio_frames = []
    last_text = ""
    whisper_model = None
    model_ready = (engine != 'whisper')

    # Load Whisper model if needed (BEFORE starting poll loop)
    if engine == 'whisper':
        from faster_whisper import WhisperModel
        model_name = engine_cfg.get('whisper_model', 'small')
        log(f"Loading Whisper model ({model_name})...")
        try:
            whisper_model = WhisperModel(model_name, device='cuda', compute_type='float16')
        except:
            whisper_model = WhisperModel(model_name, device='cpu', compute_type='int8')
        model_ready = True
        log("Whisper model ready")

    def transcribe_audio(audio_bytes):
        if engine == 'whisper':
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = whisper_model.transcribe(audio, language='zh')
            return ' '.join(s.text for s in segments).strip()
        elif engine in ('google_vad', 'google_cloud'):
            import base64 as b64
            cfg = {'encoding':'LINEAR16','sampleRateHertz':REC_RATE,'languageCode':'zh-CN','useEnhanced':True}
            m = engine_cfg.get('google_model', 'default')
            if m and m != 'default': cfg['model'] = m
            body = {'config': cfg, 'audio': {'content': b64.b64encode(audio_bytes).decode()}}
            resp = requests.post(f"https://speech.googleapis.com/v1/speech:recognize?key={engine_cfg['gkey']}",
                                json=body, timeout=15)
            if resp.status_code == 200:
                r = resp.json().get('results')
                if r: return r[0]['alternatives'][0]['transcript']
        elif engine == 'mimo':
            import base64 as b64
            wav = _pcm_to_wav(audio_bytes)
            body = {"model":"mimo-v2.5-asr","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":f"data:audio/wav;base64,{b64.b64encode(wav).decode()}"}}]}],"asr_options":{"language":engine_cfg.get('mimo_lang','zh')},"stream":True}
            resp = requests.post("https://api.xiaomimimo.com/v1/chat/completions", json=body, timeout=30, stream=True,
                                headers={"api-key":engine_cfg['mimo_key'],"Content-Type":"application/json"})
            if resp.status_code == 200:
                t = ""
                for line in resp.iter_lines(decode_unicode=False):
                    if line: line = line.decode('utf-8')
                    if line and line.startswith("data: "):
                        d = line[6:]
                        if d == "[DONE]": break
                        try:
                            c = json.loads(d).get("choices",[{}])[0].get("delta",{}).get("content","")
                            if c: t += c
                        except: pass
                return t.strip()
        return ""

    log(f"PTT ready — hold [{rec_key}] to record, press [{play_key}] to replay")
    if set_status: set_status('🟢 PTT listening...')

    play_was_down = False
    while not stop_event.is_set():
        try:
            data = stream.read(512, exception_on_overflow=False)
        except:
            continue

        r = keyboard.is_pressed(rec_key)
        p = keyboard.is_pressed(play_key)

        if r and not recording:
            recording = True
            audio_frames = [data]
            log("🔴 Recording...")
            if set_status: set_status('🔴 Recording...')
        elif r and recording:
            audio_frames.append(data)
        elif not r and recording:
            recording = False
            log("⏹ Stopped, transcribing...")
            if set_status: set_status('⏳ Transcribing...')
            if audio_frames and model_ready:
                text = transcribe_audio(b''.join(audio_frames))
                audio_frames = []
                if text and not stop_event.is_set():
                    last_text = text
                    on_text(text)
            if set_status: set_status('🟢 PTT listening...')

        if p and not play_was_down:
            play_was_down = True
            if last_text and not stop_event.is_set():
                log(f"▶ Playing: {last_text}")
                on_text(last_text)
        elif not p:
            play_was_down = False

    stream.stop_stream()
    stream.close()
    p.terminate()
CONFIG_PATH = os.path.join(os.path.dirname(__file__) or '.', 'config.json')

def _load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

def _save_config(values):
    keys = ['GKEY', 'WHISPER_MODEL', 'GOOGLE_MODEL', 'MIMO_KEY', 'MIMO_LANG',
            'VOICE', 'REF', 'GPT_URL', 'DEV_IN', 'DEV_OUT',
            'STT_W', 'STT_G', 'STT_MIMO',
            'TTS_GPT', 'TTS_EDGE', 'TTS_GOOGLE',
            'GM_VAD', 'GM_CLOUD', 'MODE_CONT', 'MODE_PTT',
            'PTT_REC', 'PTT_PLAY']
    cfg = {}
    for k in keys:
        v = values.get(k)
        if v is not None:
            cfg[k] = v
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except:
        pass
def main():
    cfg_saved = _load_config()
    in_devs = list_devices('input')
    out_devs = list_devices('output')
    def_dev_in = in_devs[0] if in_devs else ''
    def_dev_out = out_devs[0] if out_devs else ''

    layout = [
        [sg.Frame('STT Engine', [
            [sg.Radio('Whisper (local GPU)', 'STT', key='STT_W', default=True, enable_events=True),
             sg.Radio('Google Cloud STT', 'STT', key='STT_G', enable_events=True),
             sg.Radio('Xiaomi MiMo', 'STT', key='STT_MIMO', enable_events=True)],
            [sg.pin(sg.Col([[sg.Radio('Local VAD + REST', 'GOOGLE_MODE', key='GM_VAD', default=True),
              sg.Radio('Cloud process (full audio)', 'GOOGLE_MODE', key='GM_CLOUD'),
              sg.Text('  Model:'), sg.Combo(['default','command_and_search','phone_call','video'], default_value=cfg_saved.get('GOOGLE_MODEL','default'), key='GOOGLE_MODEL', size=(22,1))]],
              key='COL_GM', visible=False))],
            [sg.pin(sg.Col([[sg.Text('MiMo key:'), sg.Input(key='MIMO_KEY', size=(35,1),
                default_text=cfg_saved.get('MIMO_KEY',''), password_char='*'),
              sg.Text('Lang:'), sg.Combo(['zh','en','auto'], default_value=cfg_saved.get('MIMO_LANG','zh'), key='MIMO_LANG', size=(6,1))]],
              key='COL_MIMO', visible=False))],
            [sg.pin(sg.Col([[sg.Text('Whisper model:'), sg.Combo(['tiny','tiny.en','base','base.en','small','small.en','medium','medium.en','large-v2'], default_value=cfg_saved.get('WHISPER_MODEL','small'), key='WHISPER_MODEL', size=(15,1))]], key='COL_WHISPER', visible=True)),
             sg.Text('Input device:'), sg.Combo(in_devs, default_value=cfg_saved.get('DEV_IN', def_dev_in), key='DEV_IN', size=(40,1))],
            [sg.pin(sg.Col([[sg.Text('Google key:'), sg.Input(key='GKEY', size=(50,1),
                default_text=cfg_saved.get('GKEY',''), password_char='*')]], key='COL_GKEY', visible=False))],
        ])],
        [sg.Frame('Mode', [
            [sg.Radio('Continuous (VAD)', 'MODE', key='MODE_CONT', default=True, enable_events=True),
             sg.Radio('Push-to-Talk', 'MODE', key='MODE_PTT', enable_events=True)],
            [sg.pin(sg.Col([[sg.Text('Record key:'), sg.Input(key='PTT_REC', size=(8,1),
                default_text=cfg_saved.get('PTT_REC','F8')),
              sg.Text('Play key:'), sg.Input(key='PTT_PLAY', size=(8,1),
                default_text=cfg_saved.get('PTT_PLAY','F9'))]],
              key='COL_PTT', visible=False))],
        ])],
        [sg.Frame('TTS Engine', [
            [sg.Radio('GPT-SoVITS', 'TTS', key='TTS_GPT', default=True, enable_events=True),
             sg.Radio('Edge-TTS', 'TTS', key='TTS_EDGE', enable_events=True),
             sg.Radio('Google TTS', 'TTS', key='TTS_GOOGLE', enable_events=True)],
            [sg.pin(sg.Col([[sg.Text('Ref audio:'), sg.Input(key='REF', size=(42,1),
                default_text=cfg_saved.get('REF', os.path.join(os.path.dirname(__file__) or '.', 'ref_audio.wav'))),
             sg.FileBrowse(file_types=(("WAV", "*.wav"),))]], key='COL_REF', visible=True))],
            [sg.pin(sg.Col([[sg.Text('GPT URL:'), sg.Input(cfg_saved.get('GPT_URL', 'http://127.0.0.1:9880'), key='GPT_URL', size=(35,1))],
             [sg.Button('▶ GPT-SoVITS', key='GPT_START', size=(14,1)),
             sg.Button('■ GPT-SoVITS', key='GPT_STOP', size=(14,1), disabled=True, button_color='red'),
             sg.Text(check_gptsovits(), key='SRV', size=(16,1))]], key='COL_GPT', visible=True))],
            [sg.Combo(EDGE_VOICES, default_value='zh-CN-XiaoxiaoNeural', key='VOICE', size=(30,1))],
        ])],
        [sg.Frame('Audio Output', [
            [sg.Text('Play to:'), sg.Combo(out_devs, default_value=cfg_saved.get('DEV_OUT', def_dev_out), key='DEV_OUT', size=(50,1))],
        ])],
        [sg.Frame('Control', [
            [sg.Button('▶ Start', key='START', size=(10,1)),
             sg.Button('■ Stop', key='STOP', size=(10,1), disabled=True)],
        ])],
        [sg.Frame('Status', [
            [sg.Text('Idle', key='STATUS', size=(75,1), text_color='blue')],
        ])],
        [sg.Frame('Transcription', [
            [sg.Multiline(size=(85,4), key='TEXT', disabled=True, autoscroll=True)],
        ])],
        [sg.Frame('Log', [
            [sg.Multiline(size=(85,8), key='LOG', disabled=True, autoscroll=True)],
        ])],
    ]

    window = sg.Window('STT → TTS', layout, finalize=True)
    stop_event = threading.Event()
    gpt_process = None
    GPT_CMD = ['conda', 'run', '-n', 'GPTSoVits', 'python',
               os.path.join(os.path.dirname(__file__) or '.', 'GPT-SoVITS', 'api_v2.py'), '-a', '127.0.0.1', '-p', '9880',
               '-c', r'GPT_SoVITS/configs/tts_infer.yaml']

    def switch_tts(tts_mode):
        show_gpt = tts_mode == 'gpt'
        window['COL_REF'].update(visible=show_gpt)
        window['COL_GPT'].update(visible=show_gpt)
        if tts_mode == 'edge':
            window['VOICE'].update(values=EDGE_VOICES, value='zh-CN-XiaoxiaoNeural')
        elif tts_mode == 'google':
            window['VOICE'].update(values=GOOGLE_VOICES, value='Auto (default)')
        else:
            window['VOICE'].update(values=[], value='')

    # Restore saved settings
    if cfg_saved.get('STT_G'):
        window['STT_G'].update(value=True)
        window['COL_GM'].update(visible=True)
        window['COL_GKEY'].update(visible=True)
        window['COL_WHISPER'].update(visible=False)
        if cfg_saved.get('GM_CLOUD'):
            window['GM_CLOUD'].update(value=True)
    elif cfg_saved.get('STT_MIMO'):
        window['STT_MIMO'].update(value=True)
        window['COL_MIMO'].update(visible=True)
        window['COL_WHISPER'].update(visible=False)
    if cfg_saved.get('MODE_PTT'):
        window['MODE_PTT'].update(value=True)
        window['COL_PTT'].update(visible=True)
    if cfg_saved.get('TTS_EDGE'):
        window['TTS_EDGE'].update(value=True); switch_tts('edge')
    elif cfg_saved.get('TTS_GOOGLE'):
        window['TTS_GOOGLE'].update(value=True); switch_tts('google')
    else:
        switch_tts('gpt')

    # Auto-start PTT if was active last session
    if cfg_saved.get('MODE_PTT'):
        ptt_engine = 'whisper'
        if cfg_saved.get('STT_G'):
            ptt_engine = 'google_vad' if cfg_saved.get('GM_CLOUD') else 'google_vad'
        elif cfg_saved.get('STT_MIMO'):
            ptt_engine = 'mimo'
        ptt_cfg = {
            'stt': 'ptt', 'tts': 'gpt' if cfg_saved.get('TTS_GPT', True) else ('google' if cfg_saved.get('TTS_GOOGLE') else 'edge'),
            'dev_in': 0, 'dev_out': None,
            'ref': cfg_saved.get('REF', ''), 'gkey': cfg_saved.get('GKEY', ''),
            'voice': cfg_saved.get('VOICE', ''),
            'whisper_model': cfg_saved.get('WHISPER_MODEL', 'small'),
            'google_model': cfg_saved.get('GOOGLE_MODEL', 'default'),
            'mimo_key': cfg_saved.get('MIMO_KEY', ''), 'mimo_lang': cfg_saved.get('MIMO_LANG', 'zh'),
            'ptt_rec': cfg_saved.get('PTT_REC', 'F8'), 'ptt_play': cfg_saved.get('PTT_PLAY', 'F9'),
            'ptt_engine': ptt_engine,
        }
        window['START'].update(disabled=True)
        window['STOP'].update(disabled=False)
        window['STATUS'].update('🟢 PTT listening...', text_color='green')
        window.write_event_value('_AUTO_PTT', ptt_cfg)

    def handle_text(text, cfg):
        window.write_event_value('_TEXT', text)
        log(f"Recognized: {text}")
        threading.Thread(target=lambda: _play_tts(text, cfg), daemon=True).start()

    def _play_tts(text, cfg):
        with tts_lock:
            if cfg['tts'] == 'gpt':
                do_gpt(text, cfg)
            elif cfg['tts'] == 'google':
                do_google(text, cfg)
            else:
                do_edge(text, cfg)

    def do_gpt(text, cfg):
        if not os.path.exists(cfg['ref']):
            log("Ref audio missing!")
            return
        log(tts_gpt(text, cfg['ref'], 'ja', '', cfg.get('dev_out'), cfg.get('gpt_url', 'http://127.0.0.1:9880')))

    def do_edge(text, cfg):
        log(tts_edge(text, cfg['voice'], cfg.get('dev_out')))

    def do_google(text, cfg):
        log(tts_google(text, cfg['voice'], cfg['gkey'], cfg.get('dev_out')))

    def run(cfg):
        try:
            cb = lambda t: handle_text(t, cfg)
            if cfg['stt'] == 'whisper':
                stt_whisper(cfg['dev_in'], stop_event, cb, log, cfg.get('whisper_model', 'tiny'))
            elif cfg['stt'] == 'ptt':
                stt_ptt(cfg['dev_in'], cfg['ptt_rec'], cfg['ptt_play'],
                       cfg.get('ptt_engine', 'whisper'), cfg, stop_event, cb, log,
                       lambda m: window.write_event_value('_STATUS', m))
            elif cfg['stt'] == 'mimo':
                stt_mimo(cfg['dev_in'], cfg['mimo_key'], cfg['mimo_lang'], stop_event, cb, log)
            elif cfg['stt'] == 'google_vad':
                stt_google_vad(cfg['dev_in'], cfg['gkey'], stop_event, cb, log, cfg.get('google_model', 'default'))
            else:
                stt_google_cloud(cfg['dev_in'], cfg['gkey'], stop_event, cb, log, cfg.get('google_model', 'default'))
        except Exception as ex:
            log(f'STT thread crashed: {ex}')
            import traceback
            log(traceback.format_exc())
        window.write_event_value('_STATUS', '⏹ Stopped')


    def update_srv_status(gpt_url):
        st = check_gptsovits(gpt_url)
        window['SRV'].update(st)
        return '✅' in st
    # Initial visibility — already set via layout visible=False/True

    while True:
        event, values = window.read(timeout=2000)
        if event == sg.WINDOW_CLOSED:
            stop_event.set()
            if gpt_process:
                try: gpt_process.terminate()
                except: pass
            break
        if event in ('STT_W', 'STT_G', 'STT_MIMO'):
            try:
                is_google = values['STT_G']
                is_mimo = values['STT_MIMO']
                window['COL_GM'].update(visible=is_google)
                window['COL_MIMO'].update(visible=is_mimo)
                window['COL_GKEY'].update(visible=is_google)
                window['COL_WHISPER'].update(visible=not (is_google or is_mimo))
                _save_config(values)
            except Exception:
                pass
        if event in ('MODE_CONT', 'MODE_PTT'):
            is_ptt = values['MODE_PTT']
            window['COL_PTT'].update(visible=is_ptt)
            _save_config(values)
            if is_ptt and not stop_event.is_set():
                # Auto-start PTT
                ptt_engine = 'whisper'
                if values['STT_G']:
                    ptt_engine = 'google_vad' if values.get('GM_VAD', True) else 'google_cloud'
                elif values['STT_MIMO']:
                    ptt_engine = 'mimo'
                ptt_cfg = {
                    'stt': 'ptt', 'tts': 'gpt' if values['TTS_GPT'] else ('google' if values['TTS_GOOGLE'] else 'edge'),
                    'dev_in': int(values['DEV_IN'].split(':')[0]),
                    'dev_out': int(values['DEV_OUT'].split(':')[0]) if values['DEV_OUT'] else None,
                    'ref': values['REF'], 'gkey': values['GKEY'], 'voice': values['VOICE'],
                    'whisper_model': values.get('WHISPER_MODEL', 'small'),
                    'google_model': values.get('GOOGLE_MODEL', 'default'),
                    'mimo_key': values.get('MIMO_KEY', ''), 'mimo_lang': values.get('MIMO_LANG', 'zh'),
                    'ptt_rec': values.get('PTT_REC', 'F8'), 'ptt_play': values.get('PTT_PLAY', 'F9'),
                    'ptt_engine': ptt_engine,
                }
                window['START'].update(disabled=True)
                window['STOP'].update(disabled=False)
                window['TEXT'].update('')
                window['LOG'].update('')
                window['STATUS'].update('🟢 PTT listening...', text_color='green')
                stop_event.clear()
                threading.Thread(target=run, args=(ptt_cfg,), daemon=True).start()
            elif not is_ptt:
                stop_event.set()
                window['START'].update(disabled=False)
                window['STOP'].update(disabled=True)
                window['STATUS'].update('⏹ Stopped', text_color='blue')
        if event in ('TTS_GPT', 'TTS_EDGE', 'TTS_GOOGLE'):
            try:
                if values['TTS_GPT']: switch_tts('gpt')
                elif values['TTS_EDGE']: switch_tts('edge')
                elif values['TTS_GOOGLE']: switch_tts('google')
                _save_config(values)
            except Exception:
                pass
        if event == '__TIMEOUT__':
            gpt_url = values.get('GPT_URL', 'http://127.0.0.1:9880')
            now_running = update_srv_status(gpt_url)
            window['GPT_START'].update(disabled=now_running)
            window['GPT_STOP'].update(disabled=not now_running)
            continue

        if event == 'GPT_START':
            if gpt_process and gpt_process.poll() is None:
                log("GPT-SoVITS already running")
            else:
                log("Starting GPT-SoVITS...")
                import subprocess
                gpt_process = subprocess.Popen(GPT_CMD,
                    cwd=os.path.join(os.path.dirname(__file__) or '.', 'GPT-SoVITS'),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log(f"GPT-SoVITS starting (PID {gpt_process.pid}), wait ~40s...")
                window['GPT_START'].update(disabled=True)

        elif event == 'GPT_STOP':
            if gpt_process and gpt_process.poll() is None:
                gpt_process.terminate()
                try: gpt_process.wait(timeout=10)
                except: gpt_process.kill()
                gpt_process = None
                log("GPT-SoVITS stopped")
            else:
                log("GPT-SoVITS not running")
            update_srv_status(values.get('GPT_URL', 'http://127.0.0.1:9880'))

        if event == 'START':
            try:
                if values['MODE_PTT']:
                    stt = 'ptt'
                    ptt_engine = 'whisper'
                    if values['STT_G']:
                        ptt_engine = 'google_vad' if values.get('GM_VAD', True) else 'google_cloud'
                    elif values['STT_MIMO']:
                        ptt_engine = 'mimo'
                elif values['STT_MIMO']:
                    stt = 'mimo'
                elif values['STT_G']:
                    stt = 'google_vad' if values.get('GM_VAD', True) else 'google_cloud'
                else:
                    stt = 'whisper'
                tts = 'gpt' if values['TTS_GPT'] else ('google' if values['TTS_GOOGLE'] else 'edge')
                cfg = {
                    'stt': stt, 'tts': tts,
                    'dev_in': int(values['DEV_IN'].split(':')[0]),
                    'dev_out': int(values['DEV_OUT'].split(':')[0]) if values['DEV_OUT'] else None,
                    'ref': values['REF'], 'gkey': values['GKEY'], 'voice': values['VOICE'],
                    'gpt_url': values.get('GPT_URL', 'http://127.0.0.1:9880'),
                    'whisper_model': values.get('WHISPER_MODEL', 'small'),
                    'mimo_key': values.get('MIMO_KEY', ''),
                    'mimo_lang': values.get('MIMO_LANG', 'zh'),
                    'ptt_rec': values.get('PTT_REC', 'F8'),
                    'ptt_play': values.get('PTT_PLAY', 'F9'),
                    'ptt_engine': locals().get('ptt_engine', 'whisper'),
                }
                if stt == 'mimo' and not cfg['mimo_key']:
                    window['STATUS'].update('❌ MiMo API key required', text_color='red')
                    log('ERROR: MiMo API key required')
                    continue
                if stt.startswith('google') and (not cfg['gkey'] or 'YOUR_' in cfg['gkey']):
                    window['STATUS'].update('❌ Google API key required', text_color='red')
                    log('ERROR: Google API key required for Google STT')
                    continue
                if tts == 'google' and (not cfg['gkey'] or 'YOUR_' in cfg['gkey']):
                    window['STATUS'].update('❌ Google API key required', text_color='red')
                    log('ERROR: Google API key required for Google TTS')
                    continue
                if tts == 'gpt' and not os.path.exists(cfg['ref']):
                    window['STATUS'].update(f'❌ Ref audio not found: {cfg["ref"]}', text_color='red')
                    log(f'ERROR: Ref audio not found — {cfg["ref"]}')
                    log('HINT: Switch TTS to Edge-TTS (no ref audio needed) or place a .wav file')
                    continue
                stop_event.clear()
                window['START'].update(disabled=True)
                window['STOP'].update(disabled=False)
                window['TEXT'].update('')
                window['LOG'].update('')
                window['STATUS'].update('🟢 Running...', text_color='green')
                _save_config(values)
                threading.Thread(target=run, args=(cfg,), daemon=True).start()
            except Exception as ex:
                window['STATUS'].update(f'❌ Start failed: {ex}', text_color='red')
                log(f'ERROR on start: {ex}')
                import traceback
                log(traceback.format_exc())


        elif event == 'STOP':
            stop_event.set()
            window['START'].update(disabled=False)
            window['STOP'].update(disabled=True)
            window['STATUS'].update('⏹ Stopped', text_color='blue')

        elif event == '_TEXT':
            window['TEXT'].update(window['TEXT'].get() + values[event] + '\n')
        elif event == '_LOG':
            window['LOG'].update(window['LOG'].get() + values[event] + '\n')
        elif event == '_STATUS':
            window['STATUS'].update(values[event], text_color='blue')
        elif event == '_AUTO_PTT':
            threading.Thread(target=run, args=(values[event],), daemon=True).start()
        last_values = values
    _save_config(last_values)
    window.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback, tempfile
        tb = traceback.format_exc()
        try:
            log_path = os.path.join(tempfile.gettempdir(), 'gui_error.log')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(tb)
        except:
            pass
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"STTTS GUI crashed:\n\n{tb}", "STTTS Error", 0x10)
