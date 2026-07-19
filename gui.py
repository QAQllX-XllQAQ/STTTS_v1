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
        requests.get(f'{base_url}/tts?text=ping&text_lang=zh&ref_audio_path=none&prompt_lang=zh', timeout=2)
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
GOOGLE_VOICES = ['zh-CN-Wavenet-A','zh-CN-Wavenet-B','zh-CN-Wavenet-C','zh-CN-Wavenet-D',
                 'zh-CN-Standard-A','zh-CN-Standard-B','zh-CN-Standard-C','zh-CN-Standard-D']

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
    asyncio.run(edge_tts.Communicate(text, voice).save(tmp))
    import soundfile as sf, sounddevice as sd
    data, sr = sf.read(tmp)
    sd.play(data, sr, device=out_device)
    sd.wait()
    return f"Edge-TTS OK"

def tts_google(text, voice, api_key, out_device):
    url = f'https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}'
    body = {
        'input': {'text': text},
        'voice': {'languageCode': 'zh-CN'},
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

def stt_whisper(dev_idx, stop_event, on_text, log):
    from RealtimeSTT import AudioToTextRecorder
    log("Loading Whisper STT...")
    rec = AudioToTextRecorder(
        spinner=False, model='tiny', input_device_index=dev_idx,
        realtime_model_type='tiny', language='zh',
        silero_sensitivity=0.3, webrtc_sensitivity=2,
        post_speech_silence_duration=0.2,
        pre_recording_buffer_duration=0.5,
        min_length_of_recording=0.5, min_gap_between_recordings=0.3,
    )
    log("Whisper STT ready")
    while not stop_event.is_set():
        t = rec.text()
        if t and not stop_event.is_set():
            on_text(t)
    try: rec.shutdown()
    except: pass


def stt_google_vad(dev_idx, api_key, stop_event, on_text, log):
    """Local VAD + REST API: webrtcvad detects speech, sends only speech segments"""
    import pyaudio as pa, webrtcvad, collections
    RATE, CHUNK, VAD_MODE = 16000, 480, 1
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt16, channels=1, rate=RATE,
                    input=True, input_device_index=dev_idx,
                    frames_per_buffer=CHUNK)
    vad = webrtcvad.Vad(VAD_MODE)
    log("Google STT ready (local VAD mode)")
    PRE_BUFFER_FRAMES = int(RATE / CHUNK * 0.5)
    ring_buffer = collections.deque(maxlen=PRE_BUFFER_FRAMES)
    while not stop_event.is_set():
        frames, triggered, silence = [], False, 0
        MAX_SILENCE = int(RATE / CHUNK * 0.8)
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
        body = {
            'config': {'encoding':'LINEAR16','sampleRateHertz':RATE,'languageCode':'zh-CN'},
            'audio': {'content': base64.b64encode(b''.join(frames)).decode()}
        }
        url = f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}'
        try:
            resp = requests.post(url, json=body, timeout=15)
            if resp.status_code == 200:
                results = resp.json().get('results')
                if results:
                    text = results[0]['alternatives'][0]['transcript']
                    if text and not stop_event.is_set():
                        on_text(text)
            else:
                log(f"Google STT API error: {resp.status_code}")
        except Exception as e:
            log(f"Google STT error: {e}")
    stream.stop_stream(); stream.close(); p.terminate()

def stt_google_cloud(dev_idx, api_key, stop_event, on_text, log):
    """Cloud VAD mode: send 2s fixed chunks, let Google handle VAD"""
    import pyaudio as pa, base64 as b64
    RATE, CHUNK_DURATION = 16000, 2.0
    CHUNK_SIZE = int(RATE * CHUNK_DURATION)
    p = pa.PyAudio()
    stream = p.open(format=pa.paInt16, channels=1, rate=RATE,
                    input=True, input_device_index=dev_idx,
                    frames_per_buffer=1024)
    log("Google STT ready (cloud VAD mode)")
    buffer = []
    bytes_read = 0
    while not stop_event.is_set():
        data = stream.read(1024, exception_on_overflow=False)
        buffer.append(data)
        bytes_read += len(data)
        if bytes_read >= CHUNK_SIZE * 2:
            audio_bytes = b''.join(buffer)
            buffer = []; bytes_read = 0
            body = {
                'config': {'encoding':'LINEAR16','sampleRateHertz':RATE,'languageCode':'zh-CN',
                           'enableAutomaticPunctuation':True},
                'audio': {'content': b64.b64encode(audio_bytes).decode()}
            }
            url = f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}'
            try:
                resp = requests.post(url, json=body, timeout=15)
                if resp.status_code == 200:
                    results = resp.json().get('results')
                    if results:
                        text = results[0]['alternatives'][0]['transcript']
                        if text and not stop_event.is_set():
                            on_text(text)
            except Exception as e:
                pass  # silent
    stream.stop_stream(); stream.close(); p.terminate()

def main():
    in_devs = list_devices('input')
    out_devs = list_devices('output')
    def_dev_in = next((d for d in in_devs if 'B1' in d and 'Voicemeeter' in d), in_devs[0] if in_devs else '')
    def_dev_out = next((d for d in out_devs if 'Headphones' in d or 'Speaker' in d), out_devs[0] if out_devs else '')

    layout = [
        [sg.Frame('STT Engine', [
            [sg.Radio('Whisper (local GPU)', 'STT', key='STT_W', default=True, enable_events=True),
             sg.Radio('Google Cloud STT', 'STT', key='STT_G', enable_events=True)],
            [sg.pin(sg.Col([[sg.Radio('Local VAD + REST (save cost)', 'GOOGLE_MODE', key='GM_VAD', default=True),
              sg.Radio('Cloud process (full audio)', 'GOOGLE_MODE', key='GM_CLOUD')]], key='COL_GM', visible=True))],
            [sg.Text('Input device:'), sg.Combo(in_devs, default_value=def_dev_in, key='DEV_IN', size=(50,1))],
            [sg.Text('Google key:'), sg.Input(key='GKEY', size=(50,1),
                default_text='', password_char='*')],
        ])],
        [sg.Frame('TTS Engine', [
            [sg.Radio('GPT-SoVITS', 'TTS', key='TTS_GPT', default=True, enable_events=True),
             sg.Radio('Edge-TTS', 'TTS', key='TTS_EDGE', enable_events=True),
             sg.Radio('Google TTS', 'TTS', key='TTS_GOOGLE', enable_events=True)],
            [sg.pin(sg.Col([[sg.Text('Ref audio:'), sg.Input(key='REF', size=(42,1),
                default_text=os.path.join(os.path.dirname(__file__) or '.', 'ref_audio.wav')),
             sg.FileBrowse(file_types=(("WAV", "*.wav"),))]], key='COL_REF', visible=True))],
            [sg.pin(sg.Col([[sg.Text('GPT URL:'), sg.Input('http://127.0.0.1:9880', key='GPT_URL', size=(35,1))],
             [sg.Button('▶ GPT-SoVITS', key='GPT_START', size=(14,1)),
             sg.Button('■ GPT-SoVITS', key='GPT_STOP', size=(14,1), disabled=True, button_color='red'),
             sg.Text(check_gptsovits(), key='SRV', size=(16,1))]], key='COL_GPT', visible=True))],
            [sg.Combo(EDGE_VOICES, default_value='zh-CN-XiaoxiaoNeural', key='VOICE', size=(30,1))],
        ])],
        [sg.Frame('Audio Output', [
            [sg.Text('Play to:'), sg.Combo(out_devs, default_value=def_dev_out, key='DEV_OUT', size=(50,1))],
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
            window['VOICE'].update(values=GOOGLE_VOICES, value='zh-CN-Wavenet-A')
        else:
            window['VOICE'].update(values=[], value='')

    # Apply initial visibility
    switch_tts('gpt')

    def log(msg):
        window.write_event_value('_LOG', f"[{time.strftime('%H:%M:%S')}] {msg}")

    def handle_text(text, cfg):
        window.write_event_value('_TEXT', text)
        log(f"Recognized: {text}")
        if cfg['tts'] == 'gpt':
            threading.Thread(target=lambda: do_gpt(text, cfg), daemon=True).start()
        elif cfg['tts'] == 'google':
            threading.Thread(target=lambda: do_google(text, cfg), daemon=True).start()
        else:
            threading.Thread(target=lambda: do_edge(text, cfg), daemon=True).start()
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
        cb = lambda t: handle_text(t, cfg)
        if cfg['stt'] == 'whisper':
            stt_whisper(cfg['dev_in'], stop_event, cb, log)
        elif cfg['stt'] == 'google_vad':
            stt_google_vad(cfg['dev_in'], cfg['gkey'], stop_event, cb, log)
        else:
            stt_google_cloud(cfg['dev_in'], cfg['gkey'], stop_event, cb, log)
        window.write_event_value('_STATUS', '⏹ Stopped')


    def update_srv_status(gpt_url):
        st = check_gptsovits(gpt_url)
        window['SRV'].update(st)
        return '✅' in st

    # Initial visibility
    window['COL_GM'].update(visible=False)

    while True:
        event, values = window.read(timeout=2000)
        if event == sg.WINDOW_CLOSED:
            stop_event.set()
            if gpt_process:
                try: gpt_process.terminate()
                except: pass
            break

        # STT radio toggle → show Google mode options
        if event in ('STT_W', 'STT_G'):
            window['COL_GM'].update(visible=values['STT_G'])

        # TTS radio toggle
        if event in ('TTS_GPT', 'TTS_EDGE', 'TTS_GOOGLE'):
            if values['TTS_GPT']: switch_tts('gpt')
            elif values['TTS_EDGE']: switch_tts('edge')
            elif values['TTS_GOOGLE']: switch_tts('google')

        # Periodic server health check
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
            gm = 'google_vad' if values.get('GM_VAD', True) else 'google_cloud'
            stt = 'whisper' if values['STT_W'] else gm
            tts = 'gpt' if values['TTS_GPT'] else ('google' if values['TTS_GOOGLE'] else 'edge')
            cfg = {
                'stt': stt, 'tts': tts,
                'dev_in': int(values['DEV_IN'].split(':')[0]),
                'dev_out': int(values['DEV_OUT'].split(':')[0]) if values['DEV_OUT'] else None,
                'ref': values['REF'], 'gkey': values['GKEY'], 'voice': values['VOICE'],
                'gpt_url': values.get('GPT_URL', 'http://127.0.0.1:9880'),
            }
            if stt.startswith('google') and (not cfg['gkey'] or 'YOUR_' in cfg['gkey']):
                sg.popup_error('Google API key required!'); continue
            if tts == 'google' and (not cfg['gkey'] or 'YOUR_' in cfg['gkey']):
                sg.popup_error('Google API key required!'); continue
            if tts == 'gpt' and not os.path.exists(cfg['ref']):
                sg.popup_error('Ref audio not found!'); continue

            stop_event.clear()
            window['START'].update(disabled=True)
            window['STOP'].update(disabled=False)
            window['TEXT'].update('')
            window['LOG'].update('')
            window['STATUS'].update('🟢 Running...')
            threading.Thread(target=run, args=(cfg,), daemon=True).start()

        elif event == 'STOP':
            stop_event.set()
            window['START'].update(disabled=False)
            window['STOP'].update(disabled=True)
            window['STATUS'].update('⏹ Stopped')

        elif event == '_TEXT':
            window['TEXT'].update(window['TEXT'].get() + values[event] + '\n')
        elif event == '_LOG':
            window['LOG'].update(window['LOG'].get() + values[event] + '\n')

    window.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        with open(os.path.join(os.path.dirname(__file__) or '.', 'gui_error.log'), 'w') as f:
            traceback.print_exc(file=f)
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"STTTS GUI crashed:\n\n{traceback.format_exc()}", "STTTS Error", 0x10)
