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

def check_gptsovits():
    try:
        requests.get('http://127.0.0.1:9880/tts?text=ping&text_lang=zh&ref_audio_path=none&prompt_lang=zh', timeout=2)
        return "✅ GPT-SoVITS"
    except:
        return "❌ GPT-SoVITS"

def play_wav(path, device_idx=None):
    import soundfile as sf, sounddevice as sd
    data, sr = sf.read(path)
    sd.play(data, sr, device=device_idx)
    sd.wait()

# ── TTS ──────────────────────────────────────────────────

def tts_gpt(text, ref_audio, prompt_lang, prompt_text, out_device):
    url = 'http://127.0.0.1:9880/tts'
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

# ── STT workers ──────────────────────────────────────────

def stt_whisper(dev_idx, stop_event, on_text, log):
    from RealtimeSTT import AudioToTextRecorder
    log("Loading Whisper STT...")
    rec = AudioToTextRecorder(
        spinner=False, model='tiny', input_device_index=dev_idx,
        realtime_model_type='tiny', language='zh',
        silero_sensitivity=0.05, webrtc_sensitivity=3,
        post_speech_silence_duration=0.5,
        min_length_of_recording=1.1, min_gap_between_recordings=0,
    )
    log("Whisper STT ready")
    while not stop_event.is_set():
        t = rec.text()
        if t and not stop_event.is_set():
            on_text(t)
    try: rec.shutdown()
    except: pass

def stt_google(dev_idx, api_key, stop_event, on_text, log):
    import speech_recognition as sr
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=dev_idx)
    log("Google STT adjusting for ambient noise...")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    log("Google STT ready")
    while not stop_event.is_set():
        try:
            with mic as source:
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google_cloud(audio, api_key=api_key, language='zh-CN')
            if text and not stop_event.is_set():
                on_text(text)
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            log(f"Google STT error: {e}")

# ── main window ──────────────────────────────────────────

def main():
    in_devs = list_devices('input')
    out_devs = list_devices('output')
    def_dev_in = next((d for d in in_devs if 'B1' in d and 'Voicemeeter' in d), in_devs[0] if in_devs else '')
    def_dev_out = next((d for d in out_devs if 'Headphones' in d or 'Speaker' in d), out_devs[0] if out_devs else '')

    layout = [
        [sg.Frame('STT Engine', [
            [sg.Radio('Whisper (local GPU)', 'STT', key='STT_W', default=True),
             sg.Radio('Google Cloud STT', 'STT', key='STT_G')],
            [sg.Text('Input device:'), sg.Combo(in_devs, default_value=def_dev_in, key='DEV_IN', size=(50,1))],
            [sg.Text('Google key:'), sg.Input(key='GKEY', size=(50,1),
                default_text='AIzaSyB1C5UCP1kTB0uQoXAxKPMWrVg8ym24qyU', password_char='*')],
        ])],
        [sg.Frame('TTS Engine', [
            [sg.Radio('GPT-SoVITS', 'TTS', key='TTS_GPT', default=True),
             sg.Radio('Edge-TTS', 'TTS', key='TTS_EDGE')],
            [sg.Text('Ref audio:'), sg.Input(key='REF', size=(42,1),
                default_text=r"I:\STTTS\GPT-SoVITS_Mortis_Mutsumi_0104等3个文件\GPT-SoVITS_Mortis_Mutsumi_0104\model_Mutsumi_beta_0103\model_Mutsumi_beta_0103\サキ、ムシカが壊れたらサキも.wav"),
             sg.FileBrowse(file_types=(("WAV", "*.wav"),))],
            [sg.Text('Voice:'), sg.Combo(
                ['zh-CN-XiaoxiaoNeural','zh-CN-XiaoyiNeural','zh-CN-YunjianNeural',
                 'zh-CN-YunxiNeural','zh-CN-YunyangNeural','zh-CN-XiaohanNeural'],
                default_value='zh-CN-XiaoxiaoNeural', key='VOICE', size=(30,1))],
            [sg.Button('▶ GPT-SoVITS', key='GPT_START', size=(14,1)),
             sg.Button('■ GPT-SoVITS', key='GPT_STOP', size=(14,1), disabled=True, button_color='red'),
             sg.Text(check_gptsovits(), key='SRV', size=(16,1))],
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
               r'I:\STTTS\GPT-SoVITS\api_v2.py', '-a', '127.0.0.1', '-p', '9880',
               '-c', r'GPT_SoVITS/configs/tts_infer.yaml']


    def log(msg):
        window.write_event_value('_LOG', f"[{time.strftime('%H:%M:%S')}] {msg}")

    def handle_text(text, cfg):
        window.write_event_value('_TEXT', text)
        log(f"Recognized: {text}")
        if cfg['tts'] == 'gpt':
            threading.Thread(target=lambda: do_gpt(text, cfg), daemon=True).start()
        else:
            threading.Thread(target=lambda: do_edge(text, cfg), daemon=True).start()

    def do_gpt(text, cfg):
        if not os.path.exists(cfg['ref']):
            log("Ref audio missing!")
            return
        log(tts_gpt(text, cfg['ref'], 'ja', '', cfg.get('dev_out')))

    def do_edge(text, cfg):
        log(tts_edge(text, cfg['voice'], cfg.get('dev_out')))

    def run(cfg):
        cb = lambda t: handle_text(t, cfg)
        if cfg['stt'] == 'whisper':
            stt_whisper(cfg['dev_in'], stop_event, cb, log)
        else:
            stt_google(cfg['dev_in'], cfg['gkey'], stop_event, cb, log)
        window.write_event_value('_STATUS', '⏹ Stopped')

    def update_srv_status():
        st = check_gptsovits()
        window['SRV'].update(st)
        return '✅' in st

    while True:
        event, values = window.read(timeout=2000)
        if event == sg.WINDOW_CLOSED:
            stop_event.set()
            if gpt_process:
                try: gpt_process.terminate()
                except: pass
            break

        # Periodic server health check
        if event == '__TIMEOUT__':
            was_running = '✅' in window['SRV'].get()
            now_running = update_srv_status()
            # Auto-update button states
            if now_running:
                window['GPT_START'].update(disabled=True)
                window['GPT_STOP'].update(disabled=False)
            else:
                window['GPT_START'].update(disabled=False)
                window['GPT_STOP'].update(disabled=True)
            continue

        if event == 'GPT_START':
            if gpt_process and gpt_process.poll() is None:
                log("GPT-SoVITS already running")
            else:
                log("Starting GPT-SoVITS...")
                import subprocess
                gpt_process = subprocess.Popen(GPT_CMD,
                    cwd=r'I:\STTTS\GPT-SoVITS',
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
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
            update_srv_status()

        if event == 'START':
            cfg = {
                'stt': 'whisper' if values['STT_W'] else 'google',
                'tts': 'gpt' if values['TTS_GPT'] else 'edge',
                'dev_in': int(values['DEV_IN'].split(':')[0]),
                'dev_out': int(values['DEV_OUT'].split(':')[0]) if values['DEV_OUT'] else None,
                'ref': values['REF'],
                'gkey': values['GKEY'],
                'voice': values['VOICE'],
            }
            if cfg['stt'] == 'google' and not cfg['gkey']:
                sg.popup_error('Google API key required!'); continue
            if cfg['tts'] == 'gpt' and not os.path.exists(cfg['ref']):
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
    main()
