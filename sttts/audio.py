"""Audio utilities: device listing, playback, PCM/WAV conversion."""

import os
import struct
import tempfile
import threading

import pyaudio
import sounddevice as sd
import soundfile as sf


def list_devices(kind='input'):
    """List audio devices. kind='input' or 'output'."""
    p = pyaudio.PyAudio()
    items = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if kind == 'input' and info['maxInputChannels'] > 0:
                items.append(f"{i}: {info['name'].strip()}")
            elif kind == 'output' and info['maxOutputChannels'] > 0:
                items.append(f"{i}: {info['name'].strip()}")
    finally:
        p.terminate()
    return items


def play_wav(path, device_idx=None):
    """Play a WAV/MP3 file on the specified output device."""
    data, sr = sf.read(path)
    sd.play(data, sr, device=device_idx)
    sd.wait()


def play_audio_bytes(audio_bytes, media_type='wav', device_idx=None):
    """Play audio from raw bytes (saves to temp file first)."""
    ext = '.wav' if media_type == 'wav' else '.mp3'
    tmp = os.path.join(tempfile.gettempdir(), f'sttts_tmp{ext}')
    with open(tmp, 'wb') as f:
        f.write(audio_bytes)
    play_wav(tmp, device_idx)


def pcm_to_wav(pcm_data, sample_rate=16000, channels=1, sample_width=2):
    """Convert raw PCM data to WAV format with header."""
    data_len = len(pcm_data)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_len, b'WAVE',
        b'fmt ', 16, 1, channels, sample_rate,
        sample_rate * channels * sample_width, channels * sample_width, sample_width * 8,
        b'data', data_len,
    )
    return header + pcm_data


def parse_device_index(device_str):
    """Extract device index from 'N: name' format string. Returns int or None."""
    if not device_str:
        return None
    try:
        return int(device_str.split(':')[0])
    except (ValueError, IndexError):
        return None


def get_wav_duration(path):
    """Get duration of a WAV file in seconds."""
    try:
        data, sr = sf.read(path)
        return len(data) / sr
    except Exception:
        return 0.0
