"""STTTS CLI — Speech-to-Text → Text-to-Speech real-time voice conversion.

Usage:
    python STTTS.py                    # Run with defaults (Whisper + GPT-SoVITS)
    python STTTS.py --stt whisper --tts edge   # Use Whisper STT + Edge TTS
    python STTTS.py --help             # Show all options
"""

import argparse
import atexit
import os
import signal
import sys
import threading
import time

# Ensure package is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='STTTS — Real-time Speech-to-Text → Text-to-Speech',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stt', choices=['whisper', 'google_vad', 'google_cloud', 'mimo'],
                        default='whisper', help='STT engine (default: whisper)')
    parser.add_argument('--tts', choices=['gpt', 'edge', 'google'],
                        default='gpt', help='TTS engine (default: gpt)')
    parser.add_argument('--whisper-model', default='tiny',
                        help='Whisper model name (default: tiny)')
    parser.add_argument('--language', default='zh',
                        help='Recognition language (default: zh)')
    parser.add_argument('--ref-audio', default=None,
                        help='Reference audio for GPT-SoVITS')
    parser.add_argument('--gpt-url', default='http://127.0.0.1:9880',
                        help='GPT-SoVITS API URL')
    parser.add_argument('--device-in', type=int, default=None,
                        help='Input device index')
    parser.add_argument('--device-out', type=int, default=None,
                        help='Output device index')
    parser.add_argument('--voice', default='zh-CN-XiaoxiaoNeural',
                        help='Voice name for Edge-TTS/Google TTS')
    parser.add_argument('--extended-logging', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def main():
    args = parse_args()

    # Imports (deferred for fast --help)
    try:
        import rich
    except ImportError:
        print("This program needs the 'rich' library. Install it with: pip install rich")
        sys.exit(1)

    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    from sttts.tts import tts_gptsovits, tts_edge, tts_google
    from sttts.stt import stt_whisper, stt_google_vad, stt_google_cloud, stt_mimo

    console = Console()

    # Resolve ref audio path
    ref_audio = args.ref_audio
    if ref_audio is None:
        ref_audio = os.path.join(os.path.dirname(__file__) or '.', 'ref_audio.wav')

    # ── State ────────────────────────────────────────────
    exit_event = threading.Event()
    full_sentences = []
    displayed_text = ""

    # ── Rich display ─────────────────────────────────────
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    def update_display(current_text=""):
        """Update the Rich Live display with transcription."""
        nonlocal displayed_text
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            style = "yellow" if i % 2 == 0 else "cyan"
            rich_text += Text(sentence, style=style) + Text(" ")
        if current_text:
            rich_text += Text(current_text, style="bold yellow")

        new_plain = rich_text.plain
        if new_plain != displayed_text:
            displayed_text = new_plain
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]",
                          border_style="bold green")
            live.update(panel)

    # ── TTS playback ─────────────────────────────────────
    tts_lock = threading.Lock()

    def play_tts(text):
        """Synthesize and play text via the configured TTS engine."""
        with tts_lock:
            try:
                if args.tts == 'gpt':
                    result = tts_gptsovits(text, ref_audio, out_device=args.device_out,
                                           base_url=args.gpt_url)
                elif args.tts == 'edge':
                    result = tts_edge(text, voice=args.voice, out_device=args.device_out)
                elif args.tts == 'google':
                    result = tts_google(text, voice=args.voice, out_device=args.device_out)
                else:
                    return
                console.print(f"[bold green]{result}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]TTS error: {e}[/bold red]")

    # ── Text processing callbacks ────────────────────────
    def on_text_finalized(text):
        """Called when STT produces a final text segment."""
        full_sentences.append(text)
        update_display("")
        # Auto TTS
        if text and len(text) > 1:
            threading.Thread(target=play_tts, args=(text,), daemon=True).start()

    def on_realtime_update(text):
        """Called on real-time transcription updates (Whisper mode)."""
        update_display(text)

    # ── Commands ─────────────────────────────────────────
    def read_live_text():
        """Read all live transcribed sentences via TTS."""
        text = '，'.join(full_sentences)
        if text:
            console.print(f"[bold green]Reading live text via {args.tts}...[/bold green]")
            threading.Thread(target=play_tts, args=(text,), daemon=True).start()

    def clear_display():
        """Clear the transcription buffer."""
        full_sentences.clear()
        console.print("[bold magenta]Display cleared.[/bold magenta]")

    # ── STT thread ───────────────────────────────────────
    def run_stt():
        try:
            if args.stt == 'whisper':
                stt_whisper(args.device_in, exit_event, on_text_finalized,
                            console.print, args.whisper_model)
            elif args.stt == 'google_vad':
                stt_google_vad(args.device_in, '', exit_event, on_text_finalized,
                               console.print)
            elif args.stt == 'google_cloud':
                stt_google_cloud(args.device_in, '', exit_event, on_text_finalized,
                                 console.print)
            elif args.stt == 'mimo':
                stt_mimo(args.device_in, '', args.language, exit_event,
                         on_text_finalized, console.print)
        except Exception as e:
            console.print(f"[bold red]STT error: {e}[/bold red]")

    # ── Hotkeys (optional) ───────────────────────────────
    try:
        import keyboard
        HAS_KEYBOARD = True
    except ImportError:
        HAS_KEYBOARD = False

    if HAS_KEYBOARD:
        keyboard.add_hotkey('F1', read_live_text, suppress=True)
        keyboard.add_hotkey('F3', clear_display, suppress=True)

    # ── Cleanup ──────────────────────────────────────────
    def cleanup():
        exit_event.set()
        live.stop()
        console.print("[bold red]Exiting...[/bold red]")

    atexit.register(cleanup)

    def signal_handler(sig, frame):
        console.print("\n[bold yellow]Interrupted.[/bold yellow]")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # ── Start ────────────────────────────────────────────
    console.print(f"[bold green]STTTS CLI[/bold green] — STT: {args.stt} | TTS: {args.tts}")
    if HAS_KEYBOARD:
        console.print("[bold cyan]F1[/bold cyan]: Read live text  |  [bold cyan]F3[/bold cyan]: Clear display  |  [bold cyan]Ctrl+C[/bold cyan]: Exit")
    else:
        console.print("[dim]Install 'keyboard' for hotkey support. Press Ctrl+C to exit.[/dim]")

    stt_thread = threading.Thread(target=run_stt, daemon=True)
    stt_thread.start()

    # Keep main thread alive
    try:
        if HAS_KEYBOARD:
            keyboard.wait()
        else:
            while not exit_event.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == '__main__':
    main()
