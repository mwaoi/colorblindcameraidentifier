import threading

# Linux requires espeak: sudo apt install espeak
# macOS uses nsss engine automatically
# Windows uses SAPI5 engine automatically

_speech_thread: threading.Thread | None = None
_volume: float = 1.0
_rate: int = 150


def set_volume(v: float) -> None:
    global _volume
    _volume = max(0.0, min(1.0, v))


def set_rate(r: int) -> None:
    global _rate
    _rate = max(50, min(300, r))


def speak(text: str) -> None:
    """Speak text aloud in a background thread. Skips if already speaking."""
    global _speech_thread
    if _speech_thread is not None and _speech_thread.is_alive():
        return
    _speech_thread = threading.Thread(target=_run_speech, args=(text,), daemon=True)
    _speech_thread.start()


def _run_speech(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", _rate)
        engine.setProperty("volume", _volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass
