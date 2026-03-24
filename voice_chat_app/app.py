import threading
import warnings

import numpy as np
import sounddevice as sd
from moonshine_voice import MicTranscriber, ModelArch
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from .audio import normalize_audio, smooth_audio_edges
from .config import (
    AUDIO_SAMPLE_RATE,
    LOCAL_MODEL_FILE,
    LOCAL_VOICES_FILE,
    MODEL_ARCH,
    MODEL_PATH,
    VOICE,
)
from .engine import VoiceChatEngine
from .listener import SafeListener
from .ollama_models import OLLAMA_MODEL
from .tts_setup import build_tts


def run():
    EspeakWrapper.set_library("/opt/homebrew/lib/libespeak-ng.dylib")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    transcriber = None

    try:
        tts, selected_voice, _available_voices = build_tts(
            LOCAL_MODEL_FILE, LOCAL_VOICES_FILE, VOICE
        )
    except Exception as e:
        print(f"Failed to load TTS model: {e}")
        print("Quick troubleshooting steps:")
        print(f"  ls -lh {LOCAL_MODEL_FILE}")
        print(
            "  rm -rf ./kitten-tts-mini-0.8 && uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )
        raise

    engine = VoiceChatEngine(
        tts=tts,
        selected_voice=selected_voice,
        ollama_model=OLLAMA_MODEL,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        cooldown_seconds=0.8,
    )

    print("Voice chat started — feedback loop protection enabled")
    print(f"  LLM model : {OLLAMA_MODEL}")
    print(f"  TTS voice : {selected_voice}")
    print(f"  TTS model : {LOCAL_MODEL_FILE}")
    print("  AI should NOT talk to itself\n")

    print("TTS test...")
    try:
        test_audio = np.asarray(
            tts.generate("Test sound one two three", voice=selected_voice), dtype=np.float32
        )
        test_audio = normalize_audio(test_audio, gain=0.8)
        test_audio = smooth_audio_edges(test_audio, AUDIO_SAMPLE_RATE)
        sd.play(test_audio, samplerate=AUDIO_SAMPLE_RATE)
        sd.wait()
        print("TTS test completed.\n")
    except Exception as e:
        print(f"TTS test failed: {e}")

    try:
        print("Starting microphone listener...")
        transcriber = MicTranscriber(
            model_path=MODEL_PATH, model_arch=ModelArch(MODEL_ARCH)
        )
        transcriber.add_listener(SafeListener(engine=engine))
        transcriber.start()
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        if transcriber is not None:
            transcriber.stop()
        print("Goodbye!")
    except Exception as e:
        print(f"Launch error: {e}")
