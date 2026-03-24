VOICE = "Jasper"
MODEL_PATH = "/Users/berloga/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"
MODEL_ARCH = 5
AUDIO_SAMPLE_RATE = 24000

LOCAL_MODEL_FILE = "./kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx"
LOCAL_VOICES_FILE = "./kitten-tts-mini-0.8/voices.npz"

LEGACY_VOICE_ALIASES = {
    "Jasper": ("expr-voice-2-m", "expr-voice-3-m", "expr-voice-4-m", "expr-voice-5-m"),
    "Bella": ("expr-voice-2-f", "expr-voice-3-f", "expr-voice-4-f", "expr-voice-5-f"),
    "Luna": ("expr-voice-3-f", "expr-voice-4-f", "expr-voice-5-f", "expr-voice-2-f"),
}
