# moonshine-Kitten Voice Chat

Real-time local voice assistant in Python:

- **STT**: Moonshine Voice (microphone transcription)
- **LLM**: Ollama (local chat inference)
- **TTS**: KittenTTS (ONNX speech synthesis)

## Features

- Real-time voice loop: mic → text → LLM → spoken reply
- Fully local runtime (no cloud inference in the chat loop)
- Feedback-loop protection (`is_speaking` guard + cooldown)
- Voice compatibility fallback (legacy names like `Jasper` auto-map to current local voices)
- Smoother playback (fade-in/fade-out + small silence padding to reduce clicks/pops)

## Requirements

### 1) System dependency (phonemizer / espeak-ng)

macOS:

```bash
brew install espeak-ng
```

Ubuntu / Debian:

```bash
sudo apt update && sudo apt install espeak-ng libespeak-ng1
```

Windows:

- Install `espeak-ng` (Chocolatey or manual installer from official releases)

### 2) Python + uv

- Python `>=3.12` (from `pyproject.toml`)
- Install `uv`: https://docs.astral.sh/uv/getting-started/installation/

### 3) Ollama

Install Ollama and ensure it is running (`ollama serve` or Ollama app).

Pull at least one local model, for example:

```bash
ollama pull gemma2:2b
```

## Quick start

```bash
# 1. Clone
git clone https://github.com/berlogabob/moonshine-Kitten.git
cd moonshine-Kitten

# 2. Install dependencies
uv sync

# 3. One-time TTS model download (~75 MB)
uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8

# 4. Run the app
uv run voice_chat_v03.py
```

## Current runtime behavior (`voice_chat_v03.py`)

- Loads ONNX model from:
  - `./kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx`
- Loads local voice embeddings from:
  - `./kitten-tts-mini-0.8/voices.npz`
- If configured `VOICE` is missing, it prints a warning and selects a compatible fallback.
- Performs a startup TTS self-test before microphone listener starts.

## Configuration

Edit constants near top of `voice_chat_v03.py`:

```python
OLLAMA_MODEL = "gemma2:2b"
VOICE = "Jasper"  # legacy alias supported; auto-resolved if missing
AUDIO_SAMPLE_RATE = 24000
```

Legacy aliases currently mapped:

- `Jasper` → male `expr-voice-*`
- `Bella` / `Luna` → female `expr-voice-*`

Inspect voices available on your machine:

```bash
uv run python -c "import numpy as np; z=np.load('kitten-tts-mini-0.8/voices.npz'); print(z.files)"
```

## Troubleshooting

If TTS model files are missing or corrupted:

```bash
rm -rf ./kitten-tts-mini-0.8
uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8
```

If speech sounds wrong after dependency changes:

- Re-run `uv sync`
- Verify local voices list with the command above
- Run `uv run voice_chat_v03.py` and check startup warnings
