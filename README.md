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

# 2. Install dependencies into project-local virtual env (.venv)
UV_PROJECT_ENVIRONMENT=.venv uv sync

# 3. One-time TTS model download (~75 MB)
UV_PROJECT_ENVIRONMENT=.venv uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8

# 4. Run the app
UV_PROJECT_ENVIRONMENT=.venv uv run voice_chat.py
```

## Current runtime behavior (`voice_chat.py`)

- Loads ONNX model from:
  - `./kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx`
- Loads local voice embeddings from:
  - `./kitten-tts-mini-0.8/voices.npz`
- If configured `VOICE` is missing, it prints a warning and selects a compatible fallback.
- Performs a startup TTS self-test before microphone listener starts.
- Source is split into small modules under `voice_chat_app/`.

## Project structure

- `voice_chat.py` — minimal entrypoint
- `voice_chat_app/app.py` — startup flow and main loop
- `voice_chat_app/listener.py` — transcription event handler and LLM/TTS loop
- `voice_chat_app/tts_setup.py` — TTS model/voice loading and compatibility checks
- `voice_chat_app/audio.py` — audio normalization + click/pop smoothing
- `voice_chat_app/text_utils.py` — response text cleanup
- `voice_chat_app/ollama_models.py` — default model name
- `voice_chat_app/ollama_settings.py` — generation options + fallback
- `voice_chat_app/prompts.py` — system prompt
- `voice_chat_app/config.py` — non-LLM runtime constants

## Configuration

Edit constants in:

- `voice_chat_app/ollama_models.py` (LLM model)
- `voice_chat_app/prompts.py` (system prompt)
- `voice_chat_app/config.py` (voice/audio/model paths)

```python
OLLAMA_MODEL = "gemma2:2b"
VOICE = "Jasper"  # legacy alias supported; auto-resolved if missing
AUDIO_SAMPLE_RATE = 24000
```

## Zed editor setup (Python diagnostics)

This repo uses a project-local virtual environment at `./.venv`.
Set Zed to use it (already configured in `.zed/settings.json`):

```json
{
  "languages": {
    "Python": {
      "language_servers": ["pyright"],
      "venv": {
        "path": ".",
        "default": ".venv"
      }
    }
  }
}
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

- Re-run `UV_PROJECT_ENVIRONMENT=.venv uv sync`
- Verify local voices list with the command above
- Run `UV_PROJECT_ENVIRONMENT=.venv uv run voice_chat.py` and check startup warnings
