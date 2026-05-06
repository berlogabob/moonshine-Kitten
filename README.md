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
- `voice_chat_app/listener.py` — Moonshine event bridge to the engine
- `voice_chat_app/engine.py` — main conversation/runtime orchestration
- `voice_chat_app/state.py` — chat history state and trimming
- `voice_chat_app/llm.py` — Ollama response generation/cleaning
- `voice_chat_app/speech_pipeline.py` — chunked parallel TTS generation + playback queue
- `voice_chat_app/chunking.py` — text chunk splitting rules for low latency speech
- `voice_chat_app/tts_setup.py` — TTS model/voice loading and compatibility checks
- `voice_chat_app/audio.py` — audio normalization + click/pop smoothing
- `voice_chat_app/text_utils.py` — response text cleanup
- `voice_chat_app/ollama_models.py` — default model name
- `voice_chat_app/ollama_settings.py` — generation options + fallback
- `voice_chat_app/prompts.py` — system prompt
- `voice_chat_app/config.py` — non-LLM runtime constants

### Module flow (who calls whom)

1. `voice_chat.py` calls `voice_chat_app.app.run()`.
2. `app.py` initializes espeak, loads TTS/voices, runs startup TTS test, and starts `MicTranscriber`.
3. Moonshine emits transcript-line events to `listener.SafeListener`.
4. `listener.py` forwards text to `engine.VoiceChatEngine.process_user_text(...)`.
5. `engine.py` updates `state.ChatState`, requests reply from `llm.py`, then triggers speech via `speech_pipeline.py`.
6. `speech_pipeline.py` splits text (via `chunking.py`), generates TTS chunk-by-chunk, and plays queued audio.

### What to edit for common tasks

- Change LLM model:
  - `voice_chat_app/ollama_models.py`
- Change assistant personality/system instruction:
  - `voice_chat_app/prompts.py`
- Change Ollama generation behavior (`temperature`, `num_predict`, penalties):
  - `voice_chat_app/ollama_settings.py`
- Change chunking behavior (latency vs naturalness tradeoff):
  - `voice_chat_app/chunking.py`
- Change speech gain/fade/padding:
  - `voice_chat_app/audio.py`
  - and chunk playback parameters in `voice_chat_app/speech_pipeline.py`
- Change runtime paths and STT model config:
  - `voice_chat_app/config.py`
- Add app-level integrations (TouchDesigner hooks, control surface):
  - prefer `voice_chat_app/engine.py` and `voice_chat_app/app.py` as integration boundaries.

### TouchDesigner integration notes

For TouchDesigner, treat `VoiceChatEngine` as the core unit:

- input boundary: text coming from mic/STT or external UI
- output boundary: spoken playback in `speech_pipeline.py` (or replace with a custom audio sink)
- state boundary: `ChatState` history

If needed, you can create a small adapter that calls `engine.process_user_text(...)` directly from TD callbacks.

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


гзвфеу
