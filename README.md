# moonshine-Kitten Voice Chat
A lightweight, real-time voice chat application built with Python using fully local models:
- **Speech-to-Text (STT)** — Moonshine Voice (real-time microphone transcription) 
- **Language Model (LLM)** — Ollama (local inference) 
- **Text-to-Speech (TTS)** — KittenTTS (ONNX-based voice synthesis)
## Features
- Real-time voice input → text → LLM response → spoken output 
- Feedback loop protection (AI does not hear itself) 
- Fully local processing (no cloud APIs after initial setup) 
- Easy switching of Ollama models and TTS voices
## Requirements
### 1. System dependencies
**macOS (Apple Silicon recommended)**
```bash # Required by phonemizer (used inside KittenTTS) brew install espeak-ng ```
**Windows**
```powershell 
# Recommended via Chocolatey choco install espeak-ng
# Or download manually: 
# https://github.com/espeak-ng/espeak-ng/releases
```
**Ubuntu / Debian**
```bash
sudo apt update sudo apt install espeak-ng libespeak-ng1
```
### 2. Python & uv
- Python 3.10 – 3.13 - [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast modern Python project & dependency manager.
Install uv if you don't have it yet:
```bash
# macOS / Linux 
curl -LsSf https://astral.sh/uv/install.sh "| sh
```
### 3. Ollama (local LLM backend)
- Download and install: https://ollama.com 
- Start Ollama server: run `ollama serve` or open Ollama.app 
- Pull at least one model (example):
```bash
ollama pull gemma2:2b
# other good small models: # ollama pull phi3:mini # ollama pull llama3.2:3b # ollama pull tinyllama:1.1b
```
## Quick Start
```bash 
# 1. Clone the repository
git clone https://github.com/berlogabob/moonshine-Kitten.git cd moonshine-Kitten
# 2. Install all Python dependencies (creates .venv automatically)
uv sync
# 3. Download the TTS model (~75 MB, one-time)
uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8
# 4. Run the voice chat
uv run voice_chat_v03.py
```
## Configuration (edit in `voice_chat_v03.py`)
```python
# Which Ollama model to use (must be pulled already)
OLLAMA_MODEL = "gemma2:2b"
# alternatives: "phi3:mini", "llama3.2:3b", "tinyllama:1.1b"
# Preferred TTS voice
VOICE = "Jasper"
```

`voice_chat_v03.py` now resolves legacy voice names (for example `Jasper`, `Bella`, `Luna`) to available voices from `kitten-tts-mini-0.8/voices.npz`.
If your configured voice is unavailable, the app prints a warning and auto-selects a compatible fallback.

To inspect your local voices:
```bash
uv run python -c "import numpy as np; z=np.load('kitten-tts-mini-0.8/voices.npz'); print(z.files)"
```
