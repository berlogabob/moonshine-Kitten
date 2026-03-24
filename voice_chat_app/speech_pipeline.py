import queue
import threading

import numpy as np
import sounddevice as sd

from .audio import normalize_audio, smooth_audio_edges
from .chunking import split_text_for_tts


def _playback_worker(audio_queue: "queue.Queue[np.ndarray | None]", sample_rate: int):
    while True:
        item = audio_queue.get()
        if item is None:
            audio_queue.task_done()
            break
        sd.play(item, samplerate=sample_rate)
        sd.wait()
        audio_queue.task_done()


def speak_text_chunked(
    tts,
    text: str,
    voice: str,
    sample_rate: int,
    gain: float = 0.72,
    max_chunk_chars: int = 90,
):
    text_chunks = split_text_for_tts(text, max_chunk_chars=max_chunk_chars)
    audio_queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=2)
    player = threading.Thread(
        target=_playback_worker, args=(audio_queue, sample_rate), daemon=True
    )
    player.start()

    try:
        for chunk in text_chunks:
            raw = tts.generate(chunk, voice=voice)
            audio = np.asarray(raw, dtype=np.float32)
            audio = normalize_audio(audio, gain=gain)
            audio = smooth_audio_edges(audio, sample_rate)
            audio_queue.put(audio)
    finally:
        audio_queue.put(None)
        audio_queue.join()
        player.join(timeout=0.5)

