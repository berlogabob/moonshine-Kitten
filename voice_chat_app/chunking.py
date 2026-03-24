import re


def split_text_for_tts(text: str, max_chunk_chars: int = 90) -> list[str]:
    parts = re.split(r"(?<=[.!?;:,])\s+", text)
    chunks = []
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        while len(piece) > max_chunk_chars:
            split_at = piece.rfind(" ", 0, max_chunk_chars)
            if split_at <= 0:
                split_at = max_chunk_chars
            chunks.append(piece[:split_at].strip())
            piece = piece[split_at:].strip()
        if piece:
            chunks.append(piece)
    return chunks or [text]

