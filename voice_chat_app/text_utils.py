def clean_text(text: str) -> str:
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = (
        text.replace("🤖", "").replace("👋", "").replace("😊", "").replace("...", " ")
    )
    return " ".join(text.split()).strip()

