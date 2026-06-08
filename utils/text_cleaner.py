import re


def clean(caption: str) -> str:
    caption = caption.lower()
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption.strip()

def clean_all(captions: list[str]) -> list[str]:
    return [clean(caption) for caption in captions]