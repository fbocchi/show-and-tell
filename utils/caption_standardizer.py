import re

class CaptionStandardizer:
    def __init__(self):
        pass

    def standardize(self, caption) -> str:
        caption = caption.lower()
        caption = re.sub(r"[^\w\s]", "", caption)
        return  caption.strip()

    def standardize_all(self, captions: list[str]) -> list[str]:
        return [self.standardize(caption) for caption in captions]