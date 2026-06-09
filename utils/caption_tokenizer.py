
class CaptionTokenizer:
    def __init__(self):
        pass

    def tokenize(self, caption: str) -> list[str]:
        return caption.split()

    def tokenize_all(self, captions: list[str]) -> list[list[str]]:
        return [self.tokenize(caption) for caption in captions]