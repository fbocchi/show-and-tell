from collections import Counter


class Vocabulary():
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.vocab = ["[START]", "[END]"]

    def build(self, captions: list[list[str]]):
        freqs = Counter()

        for caption in captions:
            freqs.update(caption)

        tokens_and_freqs = [
            (token, freq) for token, freq in freqs.items()
            if freq >= self.min_freq
        ]

        tokens_and_freqs = sorted(tokens_and_freqs, key=lambda x: x[1], reverse=True)
        tokens = [token for token, _ in tokens_and_freqs]

        self.vocab.extend(tokens)

    def get_vocabulary(self) -> list[str]:
        return self.vocab.copy()

    @property
    def size(self):
        return len(self.vocab)
