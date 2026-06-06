from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []

        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.START = "<start>"
        self.END = "<end>"

    @property
    def pad_token(self):
        return self.PAD

    @property
    def unk_token(self):
        return self.UNK

    @property
    def start_token(self):
        return self.START

    @property
    def end_token(self):
        return self.END

    @property
    def end_id(self):
        return self.word_to_idx[self.END]

    @property
    def size(self):
        return len(self.vocab)

    def build(self, captions):
        counter = Counter()

        for cap in captions:
            counter.update(cap.split())

        words = [
            (w, f) for w, f in counter.items()
            if f >= self.min_freq
        ]

        # sort by frequency (desc)
        words = sorted(words, key=lambda x: x[1], reverse=True)
        words = [w for w, _ in words]

        self.vocab = [self.PAD, self.UNK, self.START, self.END] + words

        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def encode(self, tokens):
        return [
            self.word_to_idx.get(tok, self.word_to_idx[self.UNK])
            for tok in tokens
        ]