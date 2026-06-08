from collections import Counter

from utils.tokenizer import Tokenizer


class Vocabulary:
    def __init__(self, tokenizer: Tokenizer, min_freq=5):
        self.tokenizer = tokenizer
        self.min_freq = min_freq

        self.token_to_idx = {}
        self.idx_to_token = {}
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
        return self.token_to_idx[self.END]

    @property
    def size(self):
        return len(self.vocab)

    def build(self, captions: list[str]):
        frequency_counter = Counter()

        captions = self.tokenizer.tokenize_all(captions)

        for tokens in captions:
            frequency_counter.update(tokens)

        tokens = [
            (token, freq) for token, freq in frequency_counter.items()
            if freq >= self.min_freq
        ]

        # sort by frequency (desc)
        tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
        tokens = [w for w, _ in tokens]

        self.vocab = [self.PAD, self.UNK, self.START, self.END] + tokens

        self.token_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_token = {i: w for w, i in self.token_to_idx.items()}

    def vectorize(self, tokens: list[str]):
        return [
            self.token_to_idx.get(token, self.token_to_idx[self.UNK])
            for token in tokens
        ]