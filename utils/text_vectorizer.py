import numpy as np

from utils.vocabulary import Vocabulary


class TextVectorizer:
    def __init__(self, vocab: Vocabulary, max_len: int):
        self.vocab = vocab
        self.max_len = max_len

    def vectorize(self, tokens: list[str]):
        return self.vocab.vectorize(tokens)

    def vectorize_all(self, captions: list[list[str]]):
        vectors = [self._truncate_with_end(v) for v in [self.vectorize(c) for c in captions]]
        return self.pad(vectors)

    def _truncate_with_end(self, vector: list[int]):
        if len(vector) > self.max_len:
            vector = vector[:self.max_len]
            vector[-1] = self.vocab.end_id

        return vector

    def pad(self, vectors: list[list[int]]):
        padded = np.zeros((len(vectors), self.max_len), dtype=np.int32)

        for i, vector in enumerate(vectors):
            padded[i, :len(vector)] = vector

        return padded