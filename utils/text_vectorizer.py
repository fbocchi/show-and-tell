import numpy as np


class TextVectorizer:
    def __init__(self, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len

    def vectorize(self, caption):
        tokens = caption.split()
        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]
        return self.vocab.encode(tokens)

    def vectorize_all(self, captions):
        vectors = [self._truncate_with_end(v) for v in [self.vectorize(c) for c in captions]]
        return self.pad(vectors)

    def _truncate_with_end(self, vec):
        """
        Se la sequenza supera max_len:
        - tronca
        - forza ultimo token = <end>
        """
        if len(vec) > self.max_len:
            vec = vec[:self.max_len]
            vec[-1] = self.vocab.end_id

        return vec

    def pad(self, vectors):
        padded = np.zeros((len(vectors), self.max_len), dtype=np.int32)

        for i, vec in enumerate(vectors):
            padded[i, :len(vec)] = vec

        return padded