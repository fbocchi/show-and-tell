from utils.text_cleaner import clean_all
from utils.vocabulary import Vocabulary
from utils.max_len import compute_max_len
from utils.text_vectorizer import TextVectorizer

if __name__ == '__main__':

    captions = [
        "A dog, running in the park!",
        "A cat... sitting on a sofa??",
        "A man riding a bicycle!!!",
        "Two birds flying-in the sky.",
        "A child playing with a ball."
    ]

    captions = clean_all(captions)
    print(f"Clean captions: {captions}")

    vocab = Vocabulary(min_freq=1)
    vocab.build(captions)
    print(f"Vocabulary size: {vocab.size}")

    max_len = compute_max_len(captions)
    print(f"Max len: {max_len}")

    vec = TextVectorizer(vocab=vocab, max_len=max_len)
    captions = vec.vectorize_all(captions)
    print(f"Vectorized captions: {captions}")

    print(len(captions[2][:-1]))
    print(len(captions[2][1:]))

