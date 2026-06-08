from utils.add_start_and_end_tokens import add_start_and_end_tokens
from utils.text_cleaner import clean_all
from utils.tokenizer import Tokenizer
from utils.vocabulary import Vocabulary
from utils.max_len import compute_max_len
from utils.text_vectorizer import TextVectorizer


def teacher_forcing(captions):
    inputs = []
    targets = []
    for tokens in captions:
        inputs.append(tokens[:-1])
        targets.append(tokens[1:])
    return inputs, targets

def main():
    captions = [
        "A dog, running in the park!",
        "A cat... sitting on a sofa??",
        "A man riding a bicycle!!!",
        "Two birds flying-in the sky.",
        "A child playing with a ball."
    ]

    tokenizer = Tokenizer()

    vocab = Vocabulary(tokenizer, min_freq=1)

    captions = clean_all(captions)
    print(f"Clean captions: {captions}")

    vocab.build(captions)
    print(f"Vocabulary size: {vocab.size}")

    captions = tokenizer.tokenize_all(captions)

    captions = add_start_and_end_tokens(captions, vocab)
    print(f"Captions: {captions}")

    max_len = compute_max_len(captions)
    print(f"Max len: {max_len}")

    inputs, targets = teacher_forcing(captions)
    print("Teacher forcing")
    for c, (i, t) in enumerate(zip(inputs, targets)):
        print(f"{c}: input: {i}, target: {t}")

    vec = TextVectorizer(vocab, max_len)

    inputs = vec.vectorize_all(inputs)
    targets = vec.vectorize_all(targets)

    print("Vectorized captions")
    for c, (i, t) in enumerate(zip(inputs, targets)):
        print(f"{c}: input: {i}, target: {t}")

    print("Vectorized inputs")
    print(inputs)

    print("Vectorized targets")
    print(targets)

if __name__ == '__main__':
    main()

