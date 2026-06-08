from utils.vocabulary import Vocabulary


def add_start_and_end_tokens(captions: list[list[str]], vocab: Vocabulary) -> list[list[str]]:
    """
    Riceve delle caption già tokenizzate.
    """
    return [[vocab.start_token] + caption + [vocab.end_token] for caption in captions]