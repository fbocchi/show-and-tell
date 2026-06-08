import numpy as np


def compute_max_len(captions: list[list[str]], percentile=95) -> int:
    """
    Riceve delle caption già tokenizzate.
    """
    lengths = [len(caption) for caption in captions]
    return int(np.percentile(lengths, percentile))

