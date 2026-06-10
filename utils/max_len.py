import numpy as np


def compute_max_len(captions: list[list[str]], percentile: int = 95) -> int:
    lengths = [len(caption) for caption in captions]
    return int(np.percentile(lengths, percentile))
