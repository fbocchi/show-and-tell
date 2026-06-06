import numpy as np


def compute_max_len(captions, percentile=95):
    lengths = [len(c.split()) for c in captions]
    return int(np.percentile(lengths, percentile)) + 2

