import string
from random import shuffle


def oversample(group_1, group_2):
    if len(group_2) > len(group_1):
        group_1, group_2 = group_2, group_1

    if len(group_1) > len(group_2):
        while len(group_1) > len(group_2):
            group_2 += group_2
        group_2 = group_2[: len(group_1)]

    data = group_1 + group_2
    shuffle(data)
    return data


MAPPING = {k: i + 1 for i, k in enumerate(string.printable)}


def map_and_cap(smile: str, cap_len: int = 74):
    smile_ints = [MAPPING[x] for x in smile]
    if len(smile_ints) >= cap_len:
        smile_ints = smile_ints[:cap_len]
    else:
        smile_ints += [0] * (cap_len - len(smile_ints))

    return smile_ints
