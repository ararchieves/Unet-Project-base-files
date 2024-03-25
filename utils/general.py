def split_classes(t):
    return t[:, 0:1, ...], t[:, 1:2, ...]