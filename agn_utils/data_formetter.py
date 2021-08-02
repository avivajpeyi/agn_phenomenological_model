import functools


def dl_to_ld(dl):
    """dict of list to list of dict"""
    ld = [{key: value[index] for key, value in dl.items()}
          for index in range(max(map(len, dl.values())))]
    return ld


def ld_to_dl(ld):
    dl = {key: [item[key] for item in ld]
          for key in list(functools.reduce(
            lambda x, y: x.union(y),
            (set(dicts.keys()) for dicts in ld)
        ))}
    return dl