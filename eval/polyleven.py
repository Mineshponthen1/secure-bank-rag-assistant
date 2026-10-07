"""Plain-Python stand-in for the polyleven package.

Windows Application Control blocks polyleven's compiled DLL on this laptop.
autoevals only uses it for its Levenshtein text-similarity scorer, which this
project does not use, so a simple, slow-but-correct version is enough.
"""


def levenshtein(a, b, k=None):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
