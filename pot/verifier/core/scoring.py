import difflib


def normalize(text: str) -> str:
    # simple normalization; expand as needed
    return " ".join(text.strip().lower().split())


def bounded_difference(a: str, b: str) -> float:
    """
    Return a score in [0,1] where 0 ~ identical, 1 ~ very different.
    Uses 1 - difflib ratio as a cheap, dependency-free placeholder.
    Swap with your preferred token-level or fuzzy-hash distance.
    """
    na, nb = normalize(a), normalize(b)
    ratio = difflib.SequenceMatcher(a=na, b=nb).ratio()
    return max(0.0, min(1.0, 1.0 - ratio))