import re

_URL = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]", flags=re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")


def simple_clean(text: str) -> str:
    """Shared preprocessing for BOTH training and inference."""
    if not isinstance(text, str):
        return ""

    s = text.lower()
    s = _URL.sub(" URL ", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


# Backward-compatible alias (some modules may import clean_text)
def clean_text(text: str) -> str:
    return simple_clean(text)
