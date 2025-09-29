import re
import unicodedata
import uuid

from utils.constants import NAMESPACE

_ws = re.compile(r"\s+")


def norm(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    return _ws.sub(" ", s.strip())


def make_sample_id(sample: dict, keys: list[str]) -> str:
    key = "|".join([norm(sample.get(key)) for key in keys])
    return f"psiloqa-{uuid.uuid5(NAMESPACE, key)}"
