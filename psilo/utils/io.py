import json
from typing import Iterable


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_text(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()
