import random
from typing import Dict, List, Optional

from dataset.answer_generator.runner import BaseRunner

_REGISTRY: Dict[str, BaseRunner] = {}


def register(runner: BaseRunner) -> BaseRunner:
    _REGISTRY[runner.runner_id] = runner
    return runner


def all_runners() -> Dict[str, BaseRunner]:
    return _REGISTRY


def runners_for_language(lang: str) -> List[BaseRunner]:
    lang = (lang or "").lower()
    out: List[BaseRunner] = []
    for r in _REGISTRY.values():
        langs = [x.lower() for x in r.languages]
        if "any" in langs or lang in langs:
            out.append(r)
    return out


def sample_runner_for_language(lang: str, seed: Optional[int]) -> Optional[BaseRunner]:
    eligible = runners_for_language(lang)
    if not eligible:
        return None
    rnd = random.Random(seed)
    return rnd.choice(eligible)
