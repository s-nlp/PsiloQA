import random

from dataset.answer_generator.runner import BaseRunner

_REGISTRY: dict[str, BaseRunner] = {}


def register(runner: BaseRunner) -> BaseRunner:
    _REGISTRY[runner.runner_id] = runner
    return runner


def all_runners() -> dict[str, BaseRunner]:
    return _REGISTRY


def runners_for_language(lang: str) -> list[BaseRunner]:
    lang = (lang or "").lower()
    out: list[BaseRunner] = []
    for r in _REGISTRY.values():
        langs = [x.lower() for x in r.languages]
        if "any" in langs or lang in langs:
            out.append(r)
    return out


def sample_runner_for_language(lang: str, seed: int | None) -> BaseRunner | None:
    eligible = runners_for_language(lang)
    if not eligible:
        return None
    rnd = random.Random(seed)
    return rnd.choice(eligible)
