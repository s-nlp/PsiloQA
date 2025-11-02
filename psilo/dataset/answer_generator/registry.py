import random

from dataset.answer_generator.runner import BaseRunner
from loguru import logger
_REGISTRY: dict[str, BaseRunner] = {}


def register(runner: BaseRunner) -> BaseRunner:
    _REGISTRY[runner.runner_id] = runner
    return runner


def all_runners() -> dict[str, BaseRunner]:
    return _REGISTRY


def runners_for_language(lang: str, checkpoints: list[str] | None) -> list[BaseRunner]:
    lang = (lang or "").lower()
    out: list[BaseRunner] = []
    for r in _REGISTRY.values():
        langs = [x.lower() for x in r.languages]
        if "any" in langs or lang in langs:
            if not checkpoints or r.runner_id in checkpoints:
                out.append(r)
    if len(out) == 0 and checkpoints:
        logger.warning(f"Did not find suitable runners for {lang} across {checkpoints}")
    return out


def sample_runner_for_language(lang: str, seed: int | None, checkpoints: list[str] | None) -> BaseRunner | None:
    eligible = runners_for_language(lang, checkpoints)
    if not eligible:
        return None
    rnd = random.Random(seed)
    return rnd.choice(eligible)
