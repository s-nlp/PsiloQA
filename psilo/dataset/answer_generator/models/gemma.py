from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class GemmaRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "google/gemma-7b-it"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "de", "fr", "ru"]

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "do_sample": True,
        }


register(GemmaRunner())
