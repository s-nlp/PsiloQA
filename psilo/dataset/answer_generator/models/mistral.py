from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class MistralRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.3"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "de", "fr", "ru"]


register(MistralRunner())
