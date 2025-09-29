from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class SmolLM_135M(RunnerWithChatTemplate):
    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def runner_id(self) -> str:
        return "HuggingFaceTB-SmolLM2-135M-Instruct"


class SmolLM_360M(RunnerWithChatTemplate):
    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def runner_id(self) -> str:
        return "HuggingFaceTB-SmolLM2-360M-Instruct"


class SmolLM_1_7B(RunnerWithChatTemplate):
    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def runner_id(self) -> str:
        return "HuggingFaceTB-SmolLM2-1.7B-Instruct"


register(SmolLM_135M())
register(SmolLM_360M())
register(SmolLM_1_7B())
