from typing import Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate

from ..registry import register


class FalconRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "tiiuae/falcon-7b-instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def template(self) -> str:
        return "Question: {}\nAnswer: "


register(FalconRunner())
