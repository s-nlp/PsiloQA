from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class PhiMiniInstructRunner(RunnerWithChatTemplate):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "microsoft/Phi-4-mini-instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def system_prompt(self) -> str:
        return "You are a helpful AI assistant."

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "do_sample": True,
        }


register(PhiMiniInstructRunner())
