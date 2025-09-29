from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class ZephyrRunner(RunnerWithChatTemplate):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "HuggingFaceH4/zephyr-7b-beta"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def system_prompt(self) -> str:
        return "Answer to the user's question."

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }


register(ZephyrRunner())
