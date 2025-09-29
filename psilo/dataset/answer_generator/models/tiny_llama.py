from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class TinyLLaMARunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }


register(TinyLLaMARunner())
