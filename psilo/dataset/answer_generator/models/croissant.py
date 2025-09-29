from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class CroissantRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "croissantllm/CroissantLLMChat-v0.1"

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "do_sample": True, "eos_token_id": self._tokenizer.eos_token_id, "pad_token_id": self._tokenizer.eos_token_id, "top_k": 128}

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]


register(CroissantRunner())
