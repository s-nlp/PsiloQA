from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class ModelloRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "sapienzanlp/modello-italia-9b"

    @property
    def languages(self) -> Sequence[str]:
        return ["it"]

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "do_sample": True, "eos_token_id": self._tokenizer.eos_token_id, "pad_token_id": self._tokenizer.eos_token_id}


register(ModelloRunner())
