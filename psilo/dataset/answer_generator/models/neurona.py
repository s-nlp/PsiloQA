from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class NeuronaRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "Iker/Llama-3-Instruct-Neurona-8b-v2"

    @property
    def languages(self) -> Sequence[str]:
        return ["es"]

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "terminators": [
                self._tokenizer.eos_token_id,
                self._tokenizer.encode("\n")[-1],
                self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            "pad_token_id": self._tokenizer.eos_token_id,
            "do_sample": True,
        }


register(NeuronaRunner())
