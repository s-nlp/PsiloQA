from typing import Sequence

from dataset.answer_generator.runner import BaseRunner

from ..registry import register


class OcciglotRunner(BaseRunner):
    @property
    def runner_id(self) -> str:
        return "ciglot/occiglot-7b-es-en-instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "es", "ca"]

    @property
    def system_prompt(self) -> str:
        return "Contesta la pregunta següent de manera precisa i concisa, en català."

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "terminators": [
                self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
                self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                self._tokenizer.encode("\n")[-1],
            ],
            "pad_token_id": self._tokenizer.eos_token_id,
            "do_sample": True,
        }


register(OcciglotRunner())
