import os
from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class GemmaRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "google/gemma-7b-it"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "de", "fr", "ru"]

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN")).to(DEVICE)

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "do_sample": True,
            "terminators": [
                self._tokenizer.eos_token_id,
                self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                self._tokenizer.encode("\n")[-1],
            ],
        }


register(GemmaRunner())
