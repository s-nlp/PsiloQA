import os
from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class MistralRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.3"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "de", "fr", "ru"]

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN")).to(DEVICE)


register(MistralRunner())
