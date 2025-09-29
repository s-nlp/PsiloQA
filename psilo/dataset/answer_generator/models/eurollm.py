import os
from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class EuroLLMRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "utter-project/EuroLLM-9B-Instruct"

    @property
    def languages(self) -> Sequence[str]:
        return [
            "bg",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "et",
            "fi",
            "fr",
            "de",
            "el",
            "hu",
            "ga",
            "it",
            "lv",
            "lt",
            "mt",
            "pl",
            "pt",
            "ro",
            "sk",
            "sl",
            "es",
            "sv",
            "ar",
            "ca",
            "zh",
            "gl",
            "hi",
            "ja",
            "ko",
            "no",
            "ru",
            "tr",
            "uk",
        ]

    @property
    def system_prompt(self) -> str:
        return "You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers."

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN")).to(DEVICE)

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "do_sample": True}


register(EuroLLMRunner())
