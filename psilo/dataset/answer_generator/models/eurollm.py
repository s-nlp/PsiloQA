import os
from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class EuroLLMRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "utter-project-EuroLLM-9B-Instruct"

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

    def load(self) -> None:
        if self._model is not None:
            return
        name = "utter-project/EuroLLM-9B-Instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN")).to(DEVICE)

    def _format(self, q: str) -> dict[str, Any]:
        message = [
            {
                "role": "system",
                "content": "You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers.",
            },
            {"role": "user", "content": q},
        ]

        return self._tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, "call load() first"
        input = self._format(question)
        output = self._model.generate(
            input,
            max_new_tokens=512,
            do_sample=True,
        )[0]
        return GenerationResult(text=self._tokenizer.decode(output[0][len(input[0]) :], skip_special_tokens=True))


register(EuroLLMRunner())
