import os
from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class GemmaRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "google-gemma-7b-it"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "de", "fr", "ru"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "google/gemma-7b-it"
        self._tokenizer = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN")).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        message = [
            {"role": "user", "content": q},
        ]

        return self._tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, "call load() first"
        input = self._format(question)
        output = self._model.generate(
            input,
            max_new_tokens=512,
            num_return_sequences=1,
            do_sample=True,
            terminators=[
                self._tokenizer.eos_token_id,
                self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                self._tokenizer.encode("\n")[-1],
            ],
        )[0]
        return GenerationResult(text=self._tokenizer.decode(output[0][input.shape[-1] :], skip_special_tokens=True))


register(GemmaRunner())
