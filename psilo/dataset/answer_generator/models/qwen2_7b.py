# psilo/hypotheses/models/bloom_zh.py
from __future__ import annotations

from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..registry import register


class TinyLLaMARunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "Qwen-Qwen2-7B-Instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "Qwen/Qwen2-7B-Instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        self._model.eval()

    def _format(self, q: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Answer briefly on the question."},
            {"role": "user", "content": q},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tokenizer([text], return_tensors="pt").to(self._model.device)

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, (
            "call load() first"
        )
        input = self._format(question)
        output = self._model.generate(
            **input,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )[0]
        start = input["input_ids"].shape[-1]
        text = self._tokenizer.decode(output[start:-1]).strip()
        return GenerationResult(text=text)


register(TinyLLaMARunner())
