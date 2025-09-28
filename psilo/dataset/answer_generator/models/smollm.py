from __future__ import annotations
import torch

from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..registry import register

class SmolLM(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]
    
    def _format(self, q: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": q}]
        chat_str = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # ensure assistant turn is open
        )
        enc = self._tokenizer(
            chat_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return enc  # has input_ids and attention_mask


    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, (
            "call load() first"
        )
        input = self._format(question)
        output = self._model.generate(input, max_new_tokens=512)[0]
        start = input["input_ids"].shape[-1]
        text = self._tokenizer.decode(output[start:-1]).strip()
        return GenerationResult(text=text)


class SmolLM_135M(SmolLM):
    @property
    def runner_id(self) -> str:
        return "HuggingFaceTB-SmolLM2-135M-Instruct"

    def load(self) -> None:
        if self._model is not None:
            return
        name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name)
        self._model.eval()


class SmolLM_360M(SmolLM):
    @property
    def runner_id(self) -> str:
        return "HuggingFaceTB-SmolLM2-360M-Instruct"

    def load(self) -> None:
        if self._model is not None:
            return
        name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        self._model.eval()

class SmolLM_1_7B(BaseRunner):
    @property
    def runner_id(self) -> str:
        return "HuggingFaceTB-SmolLM2-1.7B-Instruct"

    def load(self) -> None:
        if self._model is not None:
            return
        name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        self._model.eval()


register(SmolLM_135M())
# register(SmolLM_360M())
# register(SmolLM_1_7B())
