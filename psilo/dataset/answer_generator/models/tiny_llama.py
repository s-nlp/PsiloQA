from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class TinyLLaMARunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "TinyLlama-TinyLlama-1.1B-Chat-v1.0"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

    def _format(self, q: str) -> dict[str, Any]:
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
        return enc.to(DEVICE)  # has input_ids and attention_mask

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, "call load() first"
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
