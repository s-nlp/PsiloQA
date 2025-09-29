from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from utils.constants import DEVICE

from ..registry import register


class BloomZhRunner(BaseRunner):
    @property
    def runner_id(self) -> str:
        return "ikala/bloom-zh-3b-chat"

    @property
    def languages(self) -> Sequence[str]:
        return ["zh"]

    def _format(self, q: str) -> dict[str, Any]:
        return self._tokenizer(f"<|prompter|>{q}</s><|assistant|>", return_tensors="pt").to(DEVICE)

    @property
    def template(self) -> str:
        return "<|prompter|>{}</s><|assistant|>"

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, "call load() first"

        inp = self._format(question)
        out_ids = self._model.generate(
            **inp,
            max_new_tokens=512,
        )[0]
        start = inp["input_ids"].shape[-1]
        text = self._tokenizer.decode(out_ids[start:-1]).strip()
        return GenerationResult(text=text)


register(BloomZhRunner())
