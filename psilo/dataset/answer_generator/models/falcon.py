from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class FalconRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "tiiuae-falcon-7b-instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "tiiuae/falcon-7b-instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(
            name,
        )
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        prompt = f"Question: {q}\nAnswer: "
        return self._tokenizer(prompt, return_tensors="pt")

    def answer_one(self, question: str) -> GenerationResult:
        input = self._format(question)
        output = self._model.generate(**input, max_new_tokens=512)[0]
        return GenerationResult(text=self._tokenizer.decode(output[0][input["input_ids"].shape[-1] :], skip_special_tokens=True).strip())


register(FalconRunner())
