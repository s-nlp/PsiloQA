from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class SalamandraRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "BSC-LT-salamandra-7b"

    @property
    def languages(self) -> Sequence[str]:
        return ["fi"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "BSC-LT/salamandra-7b"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        return self._tokenizer(f"Kysymys: {q}\nVastaa:", return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        input = self._format(question)
        output = self._model.generate(**input, max_new_tokens=512)[0]
        generated_ids = [output[len(input_ids) :] for input_ids, output_ids in zip(input.input_ids, generated_ids)]
        return GenerationResult(text=self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())


register(SalamandraRunner())
