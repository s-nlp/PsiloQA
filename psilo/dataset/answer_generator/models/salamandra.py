from typing import Any, Sequence

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

    def _format(self, q: str) -> dict[str, Any]:
        return self._tokenizer(f"Kysymys: {q}\nVastaa:", return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)
        outputs = self._model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=1.0, eos_token_id=[self._model.encode("<")[0]])
        output_str = self._model.decode(outputs[0][inputs.input_ids.shape[-1] :])[:-1].strip()
        return GenerationResult(text=output_str)


register(SalamandraRunner())
