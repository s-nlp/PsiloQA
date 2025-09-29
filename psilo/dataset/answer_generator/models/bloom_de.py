from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class BloomDeRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "malteos-bloom-6b4-clp-german-oasst-v0.1"

    @property
    def languages(self) -> Sequence[str]:
        return ["de"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "malteos/bloom-6b4-clp-german-oasst-v0.1"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> dict[str, Any]:
        prompt = q + "\n"
        return self._tokenizer(prompt, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            eos_token_id=self._tokenizer.encode("\n")[0],
            pad_token_id=self._tokenizer.eos_token_id,
        )[0]

        return GenerationResult(text=self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip())


register(BloomDeRunner())
