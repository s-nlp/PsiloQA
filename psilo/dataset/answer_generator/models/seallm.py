from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class SeaLLMRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "SeaLLMs-SeaLLM-7B-v2.5"

    @property
    def languages(self) -> Sequence[str]:
        return ["ar", "en", "vi", "id", "th", "ms", "km", "lo", "my", "tl"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "SeaLLMs/SeaLLM-7B-v2.5"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        return self._tokenizer(q + "\n", return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, "call load() first"
        input = self._format(question)
        output = self._model.generate(
            **input,
            max_new_tokens=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            eos_token_id=self._tokenizer.encode("\n")[0],
            pad_token_id=self._tokenizer.eos_token_id,
        )[0]
        return GenerationResult(text=self._tokenizer.decode(output[0][input["input_ids"].shape[-1] :], skip_special_tokens=True).strip())


register(SeaLLMRunner())
