from typing import Any, Dict, Optional, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class PhiMiniInstructRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "microsoft-Phi-4-mini-instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "microsoft/Phi-4-mini-instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        message = [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": q}]

        return self._tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str, seed: Optional[int] = None) -> GenerationResult:
        inputs = self._format(question)

        outputs = self._model.generate(
            inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            do_sample=True,
        )

        response = outputs[0][inputs.shape[-1] : -1]
        return GenerationResult(text=self._tokenizer.decode(response, skip_special_tokens=True).strip())


register(PhiMiniInstructRunner())
