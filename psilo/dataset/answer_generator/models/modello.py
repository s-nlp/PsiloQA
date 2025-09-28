from typing import Any, Dict, Optional, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class ModelloRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "sapienzanlp-modello-italia-9b"

    @property
    def languages(self) -> Sequence[str]:
        return ["it"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "sapienzanlp/modello-italia-9b"
        self._tokenizer = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN")).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        message = [{"role": "user", "content": q}]
        prompt = self._tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return self._tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str, seed: Optional[int] = None) -> GenerationResult:
        inputs = self._format(question)

        output = self._model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=True,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        response_token_ids = output[0].tolist()[len(inputs[0]) :]
        response_tokens = self._tokenizer.convert_ids_to_tokens(response_token_ids)
        return GenerationResult(text=self._tokenizer.decode(response_token_ids, skip_special_tokens=True))


register(ModelloRunner())
