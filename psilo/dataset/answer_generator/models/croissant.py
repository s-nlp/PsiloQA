from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class CroissantRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "croissantllm-CroissantLLMChat-v0.1"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "croissantllm/CroissantLLMChat-v0.1"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")

    def _format(self, q: str) -> dict[str, Any]:
        message = [{"role": "user", "content": q}]
        prompt = self._tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        return self._tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, "call load() first"
        input_ids = self._format(question)
        output = self._model.generate(input_ids, max_new_tokens=512, do_sample=True, eos_token_id=self._tokenizer.eos_token_id, pad_token_id=self._tokenizer.eos_token_id, top_k=128)
        response_token_ids = output[0].to("cpu").tolist()[len(input_ids[0]) :]
        return GenerationResult(text=self._tokenizer.decode(response_token_ids, skip_special_tokens=True))


register(CroissantRunner())
