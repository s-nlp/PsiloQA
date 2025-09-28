from typing import Any, Dict, Optional, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class MistralRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "mistralai-Mistral-7B-Instruct-v0.3"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "de", "fr", "ru"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "mistralai/Mistral-7B-Instruct-v0.3"
        self._tokenizer = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN")).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        message = [
            {"role": "user", "content": q},
        ]
        return self._tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str, seed: Optional[int] = None) -> GenerationResult:
        inputs = self._format(question)

        terminators = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self._model.generate(inputs, max_new_tokens=512, num_return_sequences=1, eos_token_id=terminators, pad_token_id=self._tokenizer.eos_token_id, do_sample=True)

        response = outputs[0][inputs.shape[-1] :]
        return GenerationResult(text=self._tokenizer.decode(response, skip_special_tokens=True))


register(MistralRunner())
