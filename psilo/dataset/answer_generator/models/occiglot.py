from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class OcciglotRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "occiglot-occiglot-7b-es-en-instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en", "es", "ca"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "occiglot/occiglot-7b-es-en-instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> dict[str, Any]:
        message = [
            {"role": "user", "content": "Contesta la pregunta següent de manera precisa i concisa, en català."},
            {"role": "assistant", "content": "Per descomptat! Quina pregunta t'agradaria respondre?"},
            {"role": "user", "content": q},
        ]

        return self._tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)
        terminators = [
            self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self._tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self._tokenizer.encode("\n")[-1],
        ]

        outputs = self._model.generate(
            inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            eos_token_id=terminators,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
        )

        response = outputs[0][inputs.shape[-1] : -1]
        return GenerationResult(text=self._tokenizer.decode(response, skip_special_tokens=True).strip())


register(OcciglotRunner())
