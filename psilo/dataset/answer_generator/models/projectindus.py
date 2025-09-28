from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class ProjectIndusRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "togethercomputer-Pythia-Chat-Base-7B-v0.16"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "togethercomputer/Pythia-Chat-Base-7B-v0.16"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": q},
        ]
        return self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)
        output = self._model.generate(
            inputs,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
            max_length=512,
            num_beams=5,
            do_sample=True,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )
        return GenerationResult(text=self._tokenizer.decode(output[0][inputs.shape[-1] : -1], skip_special_tokens=False))


register(ProjectIndusRunner())
