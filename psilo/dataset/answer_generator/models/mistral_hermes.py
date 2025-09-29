import os
from typing import Any, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class MistralHermesRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "NousResearch-Nous-Hermes-2-Mistral-7B-DPO"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
        self._tokenizer = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN")).to(DEVICE)

    def _format(self, q: str) -> dict[str, Any]:
        prompt = """<|im_start|>systemYou are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|><|im_start|>user{q}<|im_start|>assistant\n"""
        return self._tokenizer(prompt.format(q=q), return_tensors="pt").input_ids.to("cuda")

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)

        generated_ids = self._model.generate(inputs, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=self._tokenizer.eos_token_id)
        return GenerationResult(text=self._tokenizer.decode(generated_ids[0][input.shape[-1] :], skip_special_tokens=True, clean_up_tokenization_space=True))


register(MistralHermesRunner())
