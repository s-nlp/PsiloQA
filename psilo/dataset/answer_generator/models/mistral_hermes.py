import os
from typing import Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class MistralHermesRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(self.runner_id, token=os.getenv("HF_TOKEN")).to(DEVICE)

    @property
    def template(self) -> str:
        return """<|im_start|>systemYou are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|><|im_start|>user{}<|im_start|>assistant\n"""

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "temperature": 0.8, "repetition_penalty": 1.1, "do_sample": True, "eos_token_id": self._tokenizer.eos_token_id}


register(MistralHermesRunner())
