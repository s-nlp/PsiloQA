import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from dataset.settings import HFSettings
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE


@dataclass
class GenerationResult:
    text: str
    meta: dict[str, Any] | None = None


class BaseRunner(ABC):
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._settings = HFSettings()

    @property
    @abstractmethod
    def runner_id(self) -> str: ...

    @property
    def system_prompt(self) -> str:
        return None

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512}

    @property
    @abstractmethod
    def languages(self) -> Sequence[str]: ...

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.runner_id, token=self._settings.token.get_secret_value())
        self._model = AutoModelForCausalLM.from_pretrained(self.runner_id, token=self._settings.token.get_secret_value()).to(DEVICE)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def destroy(self) -> None:
        del self._model
        del self._tokenizer

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()

    @abstractmethod
    def answer_one(self, question: str) -> GenerationResult: ...

    def answer_batch(self, questions: Iterable[str]) -> list[GenerationResult]:
        return [self.answer_one(q) for q in questions]


import torch


class RunnerWithChatTemplate(BaseRunner):
    def _format(self, user_prompt: str):
        messages = []
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": user_prompt})

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(DEVICE)

        attention_mask = torch.ones_like(input_ids, device=DEVICE)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)
        output = self._model.generate(**inputs, **self.generation_params)
        start = inputs["input_ids"].shape[-1]
        return GenerationResult(
            text=self._tokenizer.decode(output[0, start:], skip_special_tokens=True),
            meta=self.generation_params,
        )


class RunnerWithCustomTemplate(BaseRunner):
    @property
    def prompt_template(self) -> str:
        return None

    def _format(self, user_prompt: str):
        prompt = self.prompt_template.format(user_prompt)
        return self._tokenizer(prompt, return_tensors="pt")

    def answer_one(self, question: str) -> GenerationResult:
        input = self._format(question)
        output = self._model.generate(**input, **self.generation_params)[0]
        return GenerationResult(text=self._tokenizer.decode(output[0][input["input_ids"].shape[-1] :], skip_special_tokens=True).strip(), meta=self.generation_params)
