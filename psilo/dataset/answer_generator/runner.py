import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

import torch
from dataset.settings import HFSettings
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE


class RunnerType(Enum):
    HF = "hf"
    VLLM = "vllm"


@dataclass
class GenerationResult:
    text: str
    meta: dict[str, Any] | None = None


class BaseRunner(ABC):
    TYPE: RunnerType = RunnerType.HF

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
            self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

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


# from typing import Iterable, List

# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams


# class VLLMRunnerWithChatTemplate(BaseRunner):
#     TYPE: RunnerType = RunnerType.VLLM

#     def _format(self, user_prompt: str) -> str:
#         messages = []
#         if getattr(self, "system_prompt", None):
#             messages.append({"role": "system", "content": self.system_prompt})
#         messages.append({"role": "user", "content": user_prompt})

#         # ВАЖНО: для instruct-моделей используем chat_template и получаем СТРОКУ
#         return self._tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=False,
#         )

#     def load(self) -> None:
#         # Токенайзер нужен для chat_template и decode (хотя vLLM уже даёт готовый текст)
#         # Если требуется приватный токен HF — возьми из self._settings.token, как у тебя принято
#         hf_token = getattr(getattr(self, "_settings", None), "token", None)
#         hf_token = hf_token.get_secret_value() if hf_token else None

#         self._tokenizer = AutoTokenizer.from_pretrained(self.runner_id, token=hf_token)
#         if self._tokenizer.pad_token is None:
#             self._tokenizer.pad_token = self._tokenizer.eos_token

#         self._model = LLM(
#             model=self.runner_id,
#             trust_remote_code=True,
#             dtype="auto",
#             tensor_parallel_size=getattr(self._settings, "tensor_parallel_size", 1),
#             max_model_len=getattr(self._settings, "max_model_len", None),
#             gpu_memory_utilization=getattr(self._settings, "gpu_memory_utilization", 0.90),
#             seed=getattr(self._settings, "seed", 0),
#         )

#     def _sampling_params(self) -> SamplingParams:
#         return SamplingParams(**(self.generation_params or {}))

#     def answer_one(self, question: str) -> GenerationResult:
#         prompt = self._format(question)
#         sp = self._sampling_params()
#         outs = self._model.generate([prompt], sampling_params=sp)
#         text = outs[0].outputs[0].text if outs and outs[0].outputs else ""
#         return GenerationResult(text=text, meta=self.generation_params)

#     def answer_batch(self, questions: Iterable[str]) -> List[GenerationResult]:
#         prompts = [self._format(q) for q in questions]
#         sp = self._sampling_params()
#         outs = self._model.generate(prompts, sampling_params=sp)

#         results: List[GenerationResult] = []
#         for o in outs:
#             text = o.outputs[0].text if o.outputs else ""
#             results.append(GenerationResult(text=text, meta=self.generation_params))
#         return results


class RunnerWithCustomTemplate(BaseRunner):
    @property
    def prompt_template(self) -> str:
        return None

    def _format(self, user_prompt: str):
        prompt = self.prompt_template.format(user_prompt)
        return self._tokenizer(prompt, return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        input = self._format(question)
        output = self._model.generate(**input, **self.generation_params)[0]
        return GenerationResult(text=self._tokenizer.decode(output[input["input_ids"].shape[-1] :], skip_special_tokens=True).strip(), meta=self.generation_params)
