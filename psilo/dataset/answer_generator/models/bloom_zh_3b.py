# psilo/hypotheses/models/bloom_zh.py
from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..registry import register

_CONFIGS = [
    ("k50_t0.1", dict(top_k=50, temperature=0.1)),
    ("k50_t0.2", dict(top_k=50, temperature=0.2)),
    ("k50_t0.5", dict(top_k=50, temperature=0.5)),
    ("k50_t1.0", dict(top_k=50, temperature=1.0)),
    ("k75_t0.1", dict(top_k=75, temperature=0.1)),
    ("k75_t0.2", dict(top_k=75, temperature=0.2)),
    ("k75_t0.5", dict(top_k=75, temperature=0.5)),
    ("k75_t1.0", dict(top_k=75, temperature=1.0)),
    ("k100_t0.1", dict(top_k=100, temperature=0.1)),
    ("k100_t0.2", dict(top_k=100, temperature=0.2)),
    ("k100_t0.5", dict(top_k=100, temperature=0.5)),
    ("k100_t1.0", dict(top_k=100, temperature=1.0)),
]


class BloomZhRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "ikala-bloom-zh-3b-chat"

    @property
    def languages(self) -> Sequence[str]:
        return ["zh"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "ikala/bloom-zh-3b-chat"
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        self._model.eval()

    def _format(self, q: str) -> Dict[str, Any]:
        return self._tokenizer(
            f"<|prompter|>{q}</s><|assistant|>", return_tensors="pt"
        ).to(self._model.device)

    def answer_one(self, question: str, seed: Optional[int] = None) -> GenerationResult:
        assert self._model is not None and self._tokenizer is not None, (
            "call load() first"
        )
        rng = random.Random(seed)
        cfg_id, cfg = rng.choice(_CONFIGS)

        inp = self._format(question)
        out_ids = self._model.generate(
            **inp,
            max_new_tokens=512,
            **cfg,
        )[0]
        start = inp["input_ids"].shape[-1]
        text = self._tokenizer.decode(out_ids[start:-1]).strip()
        return GenerationResult(text=text, meta={"cfg_id": cfg_id, **cfg})


register(BloomZhRunner())
