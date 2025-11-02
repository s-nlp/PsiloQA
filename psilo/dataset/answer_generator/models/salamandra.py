from typing import Any, Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate
from utils.constants import DEVICE

from ..registry import register


class SalamandraRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "BSC-LT/salamandra-7b"

    @property
    def languages(self) -> Sequence[str]:
        return ["fi"]

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "do_sample": False, "temperature": 1.0, "eos_token_id": [self._tokenizer.encode("<")[0]]}

    @property
    def prompt_template(self) -> str:
        return "Kysymys: {}\nVastaa:"

    def _format(self, q: str) -> dict[str, Any]:
        return self._tokenizer(f"Kysymys: {q}\nVastaa:", return_tensors="pt").to(DEVICE)


register(SalamandraRunner())
