from typing import Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate

from ..registry import register


class PythiaRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "togethercomputer/Pythia-Chat-Base-7B-v0.16"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def prompt_template(self) -> str:
        return "<human>: {}\n<bot>:"

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "do_sample": False, "eos_token_id": [self._model.encode("<")[0]], "temperature": 1.0}


register(PythiaRunner())
