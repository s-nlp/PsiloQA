from typing import Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate

from ..registry import register


class BloomDeRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "malteos/bloom-6b4-clp-german-oasst-v0.1"

    @property
    def languages(self) -> Sequence[str]:
        return ["de"]

    @property
    def prompt_template(self) -> str:
        return "{}\n"

    @property
    def generation_params(self) -> dict:
        return {"max_new_tokens": 512, "no_repeat_ngram_size": 2, "do_sample": True, "eos_token_id": self._tokenizer.encode("\n")[0], "pad_token_id": self._tokenizer.eos_token_id}


register(BloomDeRunner())
