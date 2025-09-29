from typing import Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate

from ..registry import register


class SeaLLMRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "SeaLLMs/SeaLLM-7B-v2.5"

    @property
    def languages(self) -> Sequence[str]:
        return ["ar", "en", "vi", "id", "th", "ms", "km", "lo", "my", "tl"]

    @property
    def prompt_template(self) -> str:
        return "{}\n"

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "no_repeat_ngram_size": 2,
            "do_sample": False,
            "eos_token_id": self._tokenizer.encode("\n")[0],
            "pad_token_id": self._tokenizer.eos_token_id,
        }


register(SeaLLMRunner())
