from typing import Sequence

from dataset.answer_generator.runner import RunnerWithChatTemplate

from ..registry import register


class ProjectIndusRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "togethercomputer/Pythia-Chat-Base-7B-v0.16"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    @property
    def generation_params(self) -> dict:
        return {
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.eos_token_id,
            "max_length": 512,
            "num_beams": 5,
            "do_sample": True,
            "early_stopping": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        }


register(ProjectIndusRunner())
