from typing import Sequence

from dataset.answer_generator.runner import VLLMRunnerWithChatTemplate

from ..registry import register


class Qwen2Runner(VLLMRunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "Qwen/Qwen2-7B-Instruct"

    @property
    def languages(self) -> Sequence[str]:
        return [
            "en",
            "zh",
            "ru",
            "es",
            "fr",
            "de",
            "ar",
            "ko",
            "ja",
            "th",
            "vi",
            "pt",
            "it",
            "id",
            "tr",
            "he",
            "gb",
            "ca",
            "cs",
            "pl",
            "fi",
            "hu",
            "pl",
            "uk",
        ]

    @property
    def generation_params(self) -> dict:
        return {
            "max_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }


register(Qwen2Runner())
