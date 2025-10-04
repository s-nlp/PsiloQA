from typing import Sequence

from dataset.answer_generator.runner import RunnerWithCustomTemplate

from ..registry import register


class LlamaFinnishRunner(RunnerWithCustomTemplate):
    @property
    def runner_id(self) -> str:
        return "Finnish-NLP/llama-7b-finnish-instruct-v0.2"

    @property
    def languages(self) -> Sequence[str]:
        return ["fi"]

    @property
    def prompt_template(self) -> str:
        return """<|alku|> Olet tekoälyavustaja. Seuraavaksi saat kysymyksen tai tehtävän. Kirjoita vastaus parhaasi mukaan siten että se täyttää kysymyksen tai tehtävän vaatimukset.\n<|ihminen|> Kysymys/Tehtävä:\n{}\n<|avustaja|> Vastauksesi:\n"""  # noqa: E501

    @property
    def generation_params(self) -> dict:
        return {
            "max_new_tokens": 512,
            "do_sample": True,
            "eos_token_id": [
                self._tokenizer.eos_token_id,
                self._tokenizer.convert_tokens_to_ids("<|loppu|>"),
                self._tokenizer.encode("\n")[-1],
            ],
            "pad_token_id": self._tokenizer.eos_token_id,
        }


register(LlamaFinnishRunner())
