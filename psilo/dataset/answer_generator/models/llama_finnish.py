from typing import Any, Dict, Sequence

from dataset.answer_generator.runner import BaseRunner, GenerationResult
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import DEVICE

from ..registry import register


class LlamaFinnishRunner(BaseRunner):
    def __init__(self):
        self._tokenizer = None
        self._model = None

    @property
    def runner_id(self) -> str:
        return "Finnish-NLP-llama-7b-finnish-instruct-v0.2"

    @property
    def languages(self) -> Sequence[str]:
        return ["fi"]

    def load(self) -> None:
        if self._model is not None:
            return
        name = "Finnish-NLP/llama-7b-finnish-instruct-v0.2"
        self._tokenizer = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        self._model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN")).to(DEVICE)

    def _format(self, q: str) -> Dict[str, Any]:
        alpaca_prompt = """<|alku|> Olet tekoälyavustaja. Seuraavaksi saat kysymyksen tai tehtävän. Kirjoita vastaus parhaasi mukaan siten että se täyttää kysymyksen tai tehtävän vaatimukset.
        <|ihminen|> Kysymys/Tehtävä:
        {}
        <|avustaja|> Vastauksesi:
        """
        return self._tokenizer([alpaca_prompt.format(q)], return_tensors="pt").to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)

        terminators = [
            self._tokenizer.eos_token_id,
            self._tokenizer.convert_tokens_to_ids("<|loppu|>"),
            self._tokenizer.encode("\n")[-1],
        ]

        outputs = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            num_return_sequences=1,
            eos_token_id=terminators,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
        )

        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[-1]

        response = outputs[0][input_length:]
        return GenerationResult(text=self._tokenizer.decode(response, skip_special_tokens=True))


register(LlamaFinnishRunner())
