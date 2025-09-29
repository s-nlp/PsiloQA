from typing import Any, Sequence

from dataset.answer_generator.runner import GenerationResult, RunnerWithChatTemplate
from utils.constants import DEVICE

from ..registry import register


class AprielRunner(RunnerWithChatTemplate):
    @property
    def runner_id(self) -> str:
        return "ServiceNow-AI/Apriel-5B-Instruct"

    @property
    def languages(self) -> Sequence[str]:
        return ["en"]

    def _format(self, q: str) -> dict[str, Any]:
        messages = [{"role": "system", "content": "You are a helpful AI assistant that provides accurate and concise information."}, {"role": "user", "content": q}]

        input_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(DEVICE)

    def answer_one(self, question: str) -> GenerationResult:
        inputs = self._format(question)
        outputs = self._model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=True)
        response = self._tokenizer.decode(outputs[0][inputs.shape[0] :], skip_special_tokens=True)
        return GenerationResult(text=response)


register(AprielRunner())
