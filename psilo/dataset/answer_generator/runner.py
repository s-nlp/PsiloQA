from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass
class GenerationResult:
    text: str
    meta: dict[str, Any] | None = None


class BaseRunner(ABC):
    @property
    @abstractmethod
    def runner_id(self) -> str: ...

    @property
    @abstractmethod
    def languages(self) -> Sequence[str]: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def answer_one(self, question: str) -> GenerationResult: ...

    def answer_batch(self, questions: Iterable[str]) -> list[GenerationResult]:
        return [self.answer_one(q) for q in questions]
