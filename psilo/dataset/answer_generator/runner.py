from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class GenerationResult:
    text: str
    meta: Dict[str, Any]


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
    def answer_one(
        self, question: str, seed: Optional[int] = None
    ) -> GenerationResult: ...

    def answer_batch(
        self, questions: Iterable[str], seed: Optional[int] = None
    ) -> List[GenerationResult]:
        return [self.answer_one(q, seed=seed) for q in questions]
