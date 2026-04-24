import numpy as np

from typing import Dict, List, Tuple

from collections import defaultdict
import torch.nn as nn
import string

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class Deberta:
    """
    Allows for the implementation of a singleton DeBERTa model which can be shared across
    different uncertainty estimation methods in the code.
    """

    def __init__(
        self,
        deberta_path: str = "microsoft/deberta-large-mnli",
        batch_size: int = 10,
        device: str = None,
        hf_cache: str = None,
    ):
        """
        Parameters
        ----------
        deberta_path : str
            huggingface path of the pretrained DeBERTa (default 'microsoft/deberta-large-mnli')
        device : str
            device on which the computations will take place (default 'cuda:0' if available, else 'cpu').
        """
        self.deberta_path = deberta_path
        self.batch_size = batch_size
        self._deberta = None
        self._deberta_tokenizer = None
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.hf_cache = hf_cache
        self.setup()

    @property
    def deberta(self):
        if self._deberta is None:
            self.setup()

        return self._deberta

    @property
    def deberta_tokenizer(self):
        if self._deberta_tokenizer is None:
            self.setup()

        return self._deberta_tokenizer

    def to(self, device):
        self.device = device
        if self._deberta is not None:
            self._deberta.to(self.device)

    def setup(self):
        """
        Loads and prepares the DeBERTa model from the specified path.
        """
        if self._deberta is not None:
            return
        self._deberta = AutoModelForSequenceClassification.from_pretrained(
            self.deberta_path,
            problem_type="multi_label_classification",
            cache_dir=self.hf_cache,
        )
        self._deberta_tokenizer = AutoTokenizer.from_pretrained(
            self.deberta_path, cache_dir=self.hf_cache
        )
        self._deberta.to(self.device)
        self._deberta.eval()


def _eval_nli_model(nli_queue: List[Tuple[str, str]], deberta: Deberta) -> List[str]:
    nli_set = list(set(nli_queue))

    softmax = nn.Softmax(dim=1)
    w_probs = defaultdict(lambda: defaultdict(lambda: None))
    for k in range(0, len(nli_set), deberta.batch_size):
        batch = nli_set[k : k + deberta.batch_size]
        encoded = deberta.deberta_tokenizer.batch_encode_plus(
            batch, padding=True, return_tensors="pt"
        ).to(deberta.device)
        logits = deberta.deberta(**encoded).logits
        logits = logits.detach().to(deberta.device)
        for (wi, wj), prob in zip(batch, softmax(logits).cpu().detach()):
            w_probs[wi][wj] = prob

    classes = []
    for w1, w2 in nli_queue:
        pr = w_probs[w1][w2]
        id = pr.argmax()
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
        if id == ent_id:
            str_class = "entail"
        elif id == contra_id:
            str_class = "contra"
        else:
            str_class = "neutral"
        classes.append(str_class)
    return classes


class GreedyAlternativesNLICalculator():
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the SamplingPromptCalculator.
        """
        return ["greedy_tokens_alternatives_nli"], ["greedy_tokens_alternatives"]

    def __init__(self, nli_model):
        super().__init__()
        self.nli_model = nli_model

    def _strip(self, w: str):
        return w.strip(string.punctuation + " \n")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        greedy_alternatives = dependencies["greedy_tokens_alternatives"]
        greedy_alternatives_nli = []
        for sample_alternatives in greedy_alternatives:
            nli_matrixes = []
            for w_number, word_alternatives in enumerate(sample_alternatives):
                nli_queue = []
                nli_matrix = [
                    ["" for _ in range(len(word_alternatives))]
                    for _ in range(len(word_alternatives))
                ]
                if len(word_alternatives) > 0 and not isinstance(
                    word_alternatives[0][0],
                    str,
                ):
                    word_alternatives = [
                        (dependencies["tokenizer"].decode([alt]), prob)
                        for alt, prob in word_alternatives
                    ]
                words = [self._strip(alt[0]) for alt in word_alternatives]
                for wi in words:
                    nli_queue.append((words[0], wi))
                    nli_queue.append((wi, words[0]))

                nli_classes = _eval_nli_model(nli_queue, self.nli_model)
                nli_class = defaultdict(lambda: None)
                for nli_cl, (w1, w2) in zip(nli_classes, nli_queue):
                    nli_class[w1, w2] = nli_cl

                for i, wi in enumerate(words):
                    for j, wj in enumerate(words):
                        # Only calculate NLI with greedy token
                        if i > 0 and j > 0:
                            continue
                        nli_matrix[i][j] = nli_class[wi, wj]

                nli_matrixes.append(nli_matrix)
            greedy_alternatives_nli.append(nli_matrixes)

        return {"greedy_tokens_alternatives_nli": greedy_alternatives_nli}


class GreedyAlternativesFactPrefNLICalculator():
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the SamplingPromptCalculator.
        """

        return ["greedy_tokens_alternatives_fact_pref_nli"], [
            "greedy_tokens_alternatives",
            "greedy_tokens",
            "claims",
        ]

    def __init__(self, nli_model):
        super().__init__()
        self.nli_model = nli_model

    def _strip(self, w: str):
        return w.strip(string.punctuation + " \n")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        greedy_alternatives = dependencies["greedy_tokens_alternatives"]
        greedy_tokens = dependencies["greedy_tokens"]
        claims = dependencies["claims"]
        greedy_alternatives_nli = []
        for sample_alternatives, sample_claims, sample_tokens in zip(
            greedy_alternatives,
            claims,
            greedy_tokens,
        ):
            nli_queue = []
            for claim in sample_claims:
                tokens = [sample_tokens[t] for t in claim.aligned_token_ids]
                alts = [sample_alternatives[t] for t in claim.aligned_token_ids]
                for i in range(len(tokens)):
                    for j in range(len(alts[i])):
                        text1 = model.tokenizer.decode(tokens[: i + 1])
                        text2 = model.tokenizer.decode(tokens[:i] + [alts[i][j][0]])
                        nli_queue.append((text1, text2))
                        nli_queue.append((text2, text1))

            nli_classes = _eval_nli_model(nli_queue, self.nli_model)

            nli_matrixes = []
            for claim in sample_claims:
                nli_matrixes.append([])
                tokens = [sample_tokens[t] for t in claim.aligned_token_ids]
                alts = [sample_alternatives[t] for t in claim.aligned_token_ids]
                for i in range(len(tokens)):
                    nli_matrix = []
                    for _ in range(len(alts[i])):
                        nli_matrix.append([])
                        for j in range(len(alts[i])):
                            nli_matrix[-1].append(None)
                    for j in range(len(alts[i])):
                        nli_matrix[0][j], nli_matrix[j][0] = nli_classes[:2]
                        nli_classes = nli_classes[2:]
                    nli_matrixes[-1].append(nli_matrix)
            greedy_alternatives_nli.append(nli_matrixes)

        return {"greedy_tokens_alternatives_fact_pref_nli": greedy_alternatives_nli}
