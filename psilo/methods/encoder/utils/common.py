import random

import numpy as np
import torch

from lettucedetect.datasets.hallucination_dataset import HallucinationSample

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_samples(ds_split, split_name: str) -> list[HallucinationSample]:
    samples: list[HallucinationSample] = []
    for row in ds_split:
        span_pairs = row.get("labels", []) or []
        labels = [{"start": int(p[0]), "end": int(p[1])} for p in span_pairs]

        samples.append(
            HallucinationSample(
                prompt=row["question"],
                answer=row["llm_answer"],
                labels=labels,
                split=split_name,
                task_type="qa",
                dataset="psiloqa",
                language=row["lang"],
            )
        )
    return samples
