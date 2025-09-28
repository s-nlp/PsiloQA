import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationDataset,
    HallucinationSample,
)
from lettucedetect.models.trainer import Trainer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)


def set_seed(seed: int = 42):
    """Set all seeds for reproducibility.

    Args:
        seed: The seed to use

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination detector model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Name or path of the pretrained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hallucination_detector",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--epochs", type=int, default=6, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate for training"
    )
    return parser.parse_args()


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


def main():
    # Set seeds for reproducibility
    set_seed(123)

    args = parse_args()
    ds = load_dataset("s-nlp/PsiloQA")  # token=

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    train_samples = build_samples(ds["train"], "train")
    dev_samples = build_samples(ds["validation"], "dev")

    train_dataset = HallucinationDataset(train_samples, tokenizer, max_length=8192)
    dev_dataset = HallucinationDataset(dev_samples, tokenizer, max_length=8192)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=2, trust_remote_code=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=dev_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
