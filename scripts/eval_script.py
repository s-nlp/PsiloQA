import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationDataset,
    HallucinationSample,
)
from psiloqa_eval.evaluator import (
    evaluate_detector_char_level,
    evaluate_model,
    evaluate_model_example_level,
    print_metrics,
)
from lettucedetect.models.inference import HallucinationDetector


def evaluate_task_samples(
    samples,
    evaluation_type,
    model=None,
    tokenizer=None,
    detector=None,
    device=None,
    batch_size=8
):
    print(f"\nEvaluating model on {len(samples)} samples")

    if evaluation_type in {"token_level", "example_level"}:
        # Prepare the dataset and dataloader
        test_dataset = HallucinationDataset(samples, tokenizer)
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=-100
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        eval_map = {
            "token_level": (evaluate_model, "Token-Level Evaluation"),
            "example_level": (evaluate_model_example_level, "Example-Level Evaluation"),
        }
        eval_fn, eval_title = eval_map[evaluation_type]
        print(f"\n---- {eval_title} ----")
        metrics = eval_fn(model, test_loader, device)
        print_metrics(metrics)
        return metrics

    else:  # char_level
        print("\n---- Character-Level Span Evaluation ----")
        metrics = evaluate_detector_char_level(detector, samples)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a hallucination detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="If provided, evaluate only this language (e.g., 'cs', 'es')",
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="token_level",
        help="Evaluation type (token_level, example_level or char_level)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="s-nlp/PsiloQA",
        help="HF dataset repo to load (default: s-nlp/PsiloQA)",
    )
    args = parser.parse_args()

    # Load HF dataset and build samples for the test split
    ds = load_dataset(args.dataset)

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

    test_samples = build_samples(ds["test"], "test")

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    # Setup model/detector based on evaluation type
    if args.evaluation_type in {"token_level", "example_level"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        detector = None
    else:  # char_level
        model, tokenizer, device = None, None, None
        detector = HallucinationDetector(method="transformer", model_path=args.model_path)

    language_map = {}
    for sample in test_samples:
        if sample.language not in language_map:
            language_map[sample.language] = []
        language_map[sample.language].append(sample)
    for language, lang_samples in language_map.items():
        if args.language is not None and language != args.language:
            continue
        if lang_samples:
            print(f"\n--- Language: {language} ---")
            evaluate_task_samples(
                lang_samples,
                args.evaluation_type,
                model=model,
                tokenizer=tokenizer,
                detector=detector,
                device=device,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    main()
