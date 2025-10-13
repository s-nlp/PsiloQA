import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.models.inference import HallucinationDetector

from encoder.utils.common import build_samples
from encoder.utils.evaluation import (
    evaluate_detector_char_level,
    evaluate_model,
    evaluate_model_example_level,
    print_metrics,
)


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


def run_encoder_evaluation(model_path: str, language: str, evaluation_type: str, batch_size: int, dataset):
    test_samples = build_samples(dataset, "test")

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    if evaluation_type in {"token_level", "example_level"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        detector = None
    else:  # char_level
        model, tokenizer, device = None, None, None
        detector = HallucinationDetector(method="transformer", model_path=model_path)

    language_map = {}
    for sample in test_samples:
        if sample.language not in language_map:
            language_map[sample.language] = []
        language_map[sample.language].append(sample)
    for curr_language, lang_samples in language_map.items():
        if language is not None and curr_language != language:
            continue
        if lang_samples:
            print(f"\n--- Language: {curr_language} ---")
            evaluate_task_samples(
                lang_samples,
                evaluation_type,
                model=model,
                tokenizer=tokenizer,
                detector=detector,
                device=device,
                batch_size=batch_size,
            )
