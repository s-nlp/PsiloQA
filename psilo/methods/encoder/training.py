from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.models.trainer import Trainer

from encoder.utils.common import build_samples, set_seed


def run_encoder_training(model_name: str, output_dir: str, batch_size: int, epochs: int, learning_rate: float, dataset):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    train_samples = build_samples(dataset["train"], "train")
    dev_samples = build_samples(dataset["validation"], "dev")

    train_dataset = HallucinationDataset(train_samples, tokenizer, max_length=8192)
    dev_dataset = HallucinationDataset(dev_samples, tokenizer, max_length=8192)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=dev_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=output_dir,
    )
    trainer.train()
