from typing import List

import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from src.config import BASE_MODEL_PATH
from src.utils import compute_metrics, id2label_format, label2id_format


def load_base_model(model_path: str, labels: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=2,
        id2label=id2label_format(labels),
        label2id=label2id_format(labels),
    )
    return model, tokenizer


def tokenize_dataset(tokenizer, dataset_path: str):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    dataset = datasets.load_dataset(dataset_path)
    return dataset.map(preprocess_function, batched=True)


def create_trainer(model, dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="arabert-classification",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        use_mps_device=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer
