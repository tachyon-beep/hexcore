# src/training/classifier_trainer.py
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def train_transaction_classifier(
    model,
    tokenizer,
    train_data: List[Dict[str, Any]],
    output_dir: str = "./models/transaction_classifier",
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
):
    """
    Train the transaction classifier on labeled examples.

    Args:
        model: The classifier model
        tokenizer: Tokenizer for the model
        train_data: List of training examples with 'text' and 'expert_type' fields
        output_dir: Directory to save the model
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
    """
    logger.info(
        f"Preparing to train transaction classifier with {len(train_data)} examples"
    )

    # Prepare dataset
    texts = [example["text"] for example in train_data]
    labels = [example["expert_type"] for example in train_data]

    # Convert labels to IDs
    label2id = {"REASON": 0, "EXPLAIN": 1, "TEACH": 2, "PREDICT": 3, "RETROSPECT": 4}
    label_ids = [label2id[label] for label in labels]

    # Create dataset
    dataset = Dataset.from_dict({"text": texts, "label": label_ids})

    # Split into train and validation
    splits = dataset.train_test_split(test_size=0.1)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_datasets = DatasetDict(
        {
            "train": splits["train"].map(tokenize_function, batched=True),
            "validation": splits["test"].map(tokenize_function, batched=True),
        }
    )

    # Define metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    # Create a data collator for handling the padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # Use data_collator instead of tokenizer
    )

    # Train model
    logger.info("Starting classifier training")
    trainer.train()

    # Save model
    logger.info(f"Saving classifier to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Return evaluation results
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    return eval_results
