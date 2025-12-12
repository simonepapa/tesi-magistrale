"""
Model Training
==============
Unified training script for all models (BERT, mDeBERTa, UmBERTo).

Usage:
    python train.py --model bert
    python train.py --model mdeberta --epochs 5 --batch_size 16
    python train.py --model umberto --learning_rate 1e-5
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from datasets import DatasetDict, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)

from config import LABELS, id2label, label2id, get_model_config, get_available_models


def check_gpu():
    """Check GPU availability and print info."""
    print("="*60)
    print("GPU CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA available: {gpu_name}")
        print(f"Total memory: {gpu_memory:.1f} GB")
        return torch.device("cuda")
    else:
        print("CUDA not available, using CPU (training will be slow)")
        return torch.device("cpu")


def load_dataset(path: str, test_size: float = 0.1, val_size: float = 0.1):
    """Load and prepare the dataset with train/val/test splits.

    :param path: str: Path to the dataset JSON file.
    :param test_size: float: Proportion of the dataset to include in the test split. (Default value = 0.1)
    :param val_size: float: Proportion of the dataset to include in the validation split. (Default value = 0.1)

    """
    print(f"\nLoading dataset from {path}...")
    
    df = pd.read_json(path)
    df = df.sample(frac=1.0, random_state=42)  # Shuffle
    
    # Check which labels exist
    available_labels = [l for l in LABELS if l in df.columns]
    print(f"Available labels: {len(available_labels)}/{len(LABELS)}")
    
    # Create one-hot labels
    df['one_hot_labels'] = df[available_labels].values.tolist()
    
    # Keep only content and labels
    df = df[['content', 'one_hot_labels']]
    
    # Split: first train+val vs test, then train vs val
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=42, shuffle=True
    )
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size/(1-test_size), random_state=42, shuffle=True
    )
    
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val:   {len(df_val)} samples")
    print(f"  Test:  {len(df_test)} samples")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
    val_dataset = Dataset.from_pandas(df_val, preserve_index=False)
    test_dataset = Dataset.from_pandas(df_test, preserve_index=False)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }), available_labels


def preprocess_data(examples, tokenizer):
    """Tokenize and prepare data.

    :param examples: HuggingFace Dataset
    :param tokenizer: HuggingFace Tokenizer

    """
    text = examples["content"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    encoding["labels"] = np.array(examples["one_hot_labels"], dtype=np.float32)
    return encoding


def multi_label_metrics(predictions, labels, threshold=0.5):
    """Compute multi-label classification metrics.

    :param predictions: numpy array of model predictions
    :param labels: numpy array of true labels
    :param threshold: threshold for binary classification (Default value = 0.5)

    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    
    y_true = labels
    
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    
    try:
        roc_auc = roc_auc_score(y_true, probs.numpy(), average='micro')
    except ValueError:
        roc_auc = 0.5
    
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }


def compute_metrics(p: EvalPrediction):
    """Compute metrics for Trainer.

    :param p: EvalPrediction: HuggingFace EvalPrediction object

    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(predictions=preds, labels=p.label_ids)


def train(args):
    """Main training function.

    :param args: argparse.Namespace: command line arguments

    """
    
    config = get_model_config(args.model)
    base_model = config['base_model']
    output_dir = config['checkpoint']
    results_dir = f"training_results_{args.model}"
    
    # Setup
    device = check_gpu()
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load tokenizer and model
    print(f"\nLoading model: {base_model} ({config['name']})")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        use_fast=config['use_fast_tokenizer']
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        problem_type="multi_label_classification",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Load and preprocess dataset
    dataset, available_labels = load_dataset(args.dataset)
    
    print("\nTokenizing dataset...")
    encoded_dataset = dataset.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    encoded_dataset.set_format("torch")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=results_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{results_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,  # Windows compatibility
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    
    # Train
    print("\n" + "="*60)
    print(f"STARTING TRAINING - {config['name']}")
    print("="*60)
    print(f"  Model: {base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Device: {device}")
    print("="*60 + "\n")
    
    train_result = trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_results = trainer.evaluate(encoded_dataset["test"])
    print(f"\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "model_type": args.model,
        "model_name": config['name'],
        "base_model": base_model,
        "trained_at": datetime.now().isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": len(encoded_dataset["train"]),
        "val_samples": len(encoded_dataset["validation"]),
        "test_samples": len(encoded_dataset["test"]),
        "test_results": {k: float(v) for k, v in test_results.items()},
        "labels": LABELS
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {output_dir}/")
    print(f"Training info saved to: {output_dir}/training_info.json")
    print("\nNext steps:")
    print(f"  1. Run 'python evaluate.py --model {args.model}' for detailed metrics")
    print(f"  2. Run 'python inference.py --model {args.model} --test' to test inference")


def main():
    """Main function to parse command line arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Unified training script for crime classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models(),
                        help='Model to train (default: bert)')
    parser.add_argument('--dataset', '-d', type=str, default='dataset.json',
                        help='Path to dataset JSON file')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', '-b', type=int, default=24,
                        help='Batch size (default: 24, reduce for less VRAM)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--patience', '-p', type=int, default=2,
                        help='Early stopping patience (default: 2)')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
