"""
Model Training
==============
Unified training script for all models (BERT, mDeBERTa, UmBERTo).

Supports:
- Standard training with train/val/test split
- K-Fold cross-validation
- Two-phase training for domain adaptation (synthetic → real)

Usage:
    # Standard training
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it
    
    # Two-phase training (recommended for real data)
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it --two_phase --real_data datasets/train_set_real.json
    
    # K-fold cross-validation
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it --kfold 5
    
    # Train all models
    python train.py --model all --dataset_dir datasets/gemma-3-27b-it
"""

import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
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


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Results are saved in models/results/
RESULTS_BASE_DIR = os.path.join(SCRIPT_DIR, "results")


def check_gpu():
    """Check GPU availability and print info."""
    
    print("GPU CHECK")
    
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA available: {gpu_name}")
        print(f"Total memory: {gpu_memory:.1f} GB")
        return torch.device("cuda")
    else:
        print("CUDA not available, using CPU (training will be slow)")
        return torch.device("cpu")


def get_versioned_run_name(base_dir: str, epochs: int, batch_size: int, 
                           kfold: int = 0, extra_train: str = None, 
                           two_phase: bool = False, epochs_phase2: int = 0) -> str:
    """Generate a versioned run name based on training configuration.
    
    :param base_dir: str: Base directory where runs are stored
    :param epochs: int: Number of training epochs (phase 1 for two-phase)
    :param batch_size: int: Batch size used for training
    :param kfold: int: Number of k-fold splits (0 for standard training)
    :param extra_train: str: Path to extra training data file (optional)
    :param two_phase: bool: Whether this is a two-phase training run
    :param epochs_phase2: int: Number of epochs for phase 2 (two-phase only)
    :returns: str: Versioned run name
    """
    # Build base pattern
    if two_phase:
        base_pattern = f"two_phase_e{epochs}+{epochs_phase2}_b{batch_size}"
    elif kfold > 0:
        base_pattern = f"e{epochs}_b{batch_size}_kfold{kfold}"
    else:
        base_pattern = f"e{epochs}_b{batch_size}"
    
    # Add extra training indicator (only for non-two-phase)
    if extra_train and not two_phase:
        base_pattern += "+extra"
    
    # Find existing versions
    version = 1
    if os.path.exists(base_dir):
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for d in existing_dirs:
            if d.startswith(base_pattern + "_v"):
                try:
                    v = int(d.split("_v")[-1])
                    version = max(version, v + 1)
                except ValueError:
                    pass
    
    return f"{base_pattern}_v{version}"


def load_json_files_from_folder(folder_path: str) -> list:
    """Load and combine all JSON files from a folder.
    
    :param folder_path: str: Path to the folder containing JSON files.
    :returns: list: Combined list of all articles from all JSON files.
    """
    all_articles = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {folder_path}")
    
    print(f"Found {len(json_files)} JSON file(s) in {folder_path}:")
    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            print(f"  - {os.path.basename(json_file)}: {len(articles)} articles")
            all_articles.extend(articles)
    
    print(f"Total articles loaded: {len(all_articles)}")
    return all_articles


def convert_labels_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the labels from array format to one-hot encoded columns.
    
    Supports both old format (labels as columns) and new format (labels as array).
    
    :param df: pd.DataFrame: DataFrame with articles
    :returns: pd.DataFrame: DataFrame with one-hot encoded label columns
    """
    # Check if labels column exists (new format with array)
    if 'labels' in df.columns:
        print("Detected new format: converting labels array to one-hot columns...")
        
        # Initialize all label columns with 0
        for label in LABELS:
            df[label] = 0
        
        # Convert labels array to one-hot columns
        for idx, row in df.iterrows():
            labels_list = row.get('labels', [])
            if isinstance(labels_list, list):
                for label in labels_list:
                    if label in LABELS:
                        df.at[idx, label] = 1
        
        # Drop the original labels column
        df = df.drop(columns=['labels'])
    else:
        print("Detected old format: labels already as columns")
    
    return df


def ensure_content_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a 'content' column.
    
    :param df: pd.DataFrame: DataFrame with articles
    :returns: pd.DataFrame: DataFrame with 'content' column
    """
    if 'contenuto' in df.columns and 'content' not in df.columns:
        df['content'] = df['contenuto']
    if 'content' not in df.columns:
        raise ValueError("Dataset must have a 'content' column with article text")
    return df


def load_dataset(path: str = None, folder: str = None, test_file: str = None, 
                 extra_train: str = None, test_size: float = 0.1, val_size: float = 0.1):
    """Load and prepare the dataset with train/val/test splits.

    :param path: str: Path to the dataset JSON file (optional if folder is provided).
    :param folder: str: Path to folder containing JSON files (optional if path is provided).
    :param test_file: str: Optional path to a separate test set file.
    :param extra_train: str: Optional path to additional training data (e.g., real articles).
    :param test_size: float: Proportion of the dataset to include in the test split.
    :param val_size: float: Proportion of the dataset to include in the validation split.
    :returns: Tuple of (DatasetDict, available_labels)
    """
    # Load data from folder or file
    if folder:
        print(f"\nLoading training data from folder: {folder}")
        articles = load_json_files_from_folder(folder)
        df = pd.DataFrame(articles)
    elif path:
        print(f"\nLoading training data from file: {path}")
        df = pd.read_json(path)
    else:
        raise ValueError("Either 'path' or 'folder' must be provided")
    
    # Convert labels format if needed
    df = convert_labels_format(df)
    
    # Load extra training data if provided
    if extra_train and os.path.exists(extra_train):
        print(f"Loading extra training data from: {extra_train}")
        with open(extra_train, 'r', encoding='utf-8') as f:
            extra_articles = json.load(f)
        print(f"  Loaded {len(extra_articles)} extra articles")
        extra_df = pd.DataFrame(extra_articles)
        extra_df = convert_labels_format(extra_df)
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"  Combined dataset: {len(df)} articles")
    
    # Load separate test set if provided
    df_test_separate = None
    if test_file and os.path.exists(test_file):
        print(f"Loading separate test set from: {test_file}")
        df_test_separate = pd.read_json(test_file)
        df_test_separate = convert_labels_format(df_test_separate)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42)
    
    # Check which labels exist
    available_labels = [l for l in LABELS if l in df.columns]
    print(f"Available labels: {len(available_labels)}/{len(LABELS)}")
    
    # Show label distribution
    print("\nLabel distribution (Training Set):")
    for label in available_labels:
        count = df[label].sum()
        print(f"  {label}: {count}")
    
    # Ensure content column exists
    df = ensure_content_column(df)
    if df_test_separate is not None:
        df_test_separate = ensure_content_column(df_test_separate)
    
    # Create one-hot labels
    df['one_hot_labels'] = df[available_labels].values.tolist()
    if df_test_separate is not None:
        for label in available_labels:
            if label not in df_test_separate.columns:
                df_test_separate[label] = 0
        df_test_separate['one_hot_labels'] = df_test_separate[available_labels].values.tolist()
    
    # Keep only content and labels
    df = df[['content', 'one_hot_labels']]
    if df_test_separate is not None:
        df_test_separate = df_test_separate[['content', 'one_hot_labels']]
    
    # Split logic
    if df_test_separate is not None:
        df_test = df_test_separate
        df_train, df_val = train_test_split(
            df, test_size=val_size, random_state=42, shuffle=True
        )
    else:
        df_trainval, df_test = train_test_split(
            df, test_size=test_size, random_state=42, shuffle=True
        )
        df_train, df_val = train_test_split(
            df_trainval, test_size=val_size/(1-test_size), random_state=42, shuffle=True
        )
    
    print(f"\nDataset splits:")
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
    :param threshold: threshold for binary classification
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


# Optimized parameters per model
MODEL_OPTIMAL_PARAMS = {
    'bert': {'batch_size': 32, 'learning_rate': 2e-5},
    'mdeberta': {'batch_size': 16, 'learning_rate': 1e-5},
    'umberto': {'batch_size': 32, 'learning_rate': 2e-5}
}


def train_standard(args):
    """Standard training function (single phase).

    :param args: argparse.Namespace: command line arguments
    """
    config = get_model_config(args.model)
    base_model = config['base_model']
    
    # Extract dataset name from path
    if args.dataset_dir:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    else:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Create organized output directories
    dataset_dir_path = os.path.join(RESULTS_BASE_DIR, args.model, dataset_name)
    
    # Generate versioned run name
    run_name = get_versioned_run_name(
        dataset_dir_path, args.epochs, args.batch_size, 
        extra_train=args.extra_train
    )
    
    model_dir = os.path.join(dataset_dir_path, run_name)
    output_dir = os.path.join(model_dir, "model")
    results_dir = os.path.join(model_dir, "training_logs")
    
    # Setup
    device = check_gpu()
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nOutput directory: {model_dir}")
    
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
    dataset, available_labels = load_dataset(
        path=args.dataset if not args.dataset_dir else None,
        folder=args.dataset_dir,
        test_file=args.test_file,
        extra_train=getattr(args, 'extra_train', None)
    )
    
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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{results_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none"
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
    
    print(f"STARTING TRAINING - {config['name']}")
    
    print(f"  Model: {base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Device: {device}")
    print("="*60 + "\n")
    
    train_result = trainer.train()
    
    # Evaluate on test set
    
    print("EVALUATING ON TEST SET")
    
    
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
        "run_name": run_name,
        "training_mode": "standard",
        "dataset_name": dataset_name,
        "dataset_source": args.dataset_dir or args.dataset,
        "extra_train": args.extra_train if args.extra_train else None,
        "trained_at": datetime.now().isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "train_samples": len(encoded_dataset["train"]),
        "val_samples": len(encoded_dataset["validation"]),
        "test_samples": len(encoded_dataset["test"]),
        "test_results": {k: float(v) for k, v in test_results.items()},
        "labels": LABELS
    }
    
    with open(f"{model_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    
    print("TRAINING COMPLETE!")
    
    print(f"\nResults saved to: {model_dir}/")
    print(f"  - Model: {output_dir}/")
    print(f"  - Training logs: {results_dir}/")
    print(f"  - Training info: {model_dir}/training_info.json")
    
    return training_info


def train_two_phase(args):
    """Two-phase training for domain adaptation (synthetic → real).

    :param args: argparse.Namespace: command line arguments
    """
    config = get_model_config(args.model)
    base_model = config['base_model']
    
    # Extract dataset name from synthetic path
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    
    # Create output directories
    dataset_dir_path = os.path.join(RESULTS_BASE_DIR, args.model, dataset_name)
    
    # Generate versioned run name
    run_name = get_versioned_run_name(
        dataset_dir_path, args.epochs, args.batch_size,
        two_phase=True, epochs_phase2=args.epochs_phase2
    )
    
    model_dir = os.path.join(dataset_dir_path, run_name)
    phase1_output_dir = os.path.join(model_dir, "phase1_model")
    phase2_output_dir = os.path.join(model_dir, "phase2_model")
    phase1_logs_dir = os.path.join(model_dir, "phase1_logs")
    phase2_logs_dir = os.path.join(model_dir, "phase2_logs")
    
    # Setup
    device = check_gpu()
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(phase1_output_dir, exist_ok=True)
    os.makedirs(phase2_output_dir, exist_ok=True)
    os.makedirs(phase1_logs_dir, exist_ok=True)
    os.makedirs(phase2_logs_dir, exist_ok=True)
    
    print(f"\nOutput directory: {model_dir}")
    os.environ["WANDB_DISABLED"] = "true"
    
    # PHASE 1: Train on Synthetic Data
    
    print("PHASE 1: TRAINING ON SYNTHETIC DATA")
    
    
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
    
    # Load phase 1 dataset (synthetic only - no extra_train)
    dataset_p1, available_labels = load_dataset(
        folder=args.dataset_dir,
        test_file=args.test_file
    )
    
    print("\nTokenizing Phase 1 dataset...")
    encoded_dataset_p1 = dataset_p1.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset_p1['train'].column_names
    )
    encoded_dataset_p1.set_format("torch")
    
    # Phase 1 training arguments
    training_args_p1 = TrainingArguments(
        output_dir=phase1_logs_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{phase1_logs_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none"
    )
    
    # Phase 1 trainer
    trainer_p1 = Trainer(
        model=model,
        args=training_args_p1,
        train_dataset=encoded_dataset_p1["train"],
        eval_dataset=encoded_dataset_p1["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    
    # Train Phase 1
    print(f"\n{'='*60}")
    print(f"PHASE 1 TRAINING - {config['name']}")
    print(f"{'='*60}")
    print(f"  Model: {base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {device}")
    print("="*60 + "\n")
    
    trainer_p1.train()
    
    # Evaluate Phase 1 on test set
    
    print("PHASE 1 EVALUATION ON TEST SET")
    
    
    phase1_test_results = trainer_p1.evaluate(encoded_dataset_p1["test"])
    print(f"\nPhase 1 Test Results:")
    for key, value in phase1_test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save Phase 1 model
    print(f"\nSaving Phase 1 model to {phase1_output_dir}...")
    trainer_p1.save_model(phase1_output_dir)
    tokenizer.save_pretrained(phase1_output_dir)
    
    # =========================================================================
    # PHASE 2: Fine-tune on Real Data
    # =========================================================================
    
    print("PHASE 2: FINE-TUNING ON REAL DATA")
    
    
    # Load Phase 2 dataset (real data)
    print(f"\nLoading real data from: {args.real_data}")
    with open(args.real_data, 'r', encoding='utf-8') as f:
        real_articles = json.load(f)
    print(f"Loaded {len(real_articles)} real articles")
    
    df_real = pd.DataFrame(real_articles)
    df_real = convert_labels_format(df_real)
    df_real = df_real.sample(frac=1.0, random_state=42)
    
    # Show label distribution
    print("\nLabel distribution (Phase 2 - Real):")
    for label in available_labels:
        if label in df_real.columns:
            count = df_real[label].sum()
            print(f"  {label}: {count}")
    
    df_real = ensure_content_column(df_real)
    
    # Create one-hot labels
    for label in available_labels:
        if label not in df_real.columns:
            df_real[label] = 0
    df_real['one_hot_labels'] = df_real[available_labels].values.tolist()
    df_real = df_real[['content', 'one_hot_labels']]
    
    # Split real data
    df_train_p2, df_val_p2 = train_test_split(
        df_real, test_size=0.15, random_state=42, shuffle=True
    )
    
    print(f"\nPhase 2 dataset splits:")
    print(f"  Train: {len(df_train_p2)} samples")
    print(f"  Val:   {len(df_val_p2)} samples")
    
    # Convert to HuggingFace datasets
    dataset_p2 = DatasetDict({
        "train": Dataset.from_pandas(df_train_p2, preserve_index=False),
        "validation": Dataset.from_pandas(df_val_p2, preserve_index=False)
    })
    
    print("\nTokenizing Phase 2 dataset...")
    encoded_dataset_p2 = dataset_p2.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset_p2['train'].column_names
    )
    encoded_dataset_p2.set_format("torch")
    
    # Phase 2 uses a LOWER learning rate for domain adaptation
    phase2_lr = args.phase2_lr
    
    # Phase 2 training arguments
    training_args_p2 = TrainingArguments(
        output_dir=phase2_logs_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=phase2_lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs_phase2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio_phase2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{phase2_logs_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none"
    )
    
    # Phase 2 trainer (uses the model from Phase 1)
    trainer_p2 = Trainer(
        model=model,  # Continues from Phase 1
        args=training_args_p2,
        train_dataset=encoded_dataset_p2["train"],
        eval_dataset=encoded_dataset_p2["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    
    # Train Phase 2
    print(f"\n{'='*60}")
    print(f"PHASE 2 TRAINING - {config['name']}")
    print(f"{'='*60}")
    print(f"  Epochs: {args.epochs_phase2}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {phase2_lr} (reduced for domain adaptation)")
    print(f"  Warmup ratio: {args.warmup_ratio_phase2}")
    print(f"  Device: {device}")
    print("="*60 + "\n")
    
    trainer_p2.train()
    
    # Evaluate Phase 2
    
    print("PHASE 2 EVALUATION")
    
    
    # Evaluate on Phase 2 validation (real data)
    phase2_val_results = trainer_p2.evaluate(encoded_dataset_p2["validation"])
    print(f"\nPhase 2 Validation (Real Data):")
    for key, value in phase2_val_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on Phase 1 test set (synthetic - to check retention)
    phase2_test_results = trainer_p2.evaluate(encoded_dataset_p1["test"])
    print(f"\nPhase 2 Test (Synthetic Data - checking retention):")
    for key, value in phase2_test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save Phase 2 model (final model)
    print(f"\nSaving Phase 2 model (final) to {phase2_output_dir}...")
    trainer_p2.save_model(phase2_output_dir)
    tokenizer.save_pretrained(phase2_output_dir)
    
    # Save Training Summary
    training_info = {
        "model_type": args.model,
        "model_name": config['name'],
        "base_model": base_model,
        "run_name": run_name,
        "training_mode": "two_phase",
        "dataset_name": dataset_name,
        "synthetic_source": args.dataset_dir,
        "real_data_source": args.real_data,
        "trained_at": datetime.now().isoformat(),
        "phase1": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "warmup_ratio": args.warmup_ratio,
            "train_samples": len(encoded_dataset_p1["train"]),
            "val_samples": len(encoded_dataset_p1["validation"]),
            "test_samples": len(encoded_dataset_p1["test"]),
            "test_results": {k: float(v) for k, v in phase1_test_results.items()}
        },
        "phase2": {
            "epochs": args.epochs_phase2,
            "learning_rate": phase2_lr,
            "batch_size": args.batch_size,
            "warmup_ratio": args.warmup_ratio_phase2,
            "train_samples": len(encoded_dataset_p2["train"]),
            "val_samples": len(encoded_dataset_p2["validation"]),
            "val_results_real": {k: float(v) for k, v in phase2_val_results.items()},
            "test_results_synthetic": {k: float(v) for k, v in phase2_test_results.items()}
        },
        "labels": LABELS
    }
    
    with open(f"{model_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Summary
    
    print("TWO-PHASE TRAINING COMPLETE!")
    
    
    print("\nRESULTS COMPARISON:")
    print("-"*60)
    print(f"{'Metric':<25} {'Phase 1':<15} {'Phase 2':<15} {'Change':<15}")
    print("-"*60)
    
    for metric in ['eval_f1_macro', 'eval_f1_micro', 'eval_accuracy']:
        p1_val = phase1_test_results.get(metric, 0)
        p2_val = phase2_test_results.get(metric, 0)
        change = p2_val - p1_val
        change_str = f"{change:+.4f}" if change != 0 else "0.0000"
        print(f"{metric:<25} {p1_val:<15.4f} {p2_val:<15.4f} {change_str:<15}")
    
    print("-"*60)
    
    print(f"\nResults saved to: {model_dir}/")
    print(f"  - Phase 1 Model: {phase1_output_dir}/")
    print(f"  - Phase 2 Model (Final): {phase2_output_dir}/")
    print(f"  - Training Info: {model_dir}/training_info.json")
    
    return training_info


def train_kfold(args):
    """Train model using k-fold cross-validation.
    
    :param args: argparse.Namespace: command line arguments with kfold specified
    """
    from copy import deepcopy
    
    config = get_model_config(args.model)
    base_model = config['base_model']
    n_folds = args.kfold
    
    # Use optimal parameters for this model if not explicitly specified
    optimal_params = MODEL_OPTIMAL_PARAMS.get(args.model, {})
    batch_size = args.batch_size if args.custom_batch_size else optimal_params.get('batch_size', 32)
    learning_rate = args.learning_rate if args.custom_learning_rate else optimal_params.get('learning_rate', 2e-5)
    
    # Extract dataset name from path
    if args.dataset_dir:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    else:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Create output directories
    dataset_dir_path = os.path.join(RESULTS_BASE_DIR, args.model, dataset_name)
    
    # Generate versioned run name with kfold info
    run_name = get_versioned_run_name(
        dataset_dir_path, args.epochs, batch_size, 
        kfold=n_folds, extra_train=args.extra_train
    )
    
    model_dir = os.path.join(dataset_dir_path, run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup
    device = check_gpu()
    os.environ["WANDB_DISABLED"] = "true"
    
    print("=" * 60)
    print(f"K-FOLD CROSS-VALIDATION TRAINING")
    print("=" * 60)
    print(f"Model: {config['name']}")
    print(f"Folds: {n_folds}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {model_dir}")
    print("=" * 60)
    
    # Load full dataset
    if args.dataset_dir:
        print(f"\nLoading data from folder: {args.dataset_dir}")
        articles = load_json_files_from_folder(args.dataset_dir)
        df = pd.DataFrame(articles)
    else:
        print(f"\nLoading data from file: {args.dataset}")
        df = pd.read_json(args.dataset)
    
    # Load extra training data if provided
    if args.extra_train and os.path.exists(args.extra_train):
        print(f"Loading extra training data from: {args.extra_train}")
        with open(args.extra_train, 'r', encoding='utf-8') as f:
            extra_articles = json.load(f)
        print(f"  Loaded {len(extra_articles)} extra articles")
        extra_df = pd.DataFrame(extra_articles)
        extra_df = convert_labels_format(extra_df)
        df = convert_labels_format(df)
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"  Combined dataset: {len(df)} articles")
    else:
        df = convert_labels_format(df)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Check which labels exist
    available_labels = [l for l in LABELS if l in df.columns]
    print(f"\nAvailable labels: {len(available_labels)}/{len(LABELS)}")
    
    # Ensure content column exists
    df = ensure_content_column(df)
    
    # Create one-hot labels
    df['one_hot_labels'] = df[available_labels].values.tolist()
    df = df[['content', 'one_hot_labels']]
    
    # Setup K-Fold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store metrics per fold
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print("\n" + "=" * 60)
        print(f"FOLD {fold + 1}/{n_folds}")
        print("=" * 60)
        
        # Create fold-specific directories
        fold_output_dir = os.path.join(model_dir, f"fold_{fold + 1}", "model")
        fold_results_dir = os.path.join(model_dir, f"fold_{fold + 1}", "training_logs")
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_results_dir, exist_ok=True)
        
        # Split data for this fold
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"  Train: {len(df_train)} samples")
        print(f"  Val:   {len(df_val)} samples")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
        val_dataset = Dataset.from_pandas(df_val, preserve_index=False)
        
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        # Load fresh model for each fold
        print(f"\nLoading model: {base_model}")
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
        
        # Tokenize
        print("Tokenizing dataset...")
        encoded_dataset = dataset.map(
            lambda x: preprocess_data(x, tokenizer),
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        encoded_dataset.set_format("torch")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=fold_results_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            logging_dir=fold_results_dir,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to="none"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )
        
        # Train
        print(f"\nTraining fold {fold + 1}...")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        fold_metrics.append({
            'fold': fold + 1,
            'f1_macro': eval_results.get('eval_f1_macro', 0),
            'f1_micro': eval_results.get('eval_f1_micro', 0),
            'accuracy': eval_results.get('eval_accuracy', 0)
        })
        print(f"\nFold {fold + 1} Results:")
        print(f"  F1 Macro: {fold_metrics[-1]['f1_macro']:.4f}")
        print(f"  F1 Micro: {fold_metrics[-1]['f1_micro']:.4f}")
        print(f"  Accuracy: {fold_metrics[-1]['accuracy']:.4f}")
        
        # Save best model for this fold
        trainer.save_model(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)
        
        # Clear memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute average metrics
    avg_f1_macro = np.mean([m['f1_macro'] for m in fold_metrics])
    avg_f1_micro = np.mean([m['f1_micro'] for m in fold_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    std_f1_macro = np.std([m['f1_macro'] for m in fold_metrics])
    
    # Save summary
    summary = {
        'model': args.model,
        'n_folds': n_folds,
        'dataset': dataset_name,
        'training_mode': 'kfold',
        'fold_metrics': fold_metrics,
        'average_metrics': {
            'f1_macro': avg_f1_macro,
            'f1_macro_std': std_f1_macro,
            'f1_micro': avg_f1_micro,
            'accuracy': avg_accuracy
        }
    }
    
    with open(os.path.join(model_dir, 'kfold_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("K-FOLD CROSS-VALIDATION COMPLETE!")
    print("=" * 60)
    print(f"\nAverage Results ({n_folds} folds):")
    print(f"  F1 Macro: {avg_f1_macro:.4f} (+/- {std_f1_macro:.4f})")
    print(f"  F1 Micro: {avg_f1_micro:.4f}")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"\nResults saved to: {model_dir}/")


def train_all_models(args):
    """Train all models sequentially with optimized parameters.
    
    :param args: argparse.Namespace: command line arguments
    """
    models = get_available_models()
    total_models = len(models)
    
    
    print("TRAINING ALL MODELS")
    
    print(f"\nModels to train: {', '.join(models)}")
    print(f"Dataset: {args.dataset_dir or args.dataset}")
    print(f"Training mode: {'two_phase' if args.two_phase else 'standard'}")
    print(f"Epochs: {args.epochs}")
    
    
    results_summary = {}
    
    for i, model_name in enumerate(models, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{total_models}] Training {model_name.upper()}")
        print(f"{'#'*60}")
        
        # Use optimal parameters for this model
        optimal = MODEL_OPTIMAL_PARAMS.get(model_name, {})
        
        # Create a copy of args with model-specific parameters
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        
        # Use user-specified values if provided, otherwise use optimal defaults
        if not args.custom_batch_size:
            model_args.batch_size = optimal.get('batch_size', args.batch_size)
        if not args.custom_learning_rate:
            model_args.learning_rate = optimal.get('learning_rate', args.learning_rate)
            # For two-phase, also adjust phase2_lr
            if args.two_phase:
                model_args.phase2_lr = model_args.learning_rate / 10
        
        print(f"\nParameters for {model_name}:")
        print(f"  - Batch size: {model_args.batch_size}")
        print(f"  - Learning rate: {model_args.learning_rate}")
        
        try:
            if args.two_phase:
                train_two_phase(model_args)
            else:
                train_standard(model_args)
            results_summary[model_name] = "SUCCESS"
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            results_summary[model_name] = f"FAILED: {str(e)}"
        
        # Clear GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    
    print("ALL MODELS TRAINING COMPLETE!")
    
    print("\nSummary:")
    for model_name, status in results_summary.items():
        print(f"  {model_name}: {status}")


def main():
    """Main function to parse command line arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Unified training script for crime classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard training
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it
    
    # Two-phase training (synthetic → real)
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it --two_phase --real_data datasets/train_set_real.json
    
    # K-fold cross-validation
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it --kfold 5
    
    # Train all models with two-phase
    python train.py --model all --dataset_dir datasets/gemma-3-27b-it --two_phase --real_data datasets/train_set_real.json
        """
    )
    
    # Model selection
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models() + ['all'],
                        help='Model to train, or "all" to train all models (default: bert)')
    
    # Dataset arguments
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to folder containing JSON files (alternative to --dataset)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to a separate test set JSON file (optional)')
    parser.add_argument('--extra_train', type=str, default=None,
                        help='Path to additional training data (for standard training)')
    
    # Two-phase training arguments
    parser.add_argument('--two_phase', action='store_true',
                        help='Enable two-phase training (synthetic → real)')
    parser.add_argument('--real_data', type=str, default=None,
                        help='Path to real data JSON file (required for --two_phase)')
    parser.add_argument('--epochs_phase2', type=int, default=5,
                        help='Number of epochs for Phase 2 (default: 5)')
    parser.add_argument('--phase2_lr', type=float, default=2e-6,
                        help='Learning rate for Phase 2 (default: 2e-6)')
    parser.add_argument('--warmup_ratio_phase2', type=float, default=0.2,
                        help='Warmup ratio for Phase 2 (default: 0.2)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
                        help='Batch size (default: optimized per model)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=None,
                        help='Learning rate (default: optimized per model)')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup_ratio', '-wr', type=float, default=0.1,
                        help='Warmup ratio (default: 0.1)')
    parser.add_argument('--patience', '-p', type=int, default=2,
                        help='Early stopping patience (default: 2)')
    
    # K-fold arguments
    parser.add_argument('--kfold', '-k', type=int, default=0,
                        help='Number of folds for k-fold CV. Use 0 for standard split (default: 0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset and not args.dataset_dir:
        parser.error("You must specify either --dataset or --dataset_dir")
    
    if args.two_phase:
        if not args.real_data:
            parser.error("--real_data is required when using --two_phase")
        if not os.path.exists(args.real_data):
            parser.error(f"Real data file not found: {args.real_data}")
        if args.kfold > 0:
            parser.error("Cannot combine --two_phase with --kfold")
    
    # Track if user specified custom values
    args.custom_batch_size = args.batch_size is not None
    args.custom_learning_rate = args.learning_rate is not None
    
    # Set defaults based on model-specific optimal parameters
    if args.model != 'all':
        optimal = MODEL_OPTIMAL_PARAMS.get(args.model, {})
        if args.batch_size is None:
            args.batch_size = optimal.get('batch_size', 32)
        if args.learning_rate is None:
            args.learning_rate = optimal.get('learning_rate', 2e-5)
    else:
        if args.batch_size is None:
            args.batch_size = 32
        if args.learning_rate is None:
            args.learning_rate = 2e-5
    
    # Run appropriate training mode
    if args.kfold and args.kfold > 0:
        # K-Fold Cross-Validation mode
        if args.model == 'all':
            
            print(f"TRAINING ALL MODELS WITH {args.kfold}-FOLD CROSS-VALIDATION")
            
            models = get_available_models()
            for model_name in models:
                args.model = model_name
                args.custom_batch_size = False
                args.custom_learning_rate = False
                train_kfold(args)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            args.model = 'all'
        else:
            train_kfold(args)
    elif args.two_phase:
        # Two-phase training mode
        if args.model == 'all':
            train_all_models(args)
        else:
            train_two_phase(args)
    else:
        # Standard training mode
        if args.model == 'all':
            train_all_models(args)
        else:
            train_standard(args)


if __name__ == "__main__":
    main()
