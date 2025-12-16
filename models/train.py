"""
Model Training
==============
Unified training script for all models (BERT, mDeBERTa, UmBERTo).

Usage:
    python train.py --model bert --dataset dataset.json
    python train.py --model bert --dataset_dir datasets/gemma-3-27b-it
    python train.py --model mdeberta --epochs 5 --batch_size 16
    python train.py --model umberto --dataset_dir datasets/gemma-3-27b-it --learning_rate 1e-5
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


def load_dataset(path: str = None, folder: str = None, test_file: str = None, extra_train: str = None, test_size: float = 0.1, val_size: float = 0.1):
    """Load and prepare the dataset with train/val/test splits.

    :param path: str: Path to the dataset JSON file (optional if folder is provided).
    :param folder: str: Path to folder containing JSON files (optional if path is provided).
    :param test_file: str: Optional path to a separate test set file.
    :param extra_train: str: Optional path to additional training data (e.g., real articles).
    :param test_size: float: Proportion of the dataset to include in the test split. (Default value = 0.1)
    :param val_size: float: Proportion of the dataset to include in the validation split. (Default value = 0.1)

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
    
    # Load extra training data if provided
    if extra_train and os.path.exists(extra_train):
        print(f"Loading extra training data from: {extra_train}")
        with open(extra_train, 'r', encoding='utf-8') as f:
            extra_articles = json.load(f)
        print(f"  Loaded {len(extra_articles)} extra articles")
        # Combine with main dataset
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
    
    # Convert labels format if needed
    df = convert_labels_format(df)
    
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
    def ensure_content(dframe):
        if 'contenuto' in dframe.columns and 'content' not in dframe.columns:
            dframe['content'] = dframe['contenuto']
        if 'content' not in dframe.columns:
            raise ValueError("Dataset must have a 'content' column with article text")
        return dframe

    df = ensure_content(df)
    if df_test_separate is not None:
        df_test_separate = ensure_content(df_test_separate)
    
    # Create one-hot labels
    df['one_hot_labels'] = df[available_labels].values.tolist()
    if df_test_separate is not None:
        # Create one-hot labels for test set (using same available labels)
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
        # If separate test set is provided:
        # 1. Use the separate file as Test Set
        df_test = df_test_separate
        
        # 2. Split the main dataset into Train (90%) and Val (10%)
        # Note: val_size is roughly 0.1
        df_train, df_val = train_test_split(
            df, test_size=val_size, random_state=42, shuffle=True
        )
    else:
        # Standard split: Train+Val vs Test
        df_trainval, df_test = train_test_split(
            df, test_size=test_size, random_state=42, shuffle=True
        )
        # Then Train vs Val
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
    
    # Extract dataset name from path
    if args.dataset_dir:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    else:
        # Extract from filename
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Create organized output directories
    # Structure: results/{model}/{dataset}/
    base_results_dir = "results"
    model_dir = os.path.join(base_results_dir, args.model, dataset_name)
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
        "dataset_name": dataset_name,
        "dataset_source": args.dataset_dir or args.dataset,
        "trained_at": datetime.now().isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "train_samples": len(encoded_dataset["train"]),
        "val_samples": len(encoded_dataset["validation"]),
        "test_samples": len(encoded_dataset["test"]),
        "test_results": {k: float(v) for k, v in test_results.items()},
        "labels": LABELS
    }
    
    with open(f"{model_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {model_dir}/")
    print(f"  - Model: {output_dir}/")
    print(f"  - Training logs: {results_dir}/")
    print(f"  - Training info: {model_dir}/training_info.json")
    print("\nNext steps:")
    print(f"  1. Run 'python evaluate.py --model {args.model} --dataset_dir {args.dataset_dir or args.dataset}' for detailed metrics")
    print(f"  2. Run 'python inference.py --model {args.model} --test' to test inference")


# Optimized parameters per model (used when --model all)
MODEL_OPTIMAL_PARAMS = {
    'bert': {'batch_size': 32, 'learning_rate': 2e-5},
    'mdeberta': {'batch_size': 16, 'learning_rate': 1e-5},
    'umberto': {'batch_size': 32, 'learning_rate': 2e-5}
}


def train_all_models(args):
    """Train all models sequentially with optimized parameters.
    
    :param args: argparse.Namespace: command line arguments
    """
    models = get_available_models()
    total_models = len(models)
    
    print("="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    print(f"\nModels to train: {', '.join(models)}")
    print(f"Dataset: {args.dataset_dir or args.dataset}")
    print(f"Epochs: {args.epochs}")
    print("="*60)
    
    results_summary = {}
    
    for i, model_name in enumerate(models, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{total_models}] Training {model_name.upper()}")
        print(f"{'#'*60}")
        
        # Use optimal parameters for this model (unless user specified custom values)
        optimal = MODEL_OPTIMAL_PARAMS.get(model_name, {})
        
        # Create a copy of args with model-specific parameters
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        
        # Use user-specified values if provided, otherwise use optimal defaults
        if not args.custom_batch_size:
            model_args.batch_size = optimal.get('batch_size', args.batch_size)
        if not args.custom_learning_rate:
            model_args.learning_rate = optimal.get('learning_rate', args.learning_rate)
        
        print(f"\nParameters for {model_name}:")
        print(f"  - Batch size: {model_args.batch_size}")
        print(f"  - Learning rate: {model_args.learning_rate}")
        
        try:
            train(model_args)
            results_summary[model_name] = "SUCCESS"
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            results_summary[model_name] = f"FAILED: {str(e)}"
        
        # Clear GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "="*60)
    print("ALL MODELS TRAINING COMPLETE!")
    print("="*60)
    print("\nSummary:")
    for model_name, status in results_summary.items():
        print(f"  {model_name}: {status}")
    print("\nResults saved to:")
    for model_name in models:
        if results_summary.get(model_name) == "SUCCESS":
            dataset_name = os.path.basename(os.path.normpath(args.dataset_dir)) if args.dataset_dir else os.path.splitext(os.path.basename(args.dataset))[0]
            print(f"  - results/{model_name}/{dataset_name}/")


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
    base_results_dir = "results"
    model_dir = os.path.join(base_results_dir, args.model, f"{dataset_name}_kfold{n_folds}")
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
    if 'contenuto' in df.columns and 'content' not in df.columns:
        df['content'] = df['contenuto']
    if 'content' not in df.columns:
        raise ValueError("Dataset must have a 'content' column with article text")
    
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
            weight_decay=0.01,
            warmup_ratio=0.1,
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





def main():
    """Main function to parse command line arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Unified training script for crime classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models() + ['all'],
                        help='Model to train, or "all" to train all models (default: bert)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to folder containing JSON files (alternative to --dataset)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to a separate test set JSON file (optional)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
                        help='Batch size (default: optimized per model)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=None,
                        help='Learning rate (default: optimized per model)')
    parser.add_argument('--patience', '-p', type=int, default=2,
                        help='Early stopping patience (default: 2)')
    parser.add_argument('--extra_train', type=str, default=None,
                        help='Path to additional training data file (e.g., real articles from train_set_real.json)')
    parser.add_argument('--kfold', '-k', type=int, default=5,
                        help='Number of folds for k-fold cross-validation (default: 5)')
    
    args = parser.parse_args()
    
    # Validate dataset arguments
    if not args.dataset and not args.dataset_dir:
        parser.error("You must specify either --dataset or --dataset_dir")
    
    # Track if user specified custom values
    args.custom_batch_size = args.batch_size is not None
    args.custom_learning_rate = args.learning_rate is not None
    
    # Set defaults if not specified
    if args.batch_size is None:
        args.batch_size = 32  # Default for single model training
    if args.learning_rate is None:
        args.learning_rate = 2e-5  # Default for single model training
    
    if args.kfold and args.kfold > 0:
        # K-Fold Cross-Validation mode
        if args.model == 'all':
            # Train all models with k-fold
            print("=" * 60)
            print(f"TRAINING ALL MODELS WITH {args.kfold}-FOLD CROSS-VALIDATION")
            print("WARNING: This will take a long time!")
            print("=" * 60)
            models = get_available_models()
            for model_name in models:
                print(f"\n{'='*60}")
                print(f"Training {model_name} with {args.kfold}-fold CV")
                print("=" * 60)
                args.model = model_name
                # Reset custom flags for optimal params
                args.custom_batch_size = False
                args.custom_learning_rate = False
                train_kfold(args)
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Reset model to 'all' for summary
            args.model = 'all'
            print("\n" + "=" * 60)
            print("ALL MODELS TRAINING COMPLETE!")
            print("=" * 60)
        else:
            train_kfold(args)
    else:
        # Standard training (no k-fold, uses train/val/test split)
        if args.model == 'all':
            train_all_models(args)
        else:
            train(args)


if __name__ == "__main__":
    main()

