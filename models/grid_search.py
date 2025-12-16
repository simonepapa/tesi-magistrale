"""
Grid Search for Hyperparameter Tuning
======================================
Performs grid search over hyperparameters to find the optimal configuration
for BERT, mDeBERTa, and UmBERTo models.

Usage:
    python grid_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it
    python grid_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --extra_train ../datasets/train_set_real.json
    python grid_search.py --model all --dataset_dir ../datasets/gemma-3-27b-it --quick
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)

from config import LABELS, id2label, label2id, get_model_config, get_available_models
from train import (
    check_gpu, 
    load_json_files_from_folder, 
    convert_labels_format, 
    preprocess_data,
    multi_label_metrics,
    compute_metrics
)


# Grid search parameter space
GRID_PARAMS = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32],
    'weight_decay': [0.01, 0.1],
    'warmup_ratio': [0.0, 0.1, 0.2],
}

# Quick grid (fewer combinations for faster testing)
QUICK_GRID_PARAMS = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32],
}


def load_dataset_for_grid_search(path=None, folder=None, extra_train=None, val_size=0.2):
    """Load dataset and create train/val split for grid search.
    
    :param path: Path to single JSON file
    :param folder: Path to folder with JSON files
    :param extra_train: Path to additional training data
    :param val_size: Validation set proportion
    :returns: Tuple of (train_df, val_df, available_labels)
    """
    # Load main dataset
    if folder:
        print(f"\nLoading data from folder: {folder}")
        articles = load_json_files_from_folder(folder)
        df = pd.DataFrame(articles)
    elif path:
        print(f"\nLoading data from file: {path}")
        df = pd.read_json(path)
    else:
        raise ValueError("Either 'path' or 'folder' must be provided")
    
    # Load extra training data if provided
    if extra_train and os.path.exists(extra_train):
        print(f"Loading extra training data from: {extra_train}")
        with open(extra_train, 'r', encoding='utf-8') as f:
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
    print(f"Available labels: {len(available_labels)}/{len(LABELS)}")
    
    # Ensure content column exists
    if 'contenuto' in df.columns and 'content' not in df.columns:
        df['content'] = df['contenuto']
    
    # Create one-hot labels
    df['one_hot_labels'] = df[available_labels].values.tolist()
    df = df[['content', 'one_hot_labels']]
    
    # Split
    df_train, df_val = train_test_split(df, test_size=val_size, random_state=42)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val:   {len(df_val)} samples")
    
    return df_train, df_val, available_labels


def train_single_config(model_name, df_train, df_val, config_params, epochs=5, device=None):
    """Train a single configuration and return validation metrics.
    
    :param model_name: Name of the model
    :param df_train: Training DataFrame
    :param df_val: Validation DataFrame
    :param config_params: Dictionary of hyperparameters
    :param epochs: Number of training epochs
    :param device: PyTorch device
    :returns: Dictionary with metrics
    """
    config = get_model_config(model_name)
    base_model = config['base_model']
    
    # Create temporary output directory
    temp_dir = f"temp_grid_search_{model_name}_{datetime.now().strftime('%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load tokenizer and model
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
        
        # Create datasets
        train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True), preserve_index=False)
        val_dataset = Dataset.from_pandas(df_val.reset_index(drop=True), preserve_index=False)
        
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
        # Tokenize
        encoded_dataset = dataset.map(
            lambda x: preprocess_data(x, tokenizer),
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        encoded_dataset.set_format("torch")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=temp_dir,
            eval_strategy="epoch",
            save_strategy="no",  # Don't save checkpoints during grid search
            learning_rate=config_params['learning_rate'],
            per_device_train_batch_size=config_params['batch_size'],
            per_device_eval_batch_size=config_params['batch_size'],
            num_train_epochs=epochs,
            weight_decay=config_params.get('weight_decay', 0.01),
            warmup_ratio=config_params.get('warmup_ratio', 0.1),
            load_best_model_at_end=False,
            logging_dir=temp_dir,
            logging_steps=100,
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
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        return {
            'f1_macro': eval_results.get('eval_f1_macro', 0),
            'f1_micro': eval_results.get('eval_f1_micro', 0),
            'accuracy': eval_results.get('eval_accuracy', 0),
            'loss': eval_results.get('eval_loss', float('inf'))
        }
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Clear GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_grid_search(model_name, df_train, df_val, grid_params, epochs=5, device=None):
    """Run grid search over hyperparameters.
    
    :param model_name: Name of the model
    :param df_train: Training DataFrame
    :param df_val: Validation DataFrame
    :param grid_params: Dictionary of parameter lists
    :param epochs: Number of training epochs per config
    :param device: PyTorch device
    :returns: List of results for each configuration
    """
    # Generate all combinations
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    combinations = list(product(*param_values))
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Epochs per config: {epochs}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, combo in enumerate(combinations):
        config_params = dict(zip(param_names, combo))
        
        print(f"\n[{i+1}/{len(combinations)}] Testing: {config_params}")
        
        try:
            metrics = train_single_config(
                model_name, df_train, df_val, 
                config_params, epochs, device
            )
            
            result = {
                'config': config_params,
                'metrics': metrics
            }
            results.append(result)
            
            print(f"  -> F1 Macro: {metrics['f1_macro']:.4f}, F1 Micro: {metrics['f1_micro']:.4f}")
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({
                'config': config_params,
                'metrics': None,
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models() + ['all'],
                        help='Model to tune (default: bert)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to folder containing JSON files')
    parser.add_argument('--extra_train', type=str, default=None,
                        help='Path to additional training data file')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Epochs per configuration (default: 5)')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Use quick grid (fewer combinations)')
    parser.add_argument('--output', '-o', type=str, default='grid_search_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Validate
    if not args.dataset and not args.dataset_dir:
        parser.error("You must specify either --dataset or --dataset_dir")
    
    # Select grid
    grid_params = QUICK_GRID_PARAMS if args.quick else GRID_PARAMS
    
    # Setup
    device = check_gpu()
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load data once
    df_train, df_val, available_labels = load_dataset_for_grid_search(
        path=args.dataset if not args.dataset_dir else None,
        folder=args.dataset_dir,
        extra_train=args.extra_train
    )
    
    # Run grid search
    all_results = {}
    
    models = get_available_models() if args.model == 'all' else [args.model]
    
    for model_name in models:
        results = run_grid_search(
            model_name, df_train, df_val,
            grid_params, args.epochs, device
        )
        all_results[model_name] = results
    
    # Find best configuration per model
    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    
    best_configs = {}
    
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r.get('metrics')]
        if valid_results:
            best = max(valid_results, key=lambda x: x['metrics']['f1_macro'])
            best_configs[model_name] = best
            
            print(f"\n{model_name.upper()}:")
            print(f"  Best config: {best['config']}")
            print(f"  F1 Macro:    {best['metrics']['f1_macro']:.4f}")
            print(f"  F1 Micro:    {best['metrics']['f1_micro']:.4f}")
            print(f"  Accuracy:    {best['metrics']['accuracy']:.4f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'grid_params': grid_params,
        'epochs_per_config': args.epochs,
        'all_results': all_results,
        'best_configs': {k: v for k, v in best_configs.items()}
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print("=" * 60)
    
    # Print command to retrain with best config
    if best_configs:
        print("\nTo retrain with best config:")
        for model_name, best in best_configs.items():
            cfg = best['config']
            cmd = f"python train.py --model {model_name} --dataset_dir {args.dataset_dir or args.dataset}"
            cmd += f" --learning_rate {cfg['learning_rate']} --batch_size {cfg['batch_size']}"
            if args.extra_train:
                cmd += f" --extra_train {args.extra_train}"
            print(f"  {cmd}")


if __name__ == "__main__":
    main()
