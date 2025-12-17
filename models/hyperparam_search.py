"""
Hyperparameter Search for Model Tuning
======================================
Provides three search strategies for hyperparameter optimization:
1. Grid Search - Exhaustive search over all combinations
2. Random Search - Randomly samples configurations
3. Bayesian Search - Uses Optuna for intelligent search

Usage:
    # Grid Search 
    python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method grid
    
    # Random Search
    python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method random --n_trials 10
    
    # Bayesian Search with Optuna
    python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method bayesian --n_trials 20
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import random
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


# SEARCH SPACES
# Grid search parameter space (exhaustive)
GRID_PARAMS = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32],
    'weight_decay': [0.01, 0.1],
    'warmup_ratio': [0.0, 0.1],
}  # 24 combinations

# Quick grid (fewer combinations)
QUICK_GRID_PARAMS = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32],
}  # 6 combinations

# Random/Bayesian search ranges
SEARCH_SPACE = {
    'learning_rate': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-4},
    'batch_size': {'type': 'categorical', 'choices': [8, 16, 32]},
    'weight_decay': {'type': 'loguniform', 'low': 0.001, 'high': 0.3},
    'warmup_ratio': {'type': 'uniform', 'low': 0.0, 'high': 0.2},
}


# DATA LOADING
def load_dataset_for_search(path=None, folder=None, extra_train=None, val_size=0.2):
    """Load dataset and create train/val split for hyperparameter search.
    
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


# TRAINING FUNCTION
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
    temp_dir = f"temp_search_{model_name}_{datetime.now().strftime('%H%M%S%f')}"
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
            save_strategy="no",
            learning_rate=config_params['learning_rate'],
            per_device_train_batch_size=int(config_params['batch_size']),
            per_device_eval_batch_size=int(config_params['batch_size']),
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


# GRID SEARCH
def run_grid_search(model_name, df_train, df_val, grid_params, epochs=5, device=None):
    """Run exhaustive grid search over hyperparameters.
    
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


# RANDOM SEARCH
def sample_random_config(search_space):
    """Sample a random configuration from the search space.
    
    :param search_space: Dictionary defining parameter ranges
    :returns: Dictionary with sampled hyperparameters
    """
    config = {}
    for param, spec in search_space.items():
        if spec['type'] == 'loguniform':
            config[param] = np.exp(random.uniform(np.log(spec['low']), np.log(spec['high'])))
        elif spec['type'] == 'uniform':
            config[param] = random.uniform(spec['low'], spec['high'])
        elif spec['type'] == 'categorical':
            config[param] = random.choice(spec['choices'])
        elif spec['type'] == 'int':
            config[param] = random.randint(spec['low'], spec['high'])
    return config


def run_random_search(model_name, df_train, df_val, search_space, n_trials=10, epochs=5, device=None):
    """Run random search over hyperparameters.
    
    :param model_name: Name of the model
    :param df_train: Training DataFrame
    :param df_val: Validation DataFrame
    :param search_space: Dictionary defining parameter ranges
    :param n_trials: Number of random configurations to try
    :param epochs: Number of training epochs per config
    :param device: PyTorch device
    :returns: List of results for each configuration
    """
    print(f"\n{'='*60}")
    print(f"RANDOM SEARCH: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per config: {epochs}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i in range(n_trials):
        config_params = sample_random_config(search_space)
        
        print(f"\n[{i+1}/{n_trials}] Testing: ", end="")
        print({k: f"{v:.2e}" if isinstance(v, float) else v for k, v in config_params.items()})
        
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



# BAYESIAN SEARCH (OPTUNA)
def run_bayesian_search(model_name, df_train, df_val, search_space, n_trials=20, epochs=5, device=None):
    """Run Bayesian optimization using Optuna.
    
    :param model_name: Name of the model
    :param df_train: Training DataFrame
    :param df_val: Validation DataFrame
    :param search_space: Dictionary defining parameter ranges
    :param n_trials: Number of trials
    :param epochs: Number of training epochs per config
    :param device: PyTorch device
    :returns: List of results for each configuration
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("ERROR: Optuna not installed. Install with: pip install optuna")
        print("Falling back to random search...")
        return run_random_search(model_name, df_train, df_val, search_space, n_trials, epochs, device)
    
    print(f"\n{'='*60}")
    print(f"BAYESIAN SEARCH (Optuna): {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per config: {epochs}")
    print(f"{'='*60}\n")
    
    results = []
    
    def objective(trial):
        # Sample hyperparameters
        config_params = {}
        for param, spec in search_space.items():
            if spec['type'] == 'loguniform':
                config_params[param] = trial.suggest_float(param, spec['low'], spec['high'], log=True)
            elif spec['type'] == 'uniform':
                config_params[param] = trial.suggest_float(param, spec['low'], spec['high'])
            elif spec['type'] == 'categorical':
                config_params[param] = trial.suggest_categorical(param, spec['choices'])
            elif spec['type'] == 'int':
                config_params[param] = trial.suggest_int(param, spec['low'], spec['high'])
        
        print(f"\n[Trial {trial.number + 1}/{n_trials}] Testing: ", end="")
        print({k: f"{v:.2e}" if isinstance(v, float) else v for k, v in config_params.items()})
        
        try:
            metrics = train_single_config(
                model_name, df_train, df_val, 
                config_params, epochs, device
            )
            
            results.append({
                'trial': trial.number,
                'config': config_params,
                'metrics': metrics
            })
            
            print(f"  -> F1 Macro: {metrics['f1_macro']:.4f}, F1 Micro: {metrics['f1_micro']:.4f}")
            
            return metrics['f1_macro']
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({
                'trial': trial.number,
                'config': config_params,
                'metrics': None,
                'error': str(e)
            })
            return 0.0
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Add best trial info
    print(f"\n{'='*60}")
    print("OPTUNA BEST TRIAL:")
    print(f"  Best value (F1 Macro): {study.best_trial.value:.4f}")
    print(f"  Best params: {study.best_trial.params}")
    print(f"{'='*60}")
    
    return results


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for model tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models() + ['all'],
                        help='Model to tune (default: bert)')
    parser.add_argument('--method', type=str, default='grid',
                        choices=['grid', 'random', 'bayesian'],
                        help='Search method: grid, random, or bayesian (default: grid)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to folder containing JSON files')
    parser.add_argument('--extra_train', type=str, default=None,
                        help='Path to additional training data file')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Epochs per configuration (default: 5)')
    parser.add_argument('--n_trials', '-n', type=int, default=10,
                        help='Number of trials for random/bayesian search (default: 10)')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Use quick grid for grid search (fewer combinations)')
    
    args = parser.parse_args()
    
    # Validate
    if not args.dataset and not args.dataset_dir:
        parser.error("You must specify either --dataset or --dataset_dir")
    
    # Setup
    device = check_gpu()
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load data once
    df_train, df_val, available_labels = load_dataset_for_search(
        path=args.dataset if not args.dataset_dir else None,
        folder=args.dataset_dir,
        extra_train=args.extra_train
    )
    
    # Run search
    all_results = {}
    models = get_available_models() if args.model == 'all' else [args.model]
    
    for model_name in models:
        if args.method == 'grid':
            grid_params = QUICK_GRID_PARAMS if args.quick else GRID_PARAMS
            results = run_grid_search(
                model_name, df_train, df_val,
                grid_params, args.epochs, device
            )
        elif args.method == 'random':
            results = run_random_search(
                model_name, df_train, df_val,
                SEARCH_SPACE, args.n_trials, args.epochs, device
            )
        elif args.method == 'bayesian':
            results = run_bayesian_search(
                model_name, df_train, df_val,
                SEARCH_SPACE, args.n_trials, args.epochs, device
            )
        
        all_results[model_name] = results
    
    # Find best configuration per model
    print("\n" + "=" * 60)
    print(f"{args.method.upper()} SEARCH RESULTS")
    print("=" * 60)
    
    best_configs = {}
    
    for model_name, results in all_results.items():
        valid_results = [r for r in results if r.get('metrics')]
        if valid_results:
            best = max(valid_results, key=lambda x: x['metrics']['f1_macro'])
            best_configs[model_name] = best
            
            print(f"\n{model_name.upper()}:")
            print(f"  Best config: ", end="")
            print({k: f"{v:.2e}" if isinstance(v, float) else v for k, v in best['config'].items()})
            print(f"  F1 Macro:    {best['metrics']['f1_macro']:.4f}")
            print(f"  F1 Micro:    {best['metrics']['f1_micro']:.4f}")
            print(f"  Accuracy:    {best['metrics']['accuracy']:.4f}")
    
    # Create output directory with versioned naming
    # Structure: results/{model}/{dataset}/hypersearch_{method}_{timestamp}/
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract dataset name
    if args.dataset_dir:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    else:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Build extra suffix
    extra_suffix = "+extra" if args.extra_train else ""
    
    # Save results per model
    for model_name in models:
        output_dir = os.path.join("results", model_name, dataset_name, f"hypersearch_{args.method}{extra_suffix}_{timestamp_str}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output data
        output = {
            'timestamp': datetime.now().isoformat(),
            'method': args.method,
            'model': model_name,
            'dataset_name': dataset_name,
            'dataset_source': args.dataset_dir or args.dataset,
            'extra_train': args.extra_train if args.extra_train else None,
            'extra_train_name': os.path.basename(args.extra_train) if args.extra_train else None,
            'epochs_per_config': args.epochs,
            'n_trials': args.n_trials if args.method != 'grid' else len(all_results.get(model_name, [])),
            'search_space': SEARCH_SPACE if args.method != 'grid' else (QUICK_GRID_PARAMS if args.quick else GRID_PARAMS),
            'all_results': all_results.get(model_name, []),
            'best_config': best_configs.get(model_name, None)
        }
        
        output_file = os.path.join(output_dir, "search_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n{model_name}: Results saved to: {output_dir}/")
    
    print(f"\n{'='*60}")
    
    # Print command to retrain with best config
    if best_configs:
        print("\nTo retrain with best config:")
        for model_name, best in best_configs.items():
            cfg = best['config']
            cmd = f"python train.py --model {model_name} --dataset_dir {args.dataset_dir or args.dataset}"
            cmd += f" --learning_rate {cfg['learning_rate']:.2e} --batch_size {int(cfg['batch_size'])}"
            if args.extra_train:
                cmd += f" --extra_train {args.extra_train}"
            print(f"  {cmd}")


if __name__ == "__main__":
    main()
