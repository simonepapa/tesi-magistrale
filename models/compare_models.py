"""
Compare Models
==============
Compare the performance of all supported models (BERT, mDeBERTa, UmBERTo)
for crime classification.

Features:
1. Evaluate all models on the same test set
2. Compare predictions on sample articles
3. Analyze agreement/disagreement between models
4. Generate comparison report

Usage:
    python compare_models.py --mode quick         # Quick comparison on sample text
    python compare_models.py --mode sample        # Compare on sample articles
    python compare_models.py --mode evaluate      # Full evaluation on test set
    python compare_models.py --mode full          # All comparisons
    python compare_models.py --models bert,umberto  # Compare specific models
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import LABELS, get_model_config, get_available_models
from inference import load_model, predict_with_chunking


def find_checkpoint(model_name: str, dataset_name: str = None, run_folder: str = None) -> str:
    """Find the checkpoint path for a model.
    
    :param model_name: str: model name
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folder: str: specific run folder name for this model (e.g., 'e10_b32_v1') (optional)
    :returns: str: checkpoint path
    """
    if dataset_name:
        dataset_path = os.path.join("results", model_name, dataset_name)
        
        # If run_folder specified, use it directly
        if run_folder:
            potential_path = os.path.join(dataset_path, run_folder, "model")
            if os.path.exists(potential_path):
                print(f"Using checkpoint: {potential_path}")
                return potential_path
            else:
                print(f"Warning: Run folder '{run_folder}' not found for {model_name}, searching for alternatives...")
        
        # Check if dataset_path exists and look for versioned runs
        if os.path.exists(dataset_path):
            subdirs = sorted([d for d in os.listdir(dataset_path) 
                            if os.path.isdir(os.path.join(dataset_path, d))], reverse=True)
            
            # Try versioned structure first: results/{model}/{dataset}/{run}/model
            for subdir in subdirs:
                potential_path = os.path.join(dataset_path, subdir, "model")
                if os.path.exists(potential_path):
                    print(f"Using checkpoint: {potential_path}")
                    return potential_path
            
            # Fallback: old structure results/{model}/{dataset}/model
            old_path = os.path.join(dataset_path, "model")
            if os.path.exists(old_path):
                return old_path
            
    # If no specific dataset or not found, check if there's any folder in results/{model}
    results_base = os.path.join("results", model_name)
    if os.path.exists(results_base):
        subdirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        if subdirs:
            # Check each dataset folder
            for dataset_dir in subdirs:
                dataset_path = os.path.join(results_base, dataset_dir)
                run_subdirs = sorted([d for d in os.listdir(dataset_path) 
                                     if os.path.isdir(os.path.join(dataset_path, d))], reverse=True)
                
                # Try versioned structure
                for run_dir in run_subdirs:
                    potential_path = os.path.join(dataset_path, run_dir, "model")
                    if os.path.exists(potential_path):
                        print(f"Using checkpoint: {potential_path}")
                        return potential_path
                
                # Fallback: old structure
                old_path = os.path.join(dataset_path, "model")
                if os.path.exists(old_path):
                    return old_path

    # Fallback to config default
    return None


def convert_labels_format(df):
    """Convert labels from array format to one-hot encoded columns (borrowed from train.py)"""
    if 'labels' in df.columns:
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
    return df


def load_test_data(dataset_path: str = None, test_file: str = None, test_size: float = 0.1):
    """Load test portion of the dataset.

    :param dataset_path: str: path to the full dataset (for splitting)
    :param test_file: str: path to separate test file (optional)
    :param test_size: float: size of the test set (Default value = 0.1)

    """
    if test_file and os.path.exists(test_file):
        print(f"Loading test set from file: {test_file}")
        df_test = pd.read_json(test_file)
        
    elif dataset_path and os.path.exists(dataset_path):
        print(f"Loading test set from split of: {dataset_path}")
        df = pd.read_json(dataset_path)
        df_test = convert_labels_format(df) # Convert full dataset first if needed
        df = df.sample(frac=1.0, random_state=42)  # Shuffle with same seed as training
        
        # Same split as training to get the exact test set
        df_trainval, df_test = train_test_split(
            df, test_size=test_size, random_state=42, shuffle=True
        )
    else:
        raise ValueError("Either dataset_path or test_file must be provided and exist")
        
    # Ensure labels format
    df_test = convert_labels_format(df_test)
    
    # Ensure content column
    if 'contenuto' in df_test.columns and 'content' not in df_test.columns:
        df_test['content'] = df_test['contenuto']
    
    return df_test


def evaluate_single_model(model_name: str, test_df: pd.DataFrame, dataset_name: str = None, run_folder: str = None) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate a single model on the test set.

    :param model_name: str: name of the model to evaluate
    :param test_df: pd.DataFrame: test set
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folder: str: specific run folder for this model (optional)

    """
    config = get_model_config(model_name)
    checkpoint = find_checkpoint(model_name, dataset_name, run_folder)
    display_name = f"{config['name']} ({'custom' if checkpoint else 'default'})"
    print(f"\nEvaluating {display_name}...")
    
    model, tokenizer, device, _ = load_model(model_name, checkpoint)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=config['name']):
        content = row.get("content", "")
        true_labels = [row.get(label, 0) for label in LABELS]
        
        if not content:
            preds = {label: {"value": 0, "prob": 0.0} for label in LABELS}
        else:
            preds, _ = predict_with_chunking(model, tokenizer, content, device)
        
        pred_values = [preds[label]["value"] for label in LABELS]
        pred_probs = [preds[label]["prob"] for label in LABELS]
        
        all_preds.append(pred_values)
        all_probs.append(pred_probs)
        all_labels.append(true_labels)
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        "model": config['name'],
        "f1_micro": f1_score(y_true, y_pred, average='micro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    
    # ROC AUC
    try:
        metrics["roc_auc_micro"] = roc_auc_score(y_true, y_probs, average='micro')
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_probs, average='macro')
    except ValueError:
        metrics["roc_auc_micro"] = 0.5
        metrics["roc_auc_macro"] = 0.5
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(LABELS):
        metrics[f"f1_{label}"] = per_class_f1[i]
    
    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics, y_pred, y_probs


def compare_predictions(predictions: Dict[str, Dict], content: str, title: str = None) -> Dict:
    """Compare predictions from multiple models on a single article.

    :param predictions: Dict[str, Dict]: predictions from multiple models
    :param content: str: article content
    :param title: str: article title (Default value = None)

    """
    model_names = list(predictions.keys())
    
    comparison = {
        "title": title or content[:50] + "...",
        "agreements": [],
        "disagreements": []
    }
    
    for label in LABELS:
        values = {m: predictions[m][label]["value"] for m in model_names}
        probs = {m: predictions[m][label]["prob"] for m in model_names}
        
        all_same = len(set(values.values())) == 1
        
        entry = {"label": label, "predictions": {}}
        for m in model_names:
            entry["predictions"][m] = {"value": values[m], "prob": probs[m]}
        
        if all_same and list(values.values())[0] == 1:
            comparison["agreements"].append(entry)
        elif not all_same:
            comparison["disagreements"].append(entry)
    
    return comparison


def run_quick_comparison(models_to_compare: List[str], dataset_name: str = None, run_folders: Dict[str, str] = None):
    """Quick side-by-side comparison on a single example.

    :param models_to_compare: List[str]: models to compare
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folders: Dict[str, str]: mapping of model name to run folder (optional)

    """
    if run_folders is None:
        run_folders = {}
        
    print("\n" + "="*60)
    print("QUICK COMPARISON")
    print("="*60)
    
    # Sample text
    test_text = """
    Un grave episodio di cronaca si è verificato ieri sera nel quartiere Libertà di Bari.
    Un uomo di 35 anni è stato arrestato dalla polizia con l'accusa di rapina aggravata
    e lesioni personali. Secondo le ricostruzioni, l'uomo avrebbe aggredito un passante
    nei pressi di via Manzoni, strappandogli il portafoglio e il telefono cellulare.
    La vittima, un anziano di 72 anni, è stata trasportata al Policlinico di Bari dove
    è stata medicata per le contusioni riportate. L'aggressore è stato rintracciato poco
    dopo dalle forze dell'ordine grazie alle immagini delle telecamere di sorveglianza
    presenti nella zona. Durante la perquisizione, sono stati trovati anche 50 grammi
    di cocaina, facendo scattare l'accusa di spaccio di sostanze stupefacenti.
    """
    
    # Load models and predict
    all_preds = {}
    all_chunks = {}
    
    for model_name in models_to_compare:
        config = get_model_config(model_name)
        run_folder = run_folders.get(model_name)
        checkpoint = find_checkpoint(model_name, dataset_name, run_folder)
        print(f"\nLoading {config['name']} from {checkpoint or 'default'}...")
        model, tokenizer, device, _ = load_model(model_name, checkpoint)
        
        preds, chunks = predict_with_chunking(model, tokenizer, test_text, device)
        all_preds[config['name']] = preds
        all_chunks[config['name']] = chunks
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Display results
    model_names = list(all_preds.keys())
    header = f"{'Label':<30}" + "".join([f"{m:>15}" for m in model_names])
    print(f"\n{header}")
    print("-" * (30 + 15 * len(model_names)))
    
    for label in LABELS:
        values = [all_preds[m][label] for m in model_names]
        
        # Only show if any model detected something
        if any(v["value"] == 1 for v in values) or any(v["prob"] > 0.3 for v in values):
            row = f"{label:<30}"
            for v in values:
                symbol = "Y" if v["value"] else "o"
                row += f"{symbol} {v['prob']:.2f}".rjust(15)
            
            # Mark disagreements
            if len(set(v["value"] for v in values)) > 1:
                row += " *"
            
            print(row)
    
    print(f"\nChunks used: " + ", ".join([f"{m}={all_chunks[m]}" for m in model_names]))


def run_sample_comparison(models_to_compare: List[str], num_samples: int = 10, 
                          dataset_path: str = None, test_file: str = None, dataset_name: str = None, run_folders: Dict[str, str] = None):
    """Compare models on a sample of articles.

    :param models_to_compare: List[str]: models to compare
    :param num_samples: int: number of samples to compare (Default value = 10)
    :param dataset_path: str: path to the full dataset (optional)
    :param test_file: str: path to separate test file (optional)
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folders: Dict[str, str]: mapping of model name to run folder (optional)

    """
    if run_folders is None:
        run_folders = {}
        
    print("\n" + "="*60)
    print("SAMPLE COMPARISON")
    print("="*60)
    
    # Load test data
    test_df = load_test_data(dataset_path, test_file)
    samples = test_df.sample(n=min(num_samples, len(test_df)), random_state=42)
    
    # Load all models
    loaded_models = {}
    for model_name in models_to_compare:
        config = get_model_config(model_name)
        run_folder = run_folders.get(model_name)
        checkpoint = find_checkpoint(model_name, dataset_name, run_folder)
        print(f"\nLoading {config['name']} from {checkpoint or 'default'}...")
        model, tokenizer, device, _ = load_model(model_name, checkpoint)
        loaded_models[model_name] = (model, tokenizer, device, config['name'])
    
    # Compare on samples
    total_predictions = 0
    agreement_counts = {m: 0 for m in models_to_compare}
    
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        content = row.get("content", "")
        title = row.get("title", "")
        
        if not content:
            continue
        
        print(f"\n{'─'*60}")
        print(f"Article {idx}: {title[:55]}...")
        
        predictions = {}
        for model_name, (model, tokenizer, device, display_name) in loaded_models.items():
            preds, _ = predict_with_chunking(model, tokenizer, content, device)
            predictions[display_name] = preds
        
        # Show detected crimes
        for label in LABELS:
            values = {m: predictions[m][label] for m in predictions}
            detected = [m for m, v in values.items() if v["value"] == 1]
            
            if detected:
                probs = ", ".join([f"{m}:{values[m]['prob']:.2f}" for m in detected])
                print(f"  {label}: {probs}")
    
    # Cleanup
    for model_name, (model, _, _, _) in loaded_models.items():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_full_evaluation(models_to_compare: List[str], dataset_path: str = None, 
                        test_file: str = None, dataset_name: str = None, run_folders: Dict[str, str] = None):
    """Run full evaluation on test set for all specified models.

    :param models_to_compare: List[str]: models to compare
    :param dataset_path: str: path to the full dataset (optional)
    :param test_file: str: separate test file path (optional)
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folders: Dict[str, str]: mapping of model name to run folder (optional)

    """
    if run_folders is None:
        run_folders = {}
        
    print("\n" + "="*60)
    print("FULL MODEL EVALUATION")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data(dataset_path, test_file)
    print(f"Test set size: {len(test_df)} articles")
    
    # Evaluate each model
    all_metrics = {}
    all_preds = {}
    
    for model_name in models_to_compare:
        run_folder = run_folders.get(model_name)
        metrics, preds, probs = evaluate_single_model(model_name, test_df, dataset_name, run_folder)
        config = get_model_config(model_name)
        all_metrics[config['name']] = metrics
        all_preds[config['name']] = preds
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    model_names = list(all_metrics.keys())
    
    # Main metrics comparison
    main_metrics = ["f1_micro", "f1_macro", "precision_micro", "recall_micro", 
                    "roc_auc_micro", "accuracy"]
    
    header = f"{'Metric':<20}" + "".join([f"{m:>12}" for m in model_names])
    print(f"\n{header}")
    print("-" * (20 + 12 * len(model_names)))
    
    for metric in main_metrics:
        row = f"{metric:<20}"
        values = [all_metrics[m][metric] for m in model_names]
        best_val = max(values)
        
        for v in values:
            marker = "*" if v == best_val and values.count(best_val) == 1 else " "
            row += f"{v:>11.4f}{marker}"
        
        print(row)
    
    # Per-class F1 comparison
    print(f"\n{'─'*60}")
    print("Per-Class F1 Scores:")
    print(f"{'─'*60}")
    
    header = f"{'Label':<25}" + "".join([f"{m:>12}" for m in model_names]) + f"{'Best':>10}"
    print(header)
    print("-" * (25 + 12 * len(model_names) + 10))
    
    wins = {m: 0 for m in model_names}
    
    for label in LABELS:
        row = f"{label:<25}"
        values = {m: all_metrics[m][f"f1_{label}"] for m in model_names}
        best_model = max(values, key=values.get)
        
        for m in model_names:
            row += f"{values[m]:>12.4f}"
        
        row += f"{best_model:>10}"
        print(row)
        
        # Count wins
        max_val = max(values.values())
        winners = [m for m, v in values.items() if v == max_val]
        if len(winners) == 1:
            wins[winners[0]] += 1
    
    print(f"\nCategory wins: " + ", ".join([f"{m}={wins[m]}" for m in model_names]))
    
    # Agreement analysis (pairwise)
    if len(model_names) >= 2:
        print(f"\n{'─'*60}")
        print("Prediction Agreement (pairwise):")
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i+1:]:
                agreement = np.sum(all_preds[m1] == all_preds[m2])
                total = all_preds[m1].size
                print(f"  {m1} vs {m2}: {100*agreement/total:.1f}%")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_size": len(test_df),
        "models_compared": model_names,
        "metrics": all_metrics,
        "category_wins": wins
    }
    
    output_file = "evaluation_results/model_comparison.json"
    os.makedirs("evaluation_results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Determine winner
    print("\n" + "="*60)
    f1_scores = {m: all_metrics[m]["f1_macro"] for m in model_names}
    best_model = max(f1_scores, key=f1_scores.get)
    print(f"BEST MODEL (F1 Macro): {best_model} ({f1_scores[best_model]:.4f})")
    print("="*60)
    
    return results


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare crime classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "sample", "evaluate", "full"],
                        help="Comparison mode (default: quick)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to compare (default: all)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples for sample mode (default: 10)")
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to a separate test set JSON file (optional)')
    parser.add_argument('--dataset_models', type=str, default=None,
                        help='Name of the dataset used for training (e.g. gemma-3-27b-it) to locate models in results/')
    parser.add_argument('--bert_run', type=str, default=None,
                        help='Run folder for BERT model (e.g. e10_b32_v1)')
    parser.add_argument('--mdeberta_run', type=str, default=None,
                        help='Run folder for mDeBERTa model (e.g. e10_b16_v1)')
    parser.add_argument('--umberto_run', type=str, default=None,
                        help='Run folder for UmBERTo model (e.g. e10_b32_v1)')
    
    args = parser.parse_args()
    
    # Build run_folders dict from per-model arguments
    run_folders = {}
    if args.bert_run:
        run_folders['bert'] = args.bert_run
    if args.mdeberta_run:
        run_folders['mdeberta'] = args.mdeberta_run
    if args.umberto_run:
        run_folders['umberto'] = args.umberto_run
    
    # Validate dataset arguments
    if not args.dataset and not args.test_file and args.mode in ['evaluate', 'full', 'sample']:
        # Default behavior: try dataset.json if exists
        if os.path.exists("dataset.json"):
            args.dataset = "dataset.json"
        elif os.path.exists("../datasets/test_set.json") and not args.dataset:
             args.test_file = "../datasets/test_set.json"
    
    # Parse models
    if args.models:
        models_to_compare = [m.strip() for m in args.models.split(",")]
        # Validate
        available = get_available_models()
        for m in models_to_compare:
            if m not in available:
                print(f"Error: Unknown model '{m}'. Available: {available}")
                return
    else:
        models_to_compare = get_available_models()
    
    print(f"Comparing models: {models_to_compare}")
    if args.dataset_models:
        print(f"Using models trained on: {args.dataset_models}")
    if run_folders:
        print(f"Using specific runs: {run_folders}")
    
    if args.mode == "quick":
        run_quick_comparison(models_to_compare, args.dataset_models, run_folders)
    elif args.mode == "sample":
        run_sample_comparison(models_to_compare, args.samples, args.dataset, args.test_file, args.dataset_models, run_folders)
    elif args.mode == "evaluate" or args.mode == "full":
        run_full_evaluation(models_to_compare, args.dataset, args.test_file, args.dataset_models, run_folders)


if __name__ == "__main__":
    main()
