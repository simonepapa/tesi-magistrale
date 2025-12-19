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


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Results are in models/results/
RESULTS_BASE_DIR = os.path.join(SCRIPT_DIR, "results")


def get_relative_path(absolute_path: str) -> str:
    """Convert an absolute path to a relative path from the current working directory.
    
    This avoids issues with non-ASCII characters in absolute paths (e.g., 'à' in 'Università')
    that cause problems with sentencepiece.
    
    :param absolute_path: str: The absolute path
    :returns: str: Relative path from CWD
    """
    try:
        return os.path.relpath(absolute_path)
    except ValueError:
        return absolute_path

def find_checkpoint(model_name: str, dataset_name: str = None, run_folder: str = None) -> str:
    """Find the checkpoint path for a model.
    
    :param model_name: str: model name
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folder: str: specific run folder name for this model (e.g., 'e10_b32_v1') (optional)
    :returns: str: checkpoint path
    """
    if dataset_name:
        dataset_path = os.path.join(RESULTS_BASE_DIR, model_name, dataset_name)
        
        # If run_folder specified, use it directly
        if run_folder:
            # First check for two-phase training (phase2_model)
            phase2_path = os.path.join(dataset_path, run_folder, "phase2_model")
            if os.path.exists(phase2_path):
                print(f"Using checkpoint (two-phase): {phase2_path}")
                return get_relative_path(phase2_path)
            # Then check for standard model folder
            potential_path = os.path.join(dataset_path, run_folder, "model")
            if os.path.exists(potential_path):
                print(f"Using checkpoint: {potential_path}")
                return get_relative_path(potential_path)
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
                    return get_relative_path(potential_path)
            
            # Fallback: old structure results/{model}/{dataset}/model
            old_path = os.path.join(dataset_path, "model")
            if os.path.exists(old_path):
                return get_relative_path(old_path)
            
    # If no specific dataset or not found, check if there's any folder in results/{model}
    results_base = os.path.join(RESULTS_BASE_DIR, model_name)
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
                        return get_relative_path(potential_path)
                
                # Fallback: old structure
                old_path = os.path.join(dataset_path, "model")
                if os.path.exists(old_path):
                    return get_relative_path(old_path)

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


def evaluate_single_model(model_name: str, test_df: pd.DataFrame, dataset_name: str = None, 
                          run_folder: str = None, display_name: str = None) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate a single model on the test set.

    :param model_name: str: name of the model to evaluate (bert, mdeberta, umberto)
    :param test_df: pd.DataFrame: test set
    :param dataset_name: str: dataset name used for training (optional)
    :param run_folder: str: specific run folder for this model (optional)
    :param display_name: str: custom display name for results (optional, defaults to model name)

    """
    config = get_model_config(model_name)
    checkpoint = find_checkpoint(model_name, dataset_name, run_folder)
    
    # Use custom display name if provided, otherwise use model name with run folder
    if display_name:
        display_label = display_name
    elif run_folder:
        display_label = f"{config['name']} ({run_folder})"
    else:
        display_label = f"{config['name']} ({'custom' if checkpoint else 'default'})"
    
    print(f"\nEvaluating {display_label}...")
    
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
    
    return metrics, y_pred, y_probs, display_label


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
        
    
    print("QUICK COMPARISON")
    
    
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
        
    
    print("SAMPLE COMPARISON")
    
    
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

def load_llm_results(llm_results_file: str, llm_name: str) -> Tuple[Dict, np.ndarray]:
    """Load LLM results from JSON file.
    
    :param llm_results_file: str: Path to LLM results JSON
    :param llm_name: str: Display name for the LLM
    :returns: Tuple of (metrics dict, predictions array or None)
    """
    if not llm_results_file or not os.path.exists(llm_results_file):
        return None, None
    
    print(f"\nLoading LLM results from: {llm_results_file}")
    with open(llm_results_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    llm_metrics = llm_data.get('metrics', {})
    
    # Convert per_class_f1 to expected format
    per_class = llm_metrics.get('per_class_f1', {})
    for label in LABELS:
        llm_metrics[f'f1_{label}'] = per_class.get(label, 0.0)
    
    # Load binary predictions if available
    preds = None
    if 'binary_predictions' in llm_data:
        preds = np.array(llm_data['binary_predictions'])
        print(f"Added {llm_name} with predictions for agreement analysis")
    else:
        print(f"Added {llm_name} (no predictions for agreement)")
    
    return llm_metrics, preds


def run_full_evaluation(model_entries: List[Tuple[str, str]], dataset_path: str = None, 
                        test_file: str = None, dataset_name: str = None,
                        llm_api_file: str = None, llm_local_file: str = None):
    """Run full evaluation on test set for all specified models.

    :param model_entries: List of tuples (model_name, run_folder) - allows same model multiple times
    :param dataset_path: str: path to the full dataset (optional)
    :param test_file: str: separate test file path (optional)
    :param dataset_name: str: dataset name used for training (optional)
    :param llm_api_file: str: path to LLM API results file
    :param llm_local_file: str: path to LLM Local results file

    """
    print("FULL MODEL EVALUATION")
    
    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data(dataset_path, test_file)
    print(f"Test set size: {len(test_df)} articles")
    
    # Evaluate each model entry
    all_metrics = {}
    all_preds = {}
    
    for model_name, run_folder in model_entries:
        metrics, preds, probs, display_label = evaluate_single_model(
            model_name, test_df, dataset_name, run_folder
        )
        all_metrics[display_label] = metrics
        all_preds[display_label] = preds
    
    # Add LLM API results if provided
    if llm_api_file:
        metrics, preds = load_llm_results(llm_api_file, "LLM-API")
        if metrics:
            all_metrics["LLM-API"] = metrics
            all_preds["LLM-API"] = preds
    
    # Add LLM Local results if provided
    if llm_local_file:
        metrics, preds = load_llm_results(llm_local_file, "LLM-Local")
        if metrics:
            all_metrics["LLM-Local"] = metrics
            all_preds["LLM-Local"] = preds
    
    # Create comparison table
    # Flush output and add spacing to avoid tqdm overlap
    import sys
    sys.stdout.flush()
    print("\n" * 3)  # Extra spacing after tqdm progress bars
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    
    model_names = list(all_metrics.keys())
    
    # Create abbreviated names for display
    def abbreviate_name(name):
        """Create short name for table display."""
        if "two_phase" in name:
            if "mDeBERTa" in name:
                return "mDeB-TP"
            elif "UmBERTo" in name:
                return "UmB-TP"
            elif "BERT" in name:
                return "BERT-TP"
        elif "e10" in name:
            if "mDeBERTa" in name:
                return "mDeB-Std"
            elif "UmBERTo" in name:
                return "UmB-Std"
            elif "BERT" in name:
                return "BERT-Std"
        # LLM models
        if "LLM-API" in name:
            return "LLM-API"
        if "LLM-Local" in name:
            return "LLM-Loc"
        return name[:10]  # Fallback: first 10 chars
    
    short_names = {m: abbreviate_name(m) for m in model_names}
    
    # Main metrics comparison
    main_metrics = ["f1_macro", "f1_micro", "accuracy", "precision_macro", "recall_macro", "roc_auc_macro"]
    
    print("\n📊 MAIN METRICS:")
    print("-" * 80)
    
    # Header with short names (max 10 chars each)
    header = f"{'Model':<12}"
    for metric in main_metrics:
        metric_short = metric.replace("_macro", "").replace("_micro", "").replace("roc_auc", "AUC")
        header += f"{metric_short:>10}"
    print(header)
    print("-" * 80)
    
    # Sort models by f1_macro descending
    sorted_models = sorted(model_names, key=lambda m: all_metrics[m].get("f1_macro", 0), reverse=True)
    
    for model in sorted_models:
        row = f"{short_names[model]:<12}"
        for metric in main_metrics:
            val = all_metrics[model].get(metric, 0)
            row += f"{val:>10.4f}"
        # Add star for best model
        if model == sorted_models[0]:
            row += " ⭐"
        print(row)
    
    print("-" * 80)
    print(f"\n🏆 Best: {sorted_models[0]} (F1 Macro: {all_metrics[sorted_models[0]]['f1_macro']:.4f})")
    
    # Legend
    print("\nLegend: TP=Two-Phase, Std=Standard, mDeB=mDeBERTa, UmB=UmBERTo")
    
    # Per-class F1 comparison
    print(f"\n{'─'*80}")
    print("📋 Per-Class F1 Scores:")
    print(f"{'─'*80}")
    
    # Use shortened names in header
    header = f"{'Label':<20}"
    for m in model_names:
        header += f"{short_names[m]:>10}"
    header += f"{'Best':>10}"
    print(header)
    print("-" * 80)
    
    wins = {m: 0 for m in model_names}
    
    for label in LABELS:
        row = f"{label:<20}"
        values = {m: all_metrics[m][f"f1_{label}"] for m in model_names}
        best_model = max(values, key=values.get)
        
        for m in model_names:
            row += f"{values[m]:>10.4f}"
        
        row += f"{short_names[best_model]:>10}"
        print(row)
        
        # Count wins
        max_val = max(values.values())
        winners = [m for m, v in values.items() if v == max_val]
        if len(winners) == 1:
            wins[winners[0]] += 1
    
    print(f"\nCategory wins: " + ", ".join([f"{short_names[m]}={wins[m]}" for m in model_names]))
    
    # Agreement analysis (pairwise) - only for models with predictions
    models_with_preds = [m for m in model_names if all_preds.get(m) is not None]
    if len(models_with_preds) >= 2:
        print(f"\n{'─'*80}")
        print("🤝 Prediction Agreement (pairwise):")
        for i, m1 in enumerate(models_with_preds):
            for m2 in models_with_preds[i+1:]:
                agreement = np.sum(all_preds[m1] == all_preds[m2])
                total = all_preds[m1].size
                print(f"  {short_names[m1]} vs {short_names[m2]}: {100*agreement/total:.1f}%")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_size": len(test_df),
        "dataset_name": dataset_name,
        "model_entries": [f"{m}/{f}" if f else m for m, f in model_entries],
        "models_compared": model_names,
        "metrics": all_metrics,
        "category_wins": wins
    }
    
    # Create versioned output filename
    os.makedirs("evaluation_results", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename
    output_file = f"evaluation_results/comparison_{timestamp_str}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Determine winner
    
    f1_scores = {m: all_metrics[m]["f1_macro"] for m in model_names}
    best_model = max(f1_scores, key=f1_scores.get)
    print(f"BEST MODEL (F1 Macro): {best_model} ({f1_scores[best_model]:.4f})")
    
    
    return results


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare crime classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare all models using latest checkpoints
    python compare_models.py --mode evaluate --dataset_models gemma-3-27b-it --test_file ../datasets/test_set.json
    
    # Specify run folders for each model
    python compare_models.py --mode evaluate --dataset_models gemma-3-27b-it --test_file ../datasets/test_set.json \\
        --run_folders bert=two_phase_e10+5_b32_v1 mdeberta=two_phase_e10+5_b16_v1
    
    # Compare only specific models
    python compare_models.py --mode evaluate --models bert,umberto --dataset_models gemma-3-27b-it
        """
    )
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "sample", "evaluate", "full"],
                        help="Comparison mode (default: quick)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to compare when not using --run_folders (default: all)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples for sample mode (default: 10)")
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to a separate test set JSON file (optional)')
    parser.add_argument('--dataset_models', type=str, default=None,
                        help='Name of the dataset used for training (e.g. gemma-3-27b-it) to locate models in results/')
    parser.add_argument('--run_folders', nargs='*', default=None,
                        help='Run folders in format: model/folder (e.g. bert/two_phase_e10+5_b32_v1 bert/e10_b32_v1)')
    # Legacy arguments (kept for backward compatibility)
    parser.add_argument('--bert_run', type=str, default=None,
                        help='[Legacy] Run folder for BERT model. Use --run_folders instead.')
    parser.add_argument('--mdeberta_run', type=str, default=None,
                        help='[Legacy] Run folder for mDeBERTa model. Use --run_folders instead.')
    parser.add_argument('--umberto_run', type=str, default=None,
                        help='[Legacy] Run folder for UmBERTo model. Use --run_folders instead.')
    parser.add_argument('--llm_api', type=str, default=None,
                        help='Path to LLM API results JSON (from evaluate_llm_api.py)')
    parser.add_argument('--llm_local', type=str, default=None,
                        help='Path to LLM Local results JSON (from evaluate_llm_local.py)')
    parser.add_argument('--llm_results', type=str, default=None,
                        help='[Deprecated] Use --llm_api or --llm_local instead')
    
    args = parser.parse_args()
    
    # Build model_entries list: [(model_name, run_folder), ...]
    model_entries = []
    available_models = get_available_models()
    
    # New format: --run_folders bert/folder1 bert/folder2 mdeberta/folder1
    if args.run_folders:
        for item in args.run_folders:
            if '/' in item:
                model, folder = item.split('/', 1)
                model = model.strip()
                folder = folder.strip()
                if model not in available_models:
                    print(f"Error: Unknown model '{model}'. Available: {available_models}")
                    return
                model_entries.append((model, folder))
            elif '=' in item:
                # Support old format model=folder for backward compatibility
                model, folder = item.split('=', 1)
                model = model.strip()
                folder = folder.strip()
                if model not in available_models:
                    print(f"Error: Unknown model '{model}'. Available: {available_models}")
                    return
                model_entries.append((model, folder))
            else:
                print(f"Warning: Invalid run_folder format '{item}'. Expected 'model/folder'.")
    
    # Legacy support: --bert_run, --mdeberta_run, --umberto_run
    if args.bert_run and not any(m == 'bert' for m, _ in model_entries):
        model_entries.append(('bert', args.bert_run))
    if args.mdeberta_run and not any(m == 'mdeberta' for m, _ in model_entries):
        model_entries.append(('mdeberta', args.mdeberta_run))
    if args.umberto_run and not any(m == 'umberto' for m, _ in model_entries):
        model_entries.append(('umberto', args.umberto_run))
    
    # If no run_folders specified, use --models or all models with None run_folder
    if not model_entries:
        if args.models:
            models_list = [m.strip() for m in args.models.split(",")]
            for m in models_list:
                if m not in available_models:
                    print(f"Error: Unknown model '{m}'. Available: {available_models}")
                    return
                model_entries.append((m, None))
        else:
            model_entries = [(m, None) for m in available_models]
    
    # Also build run_folders dict for backward compat with quick/sample modes
    run_folders = {m: f for m, f in model_entries if f}
    models_to_compare = list(dict.fromkeys([m for m, _ in model_entries]))  # Unique models
    
    # Validate dataset arguments
    if not args.dataset and not args.test_file and args.mode in ['evaluate', 'full', 'sample']:
        if os.path.exists("dataset.json"):
            args.dataset = "dataset.json"
        elif os.path.exists("../datasets/test_set.json"):
            args.test_file = "../datasets/test_set.json"
    
    # Print summary
    print(f"Model entries: {len(model_entries)}")
    for model, folder in model_entries:
        folder_str = folder if folder else "latest"
        print(f"  - {model}: {folder_str}")
    if args.dataset_models:
        print(f"Dataset: {args.dataset_models}")
    
    if args.mode == "quick":
        run_quick_comparison(models_to_compare, args.dataset_models, run_folders)
    elif args.mode == "sample":
        run_sample_comparison(models_to_compare, args.samples, args.dataset, args.test_file, args.dataset_models, run_folders)
    elif args.mode == "evaluate" or args.mode == "full":
        llm_api = args.llm_api or args.llm_results
        run_full_evaluation(model_entries, args.dataset, args.test_file, args.dataset_models, llm_api, args.llm_local)


if __name__ == "__main__":
    main()
