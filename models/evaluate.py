"""
Model Evaluation
================
Unified evaluation script for all models (BERT, mDeBERTa, UmBERTo).
Generates confusion matrices, classification reports, and F1 scores.

Usage:
    python evaluate.py --model bert
    python evaluate.py --model mdeberta --dataset data/test.json
    python evaluate.py --model umberto --output results_umberto
"""

import os
import json
import glob
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support,
)
from tqdm import tqdm

from config import LABELS, THRESHOLD, MAX_LENGTH, get_model_config, get_available_models
from inference import load_model as load_inference_model


def load_model(model_name: str = 'bert', checkpoint: str = None):
    """Load model for evaluation.

    :param model_name: str: model name (Default value = 'bert')
    :param checkpoint: str: custom checkpoint path (Default value = None)

    """
    model, tokenizer, device, config = load_inference_model(model_name, checkpoint)
    return model, tokenizer, device, config


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
        
        # Drop the originalv labels column
        df = df.drop(columns=['labels'])
    else:
        print("Detected old format: labels already as columns")
    
    return df


def load_dataset(dataset_path: str = None, dataset_dir: str = None):
    """Load and prepare the dataset.

    :param dataset_path: str: path to the dataset file (optional if dataset_dir provided)
    :param dataset_dir: str: path to folder containing JSON files (optional if dataset_path provided)

    """
    # Load data from folder or file
    if dataset_dir:
        print(f"Loading dataset from folder: {dataset_dir}")
        articles = load_json_files_from_folder(dataset_dir)
        df = pd.DataFrame(articles)
    elif dataset_path:
        print(f"Loading dataset from {dataset_path}...")
        df = pd.read_json(dataset_path)
    else:
        raise ValueError("Either 'dataset_path' or 'dataset_dir' must be provided")
    
    # Convert labels format if needed
    df = convert_labels_format(df)
    
    # Ensure content column exists
    if 'contenuto' in df.columns and 'content' not in df.columns:
        df['content'] = df['contenuto']
        print("Converted 'contenuto' column to 'content'")
    
    if 'content' not in df.columns:
        raise ValueError("Dataset must have a 'content' column with article text")
    
    # Show class distribution
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION IN DATASET")
    print("="*50)
    
    # Check which labels exist in dataset
    available_labels = [l for l in LABELS if l in df.columns]
    
    class_counts = df[available_labels].sum().sort_values(ascending=False)
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {int(count)} ({percentage:.2f}%)")
    
    total_positive = df[available_labels].sum().sum()
    print(f"\nTotal articles: {len(df)}")
    print(f"Total positive labels: {int(total_positive)}")
    print(f"Avg labels per article: {total_positive/len(df):.2f}")
    
    return df, available_labels


def predict_batch(model, tokenizer, texts, device, batch_size=16):
    """Run predictions on a batch of texts.

    :param model: model to use for predictions
    :param tokenizer: tokenizer to use for tokenization
    :param texts: list of texts to predict
    :param device: device to use for predictions
    :param batch_size: batch size (Default value = 16)

    """
    all_probs = []
    sigmoid = torch.nn.Sigmoid()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
        
        probs = sigmoid(outputs.logits).cpu().numpy()
        all_probs.extend(probs)
    
    return np.array(all_probs)


def evaluate_model(df, model, tokenizer, device, available_labels, model_name: str):
    """Evaluate the model and compute metrics.

    :param df: dataframe containing the dataset
    :param model: model to use for predictions
    :param tokenizer: tokenizer to use for tokenization
    :param device: device to use for predictions
    :param available_labels: list of labels to evaluate
    :param model_name: str: modle name

    """
    config = get_model_config(model_name)
    
    print("\n" + "="*50)
    print(f"RUNNING MODEL EVALUATION ({config['name']})")
    print("="*50)
    
    # Get ground truth labels
    y_true = df[available_labels].values
    
    # Get predictions
    texts = df['content'].tolist()
    y_probs = predict_batch(model, tokenizer, texts, device)
    
    # Only use columns for available labels
    label_indices = [LABELS.index(l) for l in available_labels]
    y_probs_filtered = y_probs[:, label_indices]
    y_pred = (y_probs_filtered > THRESHOLD).astype(int)
    
    # Compute metrics for each label
    results = {
        'model': config['name'],
        'threshold': THRESHOLD,
        'total_samples': len(df),
        'per_class_metrics': {},
        'overall_metrics': {}
    }
    
    print("\n" + "="*50)
    print("PER-CLASS METRICS")
    print("="*50)
    print(f"{'Label':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*75)
    
    for i, label in enumerate(available_labels):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, i], 
            y_pred[:, i], 
            average='binary',
            zero_division=0
        )
        support_count = int(y_true[:, i].sum())
        
        results['per_class_metrics'][label] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': support_count
        }
        
        f1_str = f"{f1:.4f}"
        if f1 < 0.5:
            f1_str = f"! {f1:.4f}"
        
        print(f"{label:<35} {precision:>10.4f} {recall:>10.4f} {f1_str:>10} {support_count:>10}")
    
    # Overall metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0
    )
    
    results['overall_metrics'] = {
        'micro_precision': float(precision_micro),
        'micro_recall': float(recall_micro),
        'micro_f1': float(f1_micro),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro)
    }
    
    print("-"*75)
    print(f"{'MICRO AVERAGE':<35} {precision_micro:>10.4f} {recall_micro:>10.4f} {f1_micro:>10.4f}")
    print(f"{'MACRO AVERAGE':<35} {precision_macro:>10.4f} {recall_macro:>10.4f} {f1_macro:>10.4f}")
    
    return results, y_true, y_pred, y_probs_filtered, available_labels


def plot_confusion_matrices(y_true, y_pred, labels, output_dir, model_name):
    """Generate and save confusion matrices for each label.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param labels: list of labels
    :param output_dir: output directory
    :param model_name: model name

    """
    print("\nGenerating confusion matrices...")
    
    n_cols = 4
    n_rows = (len(labels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        if cm.shape == (1, 1):
            if y_true[:, i].sum() == 0:
                cm = np.array([[cm[0, 0], 0], [0, 0]])
            else:
                cm = np.array([[0, 0], [0, cm[0, 0]]])
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=axes[i],
            xticklabels=['Neg', 'Pos'],
            yticklabels=['Neg', 'Pos']
        )
        axes[i].set_title(label.replace('_', ' ').title(), fontsize=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    for i in range(len(labels), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Confusion Matrices - {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to: {output_path}")


def plot_class_distribution(df, labels, output_dir):
    """Plot and save class distribution chart.

    :param df: dataframe containing the dataset
    :param labels: list of labels
    :param output_dir: output directory

    """
    class_counts = df[labels].sum().sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors = ['#ff6b6b' if c < 50 else '#4ecdc4' for c in class_counts.values]
    bars = plt.barh(range(len(class_counts)), class_counts.values, color=colors)
    
    plt.yticks(range(len(class_counts)), [l.replace('_', ' ').title() for l in class_counts.index])
    plt.xlabel('Number of Articles')
    plt.title('Class Distribution (Red = Underrepresented, <50 samples)')
    
    for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                 str(int(count)), va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution to: {output_path}")


def plot_f1_scores(results, output_dir, model_name):
    """Plot F1 scores for each class.

    :param results: results dictionary
    :param output_dir: output directory
    :param model_name: model name

    """
    labels = list(results['per_class_metrics'].keys())
    f1_scores = [results['per_class_metrics'][l]['f1'] for l in labels]
    
    sorted_indices = np.argsort(f1_scores)
    labels = [labels[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 8))
    colors = ['#ff6b6b' if f < 0.5 else '#ffd93d' if f < 0.7 else '#4ecdc4' for f in f1_scores]
    bars = plt.barh(range(len(labels)), f1_scores, color=colors)
    
    plt.yticks(range(len(labels)), [l.replace('_', ' ').title() for l in labels])
    plt.xlabel('F1 Score')
    plt.title(f'F1 Score per Class - {model_name} (Red <0.5, Yellow 0.5-0.7, Green >0.7)')
    plt.xlim(0, 1)
    
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9)
    
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'f1_scores.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved F1 scores chart to: {output_path}")

def find_checkpoint(model_name: str, dataset_name: str = None) -> str:
    """Find the checkpoint path for a model.
    
    First tries results/{model}/{dataset}/model/, then falls back to config checkpoint.
    
    :param model_name: str: Model name (bert, mdeberta, umberto)
    :param dataset_name: str: Dataset name (e.g., gemma-3-27b-it) (Default value = None)
    :returns: str: Path to the checkpoint
    """
    # Try results directory first
    if dataset_name:
        results_path = os.path.join("results", model_name, dataset_name, "model")
        if os.path.exists(results_path):
            return results_path
    
    # Try to find any dataset in results/{model}/
    model_results_dir = os.path.join("results", model_name)
    if os.path.exists(model_results_dir):
        for dataset_dir in os.listdir(model_results_dir):
            model_path = os.path.join(model_results_dir, dataset_dir, "model")
            if os.path.exists(model_path):
                return model_path
    
    # Fall back to config checkpoint
    config = get_model_config(model_name)
    return config['checkpoint']


def evaluate_single_model(model_name: str, args, dataset_name: str = None):
    """Evaluate a single model.
    
    :param model_name: str: Model name
    :param args: argparse.Namespace: Command line arguments
    :param dataset_name: str: Dataset name for checkpoint lookup
    """
    config = get_model_config(model_name)
    output_dir = f"evaluation_results_{model_name}"
    
    
    print(f"MODEL EVALUATION - {config['name']}")
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find checkpoint
    checkpoint = args.checkpoint if args.checkpoint else find_checkpoint(model_name, dataset_name)
    print(f"Using checkpoint: {checkpoint}")
    
    # Load model and data
    model, tokenizer, device, _ = load_model(model_name, checkpoint)
    df, available_labels = load_dataset(
        dataset_path=args.dataset if not args.dataset_dir else None,
        dataset_dir=args.dataset_dir
    )
    
    # Run evaluation
    results, y_true, y_pred, y_probs, labels = evaluate_model(
        df, model, tokenizer, device, available_labels, model_name
    )
    
    # Generate plots
    plot_class_distribution(df, labels, output_dir)
    plot_confusion_matrices(y_true, y_pred, labels, output_dir, config['name'])
    plot_f1_scores(results, output_dir, config['name'])
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation results to: {results_path}")
    
    return results


def main():
    """Main function to run the evaluation script."""
    available_models = get_available_models()
    
    parser = argparse.ArgumentParser(
        description="Unified model evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=available_models + ['all'],
                        help='Model to evaluate (default: bert). Use "all" to evaluate all models.')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Custom checkpoint path (e.g., results/bert/gemma-3-27b-it/model)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to a single dataset JSON file')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to folder containing JSON files (alternative to --dataset)')
    parser.add_argument('--dataset_models', type=str, default=None,
                        help='Name of the dataset used for training (e.g., gemma-3-27b-it) to locate models in results/')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory (default: evaluation_results_<model>)')
    
    args = parser.parse_args()
    
    # Validate dataset arguments
    if not args.dataset and not args.dataset_dir:
        parser.error("You must specify either --dataset or --dataset_dir")
    
    # Determine dataset name from dataset_dir if not specified
    dataset_name = args.dataset_models
    if not dataset_name and args.dataset_dir:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    
    if args.model == 'all':
        # Evaluate all models
        
        print("EVALUATING ALL MODELS")
        
        print(f"Models: {available_models}")
        if dataset_name:
            print(f"Dataset: {dataset_name}")
        print("="*60 + "\n")
        
        all_results = {}
        for model_name in available_models:
            try:
                results = evaluate_single_model(model_name, args, dataset_name)
                all_results[model_name] = results
                print("\n")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                print()
        
        # Print comparison summary
        
        print("EVALUATION SUMMARY")
        
        for model_name, results in all_results.items():
            if results:
                print(f"\n{model_name}:")
                print(f"  F1 Macro: {results['overall_metrics']['macro_f1']:.4f}")
                print(f"  F1 Micro: {results['overall_metrics']['micro_f1']:.4f}")
    else:
        # Evaluate single model
        config = get_model_config(args.model)
        output_dir = args.output or f"evaluation_results_{args.model}"
        
        
        print(f"MODEL EVALUATION - {config['name']}")
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find checkpoint
        checkpoint = args.checkpoint if args.checkpoint else find_checkpoint(args.model, dataset_name)
        print(f"Using checkpoint: {checkpoint}")
        
        # Load model and data
        model, tokenizer, device, _ = load_model(args.model, checkpoint)
        df, available_labels = load_dataset(
            dataset_path=args.dataset if not args.dataset_dir else None,
            dataset_dir=args.dataset_dir
        )
        
        # Run evaluation
        results, y_true, y_pred, y_probs, labels = evaluate_model(
            df, model, tokenizer, device, available_labels, args.model
        )
        
        # Generate plots
        plot_class_distribution(df, labels, output_dir)
        plot_confusion_matrices(y_true, y_pred, labels, output_dir, config['name'])
        plot_f1_scores(results, output_dir, config['name'])
        
        # Save results to JSON
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation results to: {results_path}")
        
        # Summary
        
        print("EVALUATION COMPLETE!")
        
        print(f"F1 Macro: {results['overall_metrics']['macro_f1']:.4f}")
        print(f"F1 Micro: {results['overall_metrics']['micro_f1']:.4f}")


if __name__ == "__main__":
    main()

