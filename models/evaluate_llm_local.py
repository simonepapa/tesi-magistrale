"""
Evaluate LLM via Ollama (Local)
================================
Evaluates local LLM models via Ollama for multi-label crime classification.
Mainly useful to avoid API content filters and rate limits.

Usage:
    python evaluate_llm_local.py --test_file ../datasets/test_set.json --model gemma3:12b
    python evaluate_llm_local.py --test_file ../datasets/test_set.json --model gemma3:12b --few_shot

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull gemma3:12b
    3. pip install ollama
"""

import os
import json
import time
import argparse
import re
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score
)

try:
    import ollama
except ImportError:
    print("Installing ollama...")
    os.system("pip install ollama")
    import ollama

# CONFIGURATION
LABELS = [
    'omicidio', 'omicidio_colposo', 'omicidio_stradale', 'tentato_omicidio',
    'furto', 'rapina', 'violenza_sessuale', 'aggressione', 'spaccio',
    'truffa', 'estorsione', 'contrabbando', 'associazione_di_tipo_mafioso'
]

# Recommended models with approximate VRAM requirements
RECOMMENDED_MODELS = {
    'gemma3:12b': '~10GB',
    'gemma3:4b': '~6GB',
    'llama3.2:3b': '~8GB',
    'mistral:7b': '~6GB',
    'qwen3:14b': '~12GB',
}

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {"text": "Un uomo di 45 anni è stato arrestato dai carabinieri dopo aver rubato un'auto parcheggiata in via Roma.", "labels": ["furto"]},
    {"text": "Blitz antidroga nel quartiere San Paolo: arrestato spacciatore con 50 grammi di cocaina.", "labels": ["spaccio"]},
    {"text": "Tragedia sulla statale 16: un camion ha travolto un ciclista di 28 anni. L'uomo è morto sul colpo.", "labels": ["omicidio_stradale"]},
    {"text": "Inaugurato il nuovo parco giochi nel quartiere Libertà. Il sindaco ha tagliato il nastro.", "labels": []},
    {"text": "Rapina in banca: i malviventi hanno aggredito una guardia giurata durante la fuga.", "labels": ["rapina", "aggressione"]},
]


# PROMPTS
def create_zero_shot_prompt(article_text: str) -> str:
    labels_list = "\n".join([f"- {label}" for label in LABELS])
    return f"""Sei un classificatore di articoli di cronaca italiana.

CATEGORIE VALIDE (usa ESATTAMENTE questi nomi, nessun altro):
{labels_list}

REGOLE TASSATIVE:
1. Rispondi SOLO con JSON: {{"labels": ["categoria1"]}}
2. Usa ESCLUSIVAMENTE le categorie elencate sopra - NON inventare nuove categorie
3. Se il crimine non rientra in nessuna categoria, usa la più simile
4. Se l'articolo NON parla di crimini: {{"labels": []}}
5. Multi-label solo se chiaramente presenti più crimini diversi

ARTICOLO:
{article_text}

JSON:"""


def create_few_shot_prompt(article_text: str) -> str:
    labels_list = "\n".join([f"- {label}" for label in LABELS])
    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += f"\nArticolo: {ex['text']}\nJSON: {{\"labels\": {json.dumps(ex['labels'])}}}\n"
    
    return f"""Classificatore di articoli di cronaca italiana.

CATEGORIE VALIDE (usa SOLO questi nomi esatti):
{labels_list}

ATTENZIONE: NON inventare categorie! Usa SOLO quelle elencate sopra.

ESEMPI:
{examples_text}

ARTICOLO DA CLASSIFICARE:
{article_text}

JSON:"""


# CLASSIFICATION
def parse_response(response_text: str) -> List[str]:
    try:
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            labels = data.get('labels', [])
            return [l for l in labels if l in LABELS]
    except json.JSONDecodeError:
        pass
    
    # Fallback
    found_labels = []
    response_lower = response_text.lower()
    for label in LABELS:
        if label in response_lower:
            found_labels.append(label)
    return found_labels


def classify_article(model_name: str, article_text: str, few_shot: bool = False) -> Tuple[List[str], str]:
    prompt = create_few_shot_prompt(article_text) if few_shot else create_zero_shot_prompt(article_text)
    
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        response_text = response['message']['content'].strip()
        labels = parse_response(response_text)
        return labels, response_text
    except Exception as e:
        return [], str(e)


def labels_to_binary(labels: List[str]) -> np.ndarray:
    binary = np.zeros(len(LABELS))
    for label in labels:
        if label in LABELS:
            binary[LABELS.index(label)] = 1
    return binary


# EVALUATION
def load_test_set(test_file: str) -> pd.DataFrame:
    print(f"Loading test set from: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    if 'contenuto' in df.columns and 'content' not in df.columns:
        df['content'] = df['contenuto']
    
    if 'labels' not in df.columns:
        df['labels'] = df.apply(
            lambda row: [l for l in LABELS if l in row and row[l] == 1],
            axis=1
        )
    
    print(f"Loaded {len(df)} articles")
    return df


def evaluate_llm(model_name: str, test_df: pd.DataFrame, few_shot: bool = False,
                 max_chars: int = None) -> Dict:
    print(f"\nEvaluating {'few-shot' if few_shot else 'zero-shot'} classification...")
    print(f"Model: {model_name}")
    print(f"Test size: {len(test_df)} articles")
    
    all_true = []
    all_pred = []
    all_pred_labels = []
    all_true_labels = []
    errors = []
    
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
        text = row['content']
        
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        true_labels = row['labels'] if isinstance(row['labels'], list) else []
        pred_labels, raw_response = classify_article(model_name, text, few_shot)
        
        true_binary = labels_to_binary(true_labels)
        pred_binary = labels_to_binary(pred_labels)
        
        all_true.append(true_binary)
        all_pred.append(pred_binary)
        all_true_labels.append(true_labels)
        all_pred_labels.append(pred_labels)
        
        if not pred_labels and true_labels:
            errors.append({
                'idx': idx,
                'text': text[:200],
                'true': true_labels,
                'response': raw_response[:200]
            })
    
    elapsed = time.time() - start_time
    
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = {label: float(f1) for label, f1 in zip(LABELS, per_class_f1)}
    
    try:
        metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred, average='micro')
    except:
        metrics['roc_auc_micro'] = None
    
    return {
        'metrics': metrics,
        'binary_predictions': y_pred.tolist(),
        'errors': errors[:20],
        'elapsed_seconds': elapsed,
        'articles_per_minute': len(test_df) / (elapsed / 60) if elapsed > 0 else 0
    }


def print_results(results: Dict, model_name: str, few_shot: bool):
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {model_name} ({'few-shot' if few_shot else 'zero-shot'})")
    print("=" * 60)
    
    print(f"\nPerformance:")
    print(f"  Time: {results['elapsed_seconds']:.1f}s")
    print(f"  Speed: {results['articles_per_minute']:.1f} articles/min")
    
    print(f"\nMetrics:")
    print(f"  F1 Micro:      {metrics['f1_micro']:.4f}")
    print(f"  F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"  Precision:     {metrics['precision_micro']:.4f}")
    print(f"  Recall:        {metrics['recall_micro']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    print("-" * 40)
    for label, f1 in metrics['per_class_f1'].items():
        print(f"  {label:35} {f1:.4f}")


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM via Ollama (local) for crime classification"
    )
    parser.add_argument('--test_file', '-t', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, default='gemma3:12b',
                        help='Ollama model (default: gemma3:12b)')
    parser.add_argument('--few_shot', '-f', action='store_true')
    parser.add_argument('--limit', '-l', type=int, default=None)
    parser.add_argument('--max_chars', type=int, default=None)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--list_models', action='store_true',
                        help='List recommended models for your GPU')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nRecommended models (with VRAM requirements):")
        print("-" * 40)
        for model, vram in RECOMMENDED_MODELS.items():
            print(f"  {model:20} {vram}")
        print("\nTo pull a model: ollama pull gemma3:12b")
        return
    
    # Check if Ollama is running
    try:
        ollama.list()
    except Exception as e:
        print("Error: Ollama not running. Start it with: ollama serve")
        print(f"Details: {e}")
        return
    
    # Load test set
    test_df = load_test_set(args.test_file)
    
    if args.limit:
        test_df = test_df.head(args.limit)
        print(f"Limited to {args.limit} articles")
    
    # Evaluate
    print(f"\nUsing model: {args.model}")
    if args.max_chars:
        print(f"Truncating articles to {args.max_chars} chars")
    
    results = evaluate_llm(args.model, test_df, args.few_shot, args.max_chars)
    print_results(results, args.model, args.few_shot)
    
    # Save
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "fewshot" if args.few_shot else "zeroshot"
        model_safe = args.model.replace(':', '-')
        args.output = f"evaluation_results/llm_local_{model_safe}_{mode}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'provider': 'ollama-local',
        'mode': 'few-shot' if args.few_shot else 'zero-shot',
        'test_size': len(test_df),
        'metrics': results['metrics'],
        'binary_predictions': results['binary_predictions'],
        'performance': {
            'elapsed_seconds': results['elapsed_seconds'],
            'articles_per_minute': results['articles_per_minute']
        },
        'errors_sample': results['errors']
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\nTo compare with fine-tuned models, run:")
    print(f"  python compare_models.py --mode full --test_file {args.test_file} --dataset_models gemma-3-27b-it --llm_results {args.output}")


if __name__ == "__main__":
    main()
