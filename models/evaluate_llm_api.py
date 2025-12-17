"""
Evaluate LLM via API for Crime Classification
==============================================
Evaluates Gemma 3 models via Google AI API for multi-label crime classification.
Produces metrics comparable with locally fine-tuned models.

Usage:
    python evaluate_llm_api.py --test_file ../datasets/test_set.json --model gemma-3-27b-it
    python evaluate_llm_api.py --test_file ../datasets/test_set.json --model gemma-3-12b-it --few_shot
    
Environment:
    Set GEMINI_API_KEY in .env file or as environment variable
"""

import os
import sys
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
    roc_auc_score, accuracy_score, classification_report
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

# CONFIGURATION
LABELS = [
    'omicidio', 'omicidio_colposo', 'omicidio_stradale', 'tentato_omicidio',
    'furto', 'rapina', 'violenza_sessuale', 'aggressione', 'spaccio',
    'truffa', 'estorsione', 'contrabbando', 'associazione_di_tipo_mafioso'
]

# Models available via API (Gemma 3 series)
AVAILABLE_MODELS = [
    'gemma-3-27b-it',
    'gemma-3-12b-it',
    'gemma-3-4b-it',
    'gemma-3-2b-it',
    'gemma-3-1b-it',
]

# Few-shot examples for better classification
FEW_SHOT_EXAMPLES = [
    {
        "text": "Un uomo di 45 anni è stato arrestato dai carabinieri dopo aver rubato un'auto parcheggiata in via Roma. Il veicolo è stato recuperato poche ore dopo.",
        "labels": ["furto"]
    },
    {
        "text": "Blitz antidroga nel quartiere San Paolo: arrestato spacciatore con 50 grammi di cocaina e 200 grammi di marijuana. I clienti venivano contattati via Telegram.",
        "labels": ["spaccio"]
    },
    {
        "text": "Tragedia sulla strada statale 16: un camion ha travolto un ciclista di 28 anni. L'uomo è morto sul colpo. Il conducente si è fermato a prestare soccorso.",
        "labels": ["omicidio_stradale"]
    },
    {
        "text": "Inaugurato il nuovo parco giochi nel quartiere Libertà. Il sindaco ha tagliato il nastro alla presenza di numerosi bambini e famiglie.",
        "labels": []
    },
    {
        "text": "Rapina a mano armata in banca: i malviventi hanno minacciato i dipendenti con pistole e sono fuggiti con 50.000 euro. Durante la fuga hanno aggredito una guardia giurata.",
        "labels": ["rapina", "aggressione"]
    }
]


# API SETUP
def setup_api(api_key: str = None):
    """Configure the Gemini API."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "API key not found. Set GEMINI_API_KEY in .env file or pass --api_key"
        )
    genai.configure(api_key=key)
    print(f"API configured successfully")


def create_model(model_name: str):
    """Create a GenerativeModel instance with relaxed safety settings for crime content."""
    # Safety settings to allow crime-related content (news articles about crimes)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    return genai.GenerativeModel(model_name, safety_settings=safety_settings)


# PROMPTS
def create_zero_shot_prompt(article_text: str) -> str:
    """Create a zero-shot classification prompt."""
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
    """Create a few-shot classification prompt with examples."""
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
    """Parse the model response to extract labels."""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            labels = data.get('labels', [])
            # Validate labels
            return [l for l in labels if l in LABELS]
    except json.JSONDecodeError:
        pass
    
    # Fallback: look for label names directly in response
    found_labels = []
    response_lower = response_text.lower()
    for label in LABELS:
        if label in response_lower:
            found_labels.append(label)
    
    return found_labels


def classify_article(model, article_text: str, few_shot: bool = False, 
                     max_retries: int = 3, delay: float = 1.0) -> Tuple[List[str], str]:
    """Classify a single article using the API.
    
    :returns: Tuple of (predicted_labels, raw_response)
    """
    prompt = create_few_shot_prompt(article_text) if few_shot else create_zero_shot_prompt(article_text)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            labels = parse_response(response_text)
            return labels, response_text
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                # Rate limit - wait and retry
                wait_time = delay * (2 ** attempt)
                print(f"\nRate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nError: {e}")
                if attempt == max_retries - 1:
                    return [], str(e)
                time.sleep(delay)
    
    return [], "Max retries exceeded"


def labels_to_binary(labels: List[str]) -> np.ndarray:
    """Convert label list to binary vector."""
    binary = np.zeros(len(LABELS))
    for label in labels:
        if label in LABELS:
            binary[LABELS.index(label)] = 1
    return binary


# EVALUATION
def load_test_set(test_file: str) -> pd.DataFrame:
    """Load and prepare the test set."""
    print(f"Loading test set from: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Handle content column
    if 'contenuto' in df.columns and 'content' not in df.columns:
        df['content'] = df['contenuto']
    
    # Convert labels
    if 'labels' in df.columns:
        # Already in array format
        pass
    else:
        # Convert from one-hot columns
        df['labels'] = df.apply(
            lambda row: [l for l in LABELS if l in row and row[l] == 1],
            axis=1
        )
    
    print(f"Loaded {len(df)} articles")
    return df


def evaluate_llm(model, test_df: pd.DataFrame, few_shot: bool = False,
                 rate_limit_delay: float = 0.5, save_predictions: bool = True) -> Dict:
    """Run full evaluation on test set.
    
    :param model: GenerativeModel instance
    :param test_df: Test DataFrame with 'content' and 'labels' columns
    :param few_shot: Whether to use few-shot prompting
    :param rate_limit_delay: Delay between API calls (seconds)
    :param save_predictions: Whether to save individual predictions
    :returns: Dictionary with metrics and results
    """
    print(f"\nEvaluating {'few-shot' if few_shot else 'zero-shot'} classification...")
    print(f"Test size: {len(test_df)} articles")
    print(f"Rate limit delay: {rate_limit_delay}s per request")
    
    all_true = []
    all_pred = []
    all_pred_labels = []
    all_true_labels = []
    raw_responses = []
    errors = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying"):
        text = row['content']
        true_labels = row['labels'] if isinstance(row['labels'], list) else []
        
        # Classify
        pred_labels, raw_response = classify_article(model, text, few_shot)
        
        # Convert to binary
        true_binary = labels_to_binary(true_labels)
        pred_binary = labels_to_binary(pred_labels)
        
        all_true.append(true_binary)
        all_pred.append(pred_binary)
        all_true_labels.append(true_labels)
        all_pred_labels.append(pred_labels)
        raw_responses.append(raw_response)
        
        if not pred_labels and true_labels:
            errors.append({
                'idx': idx,
                'text': text[:200],
                'true': true_labels,
                'response': raw_response
            })
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    # Convert to arrays
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    
    # Calculate metrics
    metrics = {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = {label: float(f1) for label, f1 in zip(LABELS, per_class_f1)}
    
    # Try ROC AUC
    try:
        metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred, average='micro')
    except:
        metrics['roc_auc_micro'] = None
    
    results = {
        'metrics': metrics,
        'binary_predictions': y_pred.tolist(),  # For agreement analysis
        'predictions': [
            {
                'true_labels': all_true_labels[i],
                'pred_labels': all_pred_labels[i],
            }
            for i in range(len(test_df))
        ] if save_predictions else None,
        'errors': errors[:20]
    }
    
    return results


def print_results(results: Dict, model_name: str, few_shot: bool):
    """Print evaluation results."""
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {model_name} ({'few-shot' if few_shot else 'zero-shot'})")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  F1 Micro:      {metrics['f1_micro']:.4f}")
    print(f"  F1 Macro:      {metrics['f1_macro']:.4f}")
    print(f"  Precision:     {metrics['precision_micro']:.4f}")
    print(f"  Recall:        {metrics['recall_micro']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    if metrics['roc_auc_micro']:
        print(f"  ROC AUC:       {metrics['roc_auc_micro']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    print("-" * 40)
    for label, f1 in metrics['per_class_f1'].items():
        print(f"  {label:35} {f1:.4f}")
    
    if results.get('errors'):
        print(f"\nFirst few classification errors ({len(results['errors'])} shown):")
        for err in results['errors'][:3]:
            print(f"  True: {err['true']}")
            print(f"  Response: {err['response'][:100]}...")
            print()


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM via API for crime classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--test_file', '-t', type=str, required=True,
                        help='Path to test set JSON file')
    parser.add_argument('--model', '-m', type=str, default='gemma-3-27b-it',
                        choices=AVAILABLE_MODELS,
                        help='Model to use (default: gemma-3-27b-it)')
    parser.add_argument('--few_shot', '-f', action='store_true',
                        help='Use few-shot prompting (default: zero-shot)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                        help='Delay between API calls in seconds (default: 0.5)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of articles to evaluate (for testing)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for results (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Setup API
    setup_api(args.api_key)
    
    # Load test set
    test_df = load_test_set(args.test_file)
    
    # Limit if specified
    if args.limit:
        test_df = test_df.head(args.limit)
        print(f"Limited to {args.limit} articles for testing")
    
    # Create model
    model = create_model(args.model)
    print(f"Using model: {args.model}")
    
    # Evaluate
    results = evaluate_llm(model, test_df, args.few_shot, args.delay)
    
    # Print results
    print_results(results, args.model, args.few_shot)
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "fewshot" if args.few_shot else "zeroshot"
        args.output = f"evaluation_results/llm_{args.model}_{mode}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'mode': 'few-shot' if args.few_shot else 'zero-shot',
        'test_size': len(test_df),
        'metrics': results['metrics'],
        'binary_predictions': results['binary_predictions'],  # For agreement analysis
        'errors_sample': results['errors']
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")
    
    # Compare with fine-tuned models if available
    print("\n" + "=" * 60)
    print("COMPARISON WITH FINE-TUNED MODELS")
    print("=" * 60)
    print(f"\nGemini {args.model} F1 Macro: {results['metrics']['f1_macro']:.4f}")
    print("\nTo compare, run:")
    print(f"  python compare_models.py --mode full --test_file {args.test_file} --dataset_models gemma-3-27b-it")


if __name__ == "__main__":
    main()
