"""
Inference with Chunking for long news
================================
Single inference module that works with all models (BERT, mDeBERTa, UmBERTo).

Usage:
    python inference.py --model bert --input data/articles.json --output data/labeled.json
    python inference.py --model mdeberta --input data/test.json
    python inference.py --model umberto --test  # Run test prediction
"""

import torch
import json
import argparse
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from config import (
    LABELS, MAX_LENGTH, CHUNK_OVERLAP, THRESHOLD,
    get_model_config, get_available_models
)
from extract_streets import extract_streets


def load_model(model_name: str = 'bert', checkpoint: str = None):
    """Load a fine-tuned model and tokenizer.

    :param model_name: str: One of 'bert', 'mdeberta', 'umberto' (Default value = 'bert')
    :param checkpoint: str: Optional custom checkpoint path (overrides config) (Default value = None)
    :returns: Tuple of (model, tokenizer, device, config)

    """
    config = get_model_config(model_name)
    checkpoint = checkpoint or config['checkpoint']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading {config['name']} model from: {checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, 
        use_fast=config['use_fast_tokenizer']
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        problem_type="multi_label_classification",
        num_labels=len(LABELS)
    )
    model.to(device)
    model.eval()
    
    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    return model, tokenizer, device, config


def split_into_chunks(text: str, tokenizer, max_length: int = MAX_LENGTH, 
                      overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a long text into overlapping chunks that fit within the model's max length.

    :param text: str: The input text to split
    :param tokenizer: The tokenizer
    :param max_length: int: Maximum tokens per chunk (including special tokens) (Default value = MAX_LENGTH)
    :param overlap: int: Number of tokens to overlap between chunks (Default value = CHUNK_OVERLAP)
    :returns: List of text chunks

    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    effective_max_length = max_length - 2  # Account for special tokens [CLS] and [SEP]
    if len(tokens) <= effective_max_length:
        return [text]
    
    chunks = []
    start = 0
    step = effective_max_length - overlap
    
    while start < len(tokens):
        end = min(start + effective_max_length, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        start += step
    
    return chunks


def predict_with_chunking(model, tokenizer, text: str, device, 
                          aggregation: str = 'max') -> Tuple[Dict, int]:
    """Run prediction on a text, using chunking for long texts.

    :param model: The model
    :param tokenizer: The tokenizer
    :param text: str: Input text
    :param device: torch device
    :param aggregation: str: How to aggregate chunk predictions ('max', 'mean', 'any') (Default value = 'max')
    :returns: Tuple of (predictions dict, number of chunks used)

    """
    import numpy as np
    sigmoid = torch.nn.Sigmoid()
    
    chunks = split_into_chunks(text, tokenizer)
    num_chunks = len(chunks)
    
    all_probs = []
    
    for chunk in chunks:
        encoding = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
        
        probs = sigmoid(outputs.logits.squeeze()).cpu().numpy()
        all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    
    if aggregation == 'max':
        final_probs = np.max(all_probs, axis=0)
    elif aggregation == 'mean':
        final_probs = np.mean(all_probs, axis=0)
    elif aggregation == 'any':
        final_probs = np.max(all_probs, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    predictions = {}
    for idx, label in enumerate(LABELS):
        prob = float(final_probs[idx])
        predictions[label] = {
            "value": int(prob > THRESHOLD),
            "prob": round(prob, 2)
        }
    
    return predictions, num_chunks


def label_article(article: Dict, model, tokenizer, device, 
                  aggregation: str = 'max') -> Dict:
    """Label a single article with crime categories using chunking.
    Also extracts street names.

    :param model: The model
    :param tokenizer: The tokenizer
    :param text: str: Input text
    :param device: torch device
    :param aggregation: str: How to aggregate chunk predictions ('max', 'mean', 'any') (Default value = 'max')
    :returns: Dict: Article with crime categories and street names

    """
    content = article.get("content", "")
    
    if not content:
        for label in LABELS:
            article[label] = {"value": 0, "prob": 0.0}
        article["_chunks_used"] = 0
        article["streets"] = []
        return article
    
    predictions, num_chunks = predict_with_chunking(
        model, tokenizer, content, device, aggregation
    )
    article.update(predictions)
    article["_chunks_used"] = num_chunks
    article["streets"] = extract_streets(content)
    
    return article


def label_articles_from_file(input_file: str, output_file: str,
                             model_name: str = 'bert',
                             quartiere: str = None,
                             aggregation: str = 'max',
                             verbose: bool = True,
                             checkpoint: str = None):
    """Label all articles in a JSON file using the specified model.

    :param input_file: str: Path to input JSON file
    :param output_file: str: Path to output JSON file
    :param model_name: str: One of 'bert', 'mdeberta', 'umberto' (Default value = 'bert')
    :param quartiere: str: Optional quartiere name to add to articles (Default value = None)
    :param aggregation: str: Chunk aggregation method (Default value = 'max')
    :param verbose: bool: Print progress (Default value = True)
    :param checkpoint: str: Custom checkpoint path (Default value = None)

    """
    model, tokenizer, device, config = load_model(model_name, checkpoint)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    if verbose:
        print(f"Loaded {len(articles)} articles from {input_file}")
    
    total_chunks = 0
    multi_chunk_articles = 0
    
    desc = f"Labeling ({config['name']})"
    for article in tqdm(articles, desc=desc, disable=not verbose):
        if quartiere:
            article["python_id"] = quartiere
        
        label_article(article, model, tokenizer, device, aggregation)
        
        chunks_used = article.get("_chunks_used", 1)
        total_chunks += chunks_used
        if chunks_used > 1:
            multi_chunk_articles += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    
    if verbose:
        print(f"\nResults saved to: {output_file}")
        print(f"Statistics:")
        print(f"  - Total articles: {len(articles)}")
        print(f"  - Articles requiring chunking: {multi_chunk_articles} ({100*multi_chunk_articles/len(articles):.1f}%)")
        print(f"  - Average chunks per article: {total_chunks/len(articles):.2f}")


def run_test(model_name: str = 'bert', checkpoint: str = None):
    """Run a test prediction with sample text.

    :param model_name: str: Model name (Default value = 'bert')
    :param checkpoint: str: Custom checkpoint path (Default value = None)

    """
    
    print(f"Testing {model_name.upper()} Crime Classification with Chunking")
    
    
    model, tokenizer, device, config = load_model(model_name, checkpoint)
    
    test_text = """
    Un grave episodio di cronaca si è verificato ieri sera nel quartiere Libertà di Bari.
    Un uomo di 35 anni è stato arrestato dalla polizia con l'accusa di rapina aggravata
    e lesioni personali. Secondo le ricostruzioni, l'uomo avrebbe aggredito un passante
    nei pressi di via Manzoni, strappandogli il portafoglio e il telefono cellulare.
    La vittima, un anziano di 72 anni, è stata trasportata al Policlinico di Bari dove
    è stata medicata per le contusioni riportate. Durante la perquisizione, sono stati 
    trovati anche 50 grammi di cocaina, facendo scattare l'accusa di spaccio.
    """
    
    print("\nTest prediction with chunking:")
    print("-"*60)
    
    predictions, num_chunks = predict_with_chunking(model, tokenizer, test_text, device)
    
    print(f"Chunks used: {num_chunks}")
    print("\nPredictions:")
    for label, pred in predictions.items():
        if pred["value"] == 1 or pred["prob"] > 0.3:
            status = "✓" if pred["value"] == 1 else "○"
            print(f"  {status} {label}: {pred['prob']:.2f}")


def main():
    """Main function to run inference with chunking for crime classification."""
    parser = argparse.ArgumentParser(
        description="Unified inference with chunking for crime classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models(),
                        help='Model to use (default: bert)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Custom checkpoint path (e.g., results/bert/gemma-3-27b-it/model)')
    parser.add_argument('--input', '-i', type=str,
                        help='Input JSON file with articles')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file (default: auto-generated)')
    parser.add_argument('--quartiere', '-q', type=str,
                        help='Quartiere name to add to articles')
    parser.add_argument('--aggregation', '-a', type=str, default='max',
                        choices=['max', 'mean', 'any'],
                        help='Chunk aggregation method (default: max)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run test prediction with sample text')
    
    args = parser.parse_args()
    
    if args.test:
        run_test(args.model, args.checkpoint)
    elif args.input:
        output = args.output or args.input.replace('.json', f'_labeled_{args.model}.json')
        label_articles_from_file(
            input_file=args.input,
            output_file=output,
            model_name=args.model,
            quartiere=args.quartiere,
            aggregation=args.aggregation,
            checkpoint=args.checkpoint
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
