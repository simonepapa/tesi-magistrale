#!/usr/bin/env python3
"""
Clustering-Based Article Deduplication
======================================
Uses DBSCAN clustering on sentence embeddings to find duplicate articles.

Advantages over pairwise comparison:
- Finds natural groupings in data
- No need to compare all pairs
- Robust to variations in wording
- Can identify outliers (unique articles)

Algorithm:
1. Create embeddings for all articles
2. Apply DBSCAN clustering
3. For each cluster with multiple sources -> cross-source duplicate
4. Merge duplicates keeping best version
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.config import normalizza_quartiere

# Import ML libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install sentence-transformers scikit-learn")
    sys.exit(1)

# Configuration
EMBEDDING_MODELS = {
    'mini': 'paraphrase-multilingual-MiniLM-L12-v2',  # Fast, 384 dim
    'large': 'intfloat/multilingual-e5-large',         # Accurate, 1024 dim
}
DEFAULT_MODEL = 'mini'
DBSCAN_EPS = 0.5  # Max distance between samples in same cluster (lower = stricter)
DBSCAN_MIN_SAMPLES = 2  # Minimum samples per cluster
DATE_TOLERANCE_DAYS = 14  # Max days apart for articles to be considered same event
from datetime import datetime

# Try to load spaCy for NER filtering
try:
    import spacy
    NLP_MODEL = None  # Lazy load
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    NLP_MODEL = None


class ClusteringDeduplicator:
    """Deduplicate articles using clustering on embeddings."""
    
    def __init__(self, model_name: str = None, use_ner: bool = False):
        """Initialize with embedding model.
        
        :param model_name: str: Sentence transformer model name or key ('mini', 'large')
        :param use_ner: bool: Whether to use NER filtering for cluster validation
        """
        # Resolve model name
        if model_name is None:
            model_name = EMBEDDING_MODELS[DEFAULT_MODEL]
        elif model_name in EMBEDDING_MODELS:
            model_name = EMBEDDING_MODELS[model_name]
        
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print("Model loaded")
        
        # Initialize NER if requested
        self.use_ner = use_ner and SPACY_AVAILABLE
        self.nlp = None
        if use_ner:
            if SPACY_AVAILABLE:
                print("Loading spaCy model for NER...")
                global NLP_MODEL
                if NLP_MODEL is None:
                    try:
                        NLP_MODEL = spacy.load('it_core_news_lg')
                    except OSError:
                        print("  spaCy model not found, trying to download...")
                        import subprocess
                        subprocess.run(['python', '-m', 'spacy', 'download', 'it_core_news_lg'])
                        NLP_MODEL = spacy.load('it_core_news_lg')
                self.nlp = NLP_MODEL
                print("NER model loaded")
            else:
                print("Warning: spaCy not installed, NER filtering disabled")
                print("Install with: pip install spacy && python -m spacy download it_core_news_lg")
    
    def prepare_text(self, article: dict) -> str:
        """Prepare text from article for embedding.
        
        :param article: dict: Article dictionary
        :returns: str: Text for embedding
        """
        title = article.get('title', '')
        content = article.get('content', '') or ''
        
        # Use title + first 300 chars of content
        combined = f"{title}. {content[:300]}"
        return combined.strip()
    
    def create_embeddings(self, articles: List[dict]) -> np.ndarray:
        """Create embeddings for all articles.
        
        :param articles: List[dict]: Articles to embed
        :returns: np.ndarray: Normalized embeddings
        """
        print(f"\nCreating embeddings for {len(articles)} articles...")
        texts = [self.prepare_text(a) for a in articles]
        
        # Create embeddings in batches
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True  # L2 normalize for cosine distance
        )
        
        print(f"Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        return embeddings
    
    def cluster_articles(self, embeddings: np.ndarray, 
                        eps: float = DBSCAN_EPS,
                        min_samples: int = DBSCAN_MIN_SAMPLES) -> np.ndarray:
        """Cluster articles using DBSCAN.
        
        :param embeddings: np.ndarray: Normalized embeddings
        :param eps: float: DBSCAN eps parameter (max distance)
        :param min_samples: int: Min samples per cluster
        :returns: np.ndarray: Cluster labels (-1 = noise/unique)
        """
        print(f"\nClustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        # DBSCAN with cosine distance (embeddings are normalized)
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean',  # On normalized vectors = cosine
            n_jobs=-1
        )
        
        labels = clustering.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Found {n_clusters} clusters")
        print(f"{n_noise} unique articles (noise)")
        print(f"{len(labels) - n_noise} articles in clusters")
        
        return labels
    
    def _parse_date(self, date_str: str):
        """Parse date string."""
        if not date_str:
            return None
        formats = ['%Y-%m-%d', '%d/%m/%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except (ValueError, TypeError):
                try:
                    return datetime.strptime(date_str[:len(fmt)], fmt)
                except:
                    continue
        return None
    
    def _dates_are_close(self, articles: List[dict], indices: List[int], tolerance_days: int) -> bool:
        """Check if all article dates in cluster are within tolerance."""
        dates = []
        for idx in indices:
            d = self._parse_date(articles[idx].get('date', ''))
            if d:
                dates.append(d)
        
        if len(dates) < 2:
            return True  # Not enough dates to compare, assume OK
        
        min_date = min(dates)
        max_date = max(dates)
        delta = (max_date - min_date).days
        
        return delta <= tolerance_days
    
    def _extract_entities(self, text: str) -> set:
        """Extract named entities from text using spaCy.
        
        :param text: str: Text to extract entities from
        :returns: set: Set of (entity_text, entity_type) tuples
        """
        if not self.nlp or not text:
            return set()
        
        # Limit text length for performance
        text = text[:2000]
        doc = self.nlp(text)
        
        # Extract relevant entity types: PER (person), LOC (location), ORG (organization)
        relevant_types = {'PER', 'LOC', 'GPE', 'ORG'}
        entities = set()
        for ent in doc.ents:
            if ent.label_ in relevant_types:
                # Normalize entity text (lowercase, strip)
                normalized = ent.text.lower().strip()
                if len(normalized) > 2:  # Skip very short entities
                    entities.add((normalized, ent.label_))
        
        return entities
    
    def _entities_overlap(self, articles: List[dict], indices: List[int], min_overlap: int = 1) -> bool:
        """Check if articles in cluster share at least min_overlap entities.
        
        :param articles: List[dict]: All articles
        :param indices: List[int]: Indices of articles in cluster
        :param min_overlap: int: Minimum shared entities required
        :returns: bool: True if entities overlap sufficiently
        """
        if not self.use_ner or not self.nlp:
            return True  # If NER disabled, skip check
        
        if len(indices) < 2:
            return True
        
        # Extract entities from all articles in cluster
        entity_sets = []
        for idx in indices:
            text = articles[idx].get('title', '') + ' ' + (articles[idx].get('content', '') or '')[:500]
            entities = self._extract_entities(text)
            if entities:
                entity_sets.append(entities)
        
        if len(entity_sets) < 2:
            return True  # Not enough entities to compare
        
        # Check if there's at least min_overlap shared entities across all articles: each pair must share at least one entity
        for i in range(len(entity_sets)):
            for j in range(i + 1, len(entity_sets)):
                # Check entity text overlap
                texts_i = {e[0] for e in entity_sets[i]}
                texts_j = {e[0] for e in entity_sets[j]}
                overlap = texts_i & texts_j
                if len(overlap) >= min_overlap:
                    return True  # sufficient overlap
        
        return False  # no pair has sufficient overlap
    
    def analyze_clusters(self, labels: np.ndarray, articles: List[dict], 
                        source_names: List[str], date_tolerance: int = DATE_TOLERANCE_DAYS) -> dict:
        """Analyze clusters to find cross-source duplicates with date filtering.
        
        :param labels: np.ndarray: Cluster labels
        :param articles: List[dict]: Articles
        :param source_names: List[str]: Source for each article
        :param date_tolerance: int: Max days apart for valid cluster
        :returns: dict: Cluster analysis
        """
        print("\nAnalyzing clusters...")
        
        # Group articles by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Skip noise
                clusters[label].append(idx)
        
        # Analyze each cluster
        cross_source_clusters = []
        same_source_clusters = []
        date_rejected_cross = 0
        date_rejected_same = 0
        ner_rejected = 0
        
        for cluster_id, indices in clusters.items():
            sources = set(source_names[i] for i in indices)
            
            # Check date proximity for ALL clusters - reject if dates too far apart
            if not self._dates_are_close(articles, indices, date_tolerance):
                if len(sources) > 1:
                    date_rejected_cross += 1
                else:
                    date_rejected_same += 1
                continue  # Skip this cluster - dates too far apart
            
            # Check entity overlap if NER is enabled (only for cross-source)
            if len(sources) > 1 and self.use_ner:
                if not self._entities_overlap(articles, indices):
                    ner_rejected += 1
                    continue  # Skip - no shared entities
            
            if len(sources) > 1:
                # Cross-source cluster with close dates = real duplicate!
                cross_source_clusters.append({
                    'cluster_id': cluster_id,
                    'indices': indices,
                    'sources': sources,
                    'size': len(indices)
                })
            else:
                # Same-source cluster with close dates = republication
                same_source_clusters.append({
                    'cluster_id': cluster_id,
                    'indices': indices,
                    'source': list(sources)[0],
                    'size': len(indices)
                })
        
        print(f"Cross-source clusters (real duplicates): {len(cross_source_clusters)}")
        print(f"Cross-source rejected (dates too far): {date_rejected_cross}")
        if self.use_ner:
            print(f"Cross-source rejected (no shared entities): {ner_rejected}")
        print(f"Same-source clusters (republications): {len(same_source_clusters)}")
        print(f"Same-source rejected (dates too far): {date_rejected_same}")
        
        return {
            'cross_source': cross_source_clusters,
            'same_source': same_source_clusters,
            'noise_count': list(labels).count(-1),
            'ner_rejected': ner_rejected
        }


def merge_cluster(articles: List[dict], indices: List[int], 
                  source_names: List[str]) -> dict:
    """Merge articles in a cluster into one.
    
    :param articles: List[dict]: All articles
    :param indices: List[int]: Indices of articles in cluster
    :param source_names: List[str]: Source names
    :returns: dict: Merged article
    """
    # Choose article with longest text as primary
    primary_idx = max(indices, key=lambda i: len(articles[i].get('content', '') or ''))
    primary = articles[primary_idx].copy()
    
    # Collect all sources
    sources = []
    for idx in indices:
        sources.append({
            'name': source_names[idx],
            'link': articles[idx].get('link', ''),
            'date': articles[idx].get('date', '')
        })
    
    primary['sources'] = sources
    primary['is_merged'] = True
    primary['merge_count'] = len(indices)
    
    return primary


def load_articles_from_source(source_dir: Path, source_name: str) -> Tuple[List[dict], List[str]]:
    """Load articles from source directory.
    
    :param source_dir: Path: Directory path
    :param source_name: str: Source name
    :returns: Tuple[List[dict], List[str]]: Articles and source names
    """
    articles = []
    source_names = []
    
    if not source_dir.exists():
        print(f"!!! Directory not found: {source_dir}")
        return articles, source_names
    
    json_files = list(source_dir.glob("*.json"))
    print(f"  Loading from {source_name}: {len(json_files)} files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for article in data:
                    if 'quartiere' in article and article['quartiere']:
                        normalized = normalizza_quartiere(article['quartiere'])
                        if normalized:
                            article['quartiere'] = normalized[0]
                    
                    articles.append(article)
                    source_names.append(source_name)
        
        except Exception as e:
            print(f"!!! Error loading {json_file.name}: {e}")
    
    print(f"    Loaded {len([s for s in source_names if s == source_name])} articles")
    return articles, source_names


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clustering-based article deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--scraper-dir', '-s', type=str,
                       default='scraper/news',
                       help='Directory containing scraped news')
    parser.add_argument('--output', '-o', type=str,
                       default='scraper/deduplicated_clustered.json',
                       help='Output file')
    parser.add_argument('--eps', '-e', type=float, default=DBSCAN_EPS,
                       help=f'DBSCAN eps (lower = stricter, default: {DBSCAN_EPS})')
    parser.add_argument('--min-samples', '-m', type=int, default=DBSCAN_MIN_SAMPLES,
                       help=f'DBSCAN min_samples (default: {DBSCAN_MIN_SAMPLES})')
    parser.add_argument('--cross-source-only', action='store_true',
                       help='Only merge cross-source duplicates (default: merge all)')
    parser.add_argument('--date-tolerance', '-d', type=int, default=DATE_TOLERANCE_DAYS,
                       help=f'Max days apart for valid duplicate (default: {DATE_TOLERANCE_DAYS})')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       choices=['mini', 'large'],
                       help=f'Embedding model size (default: {DEFAULT_MODEL})')
    parser.add_argument('--use-ner', action='store_true',
                       help='Enable NER filtering to validate entity overlap in clusters')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLUSTERING-BASED DEDUPLICATION")
    print("=" * 70)
    
    # Load articles
    base_dir = Path(args.scraper_dir)
    all_articles = []
    all_sources = []
    
    print("\nLoading articles from sources...")
    
    sources = [
        ('news_source1', 'Source1'),
        ('news_source2', 'Source2'),
        ('news_source3', 'Source3')
    ]
    
    for dir_name, source_name in sources:
        source_dir = base_dir / dir_name
        articles, source_names = load_articles_from_source(source_dir, source_name)
        all_articles.extend(articles)
        all_sources.extend(source_names)
    
    print(f"\nTotal articles loaded: {len(all_articles)}")
    
    if not all_articles:
        print("\n!!! No articles found. Exiting.")
        return
    
    # Initialize deduplicator
    deduplicator = ClusteringDeduplicator(model_name=args.model, use_ner=args.use_ner)
    
    # Create embeddings
    embeddings = deduplicator.create_embeddings(all_articles)
    
    # Cluster
    labels = deduplicator.cluster_articles(embeddings, eps=args.eps, min_samples=args.min_samples)
    
    # Analyze clusters with date filtering
    analysis = deduplicator.analyze_clusters(labels, all_articles, all_sources, date_tolerance=args.date_tolerance)
    
    # Build output
    print("\nBuilding deduplicated output...")
    
    merged_indices = set()
    deduplicated = []
    
    # Merge cross-source clusters (real duplicates)
    for cluster in analysis['cross_source']:
        merged = merge_cluster(all_articles, cluster['indices'], all_sources)
        deduplicated.append(merged)
        merged_indices.update(cluster['indices'])
    
    # Merge same-source clusters (similar articles)
    if not args.cross_source_only:
        for cluster in analysis['same_source']:
            merged = merge_cluster(all_articles, cluster['indices'], all_sources)
            deduplicated.append(merged)
            merged_indices.update(cluster['indices'])
    
    # Add non-merged articles
    for i, article in enumerate(all_articles):
        if i not in merged_indices:
            article_copy = article.copy()
            article_copy['sources'] = [{
                'name': all_sources[i],
                'link': article.get('link', ''),
                'date': article.get('date', '')
            }]
            article_copy['is_merged'] = False
            article_copy['merge_count'] = 1
            deduplicated.append(article_copy)
    
    # Sort by date
    deduplicated.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    
    # Final report
    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETED")
    print("=" * 70)
    print(f"Input:  {len(all_articles)} articles")
    print(f"Output: {len(deduplicated)} deduplicated articles")
    print(f"Saved:  {output_path}")
    
    cross_source_articles = sum(len(c['indices']) for c in analysis['cross_source'])
    
    print(f"\nStatistics:")
    print(f"  - Cross-source duplicates: {len(analysis['cross_source'])} groups ({cross_source_articles} articles)")
    print(f"  - Same-source duplicates:  {len(analysis['same_source'])} groups")
    print(f"  - Unique articles:         {analysis['noise_count']}")


if __name__ == "__main__":
    main()
