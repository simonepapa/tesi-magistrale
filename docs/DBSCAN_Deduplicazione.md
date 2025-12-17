# Analisi Tecnica: Scelta dell'Algoritmo DBSCAN per la Deduplicazione di Articoli

## Sommario

Questo documento illustra le motivazioni tecniche alla base della scelta dell'algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise) per l'identificazione di articoli duplicati provenienti da fonti giornalistiche multiple.

---

## 1. Contesto del Problema

### 1.1 Obiettivo

Identificare e unificare articoli che trattano lo stesso evento giornalistico, pubblicati da testate diverse o ripubblicati dalla stessa testata in momenti diversi.

### 1.2 Sfide Specifiche

1. **Variabilità lessicale**: Lo stesso evento viene descritto con parole diverse da giornalisti diversi
2. **Scala dei dati**: Dataset di decine di migliaia di articoli
3. **Assenza di etichette**: Non esistono coppie di duplicati già annotate (problema non supervisionato)
4. **Articoli unici**: La maggior parte degli articoli NON ha duplicati
5. **Eventi ricorrenti**: Eventi simili (es. mercati periodici) che NON devono essere unificati

---

## 2. Algoritmi Considerati

### 2.1 Approccio Pairwise (Confronto a Coppie)

**Descrizione**: Confrontare ogni articolo con ogni altro calcolando una similarità.

**Complessità**: O(n²) - per 28.000 articoli significa ~392 milioni di confronti

**Svantaggi**:

- Computazionalmente proibitivo su dataset grandi
- Richiede soglia arbitraria di similarità
- Non gestisce bene la transitività (se A~B e B~C, allora A~C)

### 2.2 K-Means Clustering

**Descrizione**: Partizionare gli articoli in k cluster predefiniti.

**Svantaggi**:

- Richiede di specificare k a priori (impossibile sapere quanti duplicati esistono)
- Assume cluster di forma sferica e dimensione simile
- Non gestisce outliers (articoli unici)
- Ogni articolo DEVE appartenere a un cluster

### 2.3 Hierarchical Clustering (Agglomerativo)

**Descrizione**: Costruire una gerarchia di cluster dal basso verso l'alto.

**Svantaggi**:

- Complessità O(n²) nella versione naive
- Difficile determinare dove "tagliare" il dendrogramma
- Non scala bene su grandi dataset

### 2.4 DBSCAN ✓

**Descrizione**: Raggruppare punti densamente connessi, identificando outliers come rumore.

**Vantaggi** (dettagliati nella sezione successiva)

---

## 3. Perché DBSCAN

### 3.1 Non Richiede Specificare il Numero di Cluster

A differenza di K-Means, DBSCAN scopre automaticamente il numero di gruppi di duplicati. Questo è fondamentale perché:

- Non sappiamo a priori quanti eventi hanno duplicati
- Il numero varia nel tempo e tra dataset diversi

### 3.2 Gestione Nativa degli Outliers

DBSCAN etichetta come "rumore" (-1) i punti che non appartengono a nessun cluster denso.

**Nel nostro contesto**: La maggior parte degli articoli (~80-85%) sono unici. DBSCAN li identifica correttamente come rumore, senza forzarli in cluster artificiali.

```
Esempio output:
- 23.159 articoli unici (noise)
- 4.923 articoli in cluster (duplicati potenziali)
```

### 3.3 Cluster di Forma Arbitraria

DBSCAN può identificare cluster di qualsiasi forma, non solo sferici. Questo è utile perché le variazioni lessicali creano cluster "allungati" nello spazio degli embedding.

### 3.4 Complessità Computazionale

Con strutture dati appropriate (es. ball tree), DBSCAN ha complessità:

- Caso medio: **O(n log n)**
- Caso peggiore: O(n²)

Per il nostro dataset di ~28.000 articoli, l'elaborazione richiede circa 10-15 secondi.

### 3.5 Due Soli Iperparametri

DBSCAN richiede solo due parametri:

| Parametro       | Significato                                       | Valore scelto |
| --------------- | ------------------------------------------------- | ------------- |
| **ε (eps)**     | Raggio massimo per considerare due punti "vicini" | 0.5           |
| **min_samples** | Minimo punti per formare un cluster               | 2             |

Questi parametri sono interpretabili e stabili:

- **eps = 0.5**: Distanza euclidea su embedding normalizzati ≈ similarità coseno ~0.875
- **min_samples = 2**: Un duplicato richiede almeno 2 articoli

---

## 4. Pipeline Implementata

### 4.1 Creazione degli Embedding

```
Articolo → Sentence-Transformers → Vettore 384D → Normalizzazione L2
```

Il modello `paraphrase-multilingual-MiniLM-L12-v2` è stato scelto perché:

- Supporta l'italiano
- Ottimizzato per similarità semantica
- Leggero (dimensione 384)
- Veloce (~45 articoli/secondo)

### 4.2 Clustering con DBSCAN

```python
DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
```

Su embedding normalizzati L2, la distanza euclidea è equivalente alla distanza coseno:

```
d_euclidean = √(2 - 2·cos_sim)
```

### 4.3 Post-Filtering Temporale

DBSCAN raggruppa articoli semanticamente simili, ma non considera le date.

**Problema**: Eventi ricorrenti (es. "Mercato delle pulci domenicale") producono articoli simili ogni settimana/anno, che DBSCAN raggrupperebbe erroneamente.

**Soluzione**: Dopo il clustering, si valida ogni cluster verificando che le date degli articoli siano entro una tolleranza (default: 14 giorni). Cluster con date troppo distanti vengono scartati.

```
Cluster con date 2018-09-15 e 2025-12-09 → SCARTATO (7 anni di differenza)
Cluster con date 2025-01-15 e 2025-01-16 → VALIDO (1 giorno di differenza)
```

---

## 5. Documentazione delle Funzioni

### 5.1 Classe `ClusteringDeduplicator`

Classe principale che incapsula la logica di deduplicazione.

#### `__init__(self, model_name)`

**Scopo**: Inizializza il deduplicatore caricando il modello di embedding.

**Motivazione**: Il modello `paraphrase-multilingual-MiniLM-L12-v2` viene caricato una sola volta e riutilizzato per tutti gli articoli, evitando overhead ripetuto.

```python
self.model = SentenceTransformer(model_name)
```

---

#### `prepare_text(self, article) → str`

**Scopo**: Prepara il testo di un articolo per la creazione dell'embedding.

**Input**: Dizionario articolo con chiavi `titolo` e `testo`

**Output**: Stringa combinata di titolo + primi 300 caratteri del contenuto

**Motivazione tecnica**:

- Il **titolo** è fondamentale per identificare l'evento
- I **primi 300 caratteri** catturano il lead giornalistico (chi, cosa, dove, quando)
- Limitare a 300 caratteri:
  - Riduce il rumore dei dettagli non essenziali
  - Migliora la coerenza tra articoli sulla stessa notizia
  - Velocizza la creazione degli embedding

```python
combined = f"{title}. {content[:300]}"
```

---

#### `create_embeddings(self, articles) → np.ndarray`

**Scopo**: Crea embedding vettoriali per tutti gli articoli.

**Input**: Lista di dizionari articolo

**Output**: Matrice numpy (N × 384) di embedding normalizzati

**Motivazione tecnica**:

- **Batch processing** (`batch_size=64`): Sfrutta la parallelizzazione della GPU/CPU
- **Normalizzazione L2** (`normalize_embeddings=True`): Prerequisito per usare distanza euclidea come proxy della distanza coseno
- **Dimensione 384**: Compromesso tra espressività e velocità

```python
embeddings = self.model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True
)
```

---

#### `cluster_articles(self, embeddings, eps, min_samples) → np.ndarray`

**Scopo**: Applica l'algoritmo DBSCAN sugli embedding.

**Input**:

- `embeddings`: Matrice di embedding normalizzati
- `eps`: Raggio massimo di vicinanza (default: 0.5)
- `min_samples`: Minimo articoli per cluster (default: 2)

**Output**: Array di etichette cluster (-1 = rumore/unico)

**Motivazione tecnica**:

- **`metric='euclidean'`**: Su vettori normalizzati, equivale alla distanza coseno
- **`n_jobs=-1`**: Utilizza tutti i core CPU disponibili
- **eps = 0.5** corrisponde a similarità coseno ≈ 0.875 (formula: `cos_sim = 1 - eps²/2`)

```python
clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
labels = clustering.fit_predict(embeddings)
```

---

#### `_parse_date(self, date_str) → datetime | None`

**Scopo**: Converte stringhe data in oggetti datetime.

**Input**: Stringa data in vari formati (`YYYY-MM-DD`, `DD/MM/YYYY`)

**Output**: Oggetto datetime o None se parsing fallisce

**Motivazione tecnica**:

- Gestisce formati multipli comuni nelle testate italiane
- Fallback graceful: restituisce None invece di errore
- Necessario per il post-filtering temporale

---

#### `_dates_are_close(self, articles, indices, tolerance_days) → bool`

**Scopo**: Verifica se tutti gli articoli di un cluster hanno date entro la tolleranza.

**Input**:

- `articles`: Lista completa articoli
- `indices`: Indici degli articoli nel cluster
- `tolerance_days`: Massima differenza in giorni (default: 14)

**Output**: `True` se le date sono entro tolleranza, `False` altrimenti

**Motivazione tecnica**:

- Calcola `max(date) - min(date)` per l'intero cluster
- Se ci sono meno di 2 date valide, assume OK (non penalizza articoli senza data)
- Impedisce di unire eventi ricorrenti (es. mercato annuale 2023 con 2024)

```python
delta = (max_date - min_date).days
return delta <= tolerance_days
```

---

#### `analyze_clusters(self, labels, articles, source_names, date_tolerance) → dict`

**Scopo**: Analizza i cluster identificati da DBSCAN e li classifica.

**Input**:

- `labels`: Etichette cluster da DBSCAN
- `articles`: Lista articoli
- `source_names`: Nome fonte per ogni articolo
- `date_tolerance`: Tolleranza date in giorni

**Output**: Dizionario con:

- `cross_source`: Cluster con articoli da fonti diverse
- `same_source`: Cluster con articoli dalla stessa fonte
- `noise_count`: Numero di articoli unici

**Logica di classificazione**:

```
Per ogni cluster:
    1. Verifica date → se troppo distanti → SCARTA
    2. Conta fonti diverse:
       - Se > 1 → cross_source (duplicato reale)
       - Se = 1 → same_source (ripubblicazione)
```

**Motivazione tecnica**:

- Separare cross/same-source permette strategie di merge diverse
- Il date filtering PRIMA della classificazione evita falsi positivi

---

### 5.2 Funzione `merge_cluster(articles, indices, source_names) → dict`

**Scopo**: Unisce un cluster di articoli duplicati in un singolo articolo.

**Strategia di selezione**:

- L'articolo con il **testo più lungo** diventa il primario
- Gli altri contribuiscono alla lista `sources`

**Motivazione tecnica**:

- Il testo più lungo generalmente contiene più informazioni
- Mantenere tutte le fonti preserva la provenienza e le date originali
- Facilita audit e debug post-merge

**Output format**:

```json
{
  "title": "...",
  "testo": "...",
  "data": "...",
  "sources": [
    { "name": "Source1", "link": "...", "date": "..." },
    { "name": "Source2", "link": "...", "date": "..." }
  ],
  "is_merged": true,
  "merge_count": 2
}
```

---

### 5.3 Funzione `load_articles_from_source(source_dir, source_name) → Tuple`

**Scopo**: Carica articoli da una directory di una specifica fonte.

**Input**:

- `source_dir`: Path alla directory contenente file JSON
- `source_name`: Nome identificativo della fonte

**Output**: Tupla (lista articoli, lista nomi fonte)

**Logica**:

1. Trova tutti i file `*.json` nella directory
2. Per ogni file, carica la lista di articoli
3. Normalizza il campo `quartiere` usando `normalizza_quartiere()`
4. Associa ogni articolo al nome fonte

**Motivazione tecnica**:

- Separare il caricamento per fonte permette tracciamento cross-source
- La normalizzazione quartieri garantisce consistenza nei dati

---

### 5.4 Funzione `main()`

**Scopo**: Orchestrazione dell'intera pipeline.

**Flusso di esecuzione**:

```
1. Parsing argomenti CLI
2. Caricamento articoli da tutte le fonti
3. Creazione embedding (ClusteringDeduplicator.create_embeddings)
4. Clustering DBSCAN (ClusteringDeduplicator.cluster_articles)
5. Analisi cluster con date filter (ClusteringDeduplicator.analyze_clusters)
6. Merge cluster cross-source
7. Merge cluster same-source (se non --cross-source-only)
8. Aggiunta articoli unici (non clusterizzati)
9. Ordinamento per data
10. Salvataggio JSON e report finale
```

**Argomenti CLI**:

| Argomento             | Default                               | Descrizione                  |
| --------------------- | ------------------------------------- | ---------------------------- |
| `--scraper-dir`       | `scraper/news`                        | Directory sorgente           |
| `--output`            | `scraper/deduplicated_clustered.json` | File output                  |
| `--eps`               | 0.5                                   | Parametro DBSCAN             |
| `--min-samples`       | 2                                     | Minimo articoli per cluster  |
| `--date-tolerance`    | 14                                    | Giorni massimi di differenza |
| `--cross-source-only` | False                                 | Solo duplicati cross-fonte   |

---

## 6. Risultati Sperimentali

### 6.1 Dataset di Test

- **Articoli totali**: 28.082
- **Fonti**: 3 testate giornalistiche locali
- **Periodo**: 2016-2025

### 6.2 Performance

| Metrica                         | Valore                  |
| ------------------------------- | ----------------------- |
| Tempo di elaborazione           | ~25 secondi             |
| Articoli unici identificati     | 23.159 (82.5%)          |
| Duplicati cross-source          | 10 gruppi (21 articoli) |
| Duplicati same-source           | 911 gruppi              |
| Cluster scartati (date lontane) | 516                     |
| **Riduzione totale**            | 4.8% (1.338 articoli)   |

### 6.3 Confronto DBSCAN vs HDBSCAN

È stato condotto un confronto sistematico tra DBSCAN (con diversi valori di eps) e HDBSCAN (con diversi valori di min_cluster_size):

| Metodo     | Parametri          | Cross-Src | Same-Src | Rejected | Silhouette | Tempo    |
| ---------- | ------------------ | --------- | -------- | -------- | ---------- | -------- |
| DBSCAN     | eps=0.3            | 0         | 918      | 200      | 0.94       | 2.7s     |
| DBSCAN     | eps=0.4            | 3         | 911      | 287      | 0.84       | 1.3s     |
| **DBSCAN** | **eps=0.5**        | **10**    | **911**  | **516**  | **0.55**   | **1.3s** |
| DBSCAN     | eps=0.6            | 13        | 888      | 820      | 0.12       | 1.3s     |
| DBSCAN     | eps=0.7            | 9         | 470      | 563      | -0.16      | 1.3s     |
| HDBSCAN    | min_cluster_size=2 | 27        | 1817     | 2687     | 0.31       | 84s      |
| HDBSCAN    | min_cluster_size=3 | 6         | 442      | 1343     | 0.27       | 80s      |
| HDBSCAN    | min_cluster_size=5 | 0         | 85       | 553      | 0.24       | 82s      |

**Analisi dei risultati**:

1. **DBSCAN eps=0.5** è il miglior compromesso:

   - 10 duplicati cross-source identificati
   - Veloce (1.3 secondi)
   - Silhouette accettabile (0.55)

2. **HDBSCAN** trova più cross-source (27 con min_cluster_size=2) ma:

   - ~60x più lento (84s vs 1.3s)
   - 5x più cluster rejected (2687 vs 516)
   - Silhouette peggiore (0.31 vs 0.55)

3. **eps più bassi** (0.3-0.4) sono troppo restrittivi (0-3 cross-source)

4. **eps più alti** (0.6-0.7) generano cluster di bassa qualità (silhouette < 0.2)

---

## 7. Limitazioni e Lavori Futuri

### 7.1 Limitazioni Attuali

1. **Sensibilità a eps**: Valori troppo alti raggruppano articoli non correlati; troppo bassi perdono duplicati
2. **Articoli brevi**: Titoli senza testo producono embedding meno informativi
3. **Eventi seriali**: Processi giudiziari con più udienze possono generare articoli simili ma distinti

### 7.2 Miglioramenti Implementati

Sono stati implementati i seguenti miglioramenti rispetto alla versione base:

1. **Scelta modello di embedding**: È possibile scegliere tra:

   - `mini` (default): `paraphrase-multilingual-MiniLM-L12-v2` - veloce, 384 dimensioni
   - `large`: `intfloat/multilingual-e5-large` - più accurato, 1024 dimensioni

   ```bash
   python deduplicate_clustering.py --model large
   ```

2. **Filtraggio NER**: Verifica che i cluster cross-source condividano entità nominate (persone, luoghi, organizzazioni):

   ```bash
   python deduplicate_clustering.py --use-ner
   ```

3. **Script di confronto**: `compare_deduplication.py` per benchmark automatico di diverse configurazioni

### 7.3 Possibili Sviluppi Futuri

1. **Tuning automatico di eps** tramite silhouette score su sottocampione
2. **Combinazione di embedding** (title-only + content) con pesi
3. **Active learning** per validazione manuale di cluster borderline

---

## 8. Conclusioni

L'algoritmo DBSCAN si è rivelato la scelta ottimale per il problema della deduplicazione di articoli giornalistici per i seguenti motivi:

1. **Scalabilità**: Gestisce efficientemente decine di migliaia di articoli
2. **Nessun k predefinito**: Scopre automaticamente i gruppi di duplicati
3. **Outliers nativi**: Gestisce correttamente la maggioranza di articoli unici
4. **Interpretabilità**: Parametri intuitivi e stabili
5. **Complementarietà**: Si integra bene con il post-filtering temporale

L'implementazione combinata di DBSCAN su embedding semantici con validazione temporale rappresenta un approccio robusto e accurato per l'identificazione di duplicati cross-source in corpus giornalistici.

---

## Riferimenti

1. Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." KDD-96.
2. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.
3. Schubert, E., et al. (2017). "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN." ACM TODS.
