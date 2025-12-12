"""
Shared Configuration for Crime Classification Models
=====================================================
Central configuration file for all model-related scripts.
"""

# Crime labels (shared across all models)
LABELS = [
    'omicidio', 'omicidio_colposo', 'omicidio_stradale', 'tentato_omicidio',
    'furto', 'rapina', 'violenza_sessuale', 'aggressione', 'spaccio',
    'truffa', 'estorsione', 'contrabbando', 'associazione_di_tipo_mafioso'
]

# Label mappings
id2label = {idx: label for idx, label in enumerate(LABELS)}
label2id = {label: idx for idx, label in enumerate(LABELS)}

# Inference configuration
MAX_LENGTH = 512
CHUNK_OVERLAP = 50
THRESHOLD = 0.75

# Model configurations
MODELS = {
    'bert': {
        'name': 'BERT',
        'checkpoint': 'bert_fine_tuned_model',
        'base_model': 'dbmdz/bert-base-italian-xxl-cased',
        'use_fast_tokenizer': True,
        'output_dir': 'data/labeled_bert'
    },
    'mdeberta': {
        'name': 'mDeBERTa',
        'checkpoint': 'mdeberta_fine_tuned_model',
        'base_model': 'microsoft/mdeberta-v3-base',
        'use_fast_tokenizer': False,  # mDeBERTa requires slow tokenizer
        'output_dir': 'data/labeled_mdeberta'
    },
    'umberto': {
        'name': 'UmBERTo',
        'checkpoint': 'umberto_fine_tuned_model',
        'base_model': 'Musixmatch/umberto-commoncrawl-cased-v1',
        'use_fast_tokenizer': True,
        'output_dir': 'data/labeled_umberto'
    }
}

# List of quartieri (standard names)
QUARTIERI = [
    "bari-vecchia_san-nicola",
    "carbonara",
    "carrassi",
    "catino-san-pio",
    "ceglie-del-campo",
    "japigia",
    "liberta",
    "loseto",
    "madonnella",
    "murat",
    "palese-macchie",
    "picone",
    "san-paolo",
    "san-pasquale",
    "santo-spirito",
    "stanic",
    "torre-a-mare",
    "san-girolamo_fesca"
]

# Mapping from standard quartiere name to all possible aliases
# Keys: standard name (from QUARTIERI list above)
# Values: list of aliases (as they appear in different sources)
QUARTIERI_ALIASES = {
    "bari-vecchia_san-nicola": [
        "bari-vecchia", "san-nicola", "bari vecchia", "san nicola",
        "Murat_-_San_Nicola",  # repubblica
    ],
    "carbonara": [
        "carbonara",
        "Carbonara_-_Ceglie_-_Loseto",  # repubblica
    ],
    "carrassi": [
        "carrassi",
        "Carrassi",  # repubblica
    ],
    "ceglie-del-campo": [
        "ceglie-del-campo", "ceglie del campo", "ceglie",
        "Carbonara_-_Ceglie_-_Loseto",  # repubblica
    ],
    "catino-san-pio": [
        "catino-san-pio",
    ],
    "japigia": [
        "japigia",
        "Japigia_-_Torre_a_Mare",  # repubblica
    ],
    "liberta": [
        "liberta", "libertà",
        "Libertà_-_Marconi_-_Fesca_-_San_Girolamo",  # repubblica
    ],
    "loseto": [
        "loseto",
        "Carbonara_-_Ceglie_-_Loseto",  # repubblica
    ],
    "madonnella": [
        "madonnella",
        "Madonnella",  # repubblica
    ],
    "murat": [
        "murat",
        "Murat_-_San_Nicola",  # repubblica
    ],
    "palese-macchie": [
        "palese-macchie", "palese", "macchie",
        "Palese_-_Santo_Spirito",  # repubblica
    ],
    "picone": [
        "picone",
        "Picone_-_Poggiofranco",  # repubblica
    ],
    "san-paolo": [
        "san-paolo", "san paolo",
        "San_Paolo_-_Stanic",  # repubblica
    ],
    "san-pasquale": [
        "san-pasquale", "san pasquale",
        "San_Pasquale_-_Mungivacca",  # repubblica
    ],
    "santo-spirito": [
        "santo-spirito", "santo spirito",
        "Palese_-_Santo_Spirito",  # repubblica
    ],
    "stanic": [
        "stanic",
        "San_Paolo_-_Stanic",  # repubblica
    ],
    "torre-a-mare": [
        "torre-a-mare", "torre a mare",
        "Japigia_-_Torre_a_Mare",  # repubblica
    ],
    "san-girolamo_fesca": [
        "san-girolamo_fesca", "san-girolamo", "fesca", "san girolamo",
        "Libertà_-_Marconi_-_Fesca_-_San_Girolamo",  # repubblica
    ],
}

# Reverse mapping: alias -> standard quartiere name(s) (some aliases map to MULTIPLE quartieri)
_ALIAS_TO_QUARTIERI = {}
for quartiere, aliases in QUARTIERI_ALIASES.items():
    for alias in aliases:
        alias_lower = alias.lower().replace("_", "-").replace(" ", "-")
        if alias_lower not in _ALIAS_TO_QUARTIERI:
            _ALIAS_TO_QUARTIERI[alias_lower] = []
        if quartiere not in _ALIAS_TO_QUARTIERI[alias_lower]:
            _ALIAS_TO_QUARTIERI[alias_lower].append(quartiere)


def normalizza_quartiere(nome: str) -> list[str]:
    """Normalize a neighborhood name to the standard name(s).
    
    Some sources (like Repubblica) combine multiple neighborhoods into one zone.
    In these cases, this function returns all matching standard quartieri.
    
    :param nome: str: The neighborhood name as found in the source
    :returns: list[str]: List of matching standard quartiere names, empty if not found
    
    Example:
        >>> normalizza_quartiere("carbonara")
        ['carbonara']
        >>> normalizza_quartiere("Carbonara_-_Ceglie_-_Loseto")
        ['carbonara', 'ceglie-del-campo', 'loseto']
        >>> normalizza_quartiere("unknown")
        []
    """
    nome_normalized = nome.lower().replace("_", "-").replace(" ", "-")
    return _ALIAS_TO_QUARTIERI.get(nome_normalized, [])


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model.

    :param model_name: str: model name

    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]


def get_available_models() -> list:
    """Return list of available model names."""
    return list(MODELS.keys())
