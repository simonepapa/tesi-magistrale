"""
Street Extractor
================
Extracts street addresses from Italian news articles using regex patterns.

Usage:
    from extract_streets import extract_streets
    streets = extract_streets("Un uomo è stato arrestato in Via Roma...")
    # Returns: ["Via Roma"]
"""

import re
from typing import List

# Italian street prefixes
STREET_PREFIXES = [
    r"Via", r"Viale", r"Corso", r"Piazza", r"Piazzale", r"Piazzetta",
    r"Largo", r"Vicolo", r"Traversa", r"Strada", r"Lungomare",
    r"Contrada", r"Borgata", r"Rione", r"Quartiere"
]

# Regex pattern (without house numbers)
STREET_PATTERN = re.compile(
    r'\b(' + '|'.join(STREET_PREFIXES) + r')\s+'  # Prefix (Via, Piazza, etc)
    r'((?:del(?:la|lo|le|i|gli)?|de(?:i|gli)?|dell\')?(?:\s+)?)?'  # Optional articles
    r'([A-Z][a-zàèéìòù\']+(?:\s+(?:del(?:la|lo|le|i|gli)?|de(?:i|gli)?|[A-Z][a-zàèéìòù\']+))*)',  # Name
    re.UNICODE
)


def extract_streets(text: str) -> List[str]:
    """Extract street addresses from text.

    :param text: str: Article text content
    :returns: List of streets found (e.g., ["Via Roma", "Piazza Garibaldi"])

    """
    streets = []
    seen = set()
    
    for match in STREET_PATTERN.finditer(text):
        prefix = match.group(1)
        article = (match.group(2) or "").strip()
        name = match.group(3)
        
        # Build full address
        if article:
            full_address = f"{prefix} {article} {name}"
        else:
            full_address = f"{prefix} {name}"
        
        # Avoid duplicates
        if full_address not in seen:
            seen.add(full_address)
            streets.append(full_address)
    
    return streets


# Test
if __name__ == "__main__":
    test_texts = [
        "L'incidente è avvenuto in Via Roma 25, nei pressi di Piazza Garibaldi.",
        "I carabinieri sono intervenuti in Corso Vittorio Emanuele II all'altezza del civico 45.",
        "L'uomo è stato fermato in Largo Argentina, vicino a Viale Trastevere.",
        "Il furto è stato commesso in un appartamento di Via della Libertà 12/A a Bari.",
        "Nessun indirizzo in questo testo.",
    ]
    
    print("="*60)
    print("STREET EXTRACTION TEST")
    print("="*60)
    
    for text in test_texts:
        print(f"\nText: {text[:60]}...")
        streets = extract_streets(text)
        print(f"Streets found: {streets}")
