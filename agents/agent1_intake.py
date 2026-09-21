"""
Agent 1: Intake - Entity Extraction from Incident Queries
Extracts domain-specific entities (loom_id, batch_id, metric, defect_symptom)
from natural language queries using spaCy rule-based matching.
"""

import re
import json
import spacy
from spacy.matcher import Matcher

# Load spaCy model (en_core_web_sm is lightweight and sufficient for rule-based matching)
nlp = spacy.load("en_core_web_sm")

# ── 1. Define regex patterns for domain-specific entities ───────────────────────────
# These patterns are designed to match the specific ID formats used in textile QA

PATTERNS = {
    # Loom ID: Matches "Loom 12", "Loom-12", "L12", "LOOM-12", etc.
    "loom_id": [
        r"(?:loom|L)[\s-]?(\d+)",  # Loom 12, Loom-12, L12
        r"(?:LOOM|LOOM)[\s-]?(\d+)",  # Uppercase variants
    ],
    
    # Batch ID: Matches "Batch B102", "Batch-B102", "B102", "B-102", etc.
    "batch_id": [
        r"(?:batch|BATCH)[\s-]?([A-Z]\d+)",  # Batch B102, Batch-B102
        r"\b([A-Z]\d{3,})\b",  # B102, B-102 (standalone)
    ],
    
    # Metric: Common textile quality metrics
    "metric": [
        r"(?:elongation|strength|tension|density|count|twist|evenness|hairiness)",
        r"(?:fabric weight|gsm|yarn count|fabric width)",
    ],
    
    # Defect Symptom: Descriptive terms indicating problems
    "defect_symptom": [
        r"(?:inconsistent|variable|fluctuating|unstable|irregular)",
        r"(?:high|low|excessive|insufficient|poor)",
        r"(?:defect|fault|error|issue|problem)",
        r"(?:breakage|snapping|tearing|splitting)",
    ],
}


# ── 2. Initialize spaCy Matcher ─────────────────────────────────────────────────────
matcher = Matcher(nlp.vocab)

# Add patterns to the matcher
# We'll use regex-based custom entity matching since spaCy's built-in NER
# doesn't handle domain-specific IDs well
def add_matcher_patterns():
    """Add custom patterns to spaCy matcher for entity extraction."""
    
    # Pattern for loom IDs (looking for "Loom" followed by number)
    matcher.add("LOOM_ID", [
        [{"LOWER": "loom"}, {"IS_DIGIT": True}],
        [{"LOWER": "loom"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_DIGIT": True}],
    ])
    
    # Pattern for batch IDs (looking for "Batch" followed by alphanumeric code)
    matcher.add("BATCH_ID", [
        [{"LOWER": "batch"}, {"IS_ALPHA": True, "LENGTH": 1}, {"IS_DIGIT": True, "OP": "+"}],
        [{"LOWER": "batch"}, {"IS_PUNCT": True, "OP": "?"}, {"IS_ALPHA": True, "LENGTH": 1}, {"IS_DIGIT": True, "OP": "+"}],
    ])
    
    # Pattern for metrics (common textile terms)
    metric_terms = ["elongation", "strength", "tension", "density", "count", 
                    "twist", "evenness", "hairiness", "weight", "gsm", "width"]
    matcher.add("METRIC", [
        [{"LOWER": {"IN": metric_terms}}]
    ])
    
    # Pattern for defect symptoms (problem descriptors)
    defect_terms = ["inconsistent", "variable", "fluctuating", "unstable", "irregular",
                    "high", "low", "excessive", "insufficient", "poor",
                    "defect", "fault", "error", "issue", "problem",
                    "breakage", "snapping", "tearing", "splitting"]
    matcher.add("DEFECT_SYMPTOM", [
        [{"LOWER": {"IN": defect_terms}}]
    ])

add_matcher_patterns()


# ── 3. Entity extraction function ───────────────────────────────────────────────────
def extract_entities(query: str) -> dict:
    """
    Extract entities from a natural language query using spaCy rule-based matching.
    
    Args:
        query: Natural language query about an incident (e.g., 
               "Why is Loom 12 producing inconsistent elongation in Batch B102?")
    
    Returns:
        JSON dict with extracted entities:
        {
            "loom_id": str or None,
            "batch_id": str or None,
            "metric": str or None,
            "defect_symptom": str or None
        }
    
    Note: We use rule-based matching (regex + spaCy patterns) instead of 
    standard NER because:
    - Domain-specific IDs (Loom 12, Batch B102) are not in standard NER training data
    - Textile metrics and defect terms are industry-specific vocabulary
    - Regex provides precise control over ID format matching
    - Rule-based approach is more interpretable and maintainable for this domain
    """
    # Initialize result dict
    entities = {
        "loom_id": None,
        "batch_id": None,
        "metric": None,
        "defect_symptom": None
    }
    
    # Process query with spaCy
    doc = nlp(query)
    
    # Apply matcher to find entities
    matches = matcher(doc)
    
    # Extract matched entities
    for match_id, start, end in matches:
        # Use the correct API for spaCy v3.x+
        match_label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        text = span.text
        
        if match_label == "LOOM_ID":
            # Extract the numeric part (e.g., "12" from "Loom 12")
            numbers = re.findall(r"\d+", text)
            if numbers:
                entities["loom_id"] = numbers[0]
        
        elif match_label == "BATCH_ID":
            # Extract the alphanumeric code (e.g., "B102" from "Batch B102")
            code = re.search(r"[A-Z]\d+", text)
            if code:
                entities["batch_id"] = code.group()
        
        elif match_label == "METRIC":
            entities["metric"] = text.lower()
        
        elif match_label == "DEFECT_SYMPTOM":
            entities["defect_symptom"] = text.lower()
    
    # Fallback: Use regex patterns if spaCy matcher didn't catch everything
    # This provides additional robustness for edge cases
    
    if not entities["loom_id"]:
        for pattern in PATTERNS["loom_id"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["loom_id"] = match.group(1)
                break
    
    if not entities["batch_id"]:
        for pattern in PATTERNS["batch_id"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["batch_id"] = match.group(1)
                break
    
    if not entities["metric"]:
        for pattern in PATTERNS["metric"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["metric"] = match.group().lower()
                break
    
    if not entities["defect_symptom"]:
        for pattern in PATTERNS["defect_symptom"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["defect_symptom"] = match.group().lower()
                break
    
    return entities


# ── 4. Main function for easy testing ───────────────────────────────────────────────
def parse_query(query: str) -> str:
    """
    Parse a query and return entities as JSON string.
    
    Args:
        query: Natural language query
    
    Returns:
        JSON string of extracted entities
    """
    entities = extract_entities(query)
    return json.dumps(entities, indent=2)


# ── Example usage (uncomment to test) ───────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Why is Loom 12 producing inconsistent elongation in Batch B102?",
        "Batch A505 has high tension on Loom-7",
        "What's causing poor strength in Batch C201?",
        "Loom 15 shows fluctuating density in Batch D303"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print(parse_query(query))
        print("-" * 50)
