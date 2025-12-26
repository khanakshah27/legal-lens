import spacy

nlp = spacy.load("en_core_web_sm")

def segment_clauses(text):
    """Splits a contract into individual clauses/sentences."""
    doc = nlp(text)
    clauses = [sent.text.strip() for sent in doc.sents if len(sent.text) > 20]
    return clauses

def clean_text(text):
    """Pure NLP cleaning: lowercasing and stripping whitespace."""
    return text.lower().strip()
