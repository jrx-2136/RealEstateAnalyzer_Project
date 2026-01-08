# src/rag/intent_classifier.py

INTENTS = {"FILTER", "EXPLAIN", "COMPARE", "EDUCATIONAL"}


def classify_intent(query: str) -> str:
    q = query.lower()

    if "compare" in q:
        return "COMPARE"
    if "why" in q or "explain" in q:
        return "EXPLAIN"
    if "what is" in q or "how does" in q:
        return "EDUCATIONAL"

    return "FILTER"
