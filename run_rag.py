from src.rag.intent_classifier import classify_intent
from src.rag.property_explanations import load_property_explanations
from src.rag.rag_engine import generate_rag_response
from src.rag.vector_store import build_or_load_vector_store


def main():
    explanations = load_property_explanations()
    vector_db = build_or_load_vector_store(explanations)

    while True:
        query = input("\nAsk a real estate question (or 'exit'): ")
        if query.lower() == "exit":
            break

        intent = classify_intent(query)

        docs = vector_db.similarity_search(query, k=3)
        context = [d.page_content for d in docs]

        answer = generate_rag_response(context, query)
        print("\n", answer)

if __name__ == "__main__":
    main()
