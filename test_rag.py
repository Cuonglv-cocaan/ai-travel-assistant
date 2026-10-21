from src.retriever import retrieve_docs
from src.generator import generate_answer


def main():
    query = "What are some popular places to visit in Vietnam?"

    try:
        # Step 1: Test document retrieval from Qdrant
        docs = retrieve_docs(query)
        print(f"Retrieved documents: {len(docs)}")

        if not docs:
            print("WARNING: Qdrant did not return any documents.")
        else:
            print("\nSample document:")
            print(docs[0][:300])

        # Step 2: Test answer generation using the retrieved context
        answer = generate_answer(query)

        print("\nQuestion:", query)
        print("\nAI Answer:", answer)

        if isinstance(answer, str) and answer.strip():
            print("\nRAG pipeline returned an answer.")
        else:
            print("\nWARNING: The RAG pipeline returned an empty or non-text answer.")

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
