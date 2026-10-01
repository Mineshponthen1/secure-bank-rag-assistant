import os

import cohere
import ollama
import weaviate
from dotenv import load_dotenv
from weaviate.classes.query import MetadataQuery

load_dotenv()   # reads COHERE_API_KEY from the .env file

QUESTION = "How many days of annual leave is a worker entitled to?"
ANSWER_CLUE = "annual leave with full pay"   # phrase from the clause that really answers it
RERANK_MODEL = "rerank-v4.0-pro"             # Cohere's re-ranking model


def show(title, rows):
    print(f"\n{title}")
    for rank, (score, obj) in enumerate(rows, start=1):
        p = obj.properties
        mark = "✅ answer" if ANSWER_CLUE in p["content"].lower() else ""
        print(f"  #{rank:2}  {score:6.3f}  {p['source_file']:24} pages {p['page_number']}-{p.get('page_end')}  {mark}")


api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise SystemExit("No COHERE_API_KEY found. Put it in the .env file next to this script.")
co = cohere.ClientV2(api_key=api_key)

client = weaviate.connect_to_local(host="localhost", port=8080)
try:
    collection = client.collections.get("BankKnowledge")
    vector = ollama.embeddings(model="nomic-embed-text", prompt=QUESTION)["embedding"]

    # A. What the app does now: meaning only
    old = collection.query.near_vector(
        near_vector=vector, limit=10, return_metadata=MetadataQuery(distance=True))
    show("A. NOW: vector search (score = distance, lower is better)",
         [(o.metadata.distance or 1.0, o) for o in old.objects])

    # B. New step 1: hybrid search (meaning + keywords) builds a long list of 20
    candidates = collection.query.hybrid(
        query=QUESTION, vector=vector, alpha=0.5, limit=20,
        return_metadata=MetadataQuery(score=True)).objects

    # C. New step 2: Cohere reads the question with each candidate and re-orders them
    res = co.rerank(
        model=RERANK_MODEL, query=QUESTION,
        documents=[str(o.properties["content"]) for o in candidates], top_n=10)
    reranked = [(r.relevance_score, candidates[r.index]) for r in res.results]
    show("C. NEW: hybrid + Cohere Rerank (score = relevance 0-1, higher is better)", reranked)
finally:
    client.close()