import json
import ollama
import weaviate
from weaviate.classes.query import Filter, HybridFusion

# Initialize local Weaviate client
client = weaviate.connect_to_local(host="localhost", port=8080)
COLLECTION_NAME = "BankKnowledge"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:latest"


def local_llm_rerank(user_query: str, candidates: list) -> list:
    """
    Air-gapped Re-ranker: Uses local Llama 3.2 to score candidate relevance 
    from 1 to 10 without needing compiled C/C++ DLLs.
    """
    scored_candidates = []

    for obj in candidates:
        content = str(obj.properties.get("content", ""))
        
        prompt = f"""Rate the relevance of the following document snippet to the user query on a scale from 1 to 10, where 10 is extremely relevant and 1 is completely irrelevant.
Output ONLY a JSON object in this format: {{"score": <number>}}

User Query: {user_query}
Document Snippet: {content}"""

        try:
            response = ollama.generate(
                model=LLM_MODEL, 
                prompt=prompt, 
                format="json"
            )
            data = json.loads(response["response"])
            score = float(data.get("score", 5))
        except Exception:
            score = 5.0

        scored_candidates.append((obj, score))

    # Sort descending by relevance score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates


def hybrid_search_with_reranking(user_query: str, user_department: str, top_k_retrieval: int = 6, top_k_rerank: int = 3):
    """
    Phase 2 Precision Engineering Retrieval:
    1. Generates query vector via nomic-embed-text.
    2. Executes Hybrid Search (Dense Vector + BM25 Sparse Keyword) with RBAC filtering in Weaviate.
    3. Re-ranks top candidates using local Llama 3.2 to guarantee context precision.
    """
    collection = client.collections.get(COLLECTION_NAME)

    # 1. Generate query embedding locally
    query_vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=user_query)["embedding"]

    # 2. RBAC Department Filter
    rbac_filter = Filter.by_property("allowed_departments").contains_any([user_department])

    # 3. Weaviate Hybrid Search (Dense Vector + BM25 Keyword Search)
    hybrid_results = collection.query.hybrid(
        query=user_query,
        vector=query_vector,
        target_vector="default",
        alpha=0.5,  # Equal weighting between Dense Vector and BM25 Sparse Search
        fusion_type=HybridFusion.RANKED,
        filters=rbac_filter,
        limit=top_k_retrieval,
    )

    if not hybrid_results.objects:
        return [], []

    # 4. Perform Air-Gapped Re-ranking
    scored_candidates = local_llm_rerank(user_query, hybrid_results.objects)

    # Select top_k_rerank chunks after scoring
    top_reranked = scored_candidates[:top_k_rerank]

    context_blocks = []
    citations = []

    for idx, (obj, score) in enumerate(top_reranked, start=1):
        props = obj.properties
        content = str(props.get("content", ""))
        source = str(props.get("source_file", ""))
        page = props.get("page_number", 1)

        context_blocks.append(f"[{idx}] (Source: {source}, Page {page})\n{content}")
        citations.append({
            "citation_id": idx,
            "source_file": source,
            "page_number": page,
            "paragraph": content,
            "rerank_score": score,
        })

    return context_blocks, citations


if __name__ == "__main__":
    try:
        blocks, cites = hybrid_search_with_reranking(
            user_query="What is the work from home policy?",
            user_department="HR"
        )
        print(f"\n✅ Retrieved {len(cites)} re-ranked citation blocks:")
        for c in cites:
            print(f"  [{c['citation_id']}] File: {c['source_file']} (Re-rank Score: {c['rerank_score']}/10)")
            print(f"      Snippet: \"{c['paragraph'][:100]}...\"\n")
    finally:
        client.close()