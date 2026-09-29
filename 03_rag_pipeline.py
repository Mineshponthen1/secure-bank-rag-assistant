import ollama
import weaviate
from weaviate.classes.query import Filter

# Connect to local Weaviate container
client = weaviate.connect_to_local(host="localhost", port=8080)
COLLECTION_NAME = "BankKnowledge"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:latest"


def generate_rag_response(user_query: str, user_department: str):
    """Executes dense vector retrieval with RBAC and generates a grounded answer with verifiable citations."""
    collection = client.collections.get(COLLECTION_NAME)

    # 1. Generate query embedding locally via Ollama
    query_vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=user_query)["embedding"]

    # 2. Perform Near-Vector Search enforcing RBAC department access
    rbac_filter = Filter.by_property("allowed_departments").contains_any([user_department])

    # Target the configured named vector 'default'
    results = collection.query.near_vector(
        near_vector=query_vector,
        target_vector="default",
        filters=rbac_filter,
        limit=3,
    )

    if not results.objects:
        print("\n" + "=" * 60)
        print(f"🔍 Question  : {user_query}")
        print(f"👤 Department: {user_department}")
        print("=" * 60)
        print("⛔ Access Denied or No relevant policy information found.")
        return

    # 3. Format context blocks & assemble citation metadata
    context_blocks = []
    citations = []

    for idx, obj in enumerate(results.objects, start=1):
        props = obj.properties
        content = str(props.get("content", ""))
        source = str(props.get("source_file", ""))
        page = props.get("page_number", 1)

        context_blocks.append(f"[{idx}] (Source: {source}, Page {page})\n{content}")
        citations.append(
            {
                "citation_id": idx,
                "source_file": source,
                "page_number": page,
                "paragraph": content,
            }
        )

    formatted_context = "\n\n".join(context_blocks)

    # 4. Construct grounded system prompt
    prompt = f"""You are an enterprise AI assistant for internal bank policies.
Answer the user query strictly using the provided context blocks. 
Cite your sources in your response using numeric brackets like [1], [2], corresponding to the context block index.
Do NOT use external knowledge or fabricate policies.

Context:
{formatted_context}

User Question: {user_query}
Answer:"""

    # 5. Local LLM inference generation via Ollama
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)

    # Output generation and verifiable citations
    print("\n" + "=" * 60)
    print(f"🔍 Question  : {user_query}")
    print(f"👤 Department: {user_department}")
    print("=" * 60)
    print(f"🤖 Grounded Answer:\n{response['response']}\n")
    print("📚 Verifiable Citations:")
    for cite in citations:
        print(f"  [{cite['citation_id']}] File: {cite['source_file']} | Page: {cite['page_number']}")
        print(f"      Source Paragraph: \"{cite['paragraph'][:130]}...\"\n")


if __name__ == "__main__":
    try:
        # Test 1: HR query by authorized HR user
        generate_rag_response(
            user_query="What is the work from home policy?",
            user_department="HR",
        )

        # Test 2: Finance query by authorized Finance user
        generate_rag_response(
            user_query="What is the threshold limit for expense approvals?",
            user_department="Finance",
        )

        # Test 3: Unauthorized query by Operations user attempting to read restricted Finance data
        generate_rag_response(
            user_query="What is the expense threshold requiring Board pre-approval?",
            user_department="Operations",
        )
    finally:
        client.close()