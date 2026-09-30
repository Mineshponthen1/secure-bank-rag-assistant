import ollama
import weaviate
from weaviate.classes.query import MetadataQuery

EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "BankKnowledge"

QUESTIONS = [
    # Relevant: these documents answer them
    ("relevant",   "How many days of annual leave is a worker entitled to?"),
    ("relevant",   "What is the maximum length of a probation period?"),
    ("relevant",   "How long is maternity leave for a female worker?"),
    ("relevant",   "What are the principles for risk data aggregation?"),
    ("relevant",   "What should risk reports to the board contain?"),
    ("relevant",   "What are the three lines of defence in operational risk?"),
    ("relevant",   "What is customer due diligence?"),
    ("relevant",   "What measures apply to politically exposed persons?"),
    # Irrelevant: obvious ones and bank-sounding near-misses
    ("irrelevant", "What is the capital of France?"),
    ("irrelevant", "How do I bake a chocolate cake?"),
    ("irrelevant", "What is the approval limit for expenses?"),
    ("irrelevant", "What is the office dress code?"),
    ("irrelevant", "How do I reset my VPN password?"),
    ("irrelevant", "What time does the staff cafeteria open?"),
]

client = weaviate.connect_to_local(host="localhost", port=8080)
try:
    collection = client.collections.get(COLLECTION_NAME)
    rows = []
    for label, question in QUESTIONS:
        vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=question)["embedding"]
        res = collection.query.near_vector(
            near_vector=vector, limit=1, return_metadata=MetadataQuery(distance=True)
        )
        best = res.objects[0]
        rows.append((best.metadata.distance, label, best.properties["source_file"],
                     best.properties["page_number"], question))

    print(f"\n{'distance':>8}  {'kind':10}  {'closest source':32}  question")
    for distance, label, source, page, question in sorted(rows):
        print(f"{distance:8.3f}  {label:10}  {source + ' p.' + str(page):32}  {question}")

    worst_relevant = max(d for d, l, *_ in rows if l == "relevant")
    best_irrelevant = min(d for d, l, *_ in rows if l == "irrelevant")
    print(f"\nFurthest relevant question:  {worst_relevant:.3f}")
    print(f"Closest irrelevant question: {best_irrelevant:.3f}")
    if worst_relevant < best_irrelevant:
        print(f"✅ Clear gap. A cut-off between the two would separate them.")
    else:
        print(f"⚠️  They overlap. No single cut-off separates them perfectly.")
finally:
    client.close()