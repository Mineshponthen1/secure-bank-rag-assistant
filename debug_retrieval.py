import ollama
import weaviate
from weaviate.classes.query import MetadataQuery

QUESTION = "How many days of annual leave is a worker entitled to?"
ANSWER_CLUES = ["year of service", "annual leave with full pay"]   # phrases specific to the leave entitlement clause

client = weaviate.connect_to_local(host="localhost", port=8080)
try:
    collection = client.collections.get("BankKnowledge")
    vector = ollama.embeddings(model="nomic-embed-text", prompt=QUESTION)["embedding"]
    res = collection.query.near_vector(
        near_vector=vector, limit=10, return_metadata=MetadataQuery(distance=True)
    )

    print(f"\nQuestion: {QUESTION}\n")
    for rank, obj in enumerate(res.objects, start=1):
        p = obj.properties
        content = str(p["content"]).lower()
        found = [clue for clue in ANSWER_CLUES if clue.lower() in content]
        mark = f"✅ contains {found}" if found else ""
        status = "sent to AI" if rank <= 3 and (obj.metadata.distance or 1.0) <= 0.41 else "not sent"
        print(f"#{rank:2}  {obj.metadata.distance:.3f}  {p['source_file']:24} "
              f"pages {p['page_number']}-{p.get('page_end')}  [{status:10}]  {mark}")
finally:
    client.close()