import os
import time
import cohere
import ollama
import weaviate
from dotenv import load_dotenv

load_dotenv()
RERANK_MODEL = "rerank-v4.0-pro"

QUESTIONS = [
    ("relevant",   "How many days of annual leave is a worker entitled to?"),
    ("relevant",   "What is the maximum length of a probation period?"),
    ("relevant",   "How long is maternity leave for a female worker?"),
    ("relevant",   "What are the principles for risk data aggregation?"),
    ("relevant",   "What should risk reports to the board contain?"),
    ("relevant",   "What are the three lines of defence in operational risk?"),
    ("relevant",   "What is customer due diligence?"),
    ("relevant",   "What measures apply to politically exposed persons?"),
    ("irrelevant", "What is the capital of France?"),
    ("irrelevant", "How do I bake a chocolate cake?"),
    ("irrelevant", "What is the approval limit for expenses?"),
    ("irrelevant", "What is the office dress code?"),
    ("irrelevant", "How do I reset my VPN password?"),
    ("irrelevant", "What time does the staff cafeteria open?"),
]

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
client = weaviate.connect_to_local(host="localhost", port=8080)
try:
    collection = client.collections.get("BankKnowledge")
    rows = []
    for label, question in QUESTIONS:
        vector = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
        candidates = collection.query.hybrid(query=question, vector=vector, alpha=0.5, limit=20).objects
        res = co.rerank(model=RERANK_MODEL, query=question,
                        documents=[str(o.properties["content"]) for o in candidates], top_n=1)
        time.sleep(7)   # trial key: max 10 Cohere calls per minute
        best = res.results[0]
        source = candidates[best.index].properties
        rows.append((best.relevance_score, label, f"{source['source_file']} p.{source['page_number']}", question))

    print(f"\n{'relevance':>9}  {'kind':10}  {'best source':32}  question")
    for score, label, source, question in sorted(rows, reverse=True):
        print(f"{score:9.3f}  {label:10}  {source:32}  {question}")

    lowest_relevant = min(s for s, l, *_ in rows if l == "relevant")
    highest_irrelevant = max(s for s, l, *_ in rows if l == "irrelevant")
    print(f"\nLowest relevant:     {lowest_relevant:.3f}")
    print(f"Highest irrelevant:  {highest_irrelevant:.3f}")
    print("✅ Clear gap." if lowest_relevant > highest_irrelevant else "⚠️  They overlap.")
finally:
    client.close()
