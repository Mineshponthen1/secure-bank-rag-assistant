"""Pick a judge: can each local model tell a faithful answer from an unfaithful one?"""
import sys
import time

from openai import OpenAI
from autoevals import init
from autoevals.ragas import AnswerRelevancy, Faithfulness

import judge_compat  # noqa: F401  (lets autoevals read JSON written as plain text)

init(client=OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))

QUESTION = "How many days of annual leave does a worker get?"
CONTEXT = ["Article 29: The worker is entitled to annual leave of 30 days for each year of service."]
GOOD = "A worker gets 30 days of annual leave per year [1]."
BAD = "A worker gets 45 days of annual leave per year [1]."

models = sys.argv[1:] or ["phi4-mini:latest", "mistral:latest"]
for model in models:
    print(f"\n{model}")
    checks = [
        ("faithful answer    (should be ~1)", Faithfulness(model=model), GOOD),
        ("unfaithful answer  (should be ~0)", Faithfulness(model=model), BAD),
        ("answer relevance   (should be high)",
         AnswerRelevancy(model=model, embedding_model="nomic-embed-text", strictness=1), GOOD),
    ]
    for label, scorer, answer in checks:
        start = time.perf_counter()
        try:
            result = scorer(input=QUESTION, output=answer, expected=answer, context=CONTEXT)
            outcome = f"{result.score:.2f}"
        except Exception as e:
            outcome = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"  {label}  {outcome}   ({time.perf_counter() - start:.0f}s)")
