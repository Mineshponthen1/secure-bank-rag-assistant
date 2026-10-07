"""Score a saved report card with Braintrust autoevals, using a local judge.

Adds two scores (0 to 1) to every answered question:
  faithfulness      - does the answer only say things found in the retrieved paragraphs?
  answer_relevance  - does the answer actually address the question?
Citation accuracy is already checked by citations.py (the verified citations in the report card).

Local judge notes:
  - Judge: phi4-mini via Ollama. Chosen with eval/judge_check.py: it scored a faithful answer 1.0,
    an unfaithful one 0.0, and was the fastest of the installed models.
  - judge_compat.py lets autoevals read the judge's JSON when it is written as plain text.
  - Answer relevance asks for OpenAI's "text-embedding-ada-002". On this laptop that name is an
    Ollama alias of nomic-embed-text (ollama cp nomic-embed-text text-embedding-ada-002).
"""
import argparse
import json
import pathlib
import statistics
import time

from openai import OpenAI
from autoevals import init
from autoevals.ragas import AnswerRelevancy, Faithfulness

import judge_compat  # noqa: F401  (lets autoevals read JSON written as plain text)

HERE = pathlib.Path(__file__).parent
JUDGE_MODEL = "phi4-mini:latest"
OLLAMA_URL = "http://localhost:11434/v1"


def fmt(value):
    return "  n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Add autoevals scores to a report card.")
    parser.add_argument("--file", help="results file to score (default: the newest one in eval/results)")
    parser.add_argument("--limit", type=int, help="only score the first N answers (quick test, not saved)")
    args = parser.parse_args()

    path = pathlib.Path(args.file) if args.file else sorted((HERE / "results").glob("*.json"))[-1]
    data = json.loads(path.read_text(encoding="utf-8"))

    to_score = [a for a in data["answers"] if a.get("contexts") and a.get("answer")]
    if not to_score:
        print(f"{path.name} has no saved paragraphs. Run eval/run_eval.py again first.")
        return
    if args.limit:
        to_score = to_score[:args.limit]

    # Point autoevals at the local Ollama server instead of OpenAI
    init(client=OpenAI(base_url=OLLAMA_URL, api_key="ollama"))
    scorers = {
        "faithfulness": Faithfulness(model=JUDGE_MODEL),
        # strictness=1: the judge writes 1 test question per answer instead of 3, to save time on a CPU
        "answer_relevance": AnswerRelevancy(model=JUDGE_MODEL, strictness=1),
    }

    print(f"Scoring {len(to_score)} answers from {path.name} with judge {JUDGE_MODEL}\n")
    for i, a in enumerate(to_score, 1):
        start = time.perf_counter()
        for name, scorer in scorers.items():
            try:
                # This autoevals version reads the answer from "expected" in some steps, so pass it as both
                result = scorer(input=a["question"], output=a["answer"], expected=a["answer"],
                                context=a["contexts"])
                a[name] = None if result.score is None else round(result.score, 3)
                a.pop(f"{name}_error", None)
            except Exception as e:
                a[name] = None
                a[f"{name}_error"] = str(e)[:200]
        seconds = time.perf_counter() - start
        print(f"[{i}/{len(to_score)}] {a['id']:34} faithfulness {fmt(a['faithfulness'])}   "
              f"relevance {fmt(a['answer_relevance'])}   ({seconds:.0f}s)")

    summary = {"judge_model": JUDGE_MODEL, "scored": len(to_score)}
    for name in scorers:
        values = [a[name] for a in to_score if a.get(name) is not None]
        summary[name] = round(statistics.mean(values), 3) if values else None
        summary[f"{name}_judge_failures"] = len(to_score) - len(values)

    print("\n=========== AUTOEVALS SCORES ===========")
    print(f"Faithfulness:       {fmt(summary['faithfulness'])}   (judge failures: {summary['faithfulness_judge_failures']})")
    print(f"Answer relevance:   {fmt(summary['answer_relevance'])}   (judge failures: {summary['answer_relevance_judge_failures']})")
    print("========================================")

    if args.limit:
        print("Quick test only (--limit): nothing saved.")
        return
    data["autoevals"] = summary
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved scores into {path}")


if __name__ == "__main__":
    main()
