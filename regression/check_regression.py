"""Regression gate: compare the latest promptfoo run with the approved baseline.

Fails (exit code 1) if a change made the assistant:
  - more expensive per answered question (token cost up more than MAX_COST_INCREASE), or
  - worse at citations (verified-citation rate down more than MAX_CITATION_DROP), or
  - leak documents across departments, or fall below MIN_PASS_RATE overall.
The result is also written to Langfuse as a "regression-check" trace (flagged ERROR on failure).

Usage:
  python regression/check_regression.py                  compare latest results with the baseline
  python regression/check_regression.py --set-baseline   approve the latest results as the new baseline
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "latest.json"
BASELINE = HERE / "baseline.json"
PROMPT_FILE = HERE.parent / "prompts" / "rag_answer.yaml"

MAX_COST_INCREASE = 0.10   # fail if cost per answered question rises more than 10%
MAX_CITATION_DROP = 0.05   # fail if the verified-citation rate drops more than 5 percentage points
MIN_PASS_RATE = 0.90       # fail if fewer than 90% of golden tests pass


def metadata(row):
    return (row.get("response") or {}).get("metadata") or {}


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]["results"]
    answered = [r for r in rows if metadata(r) and not metadata(r).get("refused")]
    citations = sum(metadata(r).get("citations", 0) for r in answered)
    verified = sum(metadata(r).get("verified_citations", 0) for r in answered)
    cost = sum((r.get("response") or {}).get("cost") or 0 for r in answered)
    tokens = sum(((r.get("response") or {}).get("tokenUsage") or {}).get("total", 0) for r in answered)
    security = [r for r in rows if (r.get("vars") or {}).get("category") == "security"]
    passed = sum(1 for r in rows if r.get("success"))
    return {
        "eval_id": data.get("evalId"),
        "tests": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "answered": len(answered),
        "tokens_per_answer": round(tokens / len(answered), 1) if answered else 0.0,
        "cost_per_answer_usd": cost / len(answered) if answered else 0.0,
        "citations": citations,
        "verified_citations": verified,
        "citation_accuracy": round(verified / citations, 4) if citations else 0.0,
        "security_failures": sum(1 for r in security if not r.get("success")),
        "failed_tests": [r["testCase"]["description"] for r in rows if not r.get("success")],
    }


def find_problems(now, base):
    problems = []
    if now["security_failures"]:
        problems.append(f"PERMISSION LEAK in {now['security_failures']} security test(s)")
    if now["pass_rate"] < MIN_PASS_RATE:
        problems.append(f"pass rate {now['pass_rate']:.0%} is below {MIN_PASS_RATE:.0%}")
    if base["cost_per_answer_usd"]:
        change = now["cost_per_answer_usd"] / base["cost_per_answer_usd"] - 1
        if change > MAX_COST_INCREASE:
            problems.append(f"token cost per answer up {change:.0%} (limit +{MAX_COST_INCREASE:.0%})")
    drop = base["citation_accuracy"] - now["citation_accuracy"]
    if drop > MAX_CITATION_DROP:
        problems.append(f"citation accuracy down {drop * 100:.1f} points (limit {MAX_CITATION_DROP * 100:.0f})")
    return problems


def flag_in_langfuse(now, base, problems, version):
    """Record the gate result in Langfuse. Skipped quietly where Langfuse isn't available (e.g. CI)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE.parent / ".env")
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            print("[langfuse] no keys found: skipped")
            return
        from langfuse import get_client, propagate_attributes
        lf = get_client()
        with propagate_attributes(trace_name="regression-check",
                                  tags=["regression", f"prompt-v{version}"], version=str(version)):
            root = lf.start_observation(name="regression-check", as_type="span")
        root.update(input={"baseline": base}, output={"now": now, "problems": problems},
                    level="ERROR" if problems else "DEFAULT",
                    status_message="; ".join(problems)[:500] if problems else "no regression")
        root.score_trace(name="regression_gate", value="fail" if problems else "pass")
        root.score_trace(name="gate_citation_accuracy", value=float(now["citation_accuracy"]))
        root.score_trace(name="gate_cost_per_answer_usd", value=float(now["cost_per_answer_usd"]))
        root.end()
        lf.flush()
        print("[langfuse] recorded as a 'regression-check' trace")
    except Exception as e:
        print(f"[langfuse] skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Regression gate for the bank RAG assistant.")
    parser.add_argument("--set-baseline", action="store_true", help="approve the latest results as the baseline")
    parser.add_argument("--results", default=str(RESULTS), help="promptfoo results file to check")
    args = parser.parse_args()

    now = summarize(Path(args.results))
    version = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))["version"]

    if args.set_baseline:
        baseline = {**now, "prompt_version": version, "approved_on": datetime.date.today().isoformat()}
        BASELINE.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"Baseline saved for prompt v{version}: {now['passed']}/{now['tests']} passed, "
              f"citation accuracy {now['citation_accuracy']:.1%}, "
              f"cost per answer ${now['cost_per_answer_usd']:.7f}")
        return 0

    if not BASELINE.exists():
        print("No baseline yet. Approve one first: python regression/check_regression.py --set-baseline")
        return 1
    base = json.loads(BASELINE.read_text(encoding="utf-8"))

    print(f"Regression check: prompt v{version} vs baseline (prompt v{base['prompt_version']}, {base['approved_on']})\n")
    print(f"{'Measure':<26}{'Baseline':>16}{'Now':>16}")
    print("-" * 58)
    print(f"{'Tests passed':<26}{base['passed']:>13}/{base['tests']:<2}{now['passed']:>13}/{now['tests']:<2}")
    print(f"{'Citation accuracy':<26}{base['citation_accuracy']:>16.1%}{now['citation_accuracy']:>16.1%}")
    print(f"{'Tokens per answer':<26}{base['tokens_per_answer']:>16,.0f}{now['tokens_per_answer']:>16,.0f}")
    print(f"{'Cost per answer (USD)':<26}{base['cost_per_answer_usd']:>16.7f}{now['cost_per_answer_usd']:>16.7f}")
    print(f"{'Security failures':<26}{base['security_failures']:>16}{now['security_failures']:>16}")
    print(f"\nFailed tests now: {', '.join(now['failed_tests']) or 'none'}\n")

    problems = find_problems(now, base)
    flag_in_langfuse(now, base, problems, version)

    if problems:
        print("\nREGRESSION GATE: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nREGRESSION GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())