"""Tests for the regression gate (regression/check_regression.py).

1. The committed promptfoo run must pass the gate against the approved baseline.
   If someone commits a run where cost went up or citation accuracy went down, CI fails here.
2. "Fire drills": fake runs prove the gate catches each kind of regression, and
   tolerates small normal run-to-run wobble. No model, database or network needed.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_regression", ROOT / "regression" / "check_regression.py")
assert spec is not None and spec.loader is not None, "regression/check_regression.py not found"
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

LATEST = ROOT / "regression" / "results" / "latest.json"
BASELINE = ROOT / "regression" / "baseline.json"


def row(category="hr", success=True, refused=False, citations=2, verified=2, cost=0.00002, tokens=2000):
    """One fake promptfoo result, shaped like the real results file."""
    return {
        "success": success,
        "vars": {"category": category},
        "testCase": {"description": f"fake-{category}"},
        "response": {
            "cost": cost,
            "tokenUsage": {"total": tokens},
            "metadata": {"refused": refused, "citations": citations,
                         "verified_citations": verified, "sources": []},
        },
    }


def summary(tmp_path, rows):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"evalId": "fake", "results": {"results": rows}}), encoding="utf-8")
    return gate.summarize(path)


def baseline_rows():
    return [row() for _ in range(10)] + [row(category="security")]


@pytest.mark.skipif(not (LATEST.exists() and BASELINE.exists()), reason="no promptfoo run or baseline committed")
def test_committed_run_passes_the_gate():
    now = gate.summarize(LATEST)
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = gate.find_problems(now, base)
    assert problems == [], "Regression gate failed: " + "; ".join(problems)


def test_identical_run_passes(tmp_path):
    base = summary(tmp_path, baseline_rows())
    now = summary(tmp_path, baseline_rows())
    assert gate.find_problems(now, base) == []


def test_small_cost_wobble_is_tolerated(tmp_path):
    base = summary(tmp_path, baseline_rows())
    now = summary(tmp_path, [row(cost=0.00002 * 1.05) for _ in range(10)] + [row(category="security")])
    assert gate.find_problems(now, base) == []


def test_cost_increase_fails(tmp_path):
    base = summary(tmp_path, baseline_rows())
    now = summary(tmp_path, [row(cost=0.00002 * 1.5) for _ in range(10)] + [row(category="security")])
    assert any("token cost" in p for p in gate.find_problems(now, base))


def test_citation_accuracy_drop_fails(tmp_path):
    base = summary(tmp_path, baseline_rows())
    now = summary(tmp_path, [row(verified=1) for _ in range(10)] + [row(category="security")])
    assert any("citation accuracy" in p for p in gate.find_problems(now, base))


def test_permission_leak_fails(tmp_path):
    base = summary(tmp_path, baseline_rows())
    now = summary(tmp_path, [row() for _ in range(10)] + [row(category="security", success=False)])
    assert any("PERMISSION LEAK" in p for p in gate.find_problems(now, base))