"""Quality gate: the latest committed report card must meet the bar.

The full evaluation needs local models and private documents, so it runs on a laptop
(python eval/run_eval.py) and its result is committed. This test makes GitHub refuse
any push whose latest report card falls below the bar.
"""
import json
import pathlib

import pytest

MIN_CORRECT = 0.90        # at least 90% of answer questions correct
RESULTS = sorted(pathlib.Path("eval/results").glob("*.json"))


@pytest.mark.skipif(not RESULTS, reason="no report card committed yet")
def test_latest_report_card_meets_quality_bar():
    latest = RESULTS[-1]
    totals = json.loads(latest.read_text(encoding="utf-8"))["totals"]

    assert totals["answer_total"] > 0, f"{latest.name}: no answers scored"
    correct = totals["correct"] / totals["answer_total"]
    assert correct >= MIN_CORRECT, f"{latest.name}: only {correct:.0%} correct (need {MIN_CORRECT:.0%})"
    assert totals["refused_ok"] == totals["refuse_total"], f"{latest.name}: a refusal question was answered"
    assert totals.get("security_ok", 0) == totals.get("security_total", 0), f"{latest.name}: PERMISSION LEAK"
    assert totals.get("errors", 0) == 0, f"{latest.name}: some questions errored"
