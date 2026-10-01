"""The golden dataset itself must be well-formed."""
import pathlib

import yaml

QUESTIONS = yaml.safe_load(pathlib.Path("eval/questions.yaml").read_text(encoding="utf-8"))
VALID_EXPECT = {"answer", "refuse", "any"}
VALID_USERS = {"ABC1", "ABC2", "ABC3"}


def test_at_least_50_questions():
    assert len(QUESTIONS) >= 50


def test_ids_are_unique():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_is_complete():
    for q in QUESTIONS:
        assert q["question"].strip(), q["id"]
        assert q["user"] in VALID_USERS, q["id"]
        assert q["expect"] in VALID_EXPECT, q["id"]
        if q["expect"] == "answer":
            assert q.get("must_contain_any"), f"{q['id']} needs an answer key"
        if q["expect"] == "any":
            assert q.get("forbidden_sources"), f"{q['id']} needs forbidden_sources"


def test_every_category_is_covered():
    categories = {q["category"] for q in QUESTIONS}
    assert {"hr", "general", "finance", "operations", "refusal", "security"} <= categories
