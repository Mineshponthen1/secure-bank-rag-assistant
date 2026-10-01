"""The citation verifier must keep catching wrong citations.

These are regression tests: each one locks in a real failure we found and fixed,
so a future change can't silently bring it back.
"""
from citations import numbers_in, remove_invalid_refs, verify_citations

MATERNITY = ("Article (30) Maternity Leave 1. The female Worker shall be entitled to maternity leave "
             "of (60) sixty days, according to the following: a. The first forty-five (45) days with full pay.")
ANNUAL_LEAVE = ("Article (29) Annual leave 1. The Worker shall be entitled to an annual leave with full pay "
                "of not less than: a. Thirty days for each year of his extended service. b. Two days for "
                "each month if his service period is more than six months and less than one year.")


def sources():
    return [
        {"ref": 1, "content": MATERNITY},
        {"ref": 2, "content": ANNUAL_LEAVE},
    ]


def test_numbers_written_as_words_match_digits():
    found = numbers_in("thirty (30) days, then forty-five days")
    assert {"30", "45"} <= found


def test_correct_citation_is_verified_and_highlighted():
    results = sources()
    answer = verify_citations("A worker gets at least 30 days of annual leave for each year of extended service [2].", results)
    assert "[2?]" not in answer
    assert results[1]["verified"] is True
    assert any("Thirty days" in h for h in results[1]["highlights"])


def test_article_number_trap_is_caught():
    """Real failure we found: the '30' in 'Article (30) Maternity Leave' was once accepted
    as support for '30 days of annual leave'. The cross-source check must flag it."""
    results = sources()
    answer = verify_citations("A worker gets at least 30 days of annual leave for each year of extended service [1].", results)
    assert "[1?]" in answer
    assert results[0]["highlights"] == []


def test_footnote_after_full_stop_still_works():
    results = sources()
    answer = verify_citations("A worker gets at least 30 days of annual leave for each year of extended service. [2]", results)
    assert "[2]" in answer and "[2?]" not in answer


def test_made_up_source_numbers_are_removed():
    results = sources()
    cleaned, cited, removed = remove_invalid_refs("Leave is thirty days [29]. Maternity is sixty days [1].", results)
    assert "[29]" not in cleaned
    assert "days." in cleaned          # no stray space left before the full stop
    assert removed == [29] and cited == [1]
    assert results[0]["cited"] is True and results[1]["cited"] is False
