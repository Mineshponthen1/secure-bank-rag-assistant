"""Citation checks used by app.py.

Kept in their own file so they can be tested on GitHub without Weaviate, Ollama or Cohere.
"""
import re

STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "has", "have",
    "was", "were", "will", "your", "you", "our", "its", "any", "all", "not",
    "according", "also", "additionally", "which", "their", "they", "must",
}

# So "thirty days" in a source matches "30 days" in an answer
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40", "forty-five": "45",
    "fifty": "50", "sixty": "60", "ninety": "90", "hundred": "100",
}


def key_words(text: str) -> set:
    """Lowercase words of 3+ letters, minus common filler words."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 3 and w not in STOPWORDS}


def numbers_in(text: str) -> set:
    """All numbers in the text, whether written as digits or words."""
    t = text.lower()
    found = set(re.findall(r"\d+", t))
    for word, digit in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            found.add(digit)
    return found


def split_sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def best_match(claim_words: set, claim_numbers: set, content: str):
    """Find the passage in content that best matches the claim. Returns (passage, score)."""
    sentences = split_sentences(content)
    candidates = sentences + [a + " " + b for a, b in zip(sentences, sentences[1:])]
    best, best_score = None, 0
    for cand in candidates:
        score = (len(claim_words & key_words(cand))
                 + 3 * len(claim_numbers & numbers_in(cand)))   # numbers count triple
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def verify_citations(answer_text: str, results: list) -> str:
    """Check each cited claim against its source, and against the other sources.
    Verified     -> highlight the supporting passage.
    Not verified -> mark the footnote as [n?] so the page can warn the reader."""
    by_ref = {r["ref"]: r for r in results}
    for r in results:
        r["highlights"] = []
        r["verified"] = False

    # Move footnotes that come after a full stop to before it: "year. [1]" -> "year [1]."
    tidy = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)",
                  lambda m: " " + m.group(2).strip() + m.group(1) + " ",
                  answer_text)

    for claim in split_sentences(tidy):
        refs = sorted({int(n) for n in re.findall(r"\[(\d+)\]", claim)})
        plain = re.sub(r"\[\d+\]", "", claim)
        claim_words, claim_numbers = key_words(plain), numbers_in(plain)
        if not refs or not claim_words:
            continue
        max_score = len(claim_words) + 3 * len(claim_numbers)
        short = " ".join(plain.split())[:60]

        new_claim = claim
        for n in refs:
            r = by_ref.get(n)
            if not r:
                continue
            best, score = best_match(claim_words, claim_numbers, r["content"])
            ratio = score / max_score

            # Is another retrieved source a much better match for this claim?
            other_ref, other_ratio = None, 0.0
            for other in results:
                if other["ref"] == n:
                    continue
                _, s = best_match(claim_words, claim_numbers, other["content"])
                if s / max_score > other_ratio:
                    other_ref, other_ratio = other["ref"], s / max_score
            better_elsewhere = other_ratio >= ratio + 0.20   # starting margin, to be tuned in Phase 3

            if best and ratio >= 0.5 and not better_elsewhere:
                r["verified"] = True
                if best not in r["highlights"]:
                    r["highlights"].append(best)
                print(f"[verify] [{n}] ✓ {ratio:.0%}  {short}")
            else:
                new_claim = new_claim.replace(f"[{n}]", f"[{n}?]")
                hint = f"  (source [{other_ref}] matches {other_ratio:.0%})" if better_elsewhere else ""
                print(f"[verify] [{n}] ✗ {ratio:.0%}{hint}  {short}")

        if new_claim != claim:
            tidy = tidy.replace(claim, new_claim, 1)

    return tidy


def remove_invalid_refs(answer_text: str, results: list):
    """Remove citation numbers that don't match any source, and mark which sources were cited.
    Returns (cleaned answer, valid cited refs, removed refs)."""
    valid_refs = {r["ref"] for r in results}
    cited_refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    invalid_refs = cited_refs - valid_refs

    for n in invalid_refs:                         # remove made-up source numbers (and the space before them)
        answer_text = re.sub(rf"\s*\[{n}\]", "", answer_text)
    for r in results:                              # mark which sources were actually used
        r["cited"] = r["ref"] in cited_refs

    return answer_text, sorted(cited_refs & valid_refs), sorted(invalid_refs)
