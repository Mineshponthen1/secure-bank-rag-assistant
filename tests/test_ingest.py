"""Chunking rules from ingest.py (no Weaviate or Ollama needed)."""
from ingest import MAX_TOKENS, build_chunks, clean, estimate_tokens


def test_dotted_contents_lines_are_removed():
    # Real failure we found: dotted lines in a contents page crashed the embedding model
    assert "...." not in clean("1. Introduction ................................................ 1")


def test_token_estimate_is_not_fooled_by_symbols():
    # One 'word' made of 400 dots must still count as big
    assert estimate_tokens("." * 400) >= 100


def make_sentences(n=200):
    return [(f"Sentence {i} explains a rule about annual leave and its conditions in some detail.", 1 + i // 20)
            for i in range(n)]


def test_chunks_stay_within_size_limit():
    for chunk in build_chunks(make_sentences()):
        assert estimate_tokens(chunk["text"]) <= MAX_TOKENS * 1.05   # small margin for rounding


def test_neighbouring_chunks_overlap():
    chunks = build_chunks(make_sentences())
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = current["text"].split(". ")[0]
        assert first_sentence in previous["text"]


def test_chunks_record_their_page_range():
    for chunk in build_chunks(make_sentences()):
        assert chunk["page_start"] <= chunk["page_end"]
