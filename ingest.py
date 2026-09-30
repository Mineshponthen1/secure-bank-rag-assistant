import glob
import os
import re
import sys

import ollama
import pypdf
import weaviate
from weaviate.classes.config import Configure, DataType, Property

# ==========================================
# SETTINGS (chosen from our page measurements)
# ==========================================
DOCS_DIR = "documents"
COLLECTION_NAME = "BankKnowledge"
EMBEDDING_MODEL = "nomic-embed-text"

TARGET_TOKENS = 600    # aim for the middle of the roadmap's 500-800 range
MAX_TOKENS = 800       # never go above this
OVERLAP_TOKENS = 100   # repeat ~100 tokens from the end of the previous chunk

# The folder a PDF sits in decides who can read it
FOLDER_PERMISSIONS = {
    "HR": ["HR"],
    "Finance": ["Finance"],
    "Operations": ["Operations"],
    "General": ["HR", "Finance", "Operations"],
}

PREVIEW = "--preview" in sys.argv   # python ingest.py --preview  -> look, don't store


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~3/4 of a word per token, or ~4 characters per token, whichever is bigger."""
    return int(max(len(text.split()) * 1.33, len(text) / 4))


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)                  # re-join words hyphenated across lines
    text = re.sub(r"(?:[.·…_\-]\s?){4,}", " ", text)        # remove dotted/dashed lines in contents pages
    return re.sub(r"\s+", " ", text).strip()               # collapse extra spaces and line breaks

def read_pdf_sentences(path: str) -> list:
    """Return [(sentence, page_number), ...] for the whole PDF."""
    reader = pypdf.PdfReader(path)
    items = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if not text:
            continue                                # skip blank / image-only pages
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            words = sentence.split()
            # Very long "sentences" (tables, lists without full stops) get cut into pieces
            piece = int(TARGET_TOKENS / 1.33 / 2)
            while estimate_tokens(" ".join(words)) > MAX_TOKENS:
                items.append((" ".join(words[:piece]), page_num))
                words = words[piece:]
            if words:
                items.append((" ".join(words), page_num))
    return items


def build_chunks(items: list) -> list:
    """Group sentences into ~TARGET_TOKENS chunks with ~OVERLAP_TOKENS overlap."""
    chunks, current, current_tokens = [], [], 0

    for sentence, page in items:
        t = estimate_tokens(sentence)
        if current and (current_tokens >= TARGET_TOKENS or current_tokens + t > MAX_TOKENS):
            chunks.append(current)
            # Start the next chunk with the last ~100 tokens of this one (the overlap)
            overlap, overlap_tokens = [], 0
            for s, p in reversed(current):
                if overlap_tokens + estimate_tokens(s) > OVERLAP_TOKENS:
                    break
                overlap.insert(0, (s, p))
                overlap_tokens += estimate_tokens(s)
            current, current_tokens = overlap, overlap_tokens
        current.append((sentence, page))
        current_tokens += t

    if current:
        chunks.append(current)

    return [{
        "text": " ".join(s for s, _ in c),
        "page_start": c[0][1],
        "page_end": c[-1][1],
    } for c in chunks]


def shared_words(a: str, b: str) -> int:
    """How many words at the end of chunk a are repeated at the start of chunk b."""
    wa, wb = a.split(), b.split()
    for n in range(min(len(wa), len(wb)), 0, -1):
        if wa[-n:] == wb[:n]:
            return n
    return 0
def main():
    pdfs = sorted(glob.glob(os.path.join(DOCS_DIR, "*", "*.pdf")))
    if not pdfs:
        print(f"⚠️  No PDFs found in '{DOCS_DIR}/<Department>/'")
        return

    client, collection = None, None
    if not PREVIEW:
        client = weaviate.connect_to_local(host="localhost", port=8080)
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source_file", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),   # first page of the chunk
                Property(name="page_end", data_type=DataType.INT),      # last page of the chunk
                Property(name="allowed_departments", data_type=DataType.TEXT_ARRAY),
            ],
        )
        print(f"✨ Fresh '{COLLECTION_NAME}' collection created.")

    try:
        for path in pdfs:
            folder = os.path.basename(os.path.dirname(path))
            departments = FOLDER_PERMISSIONS.get(folder)
            if not departments:
                print(f"⚠️  Skipping {path}: no permission rule for folder '{folder}'")
                continue

            chunks = build_chunks(read_pdf_sentences(path))
            sizes = [estimate_tokens(c["text"]) for c in chunks]
            print(f"\n📄 {path}  →  {len(chunks)} chunks  "
                  f"(≈tokens avg {sum(sizes) // len(sizes)}, min {min(sizes)}, max {max(sizes)})  "
                  f"readable by {departments}")

            if PREVIEW:
                overlaps = [shared_words(chunks[i]["text"], chunks[i + 1]["text"])
                            for i in range(len(chunks) - 1)]
                if overlaps:
                    none = sum(1 for o in overlaps if o == 0)
                    avg = sum(overlaps) / len(overlaps)
                    print(f"   overlap: avg ≈{avg * 1.33:.0f} tokens between neighbours, "
                          f"{none} of {len(overlaps)} pairs with no overlap")
                    if overlaps[0]:
                        example = " ".join(chunks[1]["text"].split()[:min(overlaps[0], 20)])
                        print(f"   example of shared text: \"{example}...\"")
                continue

            assert collection is not None   # always set when not in preview mode
            for i, c in enumerate(chunks, start=1):
                try:
                    vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=c["text"])["embedding"]
                except ollama.ResponseError as e:
                    print(f"   ⚠️  Skipped chunk {i} (pages {c['page_start']}-{c['page_end']}): {e}")
                    print(f"      It starts with: {c['text'][:120]}...")
                    continue
                collection.data.insert(
                    properties={
                        "content": c["text"],
                        "source_file": os.path.basename(path),
                        "page_number": c["page_start"],
                        "page_end": c["page_end"],
                        "allowed_departments": departments,
                    },
                    vector=vector,
                )
                if i % 25 == 0 or i == len(chunks):
                    print(f"   └─ embedded {i}/{len(chunks)}")

        if collection is not None:
            total = collection.aggregate.over_all(total_count=True).total_count
            print(f"\n✅ Done. Total chunks stored: {total}")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()