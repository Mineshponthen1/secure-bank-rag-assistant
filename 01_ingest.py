import os
import pypdf
import weaviate
from weaviate.classes.config import Configure, DataType, Property

# 1. Connect to Local Weaviate Container
client = weaviate.connect_to_local(host="localhost", port=8080)

COLLECTION_NAME = "BankKnowledge"


def setup_schema():
    """Defines schema with text properties for BM25 keyword search and vector indexing."""
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),  # We supply vectors via standard embedding pipeline
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="allowed_departments", data_type=DataType.TEXT),
        ],
    )


def clean_text(text: str) -> str:
    """Removes extra white spaces and formats raw extracted text cleanly."""
    return " ".join(text.split())


def chunk_text_by_sentences(text: str, target_chunk_size: int = 500, overlap: int = 100) -> list:
    """Sentence-Aware Chunking: Splits text on sentence boundaries ('. ')."""
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_str = sentence if sentence.endswith(".") else sentence + "."
        sentence_len = len(sentence_str)

        if current_length + sentence_len > target_chunk_size and current_chunk:
            completed_chunk = " ".join(current_chunk)
            chunks.append(completed_chunk)

            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_len

        current_chunk.append(sentence_str)
        current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def ingest_pdf_to_weaviate(pdf_filename: str, allowed_departments: str):
    """Ingestion Engine: Extracts text, chunks by sentence boundaries, and saves to Weaviate."""
    if not os.path.exists(pdf_filename):
        print(f"❌ File '{pdf_filename}' not found. Skipping...")
        return

    print(f"\n🚀 Ingesting into Weaviate: {pdf_filename}...")
    reader = pypdf.PdfReader(pdf_filename)
    collection = client.collections.get(COLLECTION_NAME)

    total_chunks_added = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        if not raw_text:
            continue

        cleaned_text = clean_text(raw_text)
        page_chunks = chunk_text_by_sentences(cleaned_text, target_chunk_size=500, overlap=100)

        for chunk in page_chunks:
            collection.data.insert(
                properties={
                    "content": chunk,
                    "source_file": pdf_filename,
                    "page_number": page_num,
                    "allowed_departments": allowed_departments,
                }
            )
            total_chunks_added += 1

        print(f"  └─ Page {page_num}: Stored {len(page_chunks)} sentence-aligned chunk(s).")

    print(f"✅ Ingestion complete for '{pdf_filename}'. Total new chunks stored: {total_chunks_added}")


if __name__ == "__main__":
    setup_schema()

    policy_documents = [
        {"filename": "finance_policy.pdf", "allowed_departments": "Finance,Executive"},
        {"filename": "hr_policy.pdf", "allowed_departments": "HR,Executive"},
        {"filename": "operations_policy.pdf", "allowed_departments": "Operations,Finance,HR,Executive"},
    ]

    for doc in policy_documents:
        ingest_pdf_to_weaviate(
            pdf_filename=doc["filename"],
            allowed_departments=doc["allowed_departments"],
        )

    collection = client.collections.get(COLLECTION_NAME)
    total_count = collection.aggregate.over_all(total_count=True).total_count
    print(f"\n📊 Total Vault Record Count in Weaviate: {total_count}")

    client.close()