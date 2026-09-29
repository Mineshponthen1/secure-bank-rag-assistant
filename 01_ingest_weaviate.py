import os
import pypdf
import weaviate
from weaviate.classes.config import Configure, DataType, Property
import ollama

client = weaviate.connect_to_local(host="localhost", port=8080)
COLLECTION_NAME = "BankKnowledge"
EMBEDDING_MODEL = "nomic-embed-text"

def get_permissions_for_file(filename):
    filename_lower = filename.lower()
    
    # Auto-detect department based on filename keywords
    if "hr" in filename_lower or "employee" in filename_lower:
        return ["HR"]
    elif "operation" in filename_lower or "vault" in filename_lower:
        return ["Operations"]
    elif "finance" in filename_lower or "budget" in filename_lower:
        return ["Finance"]
    elif "admin" in filename_lower:
        return ["Operations", "Finance"] # Example multi-department permission
    else:
        # Default fallback: accessible to all departments
        return ["HR", "Operations", "Finance"]
        
def reset_and_ingest():
    # 1. Delete existing collection to wipe old data
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        print(f"🗑️ Deleted existing '{COLLECTION_NAME}' collection.")

    # 2. Re-create collection schema
    collection = client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="allowed_departments", data_type=DataType.TEXT_ARRAY),
        ]
    )
    print(f"✨ Created fresh '{COLLECTION_NAME}' collection with RBAC schema.")

    # 3. Ingest PDF files with auto-detected department tagging
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in the current directory!")
        return

    for pdf_file in pdf_files:
        allowed_deps = get_permissions_for_file(pdf_file)
        print(f"\n📄 Processing: {pdf_file} (Allowed Departments: {allowed_deps})")

        reader = pypdf.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        print(f"  └─ Total pages: {total_pages}")

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text.strip():
                continue

            # Generate embedding via local Ollama
            vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)["embedding"]

            # Store chunk with explicit department metadata
            collection.data.insert(
                properties={
                    "content": text,
                    "source_file": pdf_file,
                    "page_number": page_num,
                    "allowed_departments": allowed_deps
                },
                vector=vector
            )
            
            if page_num % 10 == 0 or page_num == total_pages:
                print(f"  └─ Indexed page {page_num}/{total_pages}")

    print("\n✅ Ingestion complete with strict RBAC rules!")

if __name__ == "__main__":
    try:
        reset_and_ingest()
    finally:
        client.close()