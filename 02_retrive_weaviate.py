import weaviate
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(host="localhost", port=8080)
COLLECTION_NAME = "BankKnowledge"


def search_bank_policies(query: str, user_department: str, limit: int = 3):
    collection = client.collections.get(COLLECTION_NAME)

    print(f"\n==================================================")
    print(f"🔍 Search Query: '{query}'")
    print(f"👤 User Department Context: '{user_department}'")
    print(f"==================================================")

    # Filter checks if the array 'allowed_departments' contains the user's department
    rbac_filter = Filter.by_property("allowed_departments").contains_any([user_department])

    results = collection.query.bm25(
        query=query,
        filters=rbac_filter,
        limit=limit,
    )

    if not results.objects:
        print("⛔ Access Denied or No Matching Documents Found.")
        return

    for idx, obj in enumerate(results.objects, start=1):
        props = obj.properties
        content_text = str(props.get("content", ""))
        
        print(f"\nResult #{idx}:")
        print(f"  📄 Source File : {props.get('source_file')}")
        print(f"  📍 Page Number : {props.get('page_number')}")
        print(f"  🔒 Allowed Dept: {props.get('allowed_departments')}")
        print(f"  📝 Content     : {content_text[:180]}...")


if __name__ == "__main__":
    search_bank_policies(
        query="employee policy conduct",
        user_department="HR",
    )

    search_bank_policies(
        query="approval expenditure limit",
        user_department="Finance",
    )

    search_bank_policies(
        query="financial audit threshold",
        user_department="Operations",
    )

    client.close()

    # Ensure strict department mapping during ingestion
pdf_department_map = {
    "hr_policy.pdf": ["HR"],
    "operations_policy.pdf": ["Operations"],
    "finance_policy.pdf": ["Finance"]
}