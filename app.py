import os
import re
import sqlite3
import bcrypt
import jwt
import secrets
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import datetime
import pypdf
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
import ollama
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Secure Enterprise RAG Bank Assistant")

# CORS: only these websites may call this API from a browser
ALLOWED_ORIGINS = [
    "http://localhost:8000",   # this app's own pages
    "http://127.0.0.1:8000",   # same, by IP address
    "http://localhost:3000",   # future Next.js (React) front end
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,                       # we use wristbands, not cookies
    allow_methods=["GET", "POST"],                 # the only methods our API uses
    allow_headers=["Content-Type", "Authorization"],
)

# Connect to Local Weaviate & Ollama
client = weaviate.connect_to_local(host="localhost", port=8080)
COLLECTION_NAME = "BankKnowledge"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_DISTANCE = 0.43  # chunks further than this are not relevant (measured from test questions)
DB_FILE = "users_approval.db"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), stored_hash.encode())


# ==========================================
# TOKEN (WRISTBAND) SETTINGS
# ==========================================
SECRET_KEY = os.environ.get("JWT_SECRET") or secrets.token_hex(32)
TOKEN_HOURS = 8


def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


bearer = HTTPBearer()


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session. Please log in again.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT username, department, is_admin FROM users WHERE username = ?", (payload["sub"],))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return {"username": row[0], "department": row[1], "is_admin": bool(row[2])}


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Unauthorized: Admin access required.")
    return user


# ==========================================
# CITATION HIGHLIGHTING (match answer sentences to source sentences)
# ==========================================
STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "has", "have",
    "was", "were", "will", "your", "you", "our", "its", "any", "all", "not",
    "according", "also", "additionally", "which", "their", "they", "must",
}


def key_words(text: str) -> set:
    """Lowercase words of 3+ letters, minus common filler words."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 3 and w not in STOPWORDS}


def split_sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def find_supporting_sentences(answer_text: str, results: list) -> None:
    """For each cited source, find the source sentence that best matches the claim citing it."""
    by_ref = {r["ref"]: r for r in results}
    for r in results:
        r["highlights"] = []

        # Move footnotes that come after a full stop to before it: "year. [1]" -> "year [1]."
    tidy = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)",
                  lambda m: " " + m.group(2).strip() + m.group(1) + " ",
                  answer_text)

    for claim in split_sentences(tidy):
        refs = {int(n) for n in re.findall(r"\[(\d+)\]", claim)}
        claim_words = key_words(re.sub(r"\[\d+\]", "", claim))
        if not refs or not claim_words:
            continue

        for n in refs:
            r = by_ref.get(n)
            if not r:
                continue
            best, best_score = None, 0
            for sentence in split_sentences(r["content"]):
                score = len(claim_words & key_words(sentence))
                if score > best_score:
                    best, best_score = sentence, score
            if best and best_score >= 2 and best not in r["highlights"]:
                r["highlights"].append(best)


# ==========================================
# 1. DATABASE SETUP (USERS & PENDING TABLES)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Active approved users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            department TEXT,
            is_admin INTEGER
        )
    """)

    # Temporary pending onboarding requests table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            department TEXT
        )
    """)

    # Create default admin if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_hash = hash_password("adminpassword")
        cursor.execute("""
            INSERT INTO users (username, password_hash, department, is_admin)
            VALUES ('admin', ?, 'Admin', 1)
        """, (admin_hash,))

    conn.commit()
    conn.close()


init_db()


# Pydantic Schemas
class RegisterRequest(BaseModel):
    username: str
    password: str
    department: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ApprovalRequest(BaseModel):
    target_username: str
    assigned_department: str


class RejectRequest(BaseModel):
    target_username: str


class QueryRequest(BaseModel):
    query: str


# ==========================================
# 2. AUTHENTICATION & ADMIN ENDPOINTS
# ==========================================
@app.post("/api/register")
def register(data: RegisterRequest):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (data.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists in active accounts.")

        cursor.execute("SELECT id FROM pending_requests WHERE username = ?", (data.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="A registration request for this username is already pending.")

        hashed_pw = hash_password(data.password)
        cursor.execute("""
            INSERT INTO pending_requests (username, password_hash, department)
            VALUES (?, ?, ?)
        """, (data.username, hashed_pw, data.department))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists.")
    finally:
        conn.close()
    return {"message": "Registration successful! Account is pending admin approval."}


@app.post("/api/login")
def login(data: LoginRequest):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT username, password_hash, department, is_admin FROM users
        WHERE username = ?
    """, (data.username,))
    user = cursor.fetchone()

    if not user or not verify_password(data.password, user[1]):
        cursor.execute("SELECT id FROM pending_requests WHERE username = ?", (data.username,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=403, detail="Waiting for Admin approval.")
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    conn.close()
    username, _, department, is_admin = user

    return {
        "message": "Login successful",
        "username": username,
        "department": department,
        "is_admin": bool(is_admin),
        "token": create_token(username)
    }


@app.get("/api/admin/pending-users")
def get_pending_users(admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, department FROM pending_requests")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "department": r[2]} for r in rows]


@app.get("/api/admin/active-users")
def get_active_users(admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, department, is_admin FROM users")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "department": r[2], "is_admin": bool(r[3])} for r in rows]


@app.post("/api/admin/approve-user")
def approve_user(data: ApprovalRequest, admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT username, password_hash FROM pending_requests WHERE username = ?", (data.target_username,))
    pending = cursor.fetchone()
    if not pending:
        conn.close()
        raise HTTPException(status_code=404, detail="Pending request not found.")

    uname, pwd_hash = pending
    cursor.execute("""
        INSERT INTO users (username, password_hash, department, is_admin)
        VALUES (?, ?, ?, 0)
    """, (uname, pwd_hash, data.assigned_department))
    cursor.execute("DELETE FROM pending_requests WHERE username = ?", (data.target_username,))
    conn.commit()
    conn.close()
    return {"message": f"User '{data.target_username}' approved and assigned to department '{data.assigned_department}'."}


@app.post("/api/admin/reject-user")
def reject_user(data: RejectRequest, admin: dict = Depends(require_admin)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pending_requests WHERE username = ?", (data.target_username,))
    conn.commit()
    conn.close()
    return {"message": f"User '{data.target_username}' request rejected and username freed."}


# ==========================================
# 3. SECURE RAG QUERY ENDPOINT WITH RBAC & LLM
# ==========================================
@app.post("/api/query")
def secure_rag_query(data: QueryRequest, user: dict = Depends(get_current_user)):
    username = user["username"]
    department = user["department"]
    is_admin = user["is_admin"]

    cleaned_query = data.query.strip().lower()
    if cleaned_query in ["hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon"]:
        current_hour = datetime.datetime.now().hour

        if 5 <= current_hour < 12:
            time_greeting = "Good morning"
        elif 12 <= current_hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"

        greeting_text = f"{time_greeting}! Hello! I am your Secure Enterprise Banking Policy Assistant. How can I help you navigate our compliance and department guidelines today?"

        return {
            "user": username,
            "department": department,
            "answer": greeting_text,
            "results": []
        }

    # Build DB-Level Weaviate Security Filter
    if is_admin or department == "Admin":
        rbac_filter = None
    else:
        rbac_filter = Filter.by_property("allowed_departments").contains_any([department])

    vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=data.query)["embedding"]
    collection = client.collections.get(COLLECTION_NAME)

    response = collection.query.near_vector(
        near_vector=vector,
        filters=rbac_filter,
        limit=3,
        return_metadata=MetadataQuery(distance=True)
    )

    retrieved_contexts = []
    results = []
    print(f"\n[retrieval] Question: {data.query}")
    for obj in response.objects:
        content = obj.properties.get("content")
        source_file = obj.properties.get("source_file")
        page_number = obj.properties.get("page_number")
        allowed_depts = obj.properties.get("allowed_departments")
        distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0

        if distance > MAX_DISTANCE:
            print(f"[retrieval]   DROPPED {source_file} p.{page_number}  distance={distance:.3f}")
            continue
        print(f"[retrieval]   KEPT    {source_file} p.{page_number}  distance={distance:.3f}")

        # Give each kept source a number, like a footnote
        ref = len(results) + 1
        retrieved_contexts.append(f"[{ref}] (Source: {source_file}, page {page_number})\n{content}")
        results.append({
            "ref": ref,
            "content": content,
            "source_file": source_file,
            "page_number": page_number,
            "allowed_departments": allowed_depts,
            "distance": round(distance, 3)
        })

    # Refuse by code (not by hoping the AI notices) when no chunk is relevant enough
    if not retrieved_contexts:
        return {
            "user": username,
            "department": department,
            "answer": "I couldn't find this in the policy documents available to your department, so I can't answer it reliably. Please contact the relevant department directly.",
            "results": []
        }

    # Combine the numbered sources for the LLM
    context_str = "\n\n".join(retrieved_contexts)

    system_prompt = f"""You are a secure enterprise banking policy assistant for the {department} department.

Answer the user's question using ONLY the numbered sources below.

Citation rules:
- After every sentence that uses information from a source, add that source's number in square brackets, like [1] or [2].
- Only use the numbers of the sources listed below. Never invent a source number.
- If the sources do not contain the answer, say the information is not available in your authorized department guidelines, and do not add any citation.
- Do not make up information.

Sources:
{context_str}
"""

    try:
        ollama_res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.query}
            ]
        )
        answer_text = ollama_res['message']['content']
    except Exception as e:
        answer_text = f"Error generating response from local LLM model: {str(e)}"

    # Check the AI's citations in code
    valid_refs = {r["ref"] for r in results}
    cited_refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    invalid_refs = cited_refs - valid_refs

    for n in invalid_refs:                         # remove made-up source numbers
        answer_text = answer_text.replace(f"[{n}]", "")
    for r in results:                              # mark which sources were actually used
        r["cited"] = r["ref"] in cited_refs

    print(f"[citations] cited={sorted(cited_refs & valid_refs)}  invalid_removed={sorted(invalid_refs)}")

    # NEW: find the exact sentence in each source that supports the answer
    find_supporting_sentences(answer_text, results)
    for r in results:
        for h in r["highlights"]:
            short = " ".join(h.split())[:90]
            print(f"[highlight] [{r['ref']}] {short}")

    return {
        "user": username,
        "department": department,
        "answer": answer_text,
        "results": results
    }


# ==========================================
# 4. STATIC FILE MOUNT
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)