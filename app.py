import os
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

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

        retrieved_contexts.append(content)
        results.append({
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
    
    # Combine context chunks for the LLM
    context_str = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No authorized internal policy documents found."

    system_prompt = f"""You are a secure enterprise banking policy assistant for the {department} department.
Answer the user's question accurately using ONLY the provided policy context below.
If the answer cannot be found in the context, politely state that the information is not available in your authorized department guidelines. Do not make up information.

Context:
{context_str}
"""

    try:
        ollama_res = ollama.chat(
            model="llama3.2",  # Ensure this matches your local model or change to llama3.1 / mistral
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.query}
            ]
        )
        answer_text = ollama_res['message']['content']
    except Exception as e:
        answer_text = f"Error generating response from local LLM model: {str(e)}"

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