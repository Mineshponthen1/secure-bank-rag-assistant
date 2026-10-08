import os
import json
import time
import sqlite3
import bcrypt
import jwt
import secrets
import yaml
import cohere
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import datetime
import weaviate
from weaviate.classes.query import Filter
import ollama
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from citations import verify_citations, remove_invalid_refs
import re
from langfuse import get_client, observe, propagate_attributes

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
DB_FILE = "users_approval.db"

# Retrieval settings
load_dotenv()                      # reads COHERE_API_KEY from .env
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-pro"
CANDIDATES = 20                    # hybrid search builds a long list...
TOP_K = 5                          # ...the re-ranker keeps the best few
MIN_RELEVANCE = 0.75               # measured: relevant >= 0.805, irrelevant <= 0.675 (see measure_rerank.py)

# Load the answer prompt from its versioned config file
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "rag_answer.yaml")
with open(PROMPT_FILE, encoding="utf-8") as f:
    PROMPT = yaml.safe_load(f)
print(f"[prompt] loaded rag_answer v{PROMPT['version']} (model: {PROMPT['model']})")
lf = get_client()   # Langfuse messenger: sends every trace to the self-hosted Langfuse (keys in .env)

# Estimated cost per request (see config/pricing.yaml): token counts x a hosted per-token price
PRICING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "pricing.yaml")
with open(PRICING_FILE, encoding="utf-8") as f:
    PRICING = yaml.safe_load(f)


def estimate_cost(usage: dict) -> dict:
    """USD cost if this model were served by a hosted API. The local model itself costs $0."""
    cost_in = usage["input"] / 1_000_000 * PRICING["usd_per_million_input_tokens"]
    cost_out = usage["output"] / 1_000_000 * PRICING["usd_per_million_output_tokens"]
    return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}


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


@observe(name="rerank", capture_input=False, capture_output=False)
def rerank_with_retry(query: str, documents: list, attempts: int = 3):
    """Ask Cohere to re-rank. If the trial rate limit is hit, wait and try again."""
    for attempt in range(1, attempts + 1):
        try:
            results = co.rerank(model=RERANK_MODEL, query=query, documents=documents, top_n=TOP_K).results
            lf.update_current_span(
                input={"query": query, "candidates": len(documents)},
                output=[{"index": r.index, "score": round(r.relevance_score, 3)} for r in results],
                metadata={"rerank_model": RERANK_MODEL, "attempts": attempt},
            )
            return results
        except cohere.errors.TooManyRequestsError:
            if attempt == attempts:
                lf.update_current_span(level="ERROR", status_message="Cohere rate limit: gave up after retries")
                raise HTTPException(status_code=503, detail="The search service is busy. Please try again in a minute.")
            lf.update_current_span(level="WARNING", status_message=f"Cohere rate limited (attempt {attempt}/{attempts})")
            print(f"[rerank] rate limited, waiting 15s (attempt {attempt}/{attempts})")
            time.sleep(15)
    return []


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
# 3. SECURE RAG QUERY (shared steps + two endpoints)
# ==========================================
GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon"]


def greeting_text() -> str:
    hour = datetime.datetime.now().hour
    time_greeting = "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 17 else "Good evening"
    return (f"{time_greeting}! Hello! I am your Secure Enterprise Banking Policy Assistant. "
            f"How can I help you navigate our compliance and department guidelines today?")


@observe(name="retrieval", as_type="retriever", capture_input=False, capture_output=False)
def retrieve(query: str, department: str, is_admin: bool):
    """Hybrid search + re-ranking, respecting department permissions.
    Returns (numbered contexts for the AI, source details for the page)."""
    if is_admin or department == "Admin":
        rbac_filter = None
    else:
        rbac_filter = Filter.by_property("allowed_departments").contains_any([department])

    vector = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)["embedding"]
    collection = client.collections.get(COLLECTION_NAME)

    # Step 1: hybrid search (meaning + keywords) builds a long list, respecting department permissions
    candidates = collection.query.hybrid(
        query=query, vector=vector, alpha=0.5,
        filters=rbac_filter, limit=CANDIDATES
    ).objects

    retrieved_contexts = []
    results = []
    print(f"\n[retrieval] Question: {query}")

    # Step 2: the re-ranker reads the question with each candidate and keeps the best
    reranked = rerank_with_retry(query, [str(o.properties["content"]) for o in candidates]) if candidates else []
    for r in reranked:
        obj = candidates[r.index]
        content = obj.properties.get("content")
        source_file = obj.properties.get("source_file")
        page_number = obj.properties.get("page_number")
        allowed_depts = obj.properties.get("allowed_departments")
        relevance = r.relevance_score

        if relevance < MIN_RELEVANCE:
            print(f"[retrieval]   DROPPED {source_file} p.{page_number}  relevance={relevance:.3f}")
            continue
        print(f"[retrieval]   KEPT    {source_file} p.{page_number}  relevance={relevance:.3f}")

        # Give each kept source a number, like a footnote
        ref = len(results) + 1
        retrieved_contexts.append(f"[{ref}] (Source: {source_file}, page {page_number})\n{content}")
        results.append({
            "ref": ref,
            "content": content,
            "source_file": source_file,
            "page_number": page_number,
            "allowed_departments": allowed_depts,
            "relevance": round(relevance, 3)
        })

    # Record what retrieval found (page numbers and scores, not the full texts)
    lf.update_current_span(
        input={"query": query, "department": department, "is_admin": is_admin},
        output=[{"ref": r["ref"], "source": r["source_file"], "page": r["page_number"],
                 "relevance": r["relevance"]} for r in results],
        metadata={"candidates": len(candidates), "kept": len(results),
                  "dropped": len(reranked) - len(results), "min_relevance": MIN_RELEVANCE},
    )
    return retrieved_contexts, results


def build_messages(query: str, department: str, retrieved_contexts: list) -> list:
    """Fill in the prompt template from prompts/rag_answer.yaml."""
    context_str = "\n\n".join(retrieved_contexts)
    return [
        {"role": "system", "content": PROMPT["system"].format(department=department, sources=context_str)},
        {"role": "user", "content": PROMPT["user"].format(question=query)},
    ]


def finalize(answer_text: str, results: list) -> str:
    """Check the AI's citations in code, then verify them against the sources."""
    answer_text, cited, removed = remove_invalid_refs(answer_text, results)
    print(f"[citations] cited={cited}  invalid_removed={removed}")

    # Verify each citation in code: highlight verified passages, mark unverified ones as [n?]
    return verify_citations(answer_text, results)


def citation_counts(text: str) -> dict:
    """Count citations in a final answer: [n] is verified, [n?] is unverified."""
    cites = re.findall(r"\[(\d+)(\?)?\]", text)
    verified = sum(1 for _, flag in cites if not flag)
    return {"citations": len(cites), "verified": verified, "unverified": len(cites) - verified}


def trace_safely(action, **fields):
    """Monitoring must never break the product: if a tracing call fails, print why and carry on."""
    try:
        return action(**fields)
    except Exception as e:
        print(f"[tracing] skipped {getattr(action, '__name__', 'call')}: {e}")

def record_scores(score_fn, outcome: str, counts: dict | None = None):
    """Attach quality scores to the trace: what happened, and (for answers) citation coverage."""
    trace_safely(score_fn, name="outcome", value=outcome, data_type="CATEGORICAL")
    if counts is not None:
        coverage = counts["verified"] / counts["citations"] if counts["citations"] else 0.0
        trace_safely(score_fn, name="citation_coverage", value=round(coverage, 3), data_type="NUMERIC")


@app.post("/api/query")
def secure_rag_query(data: QueryRequest, user: dict = Depends(get_current_user)):
    """Classic endpoint: returns the whole answer at once (used by the report card)."""
    username, department, is_admin = user["username"], user["department"], user["is_admin"]
    # Labels for the whole trace: who asked, from which department, with which prompt version
    with propagate_attributes(user_id=username, trace_name="rag-query",
                              tags=["api/query", f"prompt-v{PROMPT['version']}"],
                              metadata={"department": department},
                              version=str(PROMPT["version"])):
        return answer_query(data.query, username, department, is_admin)


@observe(name="rag-query", capture_input=False, capture_output=False)
def answer_query(query: str, username: str, department: str, is_admin: bool):
    lf.update_current_span(input={"query": query})
    # The "receipt": tokens and estimated cost. Stays at zero when the AI is never called.
    usage = {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}

    if query.strip().lower() in GREETINGS:
        answer = greeting_text()
        lf.update_current_span(output={"answer": answer}, metadata={"outcome": "greeting"})
        record_scores(lf.score_current_trace, "greeting")
        return {"user": username, "department": department, "answer": answer, "results": [], "usage": usage}

    retrieved_contexts, results = retrieve(query, department, is_admin)

    # Refuse by code (not by hoping the AI notices) when no chunk is relevant enough
    if not retrieved_contexts:
        lf.update_current_span(output={"answer": PROMPT["refusal"]}, metadata={"outcome": "refused"})
        record_scores(lf.score_current_trace, "refused")
        return {"user": username, "department": department, "answer": PROMPT["refusal"], "results": [], "usage": usage}

    try:
        answer_text, usage = generate_answer(query, department, retrieved_contexts)
    except Exception as e:
        answer_text = f"Error generating response from local LLM model: {str(e)}"
        lf.update_current_span(level="ERROR", status_message=str(e)[:200])
        record_scores(lf.score_current_trace, "failed")
        return {"user": username, "department": department, "answer": answer_text, "results": results, "usage": usage}

    answer_text = check_citations(answer_text, results)
    lf.update_current_span(output={"answer": answer_text},
                           metadata={"outcome": "answered", "sources": len(results)})
    record_scores(lf.score_current_trace, "answered", citation_counts(answer_text))
    return {"user": username, "department": department, "answer": answer_text, "results": results, "usage": usage}


@observe(name="llm-answer", as_type="generation", capture_input=False, capture_output=False)
def generate_answer(query: str, department: str, retrieved_contexts: list) -> tuple[str, dict]:
    messages = build_messages(query, department, retrieved_contexts)
    res = ollama.chat(model=PROMPT["model"], messages=messages)
    text = res["message"]["content"]
    usage = {"input": res["prompt_eval_count"] or 0, "output": res["eval_count"] or 0}
    cost = estimate_cost(usage)
    lf.update_current_generation(
        model=PROMPT["model"],
        input=messages,
        output=text,
        usage_details=usage,
        cost_details=cost,
        metadata={"prompt_name": "rag_answer", "prompt_version": PROMPT["version"]},
        version=str(PROMPT["version"]),
    )
    receipt = {"input_tokens": usage["input"], "output_tokens": usage["output"],
               "estimated_cost_usd": cost["total"]}
    return text, receipt


@observe(name="citation-check", capture_input=False, capture_output=False)
def check_citations(answer_text: str, results: list) -> str:
    final = finalize(answer_text, results)
    lf.update_current_span(output=citation_counts(final))
    return final


@app.post("/api/query/stream")
def secure_rag_query_stream(data: QueryRequest, user: dict = Depends(get_current_user)):
    """Streaming endpoint: sends the answer piece by piece, one JSON message per line.
    {"type": "token", "text": "..."}                      while the AI is writing
    {"type": "final", "answer": "...", "results": [...]}  at the end, with verified citations
    {"type": "error", "detail": "..."}                    if something goes wrong"""
    username, department, is_admin = user["username"], user["department"], user["is_admin"]

    def send(event: dict) -> str:
        return json.dumps(event) + "\n"

    def events():
        # While streaming, the web server hands the work between threads, so Langfuse's automatic
        # "current step" tracking can lose its place. Here the steps are created by hand and
        # attached to `root` directly, so the trace stays in one piece.
        with propagate_attributes(user_id=username, trace_name="rag-query",
                                  tags=["api/query/stream", f"prompt-v{PROMPT['version']}"],
                                  metadata={"department": department},
                                  version=str(PROMPT["version"])):
            root = lf.start_observation(name="rag-query")
        trace_safely(root.update, input={"query": data.query})

        try:
            if data.query.strip().lower() in GREETINGS:
                answer = greeting_text()
                trace_safely(root.update, output={"answer": answer}, metadata={"outcome": "greeting"})
                record_scores(root.score_trace, "greeting")
                yield send({"type": "final", "answer": answer, "results": []})
                return

            # Search happens before the first piece is sent, so the automatic labels still work here
            try:
                with root.start_as_current_observation(name="search", as_type="span"):
                    retrieved_contexts, results = retrieve(data.query, department, is_admin)
            except HTTPException as e:
                trace_safely(root.update, level="ERROR", status_message=str(e.detail))
                record_scores(root.score_trace, "failed")
                yield send({"type": "error", "detail": e.detail})
                return

            # Refuse by code (not by hoping the AI notices) when no chunk is relevant enough
            if not retrieved_contexts:
                trace_safely(root.update, output={"answer": PROMPT["refusal"]}, metadata={"outcome": "refused"})
                record_scores(root.score_trace, "refused")
                yield send({"type": "final", "answer": PROMPT["refusal"], "results": []})
                return

            messages = build_messages(data.query, department, retrieved_contexts)
            gen = root.start_observation(name="llm-answer", as_type="generation")
            answer_text, first_token_at, last_part = "", None, None
            try:
                for part in ollama.chat(model=PROMPT["model"], messages=messages, stream=True):
                    piece = part["message"]["content"]
                    if piece and first_token_at is None:
                        first_token_at = datetime.datetime.now(datetime.timezone.utc)   # time to first token
                    answer_text += piece
                    last_part = part
                    yield send({"type": "token", "text": piece})
            except Exception as e:
                trace_safely(gen.update, level="ERROR", status_message=str(e)[:200])
                trace_safely(root.update, level="ERROR", status_message="LLM call failed")
                record_scores(root.score_trace, "failed")
                yield send({"type": "error", "detail": f"Error generating response from local LLM model: {e}"})
                return
            finally:
                usage = None
                if last_part is not None and last_part["done"]:
                    usage = {"input": last_part["prompt_eval_count"] or 0, "output": last_part["eval_count"] or 0}
                trace_safely(gen.update, model=PROMPT["model"], input=messages, output=answer_text,
                             usage_details=usage, completion_start_time=first_token_at,
                             cost_details=estimate_cost(usage) if usage else None,
                             metadata={"prompt_name": "rag_answer", "prompt_version": PROMPT["version"]},
                             version=str(PROMPT["version"]))
                trace_safely(gen.end)

            # The answer is complete: now check and verify the citations, then send the final version
            cit = root.start_observation(name="citation-check", as_type="span")
            final = finalize(answer_text, results)
            counts = citation_counts(final)
            trace_safely(cit.update, output=counts)
            trace_safely(cit.end)

            trace_safely(root.update, output={"answer": final},
                         metadata={"outcome": "answered", "sources": len(results)})
            record_scores(root.score_trace, "answered", counts)
            yield send({"type": "final", "answer": final, "results": results})
        finally:
            trace_safely(root.end)

    return StreamingResponse(events(), media_type="application/x-ndjson")


# ==========================================
# 4. STATIC FILE MOUNT
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




