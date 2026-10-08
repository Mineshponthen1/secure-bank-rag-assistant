"""promptfoo provider: asks the running bank app (/api/query) a golden question, as the right test user.

promptfoo calls call_api() once per test. It returns the answer plus the token usage, the
estimated cost (from config/pricing.yaml, calculated by the app), and citation counts,
so the regression check can compare them with the baseline.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # the project's .env (test passwords)
BASE_URL = os.getenv("BANK_APP_URL", "http://localhost:8000")
TIMEOUT = 900   # seconds: some answers take minutes on a CPU-only laptop


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def login(user):
    password = os.getenv(f"EVAL_PASSWORD_{user}")
    if not password:
        raise RuntimeError(f"Add EVAL_PASSWORD_{user}=... to the .env file")
    return post("/api/login", {"username": user, "password": password})["token"]


def call_api(prompt, options, context):
    user = context["vars"]["user"]
    try:
        resp = post("/api/query", {"query": prompt}, login(user))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    answer = resp["answer"]
    usage = resp.get("usage", {})
    cites = re.findall(r"\[(\d+)(\?)?\]", answer)
    verified = sum(1 for _, flag in cites if not flag)
    return {
        "output": answer,
        "tokenUsage": {
            "prompt": usage.get("input_tokens", 0),
            "completion": usage.get("output_tokens", 0),
            "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
        "cost": usage.get("estimated_cost_usd", 0.0),
        "metadata": {
            "sources": sorted({r["source_file"] for r in resp["results"]}),
            "citations": len(cites),
            "verified_citations": verified,
            "refused": not resp["results"],
        },
    }