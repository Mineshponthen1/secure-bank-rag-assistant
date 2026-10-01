# Secure Enterprise RAG Bank Assistant

![CI](https://github.com/Mineshponthen1/secure-bank-rag-assistant/actions/workflows/ci.yml/badge.svg)

A retrieval-augmented generation (RAG) assistant that answers employee questions from real banking and employment documents, with **department-level access control**, **verifiable citations**, and **refusal by code** when the evidence is too weak.

Built in Python as Project 1 of an Applied AI Engineer portfolio roadmap. Everything runs locally: no document text leaves the machine.


---

## What it does

- **Answers questions from real documents** (UAE Labour Law, Basel Committee principles, FATF Recommendations)
- **Respects permissions:** each department only searches the documents it is allowed to read, enforced inside the database query
- **Cites its sources:** every claim gets a numbered footnote; clicking it opens the source and highlights the supporting passage
- **Checks its own citations:** a verifier compares each claim with the cited passage, and flags weak or mismatched citations with ⚠ instead of trusting the model
- **Refuses when it doesn't know:** if no retrieved passage is relevant enough, the code declines before the model is even called

## Architecture

```mermaid
flowchart LR
    A[Browser<br/>static/index.html] -- question + JWT --> B[FastAPI<br/>app.py]
    B -- embed question --> C[Ollama<br/>nomic-embed-text]
    B -- vector search<br/>+ department filter --> D[(Weaviate<br/>246 chunks)]
    D -- top 5 chunks --> B
    B -- distance > 0.41? --> E[Refuse by code]
    B -- numbered sources + rules --> F[Ollama<br/>llama3.2]
    F -- answer with [n] citations --> G[Citation verifier]
    G -- verified: highlight<br/>unverified: ⚠ --> A
```

## Key engineering decisions (and the evidence behind them)

**Chunk size: ~600 tokens, sentence-aware, with overlap.**
Before choosing, I measured the documents (`inspect_pdfs.py`). The real documents average 400–570 tokens per page, so the roadmap's 500–800 token range fits. Chunks cross page breaks and record their page range. A `--preview` mode in `ingest.py` shows chunk sizes and overlap before anything is stored.

**Relevance cut-off: 0.41 (cosine distance).**
Measured with `measure_distances.py` on relevant questions and bank-sounding "near-miss" questions:

| Questions | Closest distance |
|---|---|
| Relevant (8 questions) | 0.190 – 0.397 |
| Near-misses (e.g. "office dress code") | 0.419 – 0.433 |
| Unrelated (e.g. "capital of France") | 0.563 – 0.597 |

The gap is narrow (0.022), which is documented as a limitation. For a bank, a false refusal is cheaper than a false answer, so the line leans strict.

**Top 5 retrieval (not 3).**
`debug_retrieval.py` showed the annual-leave entitlement clause ranked **4th**: found, but not sent to the model. The chunk mixed the end of the previous article with the start of the leave article, so it looked "less about leave" than chunks that didn't contain the answer.

**Folder-based permissions.**
The folder a PDF sits in decides who can read it (`documents/HR/`, `documents/General/`...), replacing an earlier filename-guessing rule that could mis-tag files.

## Security

- Project files are not served to the web (only `static/`)
- Passwords hashed with **bcrypt**
- **JWT** authentication on every protected endpoint: identity comes from the token, never from the request body
- **CORS** restricted to known origins
- Source text rendered safely (escaped before display; no source text inside `onclick` attributes)

## Documents

The documents are public but copyrighted, so they are **not included** in this repository. Download them and place them in these folders:

| Folder | Document | Source |
|---|---|---|
| `documents/HR/` | UAE Federal Decree-Law No. 33 of 2021 (Labour Law) | https://uaelegislation.gov.ae/en/legislations/1541/download |
| `documents/Operations/` | BCBS: Revisions to the Principles for the Sound Management of Operational Risk (2021) | https://www.bis.org/bcbs/publ/d515.pdf |
| `documents/Finance/` | BCBS 239: Principles for effective risk data aggregation and risk reporting | https://www.bis.org/publ/bcbs239.pdf |
| `documents/General/` | The FATF Recommendations | https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html |

## How to run

Requirements: Python 3.12+, Docker, and [Ollama](https://ollama.com).

```bash
# 1. Start Weaviate
docker compose up -d

# 2. Get the local models
ollama pull nomic-embed-text
ollama pull llama3.2

# 3. Install Python packages
python -m venv venv
venv\Scripts\activate          # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt

# 4. Add the documents (see table above), then preview and ingest
python ingest.py --preview
python ingest.py

# 5. Start the app, then open http://localhost:8000
python app.py
```

A default admin account (`admin` / `adminpassword`) is created on first run. **Change it before any real use.** Set a `JWT_SECRET` environment variable so sessions survive restarts.

## Tools included

| Script | Purpose |
|---|---|
| `inspect_pdfs.py` | Measure words and tokens per page before choosing a chunk size |
| `ingest.py` | Chunk, embed and store documents (`--preview` to inspect first) |
| `measure_distances.py` | Measure relevant vs. irrelevant distances to set the cut-off |
| `debug_retrieval.py` | Show the top 10 chunks for a question and where the answer ranked |

## Known limitations (honest list)

- **The small model sometimes cites the wrong source.** `llama3.2` (3B) often quotes the right fact but attaches the wrong footnote, or cites article numbers instead of source numbers.
- **The citation verifier matches words, not meaning.** It caught several wrong citations, but it was once fooled by an *article number*: the "30" in "Article (30) Maternity Leave" was accepted as support for "30 days of annual leave". Meaning-based checks are planned.
- **Narrow relevance gap.** Relevant and near-miss questions are only 0.022 apart; re-ranking is planned to widen it.
- **Overlap averages ~45–55 tokens** rather than 100, because only whole sentences are carried over.
- **PDF noise:** repeated page headers and contents pages are embedded along with the real text.
- Answers are not streamed yet, and the front end is plain HTML/JavaScript (a React version is planned).

## Roadmap status

- [x] Ingestion, chunking, vector storage
- [x] Citations with highlighting and verification
- [x] Refusal by code with a measured cut-off
- [ ] Hybrid search (vector + keyword) and re-ranking
- [ ] Prompts in versioned config files
- [ ] Evaluation: golden dataset, automated scoring, CI gate
- [ ] Streaming answers and a React front end
