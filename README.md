# Music Agent

A natural-language music recommendation system.

Users can enter queries such as:

- `Adele`
- `sad female pop`
- `similar to Taylor Swift, but less popular and more dreamy`

The system parses the query, retrieves candidate songs from a local music library, and returns recommendations.

---

## 1. Project Structure

```text
music_agent/
├─ app/                # FastAPI backend
├─ frontend/           # Next.js frontend
├─ data/               # Local runtime data (database + vector index)
│  ├─ music_v2.db
│  ├─ faiss.index
│  └─ ids.npy
├─ scripts/            # Offline scripts / test scripts
└─ .env                # Optional environment variables
```

---

## 2. Requirements

Recommended environment:

- Python 3.10+
- Node.js 18+
- npm 9+

---

## 3. Quick Start

> For normal demo usage, you do **not** need to rebuild the database.
> As long as `data/music_v2.db`, `data/faiss.index`, and `data/ids.npy` exist, the project should run directly.

### Step 1. Clone the project

```bash
git clone <your-repo-url>
cd music_agent
```

### Step 2. Start the backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.api:app --reload
```

On Windows:

```bash
.venv\Scripts\activate
```

Once started, the backend is available at:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

Quick health check:

```bash
curl "http://127.0.0.1:8000/health"
```

Expected response:

```json
{"status":"ok"}
```

### Step 3. Start the frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL:

- `http://localhost:3000`

---

## 4. How to Use

Open the frontend page and enter a natural-language query, for example:

- `Adele`
- `Jay Chou`
- `sad female pop`
- `similar to Taylor Swift, but less popular and more dreamy`

The frontend sends the request to the backend `/search` API and displays the results.

You can also test the API directly:

```bash
curl -G "http://127.0.0.1:8000/search" --data-urlencode "query=sad female pop"
```

---

## 5. Common Issues

### 1) Backend fails to start because runtime files are missing

Make sure these files exist:

```text
data/music_v2.db
data/faiss.index
data/ids.npy
```

These three files are required at runtime.

### 2) Frontend shows `Failed to fetch`

Usually this means the backend is not running, or the frontend is calling the wrong API address.

Check:

- whether the backend is running
- whether `http://127.0.0.1:8000/health` is reachable

### 3) The backend is slow on first startup

This is normal. The service loads:

- the SQLite database
- the FAISS index
- retrieval-related resources

Later queries should be faster.

---

## 6. About the Data Files

This project has two separate phases:

### Online runtime

For normal demo usage, only do the following:

- start the backend
- start the frontend
- run searches

### Offline data building

You only need to rebuild data when:

- the data source changes
- the schema changes
- embeddings / FAISS index are regenerated

In other words:

**For regular use, just reuse the existing files in `data/`. Do not rebuild them every time.**

Current offline scripts:

- `scripts/rebuild_music_library.py`: build the v2 library, derived metadata, embeddings, and FAISS index.
- `scripts/refresh_music_v2_derived_fields.py`: refresh derived v2 fields from existing raw Last.fm tags without calling Last.fm again.
- `scripts/audit_metadata.py`: print v2 metadata coverage.
- `scripts/inspect_db.py`: inspect v2 tables and sample rows.
- `scripts/smoke_test_api.py`: run API smoke queries against a local backend.
- `scripts/eval_search_quality.py`: run the fixed query eval set in `eval_queries.json`.

Run the quality eval after starting the backend:

```bash
python3 scripts/eval_search_quality.py
```

The eval endpoint requests debug fields explicitly with `include_debug=true`.
Normal frontend/API usage keeps ranking evidence and raw debug fields hidden.

---

## 7. Shortest Demo Commands

If dependencies and data files are already prepared, run:

### Backend

```bash
source .venv/bin/activate
python -m uvicorn app.api:app --reload
```

### Frontend

```bash
cd frontend
npm run dev
```

---

## 8. Tech Stack

- FastAPI
- Next.js / React
- SQLite
- FAISS
- Embedding-based retrieval

---

## 9. Current Status

This version is suitable for:

- course project demos
- local development
- basic natural-language music retrieval

It is not intended for production deployment.
