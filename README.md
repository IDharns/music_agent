# Music Agent

A natural-language music recommendation system.

Users can enter queries such as:

- `Adele`
- `sad female pop`
- `similar to Taylor Swift, but less popular and more dreamy`

The system parses the query, retrieves candidate songs from a local music library, and returns recommendations.

---

## 1. Quick Evaluation

Recommended for reviewers:

```bash
python scripts/manage.py setup
python scripts/manage.py run all
```

Then open:

```text
http://localhost:3000
```

Defaults:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://localhost:3000`
- Runtime device: CPU-first
- Runtime data: existing files under `data/`
- Secrets and API keys: `.env` in the repo root

If the machine has no network access but the Hugging Face model is already cached:

```bash
python scripts/manage.py setup --offline
python scripts/manage.py run all --offline
```

`--offline` sets `HF_HUB_OFFLINE=1` for backend runtime validation/startup. It means: use only the local Hugging Face cache and do not contact `huggingface.co`. Do not use it on a fresh machine until `sentence-transformers/all-MiniLM-L6-v2` has been downloaded once.

---

## 2. Requirements

Recommended environment:

- Python 3.10+
- Node.js 18+
- npm 9+
- Linux, macOS, or Windows with WSL2

Native Windows may work, but WSL2 is recommended for the smoothest FAISS / SentenceTransformers dependency setup.

Runtime files required by the API:

```text
data/music_v2.db
data/faiss.index
data/ids.npy
```

For normal demo usage, you do not need to rebuild these files if they already exist.

Optional API keys belong in `.env`, not shell exports:

```bash
cp .env.example .env
```

Fill `OPENROUTER_API_KEY` only if you want LLM query rewrite / rerank. Fill `LASTFM_API_KEY` only if you want Last.fm enrichment during rebuilds. `.env` is ignored by git.

---

## 3. Project Structure

```text
music_agent/
├─ app/                # FastAPI backend
├─ frontend/           # Next.js frontend
├─ data/               # Local runtime data
│  ├─ music_v2.db      # Runtime SQLite database
│  ├─ faiss.index      # Runtime FAISS vector index
│  ├─ ids.npy          # Track ids aligned with faiss.index
│  └─ embeddings.npy   # Optional saved embeddings, useful when rebuilding
├─ scripts/            # Setup, run, offline build, inspection, and eval scripts
└─ .env                # Optional environment variables
```

---

## 4. Main Entry Point

Use `scripts/manage.py` for setup, data building, running, and warmup:

```bash
python scripts/manage.py --help
```

Available commands:

```text
setup     Install dependencies and validate the local runtime.
build-db  Build or rebuild the runtime DB, embeddings, ids, and FAISS index.
run       Run backend, frontend, or both.
warmup    Ask a running backend to preload the model and FAISS index.
```

There is no separate `check` command. `setup` includes preflight checks and post-install validation, including Python/Node/npm visibility, runtime data files, dependency imports, frontend dependencies, and Hugging Face model cache status.

---

## 5. Setup

Default CPU setup:

```bash
python scripts/manage.py setup
```

This does the following:

- prints system information
- loads `.env` if it exists
- checks runtime data files
- creates `.venv` if missing
- installs CPU-only PyTorch first
- installs `requirements.txt`
- runs `npm ci` in `frontend/`
- validates backend imports
- validates frontend dependencies
- checks whether the Hugging Face model is cached

Useful setup args:

```text
--cpu            Install CPU-only PyTorch before backend deps. This is the default.
--gpu            Do not force CPU-only PyTorch; use normal PyPI dependency resolution.
--skip-backend   Skip Python backend dependency installation.
--skip-frontend  Skip frontend npm dependency installation.
--offline        Validate as an offline run; requires the HF model cache to exist.
--build-db       Run the full DB/index rebuild pipeline after dependency setup.
```

For a CPU-only setup that also rebuilds the database and index:

```bash
python scripts/manage.py setup --build-db --src-db data/music.db
```

For GPU-oriented dependency resolution:

```bash
python scripts/manage.py setup --gpu
```

CPU remains the recommended evaluation path because it avoids CUDA driver and wheel compatibility issues.

---

## 6. Run

Run backend and frontend together:

```bash
python scripts/manage.py run all
```

Run only the backend:

```bash
python scripts/manage.py run backend
```

Run only the frontend:

```bash
python scripts/manage.py run frontend
```

Useful run args:

```text
--host HOST                 Backend host. Default: 127.0.0.1.
--backend-port PORT         Backend port. Default: 8000.
--frontend-port PORT        Frontend port. Default: 3000.
--offline                   Run backend with HF_HUB_OFFLINE=1.
--no-reload                 Run backend without uvicorn --reload.
--no-force-cpu              Do not hide CUDA devices at runtime.
```

Examples:

```bash
python scripts/manage.py run all --backend-port 8010 --frontend-port 3010
python scripts/manage.py run backend --offline
python scripts/manage.py run backend --no-reload
```

By default, backend runtime hides CUDA devices so evaluation works consistently on CPU. Use `--no-force-cpu` only if you intentionally want the runtime to see available GPUs.

The frontend receives `BACKEND_API_BASE` automatically, based on `--host` and `--backend-port`. Browser requests go to the frontend's same-origin `/api/search` route, and that route proxies to the FastAPI backend. This avoids CORS issues and also works when you open the Next.js `Network` URL shown in the terminal.

---

## 7. Warmup

`warmup` is optional. It prepares the backend for the first real search by loading:

- the SentenceTransformer embedding model
- the FAISS index
- `ids.npy`
- the retriever and reranking service objects

Without warmup, the first `/search` request does this loading work and can feel slow. With warmup, you pay that cost before demoing.

Start the backend first:

```bash
python scripts/manage.py run backend
```

Then in another terminal:

```bash
python scripts/manage.py warmup
```

Useful warmup args:

```text
--host HOST          Backend host. Default: 127.0.0.1.
--backend-port PORT  Backend port. Default: 8000.
--timeout SECONDS    Warmup timeout. Default: 180.
```

Equivalent direct API call:

```bash
curl -X POST http://127.0.0.1:8000/warmup
```

Health and warmup mean different things:

- `/health` means the API process is alive.
- `/warmup` means the recommendation service has been initialized.

---

## 8. Data Build Flow

This project has two phases:

- Online runtime: start backend, start frontend, and run searches.
- Offline data building: rebuild SQLite DB, embeddings, ids, and FAISS index.

You only need to rebuild data when the source data changes, the schema changes, or embeddings/index files need to be regenerated.

### Full Rebuild

`build-db` reads a source SQLite database that contains a `tracks` table, writes the v2 runtime DB, builds embeddings, writes `ids.npy`, and writes `faiss.index`.

For the MSD path, this project assumes MSD metadata has already been imported into a SQLite source DB such as `data/music.db`. The rebuild script starts from that MSD-derived `tracks` table, then can optionally enrich contributors/albums through Last.fm.

```bash
python scripts/manage.py build-db --src-db data/music.db
```

Expanded version:

```bash
python scripts/manage.py build-db \
  --src-db data/music.db \
  --dst-db data/music_v2.db \
  --emb-path data/embeddings.npy \
  --ids-path data/ids.npy \
  --index-path data/faiss.index
```

Useful build args:

```text
--src-db PATH              Source SQLite DB with a tracks table.
--dst-db PATH              Runtime v2 SQLite DB output.
--emb-path PATH            Embeddings .npy output.
--ids-path PATH            IDs .npy output.
--index-path PATH          FAISS index output.
--lastfm                   Enable Last.fm contributor enrichment.
--album-enrich             Enable Last.fm album enrichment. Use with --lastfm.
--max-contributors N       Limit Last.fm contributor enrichment count.
--max-albums N             Limit Last.fm album enrichment count.
--batch-size N             Embedding batch size. Default: 256.
```

To enrich metadata through Last.fm during rebuild, put the key in `.env`:

```bash
LASTFM_API_KEY=your_lastfm_key
```

Then run:

```bash
python scripts/manage.py build-db \
  --src-db data/music.db \
  --lastfm \
  --album-enrich
```

For a smaller test rebuild:

```bash
python scripts/manage.py build-db \
  --src-db data/music.db \
  --max-contributors 1000 \
  --max-albums 1000
```

### Refresh Derived Fields

Use the lower-level script when `music_v2.db` already exists and you only changed derived metadata logic. This does not call Last.fm.

```bash
source .venv/bin/activate
python scripts/refresh_music_v2_derived_fields.py \
  --db-path data/music_v2.db \
  --rebuild-index \
  --emb-path data/embeddings.npy \
  --ids-path data/ids.npy \
  --index-path data/faiss.index
```

If you omit `--rebuild-index`, the script updates derived DB fields only. Rebuild the index before using the changed DB for retrieval.

---

## 9. API Usage

Backend URLs:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Search:

```bash
curl -G "http://127.0.0.1:8000/search" --data-urlencode "query=sad female pop"
```

Smoke test after backend startup:

```bash
source .venv/bin/activate
python scripts/smoke_test_api.py
```

Evaluate search quality:

```bash
source .venv/bin/activate
python scripts/eval_search_quality.py
```

Inspect data:

```bash
source .venv/bin/activate
python scripts/inspect_db.py
python scripts/audit_metadata.py
```

---

## 10. Environment Variables

Create `.env` in the repo root when needed:

```bash
cp .env.example .env
```

Example:

```bash
MUSIC_DB_PATH=data/music_v2.db
MUSIC_INDEX_PATH=data/faiss.index
MUSIC_IDS_PATH=data/ids.npy

# Optional LLM query rewrite / rerank through OpenRouter
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4.1-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
ENABLE_LLM_QUERY_REWRITE=1
ENABLE_LLM_RERANK=1

# Optional Last.fm enrichment for offline rebuilds
LASTFM_API_KEY=
```

`scripts/manage.py`, the backend, and the rebuild scripts all read `.env`. Do not put real keys in README or commit `.env`; `.gitignore` already excludes it.

If no `OPENROUTER_API_KEY` is set, the backend still works with local parsing, retrieval, and heuristic reranking.

---

## 11. Manual Commands

The `manage.py` entry point is recommended, but manual commands still work.

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.api:app --reload
```

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

---

## 12. Common Issues

### Backend says `No module named uvicorn`

Run:

```bash
python scripts/manage.py setup --skip-frontend
```

### Backend tries to reach Hugging Face in an offline environment

If the model is already cached locally:

```bash
python scripts/manage.py run backend --offline
```

If the model is not cached, run once without `--offline` on a machine with network access so `sentence-transformers/all-MiniLM-L6-v2` can be downloaded.

### Frontend says `next: not found`

Run:

```bash
python scripts/manage.py setup --skip-backend
```

### Frontend shows `Failed to fetch`

Check that:

- the backend is running
- `http://127.0.0.1:8000/health` is reachable
- frontend and backend ports match the `run` command args
- you started the frontend through `python scripts/manage.py run frontend` or `python scripts/manage.py run all`, so `BACKEND_API_BASE` is set automatically

### First search is slow

Run warmup before the demo:

```bash
python scripts/manage.py warmup
```

---

## 13. Tech Stack

- FastAPI
- Next.js / React
- SQLite
- FAISS
- SentenceTransformers
- Embedding-based retrieval

---

## 14. Current Status

This version is a baseline for local natural-language music retrieval. It is suitable for:

- course project demos
- local development
- basic natural-language music retrieval

It is not intended for production deployment.

Current limitations:

- Recommendation quality depends heavily on metadata, Last.fm tags, and derived fields.
- The eval script checks metadata consistency, not subjective listening quality.
- The app does not yet include preview playback, like/dislike feedback, long-term user profiles, or audio-content understanding.
- LLM reranking can improve result ordering and explanations, but it still only sees the metadata returned by retrieval.

The recommended next iteration is a feedback-driven discovery UI with swipe-style actions such as Like, Pass, More like this, Hide artist, and Wrong mood. See [docs/feedback_roadmap.md](docs/feedback_roadmap.md).
