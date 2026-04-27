from __future__ import annotations

import argparse
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
ENV_FILE = ROOT_DIR / ".env"
MSD_TRACK_METADATA_URL = (
    "http://labrosa.ee.columbia.edu/millionsong/sites/default/files/"
    "AdditionalFiles/track_metadata.db"
)
DEFAULT_BOOTSTRAP_TRACK_LIMIT = 10_000


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="One entry point for setup, data building, running, and warmup."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Install dependencies and validate the local runtime.")
    setup_parser.add_argument("--cpu", action="store_true", default=True, help="Install CPU-only PyTorch before backend deps. Default.")
    setup_parser.add_argument("--gpu", action="store_true", help="Do not force CPU-only PyTorch; use normal PyPI dependency resolution.")
    setup_parser.add_argument("--skip-backend", action="store_true", help="Skip Python backend dependency installation.")
    setup_parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend npm dependency installation.")
    setup_parser.add_argument("--skip-data-bootstrap", action="store_true", help="Do not auto-download/build data when runtime files are missing.")
    setup_parser.add_argument("--bootstrap-track-limit", type=int, default=bootstrap_track_limit_default(), help="MSD tracks to import when bootstrapping data. Use 0 for all rows. Default: 10000.")
    setup_parser.add_argument("--offline", action="store_true", help="Validate as an offline run; requires the HF model to already be cached.")
    add_build_args(setup_parser, include_build_flag=True)
    setup_parser.set_defaults(func=cmd_setup)

    build_parser = subparsers.add_parser("build-db", help="Build or rebuild runtime DB, embeddings, ids, and FAISS index.")
    add_build_args(build_parser, include_build_flag=False)
    build_parser.set_defaults(func=cmd_build_db)

    bootstrap_parser = subparsers.add_parser("bootstrap-data", help="Download MSD metadata, convert it, and build runtime data files.")
    bootstrap_parser.add_argument("--force-download", action="store_true", help="Redownload the raw MSD metadata SQLite file.")
    bootstrap_parser.add_argument("--force-convert", action="store_true", help="Recreate data/music.db from the raw MSD metadata DB.")
    bootstrap_parser.add_argument("--track-limit", type=int, default=bootstrap_track_limit_default(), help="MSD tracks to import. Use 0 for all rows. Default: 10000.")
    add_build_args(bootstrap_parser, include_build_flag=False)
    bootstrap_parser.set_defaults(func=cmd_bootstrap_data)

    run_parser = subparsers.add_parser("run", help="Run backend, frontend, or both.")
    run_parser.add_argument("target", choices=("backend", "frontend", "all"), help="Which service to run.")
    run_parser.add_argument("--host", default="127.0.0.1", help="Backend host. Default: 127.0.0.1.")
    run_parser.add_argument("--backend-port", type=int, default=8000, help="Backend port. Default: 8000.")
    run_parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend port. Default: 3000.")
    run_parser.add_argument("--offline", action="store_true", help="Run backend with HF_HUB_OFFLINE=1. Requires cached model.")
    run_parser.add_argument("--skip-data-bootstrap", action="store_true", help="Do not auto-download/build data when runtime files are missing.")
    run_parser.add_argument("--bootstrap-track-limit", type=int, default=bootstrap_track_limit_default(), help="MSD tracks to import when bootstrapping data. Use 0 for all rows. Default: 10000.")
    run_parser.add_argument("--no-reload", action="store_true", help="Run backend without uvicorn --reload.")
    run_parser.add_argument("--no-force-cpu", action="store_true", help="Do not hide CUDA devices at runtime.")
    run_parser.set_defaults(func=cmd_run)

    warmup_parser = subparsers.add_parser("warmup", help="Ask a running backend to preload the model and FAISS index.")
    warmup_parser.add_argument("--host", default="127.0.0.1", help="Backend host. Default: 127.0.0.1.")
    warmup_parser.add_argument("--backend-port", type=int, default=8000, help="Backend port. Default: 8000.")
    warmup_parser.add_argument("--timeout", type=int, default=180, help="Warmup timeout in seconds. Default: 180.")
    warmup_parser.set_defaults(func=cmd_warmup)

    args = parser.parse_args()
    args.func(args)


def add_build_args(parser: argparse.ArgumentParser, include_build_flag: bool) -> None:
    if include_build_flag:
        parser.add_argument("--build-db", action="store_true", help="Run the full DB/index rebuild pipeline.")
    parser.add_argument("--src-db", type=Path, default=ROOT_DIR / "data" / "music.db", help="Source SQLite DB with a tracks table.")
    parser.add_argument("--dst-db", type=Path, default=ROOT_DIR / "data" / "music_v2.db", help="Runtime v2 SQLite DB output.")
    parser.add_argument("--emb-path", type=Path, default=ROOT_DIR / "data" / "embeddings.npy", help="Embeddings .npy output.")
    parser.add_argument("--ids-path", type=Path, default=ROOT_DIR / "data" / "ids.npy", help="IDs .npy output.")
    parser.add_argument("--index-path", type=Path, default=ROOT_DIR / "data" / "faiss.index", help="FAISS index output.")
    parser.add_argument("--lastfm", action="store_true", help="Enable Last.fm contributor enrichment. Requires LASTFM_API_KEY.")
    parser.add_argument("--album-enrich", action="store_true", help="Enable Last.fm album enrichment. Use with --lastfm.")
    parser.add_argument("--max-contributors", type=int, default=None, help="Limit Last.fm contributor enrichment count.")
    parser.add_argument("--max-albums", type=int, default=None, help="Limit Last.fm album enrichment count.")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size for index rebuild. Default: 256.")


def cmd_setup(args: argparse.Namespace) -> None:
    print_header("Preflight")
    print_system_info()
    check_data_files(warn_only=not args.build_db)

    if not args.skip_backend:
        print_header("Backend Dependencies")
        ensure_venv()
        python = venv_python()
        if args.gpu:
            print("GPU mode requested: using normal PyPI dependency resolution.")
        else:
            run_cmd([
                str(python),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch",
            ])
        run_cmd([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    if not args.skip_frontend:
        print_header("Frontend Dependencies")
        require_executable("node")
        require_executable("npm")
        run_cmd(["npm", "ci"], cwd=FRONTEND_DIR)

    if args.build_db:
        cmd_build_db(args)
    elif not args.skip_data_bootstrap:
        ensure_runtime_data(args)

    print_header("Validation")
    validate_backend_dependencies(skip=args.skip_backend)
    validate_frontend_dependencies(skip=args.skip_frontend)
    check_data_files(warn_only=False)
    check_model_cache(offline=args.offline)
    print("Setup finished.")


def cmd_build_db(args: argparse.Namespace) -> None:
    print_header("Build DB")
    if args.lastfm and not os.getenv("LASTFM_API_KEY"):
        raise SystemExit("LASTFM_API_KEY is required for --lastfm. Put it in .env.")
    python = venv_python() if venv_python().exists() else Path(sys.executable)
    cmd = [
        str(python),
        str(ROOT_DIR / "scripts" / "rebuild_music_library.py"),
        "--src-db",
        str(resolve_path(args.src_db)),
        "--dst-db",
        str(resolve_path(args.dst_db)),
        "--emb-path",
        str(resolve_path(args.emb_path)),
        "--ids-path",
        str(resolve_path(args.ids_path)),
        "--index-path",
        str(resolve_path(args.index_path)),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.lastfm:
        cmd.append("--enable-lastfm")
    if args.album_enrich:
        cmd.append("--enable-album-enrich")
    if args.max_contributors is not None:
        cmd.extend(["--max-contributors", str(args.max_contributors)])
    if args.max_albums is not None:
        cmd.extend(["--max-albums", str(args.max_albums)])
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT_DIR)
        if not existing_pythonpath
        else f"{ROOT_DIR}{os.pathsep}{existing_pythonpath}"
    )
    run_cmd(cmd, cwd=ROOT_DIR, env=env)


def cmd_bootstrap_data(args: argparse.Namespace) -> None:
    print_header("Bootstrap Data")
    raw_db = ROOT_DIR / "data" / "raw" / "track_metadata.db"
    src_db = resolve_path(args.src_db)
    ensure_msd_source_db(
        raw_db=raw_db,
        src_db=src_db,
        track_limit=args.track_limit,
        force_download=args.force_download,
        force_convert=args.force_convert,
    )
    cmd_build_db(args)


def cmd_run(args: argparse.Namespace) -> None:
    if args.target in {"backend", "all"} and not args.skip_data_bootstrap:
        ensure_runtime_data(args)

    if args.target == "backend":
        run_backend(args)
        return
    if args.target == "frontend":
        run_frontend(args)
        return

    backend_env = runtime_env(args, include_backend=True)
    frontend_env = runtime_env(args, include_backend=False)
    backend_cmd = backend_command(args)
    frontend_cmd = frontend_command(args)

    print_header("Run All")
    print(f"Backend:  http://{args.host}:{args.backend_port}")
    print(f"Frontend: http://localhost:{args.frontend_port}")

    backend = subprocess.Popen(backend_cmd, cwd=ROOT_DIR, env=backend_env)
    try:
        time.sleep(2)
        frontend = subprocess.Popen(frontend_cmd, cwd=FRONTEND_DIR, env=frontend_env)
        try:
            while True:
                backend_code = backend.poll()
                frontend_code = frontend.poll()
                if backend_code is not None:
                    raise SystemExit(backend_code)
                if frontend_code is not None:
                    raise SystemExit(frontend_code)
                time.sleep(1)
        finally:
            terminate_process(frontend)
    finally:
        terminate_process(backend)


def cmd_warmup(args: argparse.Namespace) -> None:
    url = f"http://{args.host}:{args.backend_port}/warmup"
    print_header("Warmup")
    print(f"POST {url}")
    request = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Warmup failed. Is the backend running? {exc}") from exc
    print(body)


def run_backend(args: argparse.Namespace) -> None:
    print_header("Backend")
    env = runtime_env(args, include_backend=True)
    run_cmd(backend_command(args), cwd=ROOT_DIR, env=env)


def run_frontend(args: argparse.Namespace) -> None:
    print_header("Frontend")
    env = runtime_env(args, include_backend=False)
    run_cmd(frontend_command(args), cwd=FRONTEND_DIR, env=env)


def ensure_runtime_data(args: argparse.Namespace) -> None:
    if runtime_data_exists():
        return
    if getattr(args, "offline", False):
        check_data_files(warn_only=False)
    print_header("Data Bootstrap")
    print("Runtime data is missing; downloading/building the Million Song Dataset metadata slice.")
    bootstrap_args = argparse.Namespace(**vars(args))
    bootstrap_args.track_limit = getattr(args, "bootstrap_track_limit", bootstrap_track_limit_default())
    bootstrap_args.force_download = False
    bootstrap_args.force_convert = False
    cmd_bootstrap_data(bootstrap_args)


def runtime_data_exists() -> bool:
    return all(
        path.exists()
        for path in [
            ROOT_DIR / "data" / "music_v2.db",
            ROOT_DIR / "data" / "faiss.index",
            ROOT_DIR / "data" / "ids.npy",
        ]
    )


def ensure_msd_source_db(
        raw_db: Path,
        src_db: Path,
        track_limit: int,
        force_download: bool,
        force_convert: bool,
) -> None:
    raw_db.parent.mkdir(parents=True, exist_ok=True)
    src_db.parent.mkdir(parents=True, exist_ok=True)

    if force_download or not raw_db.exists():
        download_file(MSD_TRACK_METADATA_URL, raw_db)
    else:
        print(f"MSD raw metadata already exists: {raw_db}")

    if force_convert or not src_db.exists():
        convert_msd_metadata(raw_db=raw_db, src_db=src_db, track_limit=track_limit)
    else:
        print(f"Source DB already exists: {src_db}")


def download_file(url: str, dst: Path) -> None:
    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        part.unlink()
    print(f"Downloading {url}")
    print(f"       -> {dst}")
    with urllib.request.urlopen(url) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    part.replace(dst)


def convert_msd_metadata(raw_db: Path, src_db: Path, track_limit: int) -> None:
    limit_text = "all rows" if track_limit == 0 else f"{track_limit} rows"
    print(f"Converting MSD metadata into {src_db} ({limit_text})")

    raw_conn = sqlite3.connect(raw_db)
    src_conn = sqlite3.connect(src_db)
    try:
        raw_columns = table_columns(raw_conn, "songs")
        if not raw_columns:
            raise SystemExit(f"{raw_db} does not contain the expected MSD songs table.")

        src_conn.executescript(
            """
            DROP TABLE IF EXISTS tracks;
            CREATE TABLE tracks (
                id INTEGER PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                year INTEGER,
                popularity REAL,
                genre TEXT,
                style TEXT,
                mood TEXT,
                language TEXT,
                vocal_type TEXT
            );
            """
        )

        limit_sql = "" if track_limit == 0 else "LIMIT ?"
        params: tuple[int, ...] = () if track_limit == 0 else (track_limit,)
        query = f"""
            SELECT title, artist_name, release, year, artist_hotttnesss
            FROM songs
            WHERE title IS NOT NULL
              AND artist_name IS NOT NULL
            ORDER BY rowid
            {limit_sql}
        """
        rows = raw_conn.execute(query, params)

        batch: list[tuple[int, str | None, str | None, str | None, int | None, float | None]] = []
        inserted = 0
        for inserted, row in enumerate(rows, start=1):
            title, artist, album, year, hotness = row
            popularity = None
            if hotness is not None:
                try:
                    popularity = round(float(hotness) * 100.0, 4)
                except Exception:
                    popularity = None
            batch.append((inserted, title, artist, album, normalized_year(year), popularity))
            if len(batch) >= 1000:
                insert_msd_tracks(src_conn, batch)
                batch.clear()

        if batch:
            insert_msd_tracks(src_conn, batch)

        src_conn.commit()
        print(f"Converted tracks: {inserted}")
    finally:
        raw_conn.close()
        src_conn.close()


def insert_msd_tracks(
        conn: sqlite3.Connection,
        rows: list[tuple[int, str | None, str | None, str | None, int | None, float | None]],
) -> None:
    conn.executemany(
        """
        INSERT INTO tracks
        (id, title, artist, album, year, popularity, genre, style, mood, language, vocal_type)
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
        """,
        rows,
    )


def normalized_year(value: object) -> int | None:
    try:
        year = int(value) if value is not None else 0
    except Exception:
        return None
    return year if year > 0 else None


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def bootstrap_track_limit_default() -> int:
    raw = os.getenv("MUSIC_BOOTSTRAP_TRACK_LIMIT")
    if not raw:
        return DEFAULT_BOOTSTRAP_TRACK_LIMIT
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit("MUSIC_BOOTSTRAP_TRACK_LIMIT must be an integer.")
    if value < 0:
        raise SystemExit("MUSIC_BOOTSTRAP_TRACK_LIMIT must be 0 or greater.")
    return value


def backend_command(args: argparse.Namespace) -> list[str]:
    python = venv_python() if venv_python().exists() else Path(sys.executable)
    cmd = [
        str(python),
        "-m",
        "uvicorn",
        "app.api:app",
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
    ]
    if not args.no_reload:
        cmd.append("--reload")
    return cmd


def frontend_command(args: argparse.Namespace) -> list[str]:
    return ["npm", "run", "dev", "--", "--port", str(args.frontend_port)]


def runtime_env(args: argparse.Namespace, include_backend: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    if include_backend and args.offline:
        env["HF_HUB_OFFLINE"] = "1"
    if include_backend and not args.no_force_cpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    if not include_backend:
        env["BACKEND_API_BASE"] = f"http://{args.host}:{args.backend_port}"
        env["NEXT_PUBLIC_API_BASE"] = f"http://{args.host}:{args.backend_port}"
        env.setdefault("PORT", str(args.frontend_port))
    return env


def ensure_venv() -> None:
    if VENV_DIR.exists():
        print(f"Using existing virtualenv: {VENV_DIR}")
        return
    print(f"Creating virtualenv: {VENV_DIR}")
    run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT_DIR)


def validate_backend_dependencies(skip: bool) -> None:
    if skip:
        print("Skipped backend dependency validation.")
        return
    python = venv_python()
    run_cmd([str(python), "-c", "import fastapi, uvicorn, faiss, sentence_transformers; print('backend deps ok')"])


def validate_frontend_dependencies(skip: bool) -> None:
    if skip:
        print("Skipped frontend dependency validation.")
        return
    if not (FRONTEND_DIR / "node_modules" / ".bin" / ("next.cmd" if os.name == "nt" else "next")).exists():
        raise SystemExit("Frontend dependencies are missing. Run setup without --skip-frontend.")
    print("frontend deps ok")


def check_data_files(warn_only: bool) -> None:
    required = [
        ROOT_DIR / "data" / "music_v2.db",
        ROOT_DIR / "data" / "faiss.index",
        ROOT_DIR / "data" / "ids.npy",
    ]
    missing = [path for path in required if not path.exists()]
    if not missing:
        print("runtime data ok")
        return
    message = "Missing runtime data files:\n" + "\n".join(f"  - {path}" for path in missing)
    if warn_only:
        print(f"WARN: {message}")
        print("Use setup --build-db or build-db to regenerate them.")
        return
    raise SystemExit(message)


def check_model_cache(offline: bool) -> None:
    if MODEL_CACHE_DIR.exists():
        print(f"HF model cache ok: {MODEL_NAME}")
        return
    message = f"HF model cache not found for {MODEL_NAME}."
    if offline:
        raise SystemExit(message + " Do not use --offline until the model has been downloaded once.")
    print(f"WARN: {message} First /search needs network access.")


def print_system_info() -> None:
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    node = shutil.which("node")
    npm = shutil.which("npm")
    print(f"Node: {read_version([node, '--version']) if node else 'missing'}")
    print(f"npm: {read_version([npm, '--version']) if npm else 'missing'}")
    if platform.system() == "Windows":
        print("Note: WSL2 is recommended for the smoothest FAISS/SentenceTransformers setup on Windows.")
    print(f".env: {'found' if ENV_FILE.exists() else 'not found'}")


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing executable: {name}")


def read_version(cmd: list[str | None]) -> str:
    if cmd[0] is None:
        return "missing"
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        return "unknown"
    return result.stdout.strip()


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def run_cmd(
        cmd: list[str],
        cwd: Path = ROOT_DIR,
        env: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def print_header(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
