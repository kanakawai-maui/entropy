import os
import hmac
import json
import time
import uuid
import sqlite3
import hashlib
import secrets
import asyncio
import logging

from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse


# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.8:27b")
DB_PATH = os.getenv("DB_PATH", "contributions.db")
SECRET_PATH = os.getenv("SECRET_PATH", ".prompt_hash_secret")

BLOCKCHAIN_ENABLED = (
    os.getenv("BLOCKCHAIN_ENABLED", "false").lower() == "true"
)
CHAIN_RPC_URL = os.getenv("CHAIN_RPC_URL")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
API_KEY = os.getenv("API_KEY")

MAX_BODY = 1_000_000
MAX_PROMPT = 20_000
TIMEOUT = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("llm-node")


def chain_configured():
    return bool(
        BLOCKCHAIN_ENABLED
        and CHAIN_RPC_URL
        and CONTRACT_ADDRESS
        and PRIVATE_KEY
    )


def load_secret():
    path = Path(SECRET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        secret = path.read_bytes()
        if len(secret) != 32:
            raise RuntimeError("Invalid prompt hash secret")
        return secret

    secret = secrets.token_bytes(32)
    with os.fdopen(fd, "wb") as f:
        f.write(secret)
    return secret


SECRET = load_secret()


def prompt_hash(prompt):
    return hmac.new(
        SECRET,
        prompt.strip().encode(),
        hashlib.sha256,
    ).hexdigest()


def init_db():
    with sqlite3.connect(DB_PATH, timeout=10) as db:
        db.execute("PRAGMA busy_timeout=10000")

        # Create tables if they don't exist.
        db.execute("""
            CREATE TABLE IF NOT EXISTS contributions (
                commitment TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                points INTEGER NOT NULL DEFAULT 1
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commitment TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                tx_hash TEXT,
                last_error TEXT
            )
        """)

        # Inspect existing contributions schema.
        columns = {
            row[1]
            for row in db.execute(
                "PRAGMA table_info(contributions)"
            )
        }

        # Migrate older schemas without deleting data.
        if "commitment" not in columns:
            db.execute(
                "ALTER TABLE contributions ADD COLUMN commitment TEXT"
            )

        if "created_at" not in columns:
            db.execute(
                "ALTER TABLE contributions "
                "ADD COLUMN created_at REAL NOT NULL DEFAULT 0"
            )

        if "points" not in columns:
            db.execute(
                "ALTER TABLE contributions "
                "ADD COLUMN points INTEGER NOT NULL DEFAULT 1"
            )

        # Ensure the queue has the expected columns too.
        queue_columns = {
            row[1]
            for row in db.execute("PRAGMA table_info(queue)")
        }

        if "commitment" not in queue_columns:
            db.execute(
                "ALTER TABLE queue ADD COLUMN commitment TEXT"
            )

        if "status" not in queue_columns:
            db.execute(
                "ALTER TABLE queue ADD COLUMN status TEXT "
                "NOT NULL DEFAULT 'pending'"
            )

        if "attempts" not in queue_columns:
            db.execute(
                "ALTER TABLE queue ADD COLUMN attempts INTEGER "
                "NOT NULL DEFAULT 0"
            )

        if "created_at" not in queue_columns:
            db.execute(
                "ALTER TABLE queue ADD COLUMN created_at REAL "
                "NOT NULL DEFAULT 0"
            )

        if "tx_hash" not in queue_columns:
            db.execute(
                "ALTER TABLE queue ADD COLUMN tx_hash TEXT"
            )

        if "last_error" not in queue_columns:
            db.execute(
                "ALTER TABLE queue ADD COLUMN last_error TEXT"
            )

        print("Database schema migration complete.")


def record_contribution(commitment: str) -> bool:
    now = time.time()

    with sqlite3.connect(DB_PATH, timeout=10) as db:
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("BEGIN IMMEDIATE")

        cursor = db.execute(
            """
            INSERT OR IGNORE INTO contributions
                (commitment, created_at, points)
            VALUES (?, ?, 1)
            """,
            (commitment, now),
        )

        is_new = cursor.rowcount == 1

        if is_new:
            db.execute(
                """
                INSERT OR IGNORE INTO queue
                    (commitment, status, created_at)
                VALUES (?, 'pending', ?)
                """,
                (commitment, now),
            )

    return is_new


# Optional Ethereum adapter. Install web3 only when needed.
def publish_commitment(commitment):
    if not chain_configured():
        raise RuntimeError("Blockchain disabled")

    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(CHAIN_RPC_URL, request_kwargs={"timeout": 20}))
    if not w3.is_connected():
        raise RuntimeError("RPC unavailable")

    account = w3.eth.account.from_key(PRIVATE_KEY)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=[{
            "type": "function",
            "name": "submitCommitment",
            "stateMutability": "nonpayable",
            "inputs": [{"name": "commitment", "type": "bytes32"}],
            "outputs": [],
        }],
    )

    tx = contract.functions.submitCommitment(
        bytes.fromhex(commitment)
    ).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(
            account.address, "pending"
        ),
        "chainId": w3.eth.chain_id,
        "gas": 150000,
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    return w3.eth.send_raw_transaction(
        signed.raw_transaction
    ).hex()


async def queue_worker():
    while True:
        try:
            if not chain_configured():
                await asyncio.sleep(2)
                continue

            with sqlite3.connect(DB_PATH) as db:
                job = db.execute(
                    """SELECT id, commitment, attempts FROM queue
                       WHERE status='pending'
                       ORDER BY id LIMIT 1"""
                ).fetchone()

                if not job:
                    await asyncio.sleep(2)
                    continue

                job_id, commitment, attempts = job

                db.execute(
                    "UPDATE queue SET status='sending' WHERE id=?",
                    (job_id,),
                )

            try:
                tx_hash = await asyncio.to_thread(
                    publish_commitment, commitment
                )

                with sqlite3.connect(DB_PATH) as db:
                    db.execute(
                        """UPDATE queue
                           SET status='submitted', tx_hash=?
                           WHERE id=?""",
                        (tx_hash, job_id),
                    )

                log.info("Published queue job %s", job_id)

            except Exception as exc:
                attempts += 1
                status = "failed" if attempts >= 12 else "pending"

                with sqlite3.connect(DB_PATH) as db:
                    db.execute(
                        """UPDATE queue
                           SET status=?, attempts=?, last_error=?
                           WHERE id=?""",
                        (status, attempts, type(exc).__name__, job_id),
                    )

                log.warning("Queue job %s failed", job_id)
                await asyncio.sleep(min(60, 2 ** min(attempts, 6)))

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Queue worker error")
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app):
    init_db()
    worker = asyncio.create_task(queue_worker())
    log.info("Started model=%s blockchain=%s", MODEL, chain_configured())

    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Local LLM Contribution Node", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request, call_next):
    request_id = str(uuid.uuid4())
    started = time.monotonic()

    try:
        response = await call_next(request)
    except Exception:
        log.exception("Request failed id=%s", request_id)
        raise

    response.headers["X-Request-ID"] = request_id
    log.info(
        "id=%s method=%s path=%s status=%s duration=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        time.monotonic() - started,
    )
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "blockchain_enabled": chain_configured(),
        "mode": "blockchain" if chain_configured() else "local-only",
    }


@app.post("/api/chat")
async def chat(request: Request):
    if API_KEY:
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, API_KEY):
            raise HTTPException(401, "Invalid API key")

    body = await request.body()
    if len(body) > MAX_BODY:
        raise HTTPException(413, "Request too large")

    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) > 100:
        raise HTTPException(400, "Invalid messages")

    model = payload.get("model", MODEL)
    if model != MODEL:
        raise HTTPException(400, f"Only model {MODEL} is allowed")

    prompt = next(
        (
            m["content"].strip()
            for m in reversed(messages)
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ),
        None,
    )

    if prompt and len(prompt) > MAX_PROMPT:
        raise HTTPException(413, "Prompt too long")

    commitment = prompt_hash(prompt) if prompt else None

    # This prototype uses non-streaming inference.
    payload["model"] = MODEL
    payload["stream"] = False

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_URL.rstrip('/')}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

    except httpx.TimeoutException:
        raise HTTPException(504, "Ollama timed out")
    except httpx.HTTPError:
        log.exception("Ollama connection failed")
        raise HTTPException(502, "Ollama unavailable")

    is_new = record_contribution(commitment) if commitment else False

    return {
        "response": result,
        "contribution": {
            "commitment": commitment,
            "unique": is_new,
            "points_earned": int(is_new),
            "chain_status": (
                "pending" if is_new and chain_configured()
                else "local-queue" if is_new
                else "duplicate-or-no-prompt"
            ),
        },
    }


@app.get("/stats")
async def stats():
    with sqlite3.connect(DB_PATH) as db:
        total = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(points), 0) FROM contributions"
        ).fetchone()

        queue = db.execute(
            "SELECT status, COUNT(*) FROM queue GROUP BY status"
        ).fetchall()

    return {
        "unique_prompts": total[0],
        "simulated_points": total[1],
        "queue": dict(queue),
        "blockchain_enabled": chain_configured(),
    }


@app.get("/queue")
async def queue_status():
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        jobs = db.execute(
            """SELECT id, status, attempts, tx_hash, last_error, created_at
               FROM queue ORDER BY id DESC LIMIT 100"""
        ).fetchall()

    return {
        "blockchain_enabled": chain_configured(),
        "jobs": [dict(job) for job in jobs],
    }


@app.exception_handler(HTTPException)
async def handle_http_error(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )