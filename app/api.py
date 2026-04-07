from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from app import config
from app.env import TradeDeskOpenEnv
from app.logger import get_logger
from app.models import Action
from app.tasks import get_default_task_id

log = get_logger("api")

env = TradeDeskOpenEnv()
app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="Deterministic OpenEnv-style trading desk benchmark with graded tasks.",
)

# ── CORS (allows browser-based tools to call the API) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ──────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── Global exception handlers ──────────────────
@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, exc: ValidationError):
    log.warning("Validation error: %s", exc.error_count())
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "message": "Request body failed schema validation. Check the 'detail' field.",
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(_request: Request, exc: Exception):
    log.error("Unhandled error: %s — %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc),
            "message": "An unexpected error occurred.",
        },
    )


# ── Models ──────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = None


# ── Routes ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": config.APP_TITLE,
        "version": config.APP_VERSION,
        "status": "ok",
        "default_task": get_default_task_id(),
    }


@app.get("/health")
def health():
    state = env.state()
    return {
        "status": "ok",
        "task_loaded": state.task_id is not None,
        "done": state.done,
        "default_task": get_default_task_id(),
    }


@app.get("/tasks")
def tasks():
    return {"tasks": env.available_tasks()}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    try:
        task_id = req.task_id if req and req.task_id else None
        obs = env.reset(task_id)
        return obs.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state():
    return env.state().model_dump()
