"""FastAPI application factory for the OpenCredit backend.

Run from the repository root:

    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opencredit.app")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="OpenCredit AI",
        description=(
            "Investigation and decision-support API for informal-market "
            "businesses. Runs the investigation agent and ML assessment "
            "pipeline and returns an evidence-backed report."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Surface friendly, field-level messages instead of raw Pydantic errors.
        errors = []
        for err in exc.errors():
            field = ".".join(str(loc) for loc in err.get("loc", []) if loc != "body")
            message = err.get("msg", "").removeprefix("Value error, ")
            if field:
                errors.append(f"{field}: {message}" if message else field)
            elif message:
                errors.append(message)
        detail = " ".join(errors) or "Invalid request."
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        # Never leak stack traces to clients.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong on our side. Please try again."},
        )

    return app


app = create_app()
