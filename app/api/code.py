"""
/api/v1/upload          — upload a file, get its text content back
/api/v1/process         — primary endpoint
/api/v1/process/stream  — SSE streaming version (yields pipeline steps live)
"""

import asyncio
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app.schemas.models import CodeRequest, CodeResponse
from app.agents.orchestrator import orchestrator
from app.core.security import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


# ── File Upload Endpoint ──────────────────────────────────────────────────────

@router.post(
    "/upload",
    summary="Upload a file and get its text content",
    description=(
        "Upload any plain-text file (source code, document, template, CSV, etc.) "
        "and receive its content as a string. "
        "Use the returned `file_content` value in the `file_content` or `template_content` "
        "field of the `/process` request body.\n\n"
        "**Workflow:**\n"
        "1. Upload your document → copy `file_content` from response\n"
        "2. Upload your template → copy `file_content` as `template_content`\n"
        "3. Paste both into `/process` request body"
    ),
)
async def upload_file(
    file: UploadFile = File(..., description="Any plain-text file — .txt, .py, .js, .csv, .html, .md, .json etc."),
    _:    str        = Depends(require_api_key),
):
    """
    Read an uploaded file and return its text content.
    Use the returned content in /process as file_content or template_content.
    """
    MAX_SIZE = 500_000  # 500 KB — matches CodeRequest.file_content limit

    try:
        raw = await file.read()

        if len(raw) > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {len(raw):,} bytes. "
                    f"Maximum allowed: {MAX_SIZE:,} bytes (500 KB)."
                ),
            )

        # Attempt UTF-8 decode first, then latin-1 as fallback
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
                logger.warning(
                    "File '%s' decoded using latin-1 fallback — "
                    "some characters may not render correctly.",
                    file.filename,
                )
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"File '{file.filename}' does not appear to be a text file. "
                        "Only plain-text files are supported (no PDFs, images, or binaries)."
                    ),
                )

        return {
            "file_name":    file.filename,
            "file_content": text,
            "size_chars":   len(text),
            "size_bytes":   len(raw),
            "encoding":     "utf-8",
            "message": (
                "Copy 'file_content' and paste it into the "
                "'file_content' or 'template_content' field in /process."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /upload")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/process",
    response_model=CodeResponse,
    summary="Run the agentic pipeline",
    description=(
        "Classifies intent, routes to the best specialist agent, "
        "runs validation with configurable self-correction retries, "
        "and returns the final result with full pipeline telemetry."
    ),
)
async def process_code(
    req: CodeRequest,
    _:   str = Depends(require_api_key),
) -> CodeResponse:
    try:
        # Run blocking orchestrator in a thread pool so the event loop stays free
        response = await asyncio.to_thread(
            orchestrator.process,
            prompt=req.prompt,
            file_content=req.file_content,
            file_name=req.file_name,
            template_content=req.template_content,
            template_name=req.template_name,
            session_id=req.session_id,
            max_tokens=req.max_tokens,
            validation_max_retries=req.validation_max_retries,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")
    except Exception as e:
        logger.exception("Unexpected error in /process")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/process/stream",
    summary="Run the pipeline with Server-Sent Events (live step updates)",
    description=(
        "Same as /process but streams each pipeline step as a JSON SSE event "
        "so the client can show a live progress feed. "
        "The final event has type='done' and contains the full CodeResponse."
    ),
)
async def process_code_stream(
    req: CodeRequest,
    _:   str = Depends(require_api_key),
):
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        # Run orchestrator in thread; push a sentinel when done
        async def run():
            try:
                result = await asyncio.to_thread(
                    orchestrator.process,
                    prompt=req.prompt,
                    file_content=req.file_content,
                    file_name=req.file_name,
                    template_content=req.template_content,
                    template_name=req.template_name,
                    session_id=req.session_id,
                    max_tokens=req.max_tokens,
                    validation_max_retries=req.validation_max_retries,
                )
                await queue.put(("done", result))
            except Exception as e:
                await queue.put(("error", str(e)))

        asyncio.create_task(run())

        # Stream step-by-step events
        # Since the orchestrator is synchronous, we emit all steps at the end.
        # For true streaming, refactor orchestrator.process() to an async generator.
        item_type, payload = await queue.get()

        if item_type == "done":
            # Emit each pipeline step
            for step in payload.steps:
                event = json.dumps({"type": step.type, "message": step.message})
                yield f"data: {event}\n\n"
                await asyncio.sleep(0)  # yield control to event loop

            # Final event with full response
            final = json.dumps({"type": "done", "data": payload.model_dump()})
            yield f"data: {final}\n\n"
        else:
            error = json.dumps({"type": "error", "message": payload})
            yield f"data: {error}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
