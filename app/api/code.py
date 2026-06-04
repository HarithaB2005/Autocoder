"""
/api/v1/process  — primary endpoint
/api/v1/process/stream — SSE streaming version (yields pipeline steps live)
"""

import asyncio
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.models import CodeRequest, CodeResponse
from app.agents.orchestrator import orchestrator
from app.core.security import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


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
