from fastapi import APIRouter, HTTPException, Request

from ...schemas.rag import IngestRequest, IngestResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    try:
        return await request.app.state.rag_pipeline.ingest(payload.data_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    try:
        route = None if payload.route == "auto" else payload.route
        result = await request.app.state.rag_pipeline.query(payload.query, route=route)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
