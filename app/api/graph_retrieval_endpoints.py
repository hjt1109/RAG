# app/api/graph_retrieval_endpoints.py (New file: Create this file for the graph-enhanced retrieval endpoint)
from ast import Mult
from fastapi import APIRouter, HTTPException
from ..entitys.GraphS import GraphRequestbyFileId, GraphResponsebyFileId, MultiStepRequest, MultiStepItem, MultiStepResponse # Reuse your existing models
from ..Utils.rag_pipeline import RAGPipeline
from loguru import logger
from typing import List

router = APIRouter(prefix="/graph_retrieval", tags=["graph_retrieval"])

rag_pipeline = RAGPipeline()  # Initialize the pipeline

@router.post("/by_file_id", response_model=GraphResponsebyFileId)
async def graph_retrieval_by_file_id(request: GraphRequestbyFileId) -> GraphResponsebyFileId:
    """Endpoint for graph-enhanced retrieval within a file."""
    question = request.question
    file_id = request.file_id
    if not question or not file_id:
        raise HTTPException(status_code=400, detail="Missing question or file_id")
    try:
        sequence = rag_pipeline.query_in_file(question, file_id, use_graph=False)
        logger.info(f"Graph retrieval in file '{file_id}' for question '{question}': {sequence}")
        return GraphResponsebyFileId(file_id=file_id, file_name=file_id, text=sequence, score=0.0)
    except Exception as e:
        logger.error(f"Graph retrieval in file failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi_step", response_model=MultiStepResponse)
async def graph_multi_step(request: MultiStepRequest):
    """Endpoint for multi-step graph-enhanced retrieval. Returns a sequence of components."""
    try:
        sequence = rag_pipeline.query_multi_step(request.question,request.file_id, use_graph=False)
        items = [MultiStepItem(step_id=item["step_id"], step_text=item["step_text"]) for item in sequence]
        logger.info(f"Multi-step query result: {items}")
        return MultiStepResponse(question=request.question, items=items)
    except Exception as e:
        logger.error(f"Multi-step query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


