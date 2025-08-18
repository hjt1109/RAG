from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import os

from rag_pipeline import RAGPipeline
from entitys.models import (
    IngestRequest, QueryRequest, QueryResponse, 
    RecallRequest, RecallResponse, RecallItem,
    QueryByFileNameRequest, QueryByFileNameResponseItem
)
from typing import List
from config import USE_RERANKER, RERANKER_TOP_K, INITIAL_RETRIEVAL_TOP_K

# 默认模型
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "DeepSeek-R1-Distill-Qwen-32B")

router = APIRouter(prefix="/rag", tags=["RAG Operations"])

# 全局RAG实例
rag = RAGPipeline()

@router.post("/ingest", summary="Ingest documents into the RAG system",deprecated=True)
async def ingest_documents(request: IngestRequest):
    try:
        rag.ingest_documents(request.texts)
        return {"status": "Documents ingested successfully"}
    except Exception as e:
        logger.error(f"Ingest endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", summary="Query the RAG system", response_model=QueryResponse,deprecated=True)
async def query_rag(request: QueryRequest):
    try:
        answer = rag.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recall", summary="召回检索内容和相似度分数", response_model=RecallResponse)
async def recall(request: RecallRequest):
    try:
        query_embedding = rag.embedding_model.encode([request.question])[0]
        
        # 初始检索：获取更多候选文档
        initial_top_k = INITIAL_RETRIEVAL_TOP_K if rag.reranker else 5
        results = rag.milvus_client.search_similar(query_embedding, top_k=initial_top_k)
        
        # 如果启用了重排模型，进行重排
        if rag.reranker and results:
            logger.info("开始进行文档重排...")
            reranked_results = rag.reranker.rerank_with_scores(request.question, results, top_k=RERANKER_TOP_K)
            formatted = [RecallItem(content=text, rerank_score=score, initial_score=initial_score) for text, score,initial_score in reranked_results]
        else:
            formatted = [RecallItem(content=text, score=score) for text, score in results]
        
        return RecallResponse(Recall_Content=formatted)
    except Exception as e:
        logger.error(f"Recall endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query_by_file_name", response_model=List[QueryByFileNameResponseItem],deprecated=True)
async def query_by_file_name(request: QueryByFileNameRequest):
    results = rag.query_by_file_name(request.question, request.file_name, request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="未找到对应文件或内容")
    return results

@router.get("/reranker/status", summary="获取重排模型状态")
async def get_reranker_status():
    """获取重排模型的当前状态"""
    return {
        "reranker_enabled": USE_RERANKER,
        "reranker_loaded": rag.reranker is not None,
        "reranker_top_k": RERANKER_TOP_K,
        "initial_retrieval_top_k": INITIAL_RETRIEVAL_TOP_K
    } 