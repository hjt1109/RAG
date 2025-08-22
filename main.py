from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logger import setup_logger
from app.api.rag_endpoints import router as rag_router
from app.api.chat_endpoints import router as chat_router
from app.api.document_endpoints import router as document_router
from app.api.health_endpoints import router as health_router
from app.api.rerank_endpoints import router as rerank_router
from app.api.delete_endpoints import router as delete_router
from app.api.collection_endpoints import router as collection_router
from app.api.graph_retrieval_endpoints import router as graph_retrieval_router

app = FastAPI(
    title="RAG System API",
    version="1.0.0",
    description="A simple Retrieval-Augmented Generation (RAG) system API"
)

setup_logger()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地调试方便
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含API路由
app.include_router(rag_router)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(health_router)
app.include_router(rerank_router)
app.include_router(delete_router)
app.include_router(collection_router)
app.include_router(graph_retrieval_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
