from pydantic import BaseModel, Field
from typing import List, Optional

from torch.nn import init

class IngestRequest(BaseModel):
    texts: List[str] = Field(..., example=["Document 1 text", "Document 2 text"])

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is RAG?")

class QueryResponse(BaseModel):
    answer: str = Field(..., example="RAG stands for Retrieval-Augmented Generation...")

class StatusResponse(BaseModel):
    status: str = Field(..., example="healthy")

class ChatMessage(BaseModel):
    role: str = "user"
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]

class RecallRequest(BaseModel):
    question: str = Field(..., example="What is RAG?")

class RecallItem(BaseModel):
    content: str
    rerank_score: float
    initial_score: float


class RecallResponse(BaseModel):
    Recall_Content: List[RecallItem]

class DocumentUploadResponse(BaseModel):
    file_id: str = Field(..., description="唯一文件ID")
    status_code: int = Field(200, description="状态码")
    message: str = Field("插入成功", description="状态消息")
    processed_count: int = Field(..., description="处理的记录数量")

class QueryByFileNameRequest(BaseModel):
    question: str
    file_name: str
    top_k: Optional[int] = 5

class QueryByFileNameResponseItem(BaseModel):
    file_id: str
    file_name: str
    text: str
    score: float

