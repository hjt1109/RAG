from pydantic import BaseModel, Field
from typing import Optional, List

class RetrievalRequest(BaseModel):
    question: str = Field(..., description=" 问题 ")
    rerank_topk: Optional[int] = Field(None, description=" rerank topk ")
    filter_score: Optional[float] = Field(0.7, description=" 过滤分数yuzhi ")
    initial_topk: Optional[int] = Field(10, description=" 初始topk ")
    file_id: Optional[str] = Field(None, description=" 文件id ")
    use_reranker: Optional[bool] = Field(True, description=" 是否使用reranker ")

class RetrievalInfo(BaseModel):
    content: str = Field(..., description=" 内容 ")
    initial_score: Optional[float] = Field(None, description=" 初始分数 ")
    rerank_score: Optional[float] = Field(None, description=" rerank分数 ")
    file_id: Optional[str] = Field(None, description=" 文件id ")

class RetrievalResponse(BaseModel):
    question: str = Field(..., description=" 问题 ")
    retrieval_results: List[RetrievalInfo] = Field(..., description=" 检索结果 ")  
    total_time: Optional[float] = Field(None, description=" 总耗时 ")
