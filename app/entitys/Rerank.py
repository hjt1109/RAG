from pydantic import BaseModel, Field
from typing import List, Optional

# 重排相关的模型
class RerankRequest(BaseModel):
    question: str = Field(..., description="查询问题")
    top_k: Optional[int] = Field(5, description="返回前k个重排结果")
    initial_top_k: Optional[int] = Field(20, description="初始检索的文档数量")
    min_score: Optional[float] = Field(0.0, description="最小重排分数阈值")
    max_score: Optional[float] = Field(1.0, description="最大重排分数阈值")
    file_id: Optional[str] = Field(None, description="按文件ID过滤")
    file_name: Optional[str] = Field(None, description="按文件名过滤")
    include_metadata: Optional[bool] = Field(True, description="是否包含元数据信息")

class RerankItem(BaseModel):
    content: str = Field(..., description="文档内容")
    rerank_score: float = Field(..., description="重排后的分数")
    initial_score: Optional[float] = Field(None, description="初始检索分数")
    file_id: Optional[str] = Field(..., description="来源文件ID")
    file_name:Optional[str] = Field(..., description="来源文件名")

class RerankResponse(BaseModel):
    question: str = Field(..., description="原始查询问题")
    total_documents: int = Field(..., description="总文档数量")
    reranked_documents: int = Field(..., description="重排后返回的文档数量")
    initial_retrieval_time_ms: Optional[float] = Field(None, description="初始检索时间")
    rerank_time_ms: Optional[float] = Field(None, description="重排处理时间")
    total_time_ms: Optional[float] = Field(None, description="总处理时间")
    rerank_model_info: Optional[dict] = Field(None, description="重排模型信息")
    results: List[RerankItem] = Field(..., description="重排结果列表")
    metadata: Optional[dict] = Field(None, description="额外元数据信息")
    file_id : Optional[str] = Field(None, description="文件ID")
    file_name : Optional[str] = Field(None, description="文件名")

class RerankBatchRequest(BaseModel):
    questions: List[str] = Field(..., description="批量查询问题列表")
    top_k: Optional[int] = Field(5, description="每个问题返回前k个重排结果")
    batch_size: Optional[int] = Field(10, description="批处理大小")

class RerankBatchResponse(BaseModel):
    batch_id: str = Field(..., description="批处理ID")
    total_questions: int = Field(..., description="总问题数量")
    processed_questions: int = Field(..., description="已处理问题数量")
    batch_results: List[RerankResponse] = Field(..., description="批量重排结果")
    batch_processing_time_ms: Optional[float] = Field(None, description="批处理总时间")
    

