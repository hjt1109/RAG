from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# 重排相关的模型
class RerankRequest(BaseModel):
    question: str = Field(..., description="查询问题")
    top_k: Optional[int] = Field(5, description="返回前k个重排结果")
    filter_scores: Optional[float] = Field(0.8 , description="过滤分数阈值")
    initial_top_k: Optional[int] = Field(10, description="初始检索的文档数量")
    file_id: Optional[str] = Field(None, description="按文件ID过滤")
    file_name: Optional[str] = Field(None, description="按文件名过滤")
    use_reranker: Optional[bool] = Field(False, description="是否使用重排模型")


class RerankItem(BaseModel):
    content: str = Field(..., description="文档内容")
    initial_score: Optional[float] = Field(None, description="初始检索分数")
    rerank_score: float = Field(..., description="重排后的分数")
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
    

# 定义组件信息模型
class ComponentInfo(BaseModel):
    file_id: str
    file_name: str
    组件ID: str
    组件名称: str
    组件类型: str
    交易系统: str
    组件说明: str

# 定义组件结果模型
class ComponentResult(BaseModel):
    component: ComponentInfo
    initial_score: float
    rerank_score: float

# 定义响应模型
class RerankResponse_Componets(BaseModel):
    query_id: str
    question: str
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    results: Dict[str, List[ComponentResult]]
    retrieval_time_ms: float
    rerank_time_ms: float
    total_time_ms: float

class ComponentInfo_v2(BaseModel):
    file_id: Optional[str] = None
    组件ID: Optional[str] = None
    组件名称: Optional[str] = None
    组件类型: Optional[str] = None
    交易系统: Optional[str] = None
    组件说明: Optional[str] = None
    应用类型: Optional[str] = None
    输入参数: Optional[str] = None
    输出参数: Optional[str] = None

class ComponentResult_v2(BaseModel):
    component: ComponentInfo_v2
    initial_score: Optional[float] = 0.0
    rerank_score: Optional[float] = 0.0

class RerankResponse_Componets_v2(BaseModel):
    query_id: Optional[str] = None
    question: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    results: Dict[str, List[ComponentResult_v2]]
    retrieval_time: Optional[float] = 0.0
    rerank_time: Optional[float] = 0.0
    total_time: Optional[float] = 0.0

class TransactionInfo_v2(BaseModel):
    file_id: Optional[str] = None
    交易名称: Optional[str] = None
    系统名称: Optional[str] = None
    功能描述: Optional[str] = None

class TransactionResult_v2(BaseModel):
    transaction: TransactionInfo_v2
    initial_score: Optional[float] = 0.0
    rerank_score: Optional[float] = 0.0

class RerankResponse_Transaction_v2(BaseModel):
    query_id: Optional[str] = None
    question: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    results: Dict[str, List[TransactionResult_v2]]
    retrieval_time: Optional[float] = 0.0
    rerank_time: Optional[float] = 0.0
    total_time: Optional[float] = 0.0