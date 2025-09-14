from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TransactionV3Request(BaseModel):
    Question : Optional[str] = Field(None, description="查询的问题")
    RerankTopK : Optional[int] = Field(None, description="")
    InitialFilterScores: Optional[float] = Field(0.8 , description="初始的过滤分数阈值")
    InitialTopK: Optional[int] = Field(10, description="初始检索的文档数量")
    FileID: Optional[str] = Field(None, description="文件ID")
    FileName: Optional[str] = Field(None, description="文件名")
    UseReranker: Optional[bool] = Field(False, description="是否使用重排模型")

class TransactionV3Info(BaseModel):
    file_id: Optional[str] = Field(None, description="文件ID")
    交易名称: Optional[str] = Field(None, description="交易名称")
    系统名称: Optional[str] = Field(None, description="系统名称")
    功能描述: Optional[str] = Field(None, description="功能描述")
    查询依据: Optional[str] = Field(None, description="查询依据")

class TransactionV3Result(BaseModel):
    Transaction: Optional[TransactionV3Info] = Field(None, description="交易名称相关信息")
    InitialScore: Optional[float] = Field(0.0, description="初始的过滤分数")
    RerankScore: Optional[float] = Field(0.0, description="重排后的过滤分数")
    AverageScore: Optional[float] = Field(0.0, description="初始和重排的平局加权分数")

class TransactionV3Response(BaseModel):
    query_id: Optional[str] = None
    question: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    results: Dict[str, List[TransactionV3Result]]
    retrieval_time: Optional[float] = 0.0
    rerank_time: Optional[float] = 0.0
    total_time: Optional[float] = 0.0

class TransactionReturnInCaseGroups(BaseModel):
    results: Optional[List[str]] = None