from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DataItemV1Request(BaseModel):
    Question : Optional[str] = Field(None, description="查询的问题")
    RerankTopK : Optional[int] = Field(None, description="重排返回个数")
    InitialFilterScores: Optional[float] = Field(0.8 , description="初始的过滤分数阈值")
    InitialTopK: Optional[int] = Field(10, description="初始检索的文档数量")
    FileID: Optional[str] = Field(None, description="文件ID")
    FileName: Optional[str] = Field(None, description="文件名")
    UseReranker: Optional[bool] = Field(False, description="是否使用重排模型")

class  DataItemV1Info(BaseModel):
    file_id : Optional[str] = Field(None, description="文件ID")
    输入参数 : Optional[str] = Field(None, description="输入参数")
    输出参数 : Optional[str] = Field(None, description="输出参数")
    组件ID  : Optional[str] = Field(None, description="组件ID")

class DataItemV1Result(BaseModel):
    Transaction: Optional[DataItemV1Info] = Field(None, description="数据组件相关信息")
    InitialScore: Optional[float] = Field(0.0, description="初始的过滤分数")
    RerankScore: Optional[float] = Field(0.0, description="重排后的过滤分数")
    AverageScore: Optional[float] = Field(0.0, description="初始和重排的平局加权分数")

class DataItemV1Response(BaseModel):
    query_id: Optional[str] = None
    question: Optional[str] = None
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    results: Dict[str, List[DataItemV1Result]]
    retrieval_time: Optional[float] = 0.0
    rerank_time: Optional[float] = 0.0
    total_time: Optional[float] = 0.0