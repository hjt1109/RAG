from pydantic import BaseModel, Field
from typing import List, Optional

class DeleteCollectionRequest(BaseModel):
    collection_id: str = Field(..., description="知识库ID")
    collection_name: Optional[str] = Field(None, description="知识库名称")
    force: bool = Field(False, description="是否强制删除（不检查是否存在）")

class DeleteCollectionResponseData(BaseModel):
    status: str = Field(..., description="知识库删除成功/知识库删除失败")
    deleted_collection_id: str = Field(..., description="被删除的知识库ID")
    deleted_collection_name: str = Field(..., description="被删除的知识库名称")
    deleted_document_count: int = Field(0, description="删除的文档数量")

class DeleteCollectionResponse(BaseModel):
    status: bool = Field(..., description="成功失败标识")
    message: str = Field(..., description="接口提示信息")
    data: DeleteCollectionResponseData

class CollectionInfo(BaseModel):
    collection_id: str = Field(..., description="知识库ID")
    collection_name: str = Field(..., description="知识库名称")
    document_count: int = Field(..., description="文档数量")
    description: str = Field("", description="知识库描述")
    is_current: bool = Field(False, description="是否为当前使用的知识库")

class ListCollectionsResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    total_collections: int = Field(..., description="知识库总数")
    collections: List[CollectionInfo] = Field(..., description="知识库列表") 