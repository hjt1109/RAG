from pydantic import BaseModel,Field
from typing import List,Optional

class DeleFileRequest(BaseModel):
    id: str = Field(..., description="知识库ID")
    file_id: str = Field(..., description="文件ID")

class DeleFileResponseData(BaseModel):
    status: str = Field(..., description="文件删除成功/文件删除失败")

class DeleFileResponse(BaseModel):
    status: bool = Field(..., description="成功失败标识")
    message: str = Field(..., description="接口提示信息")
    data: DeleFileResponseData