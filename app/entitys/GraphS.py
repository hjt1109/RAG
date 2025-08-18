from pydantic import BaseModel, Field
from typing import List, Optional


class GraphRequestbyFileId(BaseModel):
    question: str
    file_id: str
    top_k: Optional[int] = 5

class GraphResponsebyFileId(BaseModel):
    file_id: str
    file_name: str
    text: str
    score: float


class MultiStepRequest(BaseModel):
    """Request model for multi-step graph-enhanced retrieval query."""
    question: str = Field(
        ...,
        example="#操作步骤：\n1、登录&&核心系统&&\n2、进入<存款账户信息查询>交易，查询账户A的当前余额为Y元(数据要求:有效的个人活期存款账户A)\n3、进入<个人现金存款>交易\n4、输入账户A账号和金额X元\n5、提交交易，交易成功\n6、进入<存款账户信息查询>交易，查询账户A的当前余额增加X元",
        description="The multi-step query string with numbered steps"
    )
    file_id: str = Field(..., example="1234567890", description="The file ID of the document")

class MultiStepItem(BaseModel):
    """Item model for multi-step graph-enhanced retrieval query."""
    step_id: int = Field(..., ge=1, description="The step number")
    step_text: str = Field(..., example="组件名称：登录测管", description="The graph-validated component for the step")

class MultiStepResponse(BaseModel):
    """Response model for multi-step graph-enhanced retrieval query."""
    question: str = Field(..., description="The original multi-step query")
    items: List[MultiStepItem] = Field(..., description="List of steps with their corresponding components")

    