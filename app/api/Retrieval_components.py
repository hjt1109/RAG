from fastapi import APIRouter, HTTPException
from loguru import logger
import time 
import uuid 
from ..entitys.Rerank import (
    RerankRequest,
    ComponentInfo,
    ComponentResult,
    RerankResponse_Componets
)
from ..Utils.System_Recogni import system_recogni
from ..Utils.Components_Recogni import components_recogni
from ..Utils.Mutil_Retrieval import Multi_Retrieval_withfile_id, Multi_Retrieval_withoutfile_id


router = APIRouter(prefix = "/retrieval_components", tags = ["检索组件表"])

@router.post("/retrieval", summmary = "检索组件表", response_model = RerankResponse_Componets)
async def retrieval_components(request: RerankRequest):
    start_time = time.time()
    question = request.question
    """
    question:
    #测试意图：验证：个人活期存入，交易成功

    #操作步骤：

    1、登录&&核心系统&&

    2、进入<存款账户信息查询>交易，查询账户A的当前余额为Y元(数据要求:有效的个人活期存款账户A)

    3、进入<个人现金存款>交易

    4、输入账户A账号和金额X元

    5、提交交易，交易成功

    6、进入<存款账户信息查询>交易，查询账户A的当前余额增加X元

"""
    file_id = request.file_id
    top_k = request.top_k 
    filter_score = request.filter_scores
    initial_top_k = request.initial_top_k 
    file_name = request.file_name
    use_reranker = request.use_reranker

    