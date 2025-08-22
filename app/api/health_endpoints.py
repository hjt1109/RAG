from fastapi import APIRouter
from ..entitys.models import StatusResponse
from ..entitys.ResMilvusId import MilVusInfo
from ..Utils.milvus_utils import My_MilvusClient
from typing import Dict, Any, List
from loguru import logger
router = APIRouter(tags=["Health Check"])

# 创建Milvus客户端实例
milvus_client = My_MilvusClient()

@router.get("/health", summary="Health check endpoint", response_model=StatusResponse)
async def health_check():
    return StatusResponse(status="healthy")

@router.get("/milvus/collection/info", summary="获取Milvus Collection详细信息",response_model=MilVusInfo)
async def get_collection_info() -> MilVusInfo :
    """
    获取当前Collection的详细信息，包括Collection ID
    """
    logger.debug("Attempting to get collection info...")
    info = milvus_client.get_collection_info()
    logger.debug(f"Collection info received: {info}")    # 打印日志
    return MilVusInfo(**info)

@router.get("/milvus/collection/id", summary="获取Milvus Collection ID")
async def get_collection_id() -> Dict[str, str]:
    """
    获取当前Collection的ID
    """
    collection_id = milvus_client.get_collection_id()
    return {"collection_name": "rag0", "collection_id": str(collection_id)}

@router.get("/milvus/collections", summary="列出所有Milvus Collections")
async def list_all_collections() -> List[Dict[str, Any]]:
    """
    列出所有Collection及其ID
    """
    return milvus_client.list_all_collections()

@router.get("/milvus/collections/detailed", summary="获取所有Milvus Collections的详细信息")
async def get_all_collections_detailed() -> List[Dict[str, Any]]:
    """
    获取所有Collections的详细信息，包括ID、统计信息等
    """
    info = milvus_client.get_all_collections_with_ids()
    return info