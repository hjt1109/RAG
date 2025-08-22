from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Dict, Any, List, Optional

from ..entitys.Delete_Collection import (
    DeleteCollectionRequest, 
    DeleteCollectionResponse, 
    DeleteCollectionResponseData,
    CollectionInfo,
    ListCollectionsResponse
)
from ..Utils.Collection_Utils import collection_manager

router = APIRouter(prefix="/collection", tags=["Collection Management"])


@router.post("/delete", summary="删除指定知识库", response_model=DeleteCollectionResponse)
async def delete_collection(request: DeleteCollectionRequest):
    """
    删除指定的知识库
    
    - 根据知识库名称删除整个Milvus Collection
    - 返回删除结果和统计信息
    - 不能删除当前正在使用的知识库
    """
    try:
        # 优先使用collection_name，如果没有则使用collection_id作为名称
        collection_name = request.collection_name or request.collection_id
        logger.info(f"收到删除知识库请求: collection_name={collection_name}")
        
        # 执行删除操作
        result = collection_manager.delete_collection(
            collection_name=collection_name,
            force=request.force
        )
        
        if result["success"]:
            return DeleteCollectionResponse(
                status=True,
                message=result["message"],
                data=DeleteCollectionResponseData(
                    status=f"知识库删除成功，共删除 {result['deleted_document_count']} 个文档",
                    deleted_collection_id=request.collection_id,  # 保持兼容性
                    deleted_collection_name=result["deleted_collection_name"],
                    deleted_document_count=result["deleted_document_count"]
                )
            )
        else:
            return DeleteCollectionResponse(
                status=False,
                message=result["message"],
                data=DeleteCollectionResponseData(
                    status="知识库删除失败",
                    deleted_collection_id=request.collection_id,  # 保持兼容性
                    deleted_collection_name=result["deleted_collection_name"],
                    deleted_document_count=result["deleted_document_count"]
                )
            )
            
    except Exception as e:
        logger.error(f"删除知识库API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {str(e)}")


@router.get("/list", summary="列出所有知识库", response_model=ListCollectionsResponse)
async def list_collections():
    """
    列出所有知识库信息
    
    - 返回所有知识库的ID、名称、文档数量等信息
    - 标识当前正在使用的知识库
    """
    try:
        result = collection_manager.list_all_collections()
        
        if result["success"]:
            return ListCollectionsResponse(
                success=True,
                total_collections=result["total_collections"],
                collections=result["collections"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"获取知识库列表API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取知识库列表失败: {str(e)}")


@router.get("/info/{collection_id}", summary="获取指定知识库的详细信息")
async def get_collection_info(collection_id: str):
    """
    获取指定知识库的详细信息
    
    - 返回知识库的详细信息，包括文档数量、文件数量等
    """
    try:
        result = collection_manager.get_collection_info(collection_id)
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "collection_info": result["collection_info"]
            }
        else:
            raise HTTPException(status_code=404, detail=result["message"])
            
    except Exception as e:
        logger.error(f"获取知识库信息API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取知识库信息失败: {str(e)}")


@router.delete("/{collection_name}", summary="删除指定知识库（DELETE方法）")
async def delete_collection_by_name(collection_name: str, force: bool = False):
    """
    使用DELETE方法删除指定知识库
    
    - 这是RESTful风格的删除接口
    - 直接通过URL参数指定知识库名称
    - 支持force参数强制删除
    """
    try:
        logger.info(f"收到DELETE知识库请求: collection_name={collection_name}, force={force}")
        
        # 执行删除操作
        result = collection_manager.delete_collection(
            collection_name=collection_name,
            force=force
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "deleted_collection_name": result["deleted_collection_name"],
                "deleted_document_count": result["deleted_document_count"]
            }
        else:
            raise HTTPException(status_code=404, detail=result["message"])
            
    except Exception as e:
        logger.error(f"DELETE知识库API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {str(e)}")


@router.get("/current", summary="获取当前使用的知识库信息")
async def get_current_collection():
    """
    获取当前正在使用的知识库信息
    
    - 返回当前配置的知识库详细信息
    """
    try:
        result = collection_manager.get_current_collection_info()
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "current_collection": result["current_collection"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"获取当前知识库信息API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取当前知识库信息失败: {str(e)}")


@router.post("/create", summary="创建新的知识库")
async def create_collection(collection_name: str, dim: int = 1024, description: str = ""):
    """
    创建新的知识库
    
    - 创建新的Milvus Collection
    - 支持自定义向量维度和描述
    """
    try:
        logger.info(f"收到创建知识库请求: collection_name={collection_name}, dim={dim}")
        
        result = collection_manager.create_collection(
            collection_name=collection_name,
            dim=dim,
            description=description
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "collection_name": result["collection_name"],
                "dim": result["dim"],
                "description": result["description"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"创建知识库API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建知识库失败: {str(e)}")


@router.post("/switch/{collection_name}", summary="切换到指定知识库")
async def switch_collection(collection_name: str):
    """
    切换到指定的知识库
    
    - 加载指定的Collection作为当前使用的知识库
    """
    try:
        logger.info(f"收到切换知识库请求: collection_name={collection_name}")
        
        result = collection_manager.switch_collection(collection_name=collection_name)
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "collection_name": result["collection_name"]
            }
        else:
            raise HTTPException(status_code=404, detail=result["message"])
            
    except Exception as e:
        logger.error(f"切换知识库API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"切换知识库失败: {str(e)}") 