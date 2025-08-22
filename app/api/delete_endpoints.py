from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Dict, Any, List

from ..entitys.Dele_File import DeleFileRequest, DeleFileResponse, DeleFileResponseData
from ..Utils.milvus_utils import My_MilvusClient

router = APIRouter(prefix="/delete", tags=["Delete Operations"])

# 创建Milvus客户端实例
milvus_client = My_MilvusClient()


class FileDeleterAPI:
    """文件删除API类"""
    
    def __init__(self):
        self.milvus_client = milvus_client
    
    def delete_file_by_id(self, file_id: str) -> Dict[str, Any]:
        """
        根据文件ID删除文档
        
        Args:
            file_id: 要删除的文件ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            logger.info(f"开始删除文件ID: {file_id}")
            
            # 首先查询该文件ID是否存在
            existing_docs = self.milvus_client.client.query(
                collection_name=self.milvus_client.collection_name,
                filter=f'file_id == "{file_id}"',
                output_fields=["id", "file_id", "file_name"]
            )
            
            if not existing_docs:
                logger.warning(f"文件ID {file_id} 不存在")
                return {
                    "success": False,
                    "message": f"文件ID {file_id} 不存在",
                    "deleted_count": 0,
                    "file_id": file_id
                }
            
            # 获取要删除的文档ID列表
            doc_ids = [doc["id"] for doc in existing_docs]
            file_name = existing_docs[0].get("file_name", "未知文件")
            
            logger.info(f"找到 {len(doc_ids)} 个文档需要删除，文件名: {file_name}")
            
            # 执行删除操作
            delete_result = self.milvus_client.client.delete(
                collection_name=self.milvus_client.collection_name,
                pks=doc_ids
            )
            
            logger.info(f"删除操作完成，删除结果: {delete_result}")
            
            return {
                "success": True,
                "message": f"成功删除文件 {file_name} (ID: {file_id})",
                "deleted_count": len(doc_ids),
                "file_id": file_id,
                "file_name": file_name,
                "delete_result": delete_result
            }
            
        except Exception as e:
            logger.error(f"删除文件ID {file_id} 时发生错误: {e}")
            return {
                "success": False,
                "message": f"删除失败: {str(e)}",
                "deleted_count": 0,
                "file_id": file_id
            }
    
    def list_all_files(self) -> Dict[str, Any]:
        """
        列出所有文件信息
        
        Returns:
            Dict[str, Any]: 文件列表信息
        """
        try:
            # 查询所有文档，按文件ID分组
            all_docs = self.milvus_client.client.query(
                collection_name=self.milvus_client.collection_name,
                output_fields=["id", "file_id", "file_name", "text"],
                limit=16383
            )
            
            # 按文件ID分组
            files_info = {}
            for doc in all_docs:
                file_id = doc["file_id"]
                if file_id not in files_info:
                    files_info[file_id] = {
                        "file_id": file_id,
                        "file_name": doc["file_name"],
                        "doc_count": 0,
                        "sample_texts": []
                    }
                
                files_info[file_id]["doc_count"] += 1
                # 保存前3个文本作为示例
                if len(files_info[file_id]["sample_texts"]) < 3:
                    files_info[file_id]["sample_texts"].append(doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"])
            
            return {
                "success": True,
                "total_files": len(files_info),
                "files": list(files_info.values())
            }
            
        except Exception as e:
            logger.error(f"获取文件列表时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取文件列表失败: {str(e)}",
                "total_files": 0,
                "files": []
            }
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        获取指定文件ID的详细信息
        
        Args:
            file_id: 文件ID
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        try:
            docs = self.milvus_client.client.query(
                collection_name=self.milvus_client.collection_name,
                filter=f'file_id == "{file_id}"',
                output_fields=["id", "file_id", "file_name", "text"]
            )
            
            if not docs:
                return {
                    "success": False,
                    "message": f"文件ID {file_id} 不存在",
                    "file_info": None
                }
            
            file_info = {
                "file_id": file_id,
                "file_name": docs[0]["file_name"],
                "doc_count": len(docs),
                "sample_texts": [doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"] for doc in docs[:5]]
            }
            
            return {
                "success": True,
                "message": "获取文件信息成功",
                "file_info": file_info
            }
            
        except Exception as e:
            logger.error(f"获取文件信息时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取文件信息失败: {str(e)}",
                "file_info": None
            }


# 创建文件删除器实例
file_deleter = FileDeleterAPI()


@router.post("/file", summary="删除指定文件ID的文档", response_model=DeleFileResponse)
async def delete_file(request: DeleFileRequest):
    """
    删除指定文件ID的所有文档
    
    - 根据文件ID删除Milvus中的所有相关文档
    - 返回删除结果和统计信息
    """
    try:
        logger.info(f"收到删除文件请求: file_id={request.file_id}")
        
        # 执行删除操作
        result = file_deleter.delete_file_by_id(request.file_id)
        
        if result["success"]:
            return DeleFileResponse(
                status=True,
                message=result["message"],
                data=DeleFileResponseData(
                    status=f"文件删除成功，共删除 {result['deleted_count']} 个文档"
                )
            )
        else:
            return DeleFileResponse(
                status=False,
                message=result["message"],
                data=DeleFileResponseData(
                    status="文件删除失败"
                )
            )
            
    except Exception as e:
        logger.error(f"删除文件API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")


@router.get("/files", summary="列出所有文件")
async def list_files():
    """
    列出所有已上传的文件信息
    
    - 返回所有文件的ID、名称、文档数量等信息
    - 包含示例文本用于预览
    """
    try:
        result = file_deleter.list_all_files()
        
        if result["success"]:
            return {
                "success": True,
                "total_files": result["total_files"],
                "files": result["files"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"获取文件列表API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")


@router.get("/file/{file_id}", summary="获取指定文件ID的详细信息")
async def get_file_info(file_id: str):
    """
    获取指定文件ID的详细信息
    
    - 返回文件的详细信息，包括文档数量、示例文本等
    """
    try:
        result = file_deleter.get_file_info(file_id)
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "file_info": result["file_info"]
            }
        else:
            raise HTTPException(status_code=404, detail=result["message"])
            
    except Exception as e:
        logger.error(f"获取文件信息API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文件信息失败: {str(e)}")


@router.delete("/file/{file_id}", summary="删除指定文件ID的文档（DELETE方法）")
async def delete_file_by_id(file_id: str):
    """
    使用DELETE方法删除指定文件ID的文档
    
    - 这是RESTful风格的删除接口
    - 直接通过URL参数指定文件ID
    """
    try:
        logger.info(f"收到DELETE请求: file_id={file_id}")
        
        # 执行删除操作
        result = file_deleter.delete_file_by_id(file_id)
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "deleted_count": result["deleted_count"],
                "file_id": file_id,
                "file_name": result.get("file_name", "")
            }
        else:
            raise HTTPException(status_code=404, detail=result["message"])
            
    except Exception as e:
        logger.error(f"DELETE文件API调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}") 