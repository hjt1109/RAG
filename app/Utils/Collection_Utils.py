from loguru import logger
from typing import Dict, Any, List, Optional
from .milvus_utils import My_MilvusClient
from ..config import MILVUS_COLLECTION
from ..entitys.Delete_Collection import CollectionInfo


class CollectionManager:
    """知识库管理器，用于管理Milvus中的Collection"""
    
    def __init__(self):
        """初始化Milvus客户端"""
        try:
            self.milvus_client = My_MilvusClient()
            logger.info("知识库管理器初始化成功")
        except Exception as e:
            logger.error(f"初始化知识库管理器失败: {e}")
            raise
    
    def delete_collection(self, collection_name: str, force: bool = False) -> Dict[str, Any]:
        """
        删除指定的知识库
        
        Args:
            collection_name: 知识库名称
            force: 是否强制删除
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            logger.info(f"开始删除知识库: collection_name={collection_name}")
            
            # 检查是否是当前使用的Collection
            if collection_name == MILVUS_COLLECTION and not force:
                logger.warning(f"尝试删除当前使用的知识库: {MILVUS_COLLECTION}")
                return {
                    "success": False,
                    "message": f"不能删除当前正在使用的知识库: {MILVUS_COLLECTION}",
                    "deleted_collection_name": collection_name,
                    "deleted_document_count": 0
                }
            
            # 检查Collection是否存在
            if not self.milvus_client.client.has_collection(collection_name=collection_name) and not force:
                logger.warning(f"知识库不存在: {collection_name}")
                return {
                    "success": False,
                    "message": f"知识库不存在: {collection_name}",
                    "deleted_collection_name": collection_name,
                    "deleted_document_count": 0
                }
            
            # 获取Collection中的文档数量
            document_count = 0
            try:
                docs = self.milvus_client.client.query(
                    collection_name=collection_name,
                    output_fields=["id"],
                    limit=16383
                )
                document_count = len(docs)
                logger.info(f"知识库 {collection_name} 包含 {document_count} 个文档")
            except Exception as e:
                logger.warning(f"无法获取Collection文档数量: {e}")
            
            # 执行删除操作
            try:
                # 直接删除Collection
                self.milvus_client.client.drop_collection(collection_name=collection_name)
                logger.info(f"已删除Collection: {collection_name}")
                
                return {
                    "success": True,
                    "message": f"成功删除知识库: {collection_name}",
                    "deleted_collection_name": collection_name,
                    "deleted_document_count": document_count
                }
                
            except Exception as e:
                logger.error(f"删除Collection失败: {e}")
                return {
                    "success": False,
                    "message": f"删除知识库失败: {str(e)}",
                    "deleted_collection_name": collection_name,
                    "deleted_document_count": document_count
                }
                
        except Exception as e:
            logger.error(f"删除知识库时发生错误: {e}")
            return {
                "success": False,
                "message": f"删除知识库失败: {str(e)}",
                "deleted_collection_name": collection_name,
                "deleted_document_count": 0
            }
    
    def list_all_collections(self) -> Dict[str, Any]:
        """
        列出所有知识库信息
        
        Returns:
            Dict[str, Any]: 知识库列表信息
        """
        try:
            # 获取所有Collection信息
            all_collections = self.milvus_client.get_all_collections_with_ids()
            
            collections_info = []
            for collection in all_collections:
                # 获取Collection的文档数量
                document_count = 0
                try:
                    docs = self.milvus_client.client.query(
                        collection_name=collection["name"],
                        output_fields=["id"],
                        limit=16383
                    )
                    document_count = len(docs)
                except Exception as e:
                    logger.warning(f"无法获取Collection {collection['name']} 的文档数量: {e}")
                
                collection_info = CollectionInfo(
                    collection_id=collection["id"],
                    collection_name=collection["name"],
                    document_count=document_count,
                    description=collection.get("description", ""),
                    is_current=collection["name"] == MILVUS_COLLECTION
                )
                collections_info.append(collection_info)
            
            return {
                "success": True,
                "total_collections": len(collections_info),
                "collections": collections_info
            }
            
        except Exception as e:
            logger.error(f"获取知识库列表时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取知识库列表失败: {str(e)}",
                "total_collections": 0,
                "collections": []
            }
    
    def get_collection_info(self, collection_id: str) -> Dict[str, Any]:
        """
        获取指定知识库的详细信息
        
        Args:
            collection_id: 知识库ID
            
        Returns:
            Dict[str, Any]: 知识库信息
        """
        try:
            # 获取所有Collection信息
            all_collections = self.milvus_client.get_all_collections_with_ids()
            
            # 查找指定的Collection
            target_collection = None
            for collection in all_collections:
                if collection["id"] == collection_id:
                    target_collection = collection
                    break
            
            if not target_collection:
                return {
                    "success": False,
                    "message": f"知识库不存在: collection_id={collection_id}",
                    "collection_info": None
                }
            
            # 获取Collection的详细信息
            document_count = 0
            try:
                docs = self.milvus_client.client.query(
                    collection_name=target_collection["name"],
                    output_fields=["id", "file_id", "file_name"],
                    limit=16383
                )
                document_count = len(docs)
                
                # 统计文件数量
                file_ids = set()
                for doc in docs:
                    if "file_id" in doc:
                        file_ids.add(doc["file_id"])
                
                collection_info = {
                    "collection_id": target_collection["id"],
                    "collection_name": target_collection["name"],
                    "document_count": document_count,
                    "file_count": len(file_ids),
                    "description": target_collection.get("description", ""),
                    "is_current": target_collection["name"] == MILVUS_COLLECTION,
                    "statistics": {
                        "total_documents": document_count,
                        "total_files": len(file_ids),
                        "is_active": target_collection["name"] == MILVUS_COLLECTION
                    }
                }
                
                return {
                    "success": True,
                    "message": "获取知识库信息成功",
                    "collection_info": collection_info
                }
                
            except Exception as e:
                logger.error(f"获取Collection详细信息失败: {e}")
                return {
                    "success": False,
                    "message": f"获取知识库详细信息失败: {str(e)}",
                    "collection_info": None
                }
                
        except Exception as e:
            logger.error(f"获取知识库信息时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取知识库信息失败: {str(e)}",
                "collection_info": None
            }
    
    def get_current_collection_info(self) -> Dict[str, Any]:
        """
        获取当前使用的知识库信息
        
        Returns:
            Dict[str, Any]: 当前知识库信息
        """
        try:
            # 获取当前Collection信息
            collection_info = self.milvus_client.get_collection_info()
            
            if "error" in collection_info:
                return {
                    "success": False,
                    "message": collection_info["error"],
                    "current_collection": None
                }
            
            # 获取文档数量
            document_count = 0
            try:
                docs = self.milvus_client.client.query(
                    collection_name=MILVUS_COLLECTION,
                    output_fields=["id"],
                    limit=16383
                )
                document_count = len(docs)
            except Exception as e:
                logger.warning(f"无法获取当前Collection文档数量: {e}")
            
            return {
                "success": True,
                "message": "获取当前知识库信息成功",
                "current_collection": {
                    "collection_id": collection_info["collection_id"],
                    "collection_name": collection_info["collection_name"],
                    "document_count": document_count,
                    "description": collection_info["description"],
                    "is_current": True,
                    "statistics": collection_info["statistics"]
                }
            }
            
        except Exception as e:
            logger.error(f"获取当前知识库信息时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取当前知识库信息失败: {str(e)}",
                "current_collection": None
            }
    
    def create_collection(self, collection_name: str, dim: int = 1024, description: str = "") -> Dict[str, Any]:
        """
        创建新的知识库
        
        Args:
            collection_name: 知识库名称
            dim: 向量维度
            description: 知识库描述
            
        Returns:
            Dict[str, Any]: 创建结果
        """
        try:
            logger.info(f"开始创建知识库: collection_name={collection_name}, dim={dim}")
            
            # 检查Collection是否已存在
            if self.milvus_client.client.has_collection(collection_name=collection_name):
                return {
                    "success": False,
                    "message": f"知识库已存在: {collection_name}",
                    "collection_name": collection_name
                }
            
            # 创建Collection Schema
            schema = self.milvus_client.client.create_schema(auto_id=False)
            schema.add_field("id", "VARCHAR", max_length=36, is_primary=True)
            schema.add_field("embedding", "FLOAT_VECTOR", dim=dim)
            schema.add_field("text", "VARCHAR", max_length=65535)
            schema.add_field("file_id", "VARCHAR", max_length=36)
            schema.add_field("file_name", "VARCHAR", max_length=255)
            
            # 创建索引参数
            index_params = self.milvus_client.client.prepare_index_params()
            index_params.add_index(
                "embedding",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 1024}
            )
            
            # 创建Collection
            self.milvus_client.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"成功创建知识库: {collection_name}")
            
            return {
                "success": True,
                "message": f"成功创建知识库: {collection_name}",
                "collection_name": collection_name,
                "dim": dim,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"创建知识库失败: {e}")
            return {
                "success": False,
                "message": f"创建知识库失败: {str(e)}",
                "collection_name": collection_name
            }
    
    def switch_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        切换到指定的知识库
        
        Args:
            collection_name: 要切换到的知识库名称
            
        Returns:
            Dict[str, Any]: 切换结果
        """
        try:
            logger.info(f"开始切换到知识库: {collection_name}")
            
            # 检查Collection是否存在
            if not self.milvus_client.client.has_collection(collection_name=collection_name):
                return {
                    "success": False,
                    "message": f"知识库不存在: {collection_name}",
                    "collection_name": collection_name
                }
            
            # 加载Collection
            self.milvus_client.client.load_collection(collection_name=collection_name)
            
            logger.info(f"成功切换到知识库: {collection_name}")
            
            return {
                "success": True,
                "message": f"成功切换到知识库: {collection_name}",
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"切换知识库失败: {e}")
            return {
                "success": False,
                "message": f"切换知识库失败: {str(e)}",
                "collection_name": collection_name
            }


# 创建全局实例
collection_manager = CollectionManager()