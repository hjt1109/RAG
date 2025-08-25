from pymilvus import MilvusClient , DataType
from loguru import logger
from pymilvus.orm import collection
from ..config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
import uuid
from typing import List, Tuple, Dict, Any,Optional
import time
import numpy as np

class My_MilvusClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(My_MilvusClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, dim: int = 1024, collection_name: str = MILVUS_COLLECTION):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        try:
            self.client = MilvusClient(
                uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
                timeout=10
            )
        except Exception as e:
            logger.error(f"Failed to initialize MilvusClient: {e}")
            raise
        
        self.dim = dim
        self.collection_name = collection_name
        self._initialized = False
        self.initialize_collection()

    def initialize_collection(self):
        if self._initialized:
            logger.info(f"Collection {self.collection_name} already initialized.")
            return
        
        self._prepare_collection(collection_name=self.collection_name)
        self._initialized = True
       

    def _prepare_collection(self , collection_name: str ):
        #使用self.clent 统一操作
        has_collection = self.client.has_collection(collection_name=self.collection_name)
        if has_collection:
            logger.info(f"Collection '{collection_name}' already exists.")
            # 确保 collection 已加载
            self.client.load_collection(collection_name=self.collection_name)
            return
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Creating...")
            schema = MilvusClient.create_schema(auto_id=False)
            schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("file_id", DataType.VARCHAR, max_length=36)
            schema.add_field("file_name", DataType.VARCHAR, max_length=255)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                "embedding",
                index_type="IVF_FLAT",
                # metric_type="L2",#欧氏距离
                metric_type="COSINE" ,#余弦相似度 
                params={"nlist": 1024}
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"Collection {self.collection_name} created ")
            
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} loaded.")   

            return
    

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            all_collections = self.client.list_collections() # 也可以用 client 获取

            #  正确方式：使用 MilvusClient 的 describe_collection
            collection_info = self.client.describe_collection(MILVUS_COLLECTION)
            collection_stats = self.client.get_collection_stats(MILVUS_COLLECTION)

            info = {
                "collection_name": MILVUS_COLLECTION,
                "collection_id": str(collection_info.get("collection_id", "N/A")),
                "description": collection_info.get("description", ""),
                "collection_schema": collection_info.get("fields", []),
                "statistics": collection_stats,
                "all_collections": all_collections,
                "total_collections": len(all_collections),
                "is_exist": True
            }

            logger.debug(f"Collection info: {info}")
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
        
    
    def get_collection_id(self) -> str:
        """
        获取当前Collection的ID
        """
        try:
            # 尝试从MilvusClient获取Collection ID
            collection_id = "N/A"
            
            try:
                # 使用MilvusClient的describe_collection方法
                client_info = self.client.describe_collection(MILVUS_COLLECTION)
                logger.debug(f"Client collection info for ID: {client_info}")
                
                # 尝试从不同字段获取collection_id
                if isinstance(client_info, dict):
                    collection_id = str(client_info.get("collection_id", client_info.get("id", "N/A")))
                elif hasattr(client_info, 'collection_id'):
                    collection_id = str(client_info.collection_id)
                elif hasattr(client_info, 'id'):
                    collection_id = str(client_info.id)
                    
            except Exception as e:
                logger.warning(f"Failed to get collection ID via client: {e}")
                
            
            logger.info(f"Collection ID for '{MILVUS_COLLECTION}': {collection_id}")
            return collection_id
        except Exception as e:
            logger.error(f"Failed to get collection ID: {e}")
            return "N/A"
    
    def get_all_collections_with_ids(self) -> List[Dict[str, Any]]:
        """
        获取所有Collections及其ID的详细信息
        """
        try:
            collections = self.client.list_collections()
            collection_details = []
            
            for collection_name in collections:
                try:
                    # 尝试获取Collection信息
                    collection_info = {}
                    collection_stats = {}
                    collection_id = "N/A"
                    
                    
                    # 尝试从MilvusClient获取Collection ID
                    try:
                        client_info = self.client.describe_collection(collection_name)
                        if isinstance(client_info, dict):
                            collection_id = str(client_info.get("collection_id", client_info.get("id", "N/A")))
                        elif hasattr(client_info, 'collection_id'):
                            collection_id = str(client_info.collection_id)
                        elif hasattr(client_info, 'id'):
                            collection_id = str(client_info.id)
                    except Exception as e:
                        logger.warning(f"Failed to get ID for collection {collection_name}: {e}")
                    
                    collection_details.append({
                        "name": collection_name,
                        "id": collection_id,
                        "description": collection_info.get("description", ""),
                        "statistics": collection_stats,
                        "is_current": collection_name == MILVUS_COLLECTION
                    })
                except Exception as e:
                    logger.warning(f"Failed to get details for collection {collection_name}: {e}")
                    collection_details.append({
                        "name": collection_name,
                        "id": "N/A",
                        "description": "Error getting details",
                        "statistics": {},
                        "is_current": collection_name == MILVUS_COLLECTION
                    })
            
            return collection_details
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def list_all_collections(self) -> List[Dict[str, Any]]:
        """
        列出所有Collection及其ID（向后兼容）
        """
        return self.get_all_collections_with_ids()

    def normalize_distance(self, distances: List[float]) -> List[float]:
        """
        对距离进行归一化处理
        使用min-max归一化，将距离值映射到[0,1]区间
        """
        if not distances:
            return []
        
        min_dist = min(distances)
        max_dist = max(distances)
        
        # 避免除零错误
        if max_dist == min_dist:
            return [1.0] * len(distances)
        
        # min-max归一化，距离越小相似度越高，所以用1减去归一化值
        normalized = [0.01 + (d - min_dist) * 0.99 / (max_dist - min_dist) for d in distances]
        return normalized   
    
     # ---------- 插入 ----------
    def insert_documents(self, texts: List[str], embeddings: List[List[float]], file_id: str, file_name: str):
        data = [
            {"id": str(uuid.uuid4()), "embedding": emb, "text": txt, "file_id": file_id, "file_name": file_name}
            for txt, emb in zip(texts, embeddings)
        ]
        self.client.insert(collection_name=MILVUS_COLLECTION, data=data)
        logger.info(f"Inserted {len(texts)} docs with file_id {file_id} and file_name {file_name}.")
   
      # ---------- 检索 ----------
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        results = self.client.search(
            collection_name=MILVUS_COLLECTION,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text","file_id", "file_name"]
        )[0]
        logger.info(f"Search results: {results}")
        distances = [hit["distance"] for hit in results]
        normalized_scores = self.normalize_distance(distances)
        search_results = [(hit["entity"]["text"], score) for hit, score in zip(results, normalized_scores)]
        logger.info(f"Search results with normalized scores: {search_results}")
        return search_results

    def search_similar_in_file(self, query_embedding: List[float], file_id: str, top_k: int ) -> List[Tuple[str, float]]:
        results = self.client.search(
            collection_name=MILVUS_COLLECTION,
            data=[query_embedding],
            limit=top_k,
            filter=f'file_id == "{file_id}"',
            output_fields=["text","file_id", "file_name"]
        )[0]
        file_name = results[0]["entity"]["file_name"]
        logger.info(f"Search in file_name={file_name} and file_id={file_id} ----> results: {results}")
        distances = [hit["distance"] for hit in results]
        # normalized_scores = self.normalize_distance(distances)
        search_results_0 = [(hit["entity"]["text"], score) for hit, score in zip(results, distances)]
        # search_results_1 = [(hit["entity"]["text"], score) for hit, score in zip(results, distances)]
        logger.info(f"Search results with normalized scores: {search_results_0}")
        return search_results_0

    def get_file_id_by_name(self, file_name: str) -> str:
        results = self.client.query(
            collection_name=MILVUS_COLLECTION,
            filter=f'file_name == "{file_name}"',
            output_fields=["file_id"]
        )
        return results[0]["file_id"] if results else None
    
    def search_similar_texts_only(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        向后兼容的方法，只返回文本列表
        """
        results = self.search_similar(query_embedding, top_k)
        texts = [text for text, _ in results]
        logger.info(f"检索到的内容: {texts}")
        return  texts

    def search_similar_by_filename(
        self,
        query_embedding: List[float],
        file_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        根据文件名直接检索，内部自动解析 file_id。
        若文件名不存在，返回空列表。
        """
        file_id = self.resolve_filename_to_id(file_name)
        if not file_id:
            logger.warning("未找到文件名: {}", file_name)
            return []

        return self.search_similar_in_file(query_embedding, file_id, top_k)

    def refresh_filename_map(self) -> Dict[str, str]:
            """
            实时到 Milvus 里做一次去重查询，把最新的 {文件名: file_id} 缓存到内存。
            返回映射字典，方便调用方直接使用。
            """
            try:
                # 只取 file_name 与 file_id 两列，按 file_name 分组即可
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="",              # 查全部
                    output_fields=["file_name", "file_id"],
                    limit=16383             # 必须带 limit，避免 Milvus 1100 报错
                )

                # 按 file_name 去重（同文件名只保留一条即可）
                mapping = {}
                for row in results:
                    mapping[row["file_name"]] = row["file_id"]

                logger.info("已刷新文件名映射，共 {} 条", len(mapping))
                return mapping
            except Exception as e:
                logger.error("刷新文件名映射失败: {}", e)
                return {}

    def resolve_filename_to_id(self, raw: str) -> Optional[str]:
        """
        把用户输入的“文件名”实时解析成真正的 file_id。
        空字符串 -> None（查全部）
        找不到   -> None（调用方可据此提示）
        已像 file_id 的串 -> 原样返回（向下兼容）
        """
        raw = raw.strip()
        if not raw:
            return None

        # 实时拉映射
        mapping = self.refresh_filename_map()

        # 1. 先按文件名匹配
        if raw in mapping:
            return mapping[raw]

        # 2. 像 file_id 就直接返回
        if raw.startswith("file_"):
            return raw

        # 3. 文件不存在
        return None
    def get_file_name_by_id(self, file_id: str) -> Optional[str]:
        """根据 file_id 返回文件名"""
        if not file_id:
            return None
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f'file_id == "{file_id}"',
            output_fields=["file_name"],
            limit=1
        )
        return rows[0].get("file_name") if rows else None

