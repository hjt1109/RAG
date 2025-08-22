from typing import List, Dict, Any
from loguru import logger
from .milvus_utils_v2 import My_MilvusClient
from .embedding_utils import EmbeddingModel

class DocumentUtils:

    def __init__(self):
        self.milvus_client = My_MilvusClient()
        self.embedding_model = EmbeddingModel()


    def ingest_document(self, texts: Dict[str, List[str]], file_id: str = None, file_name: str = None):
        """插入数据库：只对 '组件名称' 列进行嵌入化，文档中的表头作为数据库表的字段名称，内容按行插入"""
        try:
            if "组件名称" not in texts:
                logger.error("未找到 '组件名称' 列，无法进行嵌入化")
                raise ValueError("texts 字典中必须包含 '组件名称' 列")
            
            # 只对“组件名称”列生成嵌入向量
            component_texts = texts["组件名称"]
            embeddings = self.embedding_model.encode(component_texts)
            
            # 传递完整的 texts 字典给 insert_documents，包括所有表头和内容
            self.milvus_client.insert_documents(texts, embeddings, file_id or "", file_name or "")
            logger.info(f"Documents ingested successfully: {len(component_texts)} docs, file_id={file_id}, file_name={file_name}")
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise
