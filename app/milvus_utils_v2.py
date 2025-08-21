from pymilvus import MilvusClient, DataType
from loguru import logger
import uuid
from typing import List, Tuple, Dict, Any, Optional
from config import MILVUS_HOST, MILVUS_PORT
from pypinyin import pinyin, Style


class My_MilvusClient:
    def __init__(self, dim: int = 1024, collection_name: str = "Component_Table"):
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
        self.field_name_mapping = {}  # 存储原始字段名到规范化字段名的映射

    def _normalize_field_name(self, field_name: str) -> str:
        """将字段名规范化，确保只包含数字、字母和下划线"""
        if not field_name:
            return f"field_{uuid.uuid4().hex[:8]}"
        
        # 如果字段名已经符合要求（只包含字母、数字、下划线），直接返回
        if field_name.encode('ascii', 'ignore').decode('ascii') == field_name and field_name[0] in '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return field_name
        
        # 将汉字转换为拼音（无音调）
        try:
            pinyin_parts = pinyin(field_name, style=Style.NORMAL)
            normalized_name = '_'.join([part[0] for part in pinyin_parts])
        except Exception as e:
            logger.warning(f"Failed to convert field name '{field_name}' to pinyin: {e}")
            normalized_name = f"field_{uuid.uuid4().hex[:8]}"
        
        # 确保以字母或下划线开头，替换非法字符
        if not normalized_name[0].isalpha() and normalized_name[0] != '_':
            normalized_name = f"_{normalized_name}"
        normalized_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized_name)
        
        # 避免字段名重复
        base_name = normalized_name
        counter = 1
        while normalized_name in self.field_name_mapping.values():
            normalized_name = f"{base_name}_{counter}"
            counter += 1
        
        return normalized_name

    def _prepare_collection(self, collection_name: str, headers: Optional[List[str]] = None):
        has_collection = self.client.has_collection(collection_name=collection_name)
        if has_collection:
            logger.info(f"Collection '{collection_name}' already exists.")
            self.client.load_collection(collection_name=collection_name)
            # 加载现有字段映射（如果需要）
            try:
                collection_info = self.client.describe_collection(collection_name)
                fields = collection_info.get("fields", [])
                for field in fields:
                    field_name = field["name"]
                    if field_name.startswith("field_") or field_name in ["id", "embedding", "file_id", "file_name"]:
                        continue
                    # 假设原始字段名与规范化字段名一致（可根据需要扩展）
                    self.field_name_mapping[field_name] = field_name
            except Exception as e:
                logger.warning(f"Failed to load field mappings for existing collection: {e}")
            return

        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        
        # 清空字段名映射
        self.field_name_mapping = {}
        
        # 动态添加表头字段
        if headers:
            for header in headers:
                normalized_header = self._normalize_field_name(header)
                schema.add_field(normalized_header, DataType.VARCHAR, max_length=65535)
                self.field_name_mapping[header] = normalized_header
                logger.debug(f"Mapped original field '{header}' to normalized field '{normalized_header}'")
        else:
            # 默认字段，保持向后兼容
            default_fields = ["组件名称", "组件ID", "组件类型", "交易系统", "组件说明"]
            for field in default_fields:
                normalized_field = self._normalize_field_name(field)
                schema.add_field(normalized_field, DataType.VARCHAR, max_length=65535)
                self.field_name_mapping[field] = normalized_field
                logger.debug(f"Mapped default field '{field}' to normalized field '{normalized_field}'")
        
        schema.add_field("file_id", DataType.VARCHAR, max_length=36)
        schema.add_field("file_name", DataType.VARCHAR, max_length=255)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024}
        )

        try:
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"Collection {collection_name} created & loaded.")
        except Exception as e:
            logger.error(f"Failed to create collection: {collection_name}, error: {e}")
            raise

    def insert_documents(self, texts: Dict[str, List[str]], embeddings: List[List[float]], file_id: str, file_name: str):
        if "组件名称" not in texts:
            logger.error("未找到 '组件名称' 列，无法进行嵌入化处理")
            raise ValueError("texts 字典中必须包含 '组件名称' 列")
        
        # 获取表头并规范化
        headers = list(texts.keys())
        self._prepare_collection(collection_name=self.collection_name, headers=headers)
        
        component_texts = texts["组件名称"]
        if len(component_texts) != len(embeddings):
            logger.error(f"嵌入向量长度 ({len(embeddings)}) 与 '组件名称' 列长度 ({len(component_texts)}) 不匹配")
            raise ValueError("嵌入向量长度必须与 '组件名称' 列长度一致")
        
        num_rows = len(component_texts)
        data = []
        for i in range(num_rows):
            row_data = {
                "id": str(uuid.uuid4()),
                "embedding": embeddings[i],
                "file_id": file_id,
                "file_name": file_name
            }
            # 使用规范化后的字段名插入数据
            for header in headers:
                normalized_header = self.field_name_mapping.get(header, self._normalize_field_name(header))
                row_data[normalized_header] = texts[header][i] if i < len(texts[header]) else ""
            data.append(row_data)
        
        self.client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"Inserted {len(data)} rows with file_id {file_id} and file_name {file_name}.")

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["file_id", "file_name"] + list(self.field_name_mapping.values())
        )[0]
        logger.info(f"Search results: {results}")
        distances = [hit["distance"] for hit in results]
        normalized_scores = self.normalize_distance(distances)
        
        # 将规范化字段名转换回原始字段名
        search_results = []
        for hit, score in zip(results, normalized_scores):
            entity = hit["entity"]
            # 转换字段名
            converted_entity = {
                "file_id": entity["file_id"],
                "file_name": entity["file_name"]
            }
            for original_field, normalized_field in self.field_name_mapping.items():
                if normalized_field in entity:
                    converted_entity[original_field] = entity[normalized_field]
            search_results.append((converted_entity, score))
        
        logger.info(f"Search results with normalized scores: {search_results}")
        return search_results

    def search_similar_in_file(self, query_embedding: List[float], file_id: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            filter=f'file_id == "{file_id}" AND ',
            output_fields=["file_id", "file_name"] + list(self.field_name_mapping.values())
        )[0]
        file_name = results[0]["entity"]["file_name"] if results else ""
        logger.info(f"Search in file_name={file_name} and file_id={file_id} ----> results: {results}")
        distances = [hit["distance"] for hit in results]
        
        # 将规范化字段名转换回原始字段名
        search_results = []
        for hit, score in zip(results, distances):
            entity = hit["entity"]
            converted_entity = {
                "file_id": entity["file_id"],
                "file_name": entity["file_name"]
            }
            for original_field, normalized_field in self.field_name_mapping.items():
                if normalized_field in entity:
                    converted_entity[original_field] = entity[normalized_field]
            search_results.append((converted_entity, score))
        
        logger.info(f"Search results with normalized scores: {search_results}")
        return search_results

    def get_file_id_by_name(self, file_name: str) -> str:
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'file_name == "{file_name}"',
            output_fields=["file_id"]
        )
        return results[0]["file_id"] if results else None
    
    def search_similar_texts_only(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        results = self.search_similar(query_embedding, top_k)
        texts = [str(entity) for entity, _ in results]
        logger.info(f"检索到的内容: {texts}")
        return texts

    def search_similar_by_filename(self, query_embedding: List[float], file_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        file_id = self.resolve_filename_to_id(file_name)
        if not file_id:
            logger.warning(f"未找到文件名: {file_name}")
            return []
        return self.search_similar_in_file(query_embedding, file_id, top_k)

    def refresh_filename_map(self) -> Dict[str, str]:
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["file_name", "file_id"],
                limit=16383
            )
            mapping = {}
            for row in results:
                mapping[row["file_name"]] = row["file_id"]
            logger.info(f"已刷新文件名映射，共 {len(mapping)} 条")
            return mapping
        except Exception as e:
            logger.error(f"刷新文件名映射失败: {e}")
            return {}

    def resolve_filename_to_id(self, raw: str) -> Optional[str]:
        raw = raw.strip()
        if not raw:
            return None
        mapping = self.refresh_filename_map()
        if raw in mapping:
            return mapping[raw]
        if raw.startswith("file_"):
            return raw
        return None

    def get_file_name_by_id(self, file_id: str) -> Optional[str]:
        if not file_id:
            return None
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f'file_id == "{file_id}"',
            output_fields=["file_name"],
            limit=1
        )
        return rows[0].get("file_name") if rows else None