from .Milvus_Connection import  MilvusConnection
from typing import List, Tuple
from loguru import logger
class InitialRetrieval:
    def __init__(self, collection_name: str = None):

        self.collection_name = collection_name
        self.milvus = MilvusConnection()

    def search_by_fileid(self,query_embedding: List[float], file_id: str, filter_score: float, top_k: int = 5) -> List[Tuple[str, float]]:

        results = self.milvus.client.search(
            collection_name = self.collection_name,
            data = [query_embedding],
            limit = top_k,
            filter = f"file_id == '{file_id}'",
            output_fields = ["file_id", "file_name", "zu_jian_ming_cheng"]
        )
        logger.info(f"Search results: {results}")
        results = results[0]
        distances = [hit["distance"] for hit in results]
        filtered_results = [hit for hit in results if hit["distance"] >= filter_score]
        search_results = [(hit["entity"]["zu_jian_ming_cheng"], distance) for hit, distance in zip(filtered_results, distances) ]
        logger.info(f"Filtered search results: {search_results}")
        return search_results

    def search_no_fileid(self,query_embedding: List[float], filter_score: float, top_k: int = 5) -> List[Tuple[str, float]]:
        results = self.milvus.client.search(
            collection_name = self.collection_name,
            data = [query_embedding],
            limit = top_k,
            output_fields = ["file_id", "file_name", "zu_jian_ming_cheng"]
        )
        logger.info(f"Search results: {results}")
        results = results[0]
        distances = [hit["distance"] for hit in results]
        filtered_results = [hit for hit in results if hit["distance"] >= filter_score]
        search_results = [(hit["entity"]["zu_jian_ming_cheng"], distance) for hit, distance in zip(filtered_results, distances) ]
        logger.info(f"Filtered search results: {search_results}")
        return search_results