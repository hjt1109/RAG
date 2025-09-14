from .milvus_utils_v2 import My_MilvusClient
from loguru import logger
from typing import List, Tuple, Dict, Any
from .embedding_utils  import EmbeddingModel

embedding_model = EmbeddingModel()
milvus_client = My_MilvusClient()

def Multi_Retrieval_withfile_id(components : List[str], system_name : str, file_id : str, filter_score : float, top_k : int = 5) ->  Dict[str, List] :

    """
    This function is used to retrieve the similar documents based on the given components and system_name.
    :param components: A list of components.
    :param system_name: The name of the system.
    :param file_id: The file_id of the system_name.
    :param filter_score: The minimum score of the retrieved documents.
    :param top_k: The number of retrieved documents.
    :return: A list of retrieved documents and their scores.
    """
    num = len(components)
    logger.info(f"Number of components: {num}")
    all_results = {}
    for i in range(num):
        component = components[i]
        query_embedding = embedding_model.encode([component])[0]
        logger.info(f"Query embedding: {query_embedding}")
        results = milvus_client.search_similar_in_file(system_name, query_embedding, top_k, filter_score, file_id)
        all_results[component] = results
    return all_results

def Multi_Retrieval_withoutfile_id(components : List[str], system_name : str, filter_score : float, top_k : int = 5) ->  Dict[str, List] :

    num = len(components)
    logger.info(f"Number of components: {num}")
    all_results = {}
    for i in range(num):
        component = components[i]
        query_embedding = embedding_model.encode([component])[0]
        logger.info(f"Query embedding: {query_embedding}")
        results = milvus_client.search_similar(system_name, query_embedding, top_k, filter_score)
        all_results[component] = results
    return all_results


