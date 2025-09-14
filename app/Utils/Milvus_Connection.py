from pymilvus import MilvusClient 
from loguru import logger
from ..config import MILVUS_HOST, MILVUS_PORT



class MilvusConnection:
    """
    单纯为了解决链接数据库，选择使用哪个collection，进行初次检索筛选的功能
    """
    def __init__(self):
     
        try:
            self.client = MilvusClient(
                uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}",
                timeout = 10
            )

        except Exception as e:
            logger.error(f"Milvus connection error: {e}")
            raise e
    



    
