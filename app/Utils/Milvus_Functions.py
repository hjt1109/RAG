from marshmallow import schema
from .Milvus_Connection import MilvusConnection
from pymilvus import DataType
from loguru import logger

class MilvusFunctions:
    """
    all the functions related to milvus
    """
    def __init__(self):
        self.client = MilvusConnection().client

    # 创建组件名称的collection
    def create_collection(self, collection_name, dimension = 1024, metric_type = "COSINE"):
        
        if self.client.has_collection(collection_name):
            logger.info(f"collection {collection_name} already exists")
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension, is_primary=False)
        schema.add_field("zu_jian_ming_cheng", DataType.VARCHAR, max_length=65535)
        schema.add_field("jiao_yi_xi_tong", DataType.VARCHAR, max_length=65535)
        schema.add_field("file_id", DataType.VARCHAR, max_length=36)
        schema.add_field("file_name", DataType.VARCHAR, max_length=255)
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={"nlist": 1024}
        )
        try:
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"collection {collection_name} created successfully")
        except Exception as e:
            logger.error(f"failed to create collection {collection_name}: {e}")
            raise e
        

    # 创建交易名称的collection
    def create_transaction_collection(self, collection_name, dimension = 1024, metric_type = "COSINE"):
        if self.client.has_collection(collection_name):
            logger.info(f"collection {collection_name} already exists")
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension, is_primary=False)
        schema.add_field("jiao_yi_ming_cheng", DataType.VARCHAR, max_length=65535)
        schema.add_field("file_id", DataType.VARCHAR, max_length=36)
        schema.add_field("file_name", DataType.VARCHAR, max_length=255)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={"nlist": 1024}
        )
        try:
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"collection {collection_name} created successfully")
        except Exception as e:
            logger.error(f"failed to create collection {collection_name}: {e}")
            raise e

#交易名称创建v3版本的collection
    def create_transaction_collection_v3(self, collection_name, dimension = 1024, metric_type = "COSINE"):
        if self.client.has_collection(collection_name):
            logger.info(f"collection {collection_name} already exists")
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field("Transactionembedding", DataType.FLOAT_VECTOR, dim=dimension, is_primary=False)
        schema.add_field("Functionembedding", DataType.FLOAT_VECTOR, dim=dimension, is_primary=False)
        schema.add_field("jiao_yi_ming_cheng", DataType.VARCHAR, max_length=65535)
        schema.add_field("FunctionDescription", DataType.VARCHAR, max_length=65535)
        schema.add_field("file_id", DataType.VARCHAR, max_length=36)
        schema.add_field("file_name", DataType.VARCHAR, max_length=255)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "Transactionembedding",
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={"nlist": 300}
            )
        index_params.add_index(
            "Functionembedding",
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={"nlist": 300}
            )
        try:
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"collection {collection_name} created successfully")
        except Exception as e:
            logger.error(f"failed to create collection {collection_name}: {e}")
            raise e
        
        
        




        """
        组件信息表涉及到输入参数和输出参数的创建collection的建表操作
        """
    def create_dataItem_v1(self, collection_name, dimension = 1024, metric_type = "COSINE"):
        if self.client.has_collection(collection_name):
            logger.info(f"collection {collection_name} already exists")
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field("InputParameterEmbedding", DataType.FLOAT_VECTOR, dim=dimension, is_primary=False)
        schema.add_field("OutputParameterEmbedding", DataType.FLOAT_VECTOR, dim=dimension, is_primary=False)
        schema.add_field("InputParameter", DataType.VARCHAR, max_length=65535)
        schema.add_field("OutputParameter", DataType.VARCHAR, max_length=65535)
        schema.add_field("ComponentID",DataType.VARCHAR, max_length=65535)
        schema.add_field("file_id", DataType.VARCHAR, max_length=36)
        schema.add_field("file_name", DataType.VARCHAR, max_length=255)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            "InputParameterEmbedding",
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={"nlist": 300}
            )
        index_params.add_index(
            "OutputParameterEmbedding",
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={"nlist": 300}
            )
        try:
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"collection {collection_name} created successfully")
        except Exception as e:
            logger.error(f"failed to create collection {collection_name}: {e}")
            raise e

    def insert(self, collection_name, data):

        if not self.client.has_collection(collection_name):
            logger.error(f"collection {collection_name} does not exist")
            raise Exception(f"collection {collection_name} does not exist")

        try:
            self.client.insert(collection_name, data)
            logger.info(f"data inserted into collection {collection_name} sussessfully")
        except Exception as e:
            logger.error(f"failed to insert data into collection {collection_name}: {e}")
            raise e



    # def delete(self, collection_name)
    #     self.client.delete


    # def update()
