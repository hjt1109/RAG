from unittest import result
from numpy.random import f
from pymilvus.orm import collection
from milvus_utils import My_MilvusClient

mc = My_MilvusClient(dim = 128, collection_name="test_collection")
class TestMilvus:
    def __init__(self) :
        self.dim = 128
        self.collection_name = "test_collection"
    
    def searchinMy_MilvusClient(self, top_k):
        results = mc.client.search(
            collection_name = "test_collection",
            data = [[1.0]*self.dim],
            limit=top_k,
            output_fields = ["e"]
        )
        return results[0]



if __name__ == '__main__':
    test = TestMilvus()

    print(test.searchinMy_MilvusClient(1))