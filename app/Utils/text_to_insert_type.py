from pyclbr import Function
from .embedding_utils import  embedding_model
from typing import Dict, List


class TextToInsertType:
    def __init__(self):
        self.mapping_dict = {
            "组件名称": "zu_jian_ming_cheng",
            "交易系统": "jiao_yi_xi_tong",
            "交易名称": "jiao_yi_ming_cheng",
            "功能描述": "FunctionDescription",
            "输入参数": "InputParameter",
            "输出参数": "OutputParameter",
            "组件ID"  : "ComponentID"
        }

    # 组件名称的插入
    def text_to_insert_type(self, texts: Dict[str, List[str]], file_id: str, file_name: str) -> List[Dict]:
        
        components = texts["组件名称"]
        component_length = len(components)
        embeddings = embedding_model.encode(components)
        data = []
        for i in range(component_length):
            row_data ={
                "file_id": file_id,
                "file_name": file_name,
                "embedding": embeddings[i],
            }
            for key in texts.keys():
                if key == "组件名称" or key == "交易系统":
                    row_data[self.mapping_dict[key]] = texts[key][i]
                else:
                    row_data[key] = texts[key][i]
            data.append(row_data)
        return data

    # 交易名称的插入
    def text_to_insert_transaction_type(self,texts: Dict[str, List[str]], file_id: str, file_name: str) -> List[Dict]:
        transaction_name = texts["交易名称"]
        transaction_length = len(transaction_name)
        embeddings = embedding_model.encode(transaction_name)
        data = []
        for i in range(transaction_length):
            row_data ={
                "file_id": file_id,
                "file_name": file_name,
                "embedding": embeddings[i],
            }
            for key in texts.keys():
                if key == "交易名称":
                    row_data[self.mapping_dict[key]] = texts[key][i]
                else:
                    row_data[key] = texts[key][i]
            data.append(row_data)
        return data


    def text_to_insert_transaction_type_v2(self, texts: Dict[str, List[str]], file_id: str, file_name: str) -> List[Dict]:
        transaction_name = texts["交易名称"]
        function_description = texts["功能描述"]
        transactionAndFunction = [f"{name}:{description}" for name, description in zip(transaction_name, function_description)]
        transaction_length = len(transactionAndFunction)
        embeddings = embedding_model.encode(transactionAndFunction)
        data = []
        for i in range(transaction_length):
            row_data ={
                "file_id": file_id,
                "file_name": file_name,
                "embedding": embeddings[i],
            }
            for key in texts.keys():
                if key == "交易名称" :
                    row_data[self.mapping_dict[key]] = texts[key][i]
                else:
                    row_data[key] = texts[key][i]
            data.append(row_data)
        return data

    #交易名称v3版本
    def text_to_insert_transaction_type_v3(self, texts: Dict[str, List[str]], file_id: str, file_name: str) -> List[Dict]:
        transaction_name = texts["交易名称"]
        function_description = texts["功能描述"]
        transaction_length = len(transaction_name)
        Transactionembedding = embedding_model.encode(transaction_name)
        Functionembedding = embedding_model.encode(function_description)
        data = []
        for i in range(transaction_length):
            row_data ={
                "file_id": file_id,
                "file_name": file_name,
                "Transactionembedding": Transactionembedding[i],
                "Functionembedding": Functionembedding[i],
                }
            for key in texts.keys():
                if key == "交易名称" or key == "功能描述":
                    row_data[self.mapping_dict[key]] = texts[key][i]
                else:
                    row_data[key] = texts[key][i]
            data.append(row_data)
        return data

    """
    将入参、出参向量化
    """
    def text_to_insert_dataItem_type_v1(self, texts: Dict[str, List[str]], file_id: str, file_name: str) -> List[Dict]:
        input_parameter = texts["输入参数"]
        output_parameter = texts["输出参数"]
        transaction_length = len(input_parameter)
        InputParameterEmbedding = embedding_model.encode(input_parameter)
        OutputParameterEmbedding = embedding_model.encode(output_parameter)
        data = []
        for i in range(transaction_length):
            row_data ={
                "file_id": file_id,
                "file_name": file_name,
                "InputParameterEmbedding": InputParameterEmbedding[i],
                "OutputParameterEmbedding": OutputParameterEmbedding[i],
                }
            for key in texts.keys():
                if key == "输入参数" or key == "输出参数" or key == "组件ID":
                    row_data[self.mapping_dict[key]] = texts[key][i]
                else:
                    row_data[key] = texts[key][i]
            data.append(row_data)
        return data
    