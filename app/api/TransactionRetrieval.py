from ..entitys.TransactionV3 import (
    TransactionV3Request,
    TransactionV3Info,
    TransactionV3Result,
    TransactionV3Response,
    TransactionReturnInCaseGroups
)
from fastapi import APIRouter, HTTPException, APIRouter
from loguru import logger
import time
import uuid
from ..services.RerankerService import RerankerService
from ..services.Muti_Retrieval_Service import Muti_Retrieval_Service
from ..Utils.TransactionStepParse import transactionStepParse
from ..services.MultiTransactionRetrievalV3 import MultiTransactionRetrieval
from collections import defaultdict


rerankerService = RerankerService()
router = APIRouter(prefix = "/TransactionRetrieval", tags = ["Transaction Retrieval API"])
service = Muti_Retrieval_Service()
multiTransactionRetrieval = MultiTransactionRetrieval()



#交易名称召回
@router.post("/transaction_retrieval", summary="交易名称召回", response_model= TransactionV3Response)
async def transaction_retrieval(request: TransactionV3Request):
    question = request.Question
    file_id = request.FileID
    rerank_top_k = request.RerankTopK
    filter_score = request.InitialFilterScores
    initial_top_k = request.InitialTopK
    file_name = request.FileName      
    use_reranker = request.UseReranker
    try :
        start_time = time.time()
        if not rerankerService:
            raise HTTPException(status_code=500, detail="Reranker Model is not loaded")
        query_id = str(uuid.uuid4())
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Question: {question}")
        retrieval_start_time = time.time()
        steps = transactionStepParse.transactionToSteps(question)
        logger.info(f"Steps: {steps}")
        if file_id:
            initial_results_transaction = multiTransactionRetrieval.multiTransactionRetrieval(steps = steps, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
            initial_results_function = multiTransactionRetrieval.multiFunctionDescriptionRetrievval(steps = steps, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
        else:
            initial_results_transaction = multiTransactionRetrieval.multiTransactionRetrievalNoFileId(steps = steps, top_k = initial_top_k, filter_score = filter_score)
            initial_results_function = multiTransactionRetrieval.multiFunctionDescriptionRetrievalNoFileId(steps = steps, top_k = initial_top_k, filter_score = filter_score)
       

        retrieval_end_time = time.time()
        reranke_start_time = time.time()
        if use_reranker:
            reranked_results_transaction = rerankerService.rerank_transactions(initial_results_transaction, rerank_top_k)
            reranked_results_function = rerankerService.rerank_function_description(initial_results_function, rerank_top_k)
        else:
            reranked_results_transaction = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results_transaction.items()
            }
            reranked_results_function = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results_function.items()
            }

        reranked_results = defaultdict(list)
        for k, v in reranked_results_transaction.items():
            reranked_results[k].extend(v)
        for k, v in reranked_results_function.items():
            reranked_results[k].extend(v)
              
        reranked_results = dict(reranked_results)

        reranke_end_time = time.time()
        total_time = (reranke_end_time - start_time) 
        retrieval_time = (retrieval_end_time - retrieval_start_time) 
        rerank_time = (reranke_end_time - reranke_start_time) 
        response = TransactionV3Response(
            query_id=query_id,
            question=question,
            file_id=file_id,
            file_name=file_name,
            results={
                key: [
                    TransactionV3Result(
                        Transaction=TransactionV3Info(**t[0]),
                        InitialScore=t[1],
                        RerankScore=t[2],
                        AverageScore=(t[1] + t[2]) / 2.0
                    )
                    for t in tuples
                ]
                for key, tuples in reranked_results.items()
            },
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            total_time=total_time
        )

        logger.info(f"查询 {query_id} 完成，总耗时 {total_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"查询 {query_id} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


"""
召回逻辑不改变。只修改返回数据的形式，按照测试案例进行返回整个内容形式是：
知识库：
系统名称: 核心系统
交易名称: 负债产品拷贝
功能描述: 用于完成负债产品的复制，通过复制现有产品快速生成新产品，需后续定制参数。

系统名称: 核心系统
交易名称: 负债产品维护
功能描述: 用于修改存款产品参数配置，支持按产品编号或币种查询并维护产品属性及控制对象信息。

系统名称: 核心系统
交易名称: 负债产品删除
功能描述: 用于删除不再使用的存款产品，若产品存在正常账户或为传统类产品则不可删除
"""
@router.post("/RetrievalByCases", summary="交易名称召回,返回内容形式通过测试案例整体返回并去重", response_model= TransactionReturnInCaseGroups)
async def transaction_retrievalRetByCase(request: TransactionV3Request):
    question = request.Question
    file_id = request.FileID
    rerank_top_k = request.RerankTopK
    filter_score = request.InitialFilterScores
    initial_top_k = request.InitialTopK
    file_name = request.FileName      
    use_reranker = request.UseReranker
    try :
        # start_time = time.time()
        if not rerankerService:
            raise HTTPException(status_code=500, detail="Reranker Model is not loaded")
        query_id = str(uuid.uuid4())
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Question: {question}")
        # retrieval_start_time = time.time()
        steps = transactionStepParse.transactionToSteps(question)
        logger.info(f"Steps: {steps}")
        if file_id:
            initial_results_transaction = multiTransactionRetrieval.multiTransactionRetrieval(steps = steps, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
            initial_results_function = multiTransactionRetrieval.multiFunctionDescriptionRetrievval(steps = steps, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
        else:
            initial_results_transaction = multiTransactionRetrieval.multiTransactionRetrievalNoFileId(steps = steps, top_k = initial_top_k, filter_score = filter_score)
            initial_results_function = multiTransactionRetrieval.multiFunctionDescriptionRetrievalNoFileId(steps = steps, top_k = initial_top_k, filter_score = filter_score)
       

        # retrieval_end_time = time.time()
        # reranke_start_time = time.time()
        if use_reranker:
            reranked_results_transaction = rerankerService.rerank_transactions(initial_results_transaction, rerank_top_k)
            reranked_results_function = rerankerService.rerank_function_description(initial_results_function, rerank_top_k)
        else:
            reranked_results_transaction = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results_transaction.items()
            }
            reranked_results_function = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results_function.items()
            }

        reranked_results = defaultdict(list)
        for k, v in reranked_results_transaction.items():
            reranked_results[k].extend(v)
        for k, v in reranked_results_function.items():
            reranked_results[k].extend(v)
              
        reranked_results = dict(reranked_results)
        

        results = []
        for value in reranked_results.values():
            for t in value:
                t_dict = t[0]
                if "系统名称" in t_dict.keys():
                    string1 = f"系统名称: {t_dict.get('系统名称', '')}"
                if "交易名称" in t_dict.keys():
                    string2 = f"交易名称: {t_dict.get('交易名称', '')}"
                if  "功能描述" in t_dict.keys():
                    string3 = f"功能描述: {t_dict.get('功能描述', '')}"
                string_result =  f"{string1}\n{string2}\n{string3}"
                results.append(string_result)
        results = list(set(results))
   
        # reranke_end_time = time.time()
        # total_time = (reranke_end_time - start_time) 
        # retrieval_time = (retrieval_end_time - retrieval_start_time) 
        # rerank_time = (reranke_end_time - reranke_start_time) 

        response = TransactionReturnInCaseGroups(
            results=results
        )

        return response



    except Exception as e:
        logger.error(f"查询 {query_id} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
