from sqlalchemy.orm import query
from ..entitys.Rerank import (
    RerankRequest,
    ComponentInfo_v2,
    ComponentResult_v2,
    RerankResponse_Componets_v2,
    TransactionInfo_v2,
    TransactionResult_v2,
    RerankResponse_Transaction_v2
)
from fastapi import APIRouter, HTTPException, APIRouter
from loguru import logger
import time
import uuid
from ..Utils.reranker_utils import RerankerModel
from ..Utils.System_Recogni import system_recogni
from ..Utils.Components_Recogni import components_recogni
from ..services.Muti_Retrieval_Service import Muti_Retrieval_Service
from ..Utils.TransactionStepParse import transactionStepParse
from ..services.MultiTransactionRetrieval import MultiTransactionRetrieval

reranker = RerankerModel()
router = APIRouter(prefix = "/retrieval_v2", tags = ["Retrieval API v2"])
service = Muti_Retrieval_Service()
multiTransactionRetrieval = MultiTransactionRetrieval()

@router.post("/retrieval", summary="检索召回", response_model=RerankResponse_Componets_v2)
async def retrieval(request: RerankRequest):
    question = request.question
    file_id = request.file_id
    top_k = request.top_k
    filter_score = request.filter_scores
    initial_top_k = request.initial_top_k
    file_name = request.file_name       
    use_reranker = request.use_reranker
    try :
        start_time = time.time()
        if not reranker:
            raise HTTPException(status_code=500, detail="Reranker Model is not loaded")
        
        query_id = str(uuid.uuid4())
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Question: {question}")
        retrieval_start_time = time.time()

        system_name = system_recogni.extract_system_name(question)
        if system_name:
            logger.info(f"System Name: {system_name}")
        else:
            logger.info("System Name: None")
        components = components_recogni.get_components(question)
        if components:
            logger.info(f"Components: {components}")
        else:
            logger.info("Components: None")
        if file_id:
            initial_results =  service.Multi_Retrieval_withfile_id(components = components, system_name = system_name, file_id = file_id, filter_score = filter_score, top_k = initial_top_k)
        else:
            initial_results =  service.Multi_Retrieval_withoutfile_id(components = components, system_name = system_name, filter_score = filter_score, top_k = initial_top_k)
        logger.info(f"Initial Results: {initial_results}")
        retrieval_end_time = time.time()
        reranke_start_time = time.time()
        if use_reranker:
            reranked_results = reranker.rerank_components(initial_results, top_k)
        else:
            reranked_results = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results.items()
            }
        reranke_end_time = time.time()
        total_time = (reranke_end_time - start_time) 
        retrieval_time = (retrieval_end_time - retrieval_start_time) 
        rerank_time = (reranke_end_time - reranke_start_time) 
        response = RerankResponse_Componets_v2(
                        query_id=query_id,
                        question=question,
                        file_id=file_id,
                        file_name=file_name,
                        results={
                            key: [
                                ComponentResult_v2(
                                    component=ComponentInfo_v2(**t[0]),
                                    initial_score=t[1],
                                    rerank_score=t[2]
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


#交易名称召回
@router.post("/transaction_retrieval", summary="交易名称召回", response_model=RerankResponse_Transaction_v2)
async def transaction_retrieval(request: RerankRequest):
    question = request.question
    file_id = request.file_id
    top_k = request.top_k
    filter_score = request.filter_scores
    initial_top_k = request.initial_top_k
    file_name = request.file_name       
    use_reranker = request.use_reranker
    try :
        start_time = time.time()
        if not reranker:
            raise HTTPException(status_code=500, detail="Reranker Model is not loaded")
        query_id = str(uuid.uuid4())
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Question: {question}")
        retrieval_start_time = time.time()
        steps = transactionStepParse.transactionToSteps(question)
        logger.info(f"Steps: {steps}")
        if file_id:
            initial_results = multiTransactionRetrieval.multiTransactionRetrieval(steps = steps, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
        else:
            initial_results = multiTransactionRetrieval.multiTransactionRetrievalNoFileId(steps = steps, top_k = initial_top_k, filter_score = filter_score)
        logger.info(f"Initial Results: {initial_results}")
        retrieval_end_time = time.time()
        reranke_start_time = time.time()
        if use_reranker:
            reranked_results = reranker.rerank_transactions(initial_results, top_k)
        else:
            reranked_results = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results.items()
            }
        reranke_end_time = time.time()
        total_time = (reranke_end_time - start_time) 
        retrieval_time = (retrieval_end_time - retrieval_start_time) 
        rerank_time = (reranke_end_time - reranke_start_time) 
        response = RerankResponse_Transaction_v2(
            query_id=query_id,
            question=question,
            file_id=file_id,
            file_name=file_name,
            results={
                key: [
                    TransactionResult_v2(
                        transaction=TransactionInfo_v2(**t[0]),
                        initial_score=t[1],
                        rerank_score=t[2]
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