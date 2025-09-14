from ..entitys.DataItemV1 import (
    DataItemV1Info,
    DataItemV1Request,
    DataItemV1Response,
    DataItemV1Result
)
from fastapi import APIRouter, HTTPException, APIRouter
from loguru import logger
import time
import uuid
from ..services.RerankerService import RerankerService
from ..services.MultiDataItemRetrievalV1 import MultiDataItemRetrievalV1
from collections import defaultdict
from ..Utils.DataComponentParse  import questionParse

rerankerService = RerankerService()
router = APIRouter(prefix = "/DataItem_retrieval", tags = ["DataItem Retrieval API"])
multidataItemRetrieval = MultiDataItemRetrievalV1()



#交易名称召回
@router.post("/dataitem_retrieval", summary="DataItem(输入参数&&输出参数)召回", response_model= DataItemV1Response)
async def dataitem_retrieval(request: DataItemV1Request):
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


        """
        问题解析功能待完成

        """


        InputItems , OutputItems= questionParse(question)
        logger.info(f"InputItems: {InputItems}")
        logger.info(f"OutputItems: {OutputItems}")
        if file_id:

            initial_results_inputParams = multidataItemRetrieval.multInputParameterRetrieval(DataItems = InputItems, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
            initial_results_outputParams = multidataItemRetrieval.multiOutputParameterRetrieval(DataItems = OutputItems, file_id = file_id, top_k = initial_top_k, filter_score = filter_score)
            
        else:

            initial_results_inputParams = multidataItemRetrieval.multInputParameterRetrievalNoFileId(DataItems = InputItems, top_k = initial_top_k, filter_score = filter_score)
            initial_results_outputParams = multidataItemRetrieval.multiOutputParameterRetrievalNoFileId(DataItems = OutputItems, top_k = initial_top_k, filter_score = filter_score)

        retrieval_end_time = time.time()
        reranke_start_time = time.time()
        if use_reranker:
            reranked_results_inputParams = rerankerService.rerank_dataItem_inputParameter(initial_results_inputParams, rerank_top_k)
            reranked_results_outputParams = rerankerService.rerank_dataItem_outputParameter(initial_results_outputParams, rerank_top_k)

        else:
            reranked_results_inputParams = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results_inputParams .items()
            }
            reranked_results_outputParams = {
                query : [(comp, score ,score) for comp, score in components]
                for query, components in initial_results_outputParams.items()
            }

        reranked_results = defaultdict(list)
        for k, v in reranked_results_inputParams.items():
            reranked_results[k].extend(v)
        for k, v in reranked_results_outputParams.items():
            reranked_results[k].extend(v)
              
        reranked_results = dict(reranked_results)

        reranke_end_time = time.time()
        total_time = (reranke_end_time - start_time) 
        retrieval_time = (retrieval_end_time - retrieval_start_time) 
        rerank_time = (reranke_end_time - reranke_start_time) 
        response = DataItemV1Response(
            query_id=query_id,
            question=question,
            file_name=file_name,
            results={
                key: [
                    DataItemV1Result(
                        Transaction=DataItemV1Info(**t[0]),
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