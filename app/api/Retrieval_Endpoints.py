from fastapi import APIRouter, HTTPException
from loguru import logger
import time
import uuid


from ..Utils.embedding_utils import EmbeddingModel
from ..Utils.reranker_utils import RerankerModel
from ..Utils.Initial_Retrieval import InitialRetrieval
from ..entitys.Retrieval_Code import(
    RetrievalRequest,
    RetrievalInfo,
    RetrievalResponse
)


router = APIRouter(prefix = "/retrieval", tags = ["Retrieval Functions"])
embedding_model = EmbeddingModel()
initial_retrieval = InitialRetrieval("Component_Table")
reranker_model = RerankerModel()

@router.post("/search", summary = "cha xun ma zhi", response_model = RetrievalResponse)
async def search(request: RetrievalRequest):
    file_id = request.file_id
    question = request.question
    rerank_topk = request.rerank_topk
    filter_score = request.filter_score
    initial_topk = request.initial_topk
    use_reranker = request.use_reranker

    start_time = time.time()
    
    if not reranker_model:
        raise HTTPException(status_code=404, detail="Reranker Model is not loaded")

    query_id = str(uuid.uuid4())
    logger.info(f"Query ID: {query_id}")
    if file_id:
        try :
            query_embedding  = embedding_model.encode([question])[0]
            initial_results = initial_retrieval.search_by_fileid(query_embedding, file_id, filter_score, initial_topk)
            logger.info(f"Initial results: {initial_results}")
            if use_reranker:
                rerank_results = reranker_model.rerank_with_scores(question, initial_results, rerank_topk)
                logger.info(f"Rerank results: {rerank_results}")
                end_time = time.time()
                retriavalinfo = [ RetrievalInfo(content = content, initial_score = initial_score, rerank_score = rerank_score, file_id = file_id) for content, initial_score, rerank_score in rerank_results]
                total_time = end_time - start_time
                response = RetrievalResponse(
                    retrieval_results = retriavalinfo,
                    question = question,
                    total_time = total_time
                )
                logger.info(f"完成重排序,结果 : {response}")
                return response
            else:
                end_time = time.time()
                retriavalinfo = [ RetrievalInfo(content = content, initial_score = initial_score) for content, initial_score in initial_results]
                total_time = end_time - start_time
                response = RetrievalResponse(
                    retrieval_results = retriavalinfo,
                    question = question,
                    total_time = total_time
                )
                logger.info(f"未开启重排模型完成初始排序,结果 : {response}")
                return response
        except Exception as e:
            logger.error(f"检索异常query_id:{query_id} error:{e}")
            raise HTTPException(status_code=500, detail= f"检索异常,error:{e}")
    else:
        try :
            query_embedding  = embedding_model.encode([question])[0]
            initial_results = initial_retrieval.search_no_fileid(query_embedding, filter_score, initial_topk)
            logger.info(f"Initial results: {initial_results}")
            if use_reranker:
                rerank_results = reranker_model.rerank_with_scores(question, initial_results, rerank_topk)
                logger.info(f"Rerank results: {rerank_results}")
                end_time = time.time()
                retriavalinfo = [ RetrievalInfo(content = content, initial_score = initial_score, rerank_score = rerank_score) for content, initial_score, rerank_score in rerank_results]
                total_time = end_time - start_time
                response = RetrievalResponse(
                    retrieval_results = retriavalinfo,
                    question = question,
                    total_time = total_time
                )
                logger.info(f"完成重排序,结果 : {response}")
                return response
            else:
                end_time = time.time()
                retriavalinfo = [ RetrievalInfo(content = content, initial_score = initial_score) for content, initial_score in initial_results]
                total_time = end_time - start_time
                response = RetrievalResponse(
                    retrieval_results = retriavalinfo,
                    question = question,
                    total_time = total_time
                )
                logger.info(f"未开启重排模型完成初始排序,结果 : {response}")
                return response
        except Exception as e:
            logger.error(f"检索异常query_id:{query_id} error:{e}")
            raise HTTPException(status_code=500, detail= f"检索异常,error:{e}")
