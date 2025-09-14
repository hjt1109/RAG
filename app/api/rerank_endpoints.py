from fastapi import APIRouter, HTTPException
from loguru import logger
import time
import uuid
from ..Utils.reranker_utils import RerankerModel
from ..Utils.milvus_utils_v2 import My_MilvusClient
from ..Utils.rag_pipeline import RAGPipeline
from ..entitys.Rerank import(
    ComponentInfo,
    ComponentResult,
    RerankRequest,
    RerankResponse,
    RerankItem,
    RerankBatchRequest,
    RerankBatchResponse,
    RerankResponse_Componets
)
from ..config import USE_RERANKER, RERANKER_TOP_K, INITIAL_RETRIEVAL_TOP_K
from ..Utils.System_Recogni import system_recogni
from ..Utils.Components_Recogni import components_recogni
from ..Utils.Mutil_Retrieval import Multi_Retrieval_withfile_id, Multi_Retrieval_withoutfile_id


router = APIRouter(prefix="/rerank", tags=["Rerank Operations"])

rag = RAGPipeline()
Milvus_Components = My_MilvusClient()
reranker = RerankerModel()

@router.post("/rerank_by_file_id", summary="根据文件ID重排查询", response_model=RerankResponse)
async def rerank_by_file_id(request: RerankRequest):
    """
    根据文件ID重排查询
    
    - 支持自定义top_k参数
    - 支持分数阈值过滤
    """
    try:
        start_time = time.time()
        
        # 检查重排模型是否可用
        if not rag.reranker:
            raise HTTPException(status_code=503, detail="重排模型未加载或不可用")
        
        # 生成查询ID
        query_id = str(uuid.uuid4())
        logger.info(f"开始重排查询 {query_id}: {request.question}")
         # 初始检索
        retrieval_start = time.time()
        query_embedding = rag.embedding_model.encode([request.question])[0]
        initial_top_k = request.initial_top_k or INITIAL_RETRIEVAL_TOP_K
        if request.file_id:
            effective_file_id = request.file_id
            initial_results = rag.milvus_client.search_similar_in_file(query_embedding, file_id=effective_file_id, top_k=initial_top_k)
        else:
            effective_file_id = None
            initial_results = rag.milvus_client.search_similar(query_embedding, top_k=initial_top_k)
       
        retrieval_time = (time.time() - retrieval_start) * 1000

        logger.info(f"初始检索完成，找到 {len(initial_results)} 个文档")
        # 重排处理
        rerank_start = time.time()
        reranked_results = rag.reranker.rerank_with_scores(
            request.question, 
            initial_results, 
            top_k=request.top_k or RERANKER_TOP_K
        )
        rerank_time = (time.time() - rerank_start) * 1000
        logger.info(f"重排完成，返回 {len(reranked_results)} 个文档")
        
        ''' 
        rerank_results  = list(text,score)
        '''    
        rerank_results = [RerankItem(content=text, rerank_score=score, initial_score=initial_score,file_id=effective_file_id,file_name=request.file_name) for text, score,initial_score in reranked_results]
        return RerankResponse(
            question=request.question,
            total_documents=len(initial_results),
            reranked_documents=len(reranked_results),
            initial_retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=retrieval_time + rerank_time,
            results=rerank_results
        )
    except Exception as e:
        logger.error(f"重排查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"重排查询失败: {str(e)}")


@router.post("/Retrieval_In_Componets_Table", summary="组件信息表查询", response_model=RerankResponse_Componets)
async def rerank_in_componets_table(request: RerankRequest):
    question = request.question
    """
    question:
    #测试意图：验证：个人活期存入，交易成功

    #操作步骤：

    1、登录&&核心系统&&

    2、进入<存款账户信息查询>交易，查询账户A的当前余额为Y元(数据要求:有效的个人活期存款账户A)

    3、进入<个人现金存款>交易

    4、输入账户A账号和金额X元

    5、提交交易，交易成功

    6、进入<存款账户信息查询>交易，查询账户A的当前余额增加X元

"""
    file_id = request.file_id
    top_k = request.top_k or RERANKER_TOP_K
    filter_score = request.filter_scores
    initial_top_k = request.initial_top_k or INITIAL_RETRIEVAL_TOP_K
    file_name = request.file_name
    use_reranker = request.use_reranker 
    try:
        start_time = time.time()

        # 检查重排模型是否可用
        if not reranker:
            raise HTTPException(status_code=503, detail="重排模型未加载或不可用")
        
        # 生成查询ID
        query_id = str(uuid.uuid4())
        logger.info(f"开始重排查询 {query_id}: {question}")
        
        # # 1️⃣ 根据 file_name 解析 file_id
        # effective_file_id = None
        # if file_name and file_name.strip():
        #     effective_file_id = Milvus_Components.get_file_id_by_name(file_name.strip())
        #     if not effective_file_id:
        #         raise HTTPException(status_code=404, detail=f"未找到文件 {file_name}")
        
        # 2️⃣ 初始检索
        retrieval_start = time.time()
        system_name = system_recogni.extract_system_name(question)
        if system_name:
            logger.info(f"交易系统名称: {system_name}")
        else:
            logger.info(f"未识别到交易系统名称")
    
        components = components_recogni.get_components(question)
        if components:
            logger.info(f"组件内容: {components}")
        else:
            logger.info(f"未识别到组件内容")
        
        if  file_id:
            initial_results = Multi_Retrieval_withfile_id(components, system_name, file_id=file_id, filter_score=filter_score, top_k=initial_top_k)
        else:
            initial_results = Multi_Retrieval_withoutfile_id(components, system_name, filter_score=filter_score, top_k=initial_top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        logger.info(f"初始检索完成，找到 {len(initial_results)} 个文档")
        
        # 3️⃣ 重排处理
        rerank_start = time.time()
        if use_reranker:
            reranked_results = reranker.rerank_components(initial_results, top_k)
        else:
            reranked_results = {
                query: [(comp, score, score) for comp, score in components]
                for query, components in initial_results.items()
            }
        rerank_time = (time.time() - rerank_start) * 1000
        
        # 4️⃣ 封装响应
        total_time = (time.time() - start_time) * 1000
        response = RerankResponse_Componets(
            query_id=query_id,
            question=question,
            file_id=file_id,
            file_name=file_name,
            results={
                query: [
                    ComponentResult(
                        component=ComponentInfo(**comp),
                        initial_score=initial_score,
                        rerank_score=rerank_score
                    )
                    for comp, initial_score, rerank_score in components
                ]
                for query, components in reranked_results.items()
            },
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time
        )
        
        logger.info(f"查询 {query_id} 完成，总耗时 {total_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"查询 {query_id} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")



        

@router.post("/single", summary="单次重排查询", response_model=RerankResponse)
async def rerank_single(request: RerankRequest):
    """
    对单个查询进行重排
    
    - 支持自定义top_k参数
    - 支持分数阈值过滤
    - 支持按文件过滤
    """
    try:
        start_time = time.time()
        
        # 检查重排模型是否可用
        if not rag.reranker:
            raise HTTPException(status_code=503, detail="重排模型未加载或不可用")
        
        # 生成查询ID
        query_id = str(uuid.uuid4())
        logger.info(f"开始重排查询 {query_id}: {request.question}")

         # 1️⃣ 根据 file_name 解析 file_id
        effective_file_id = None
        if request.file_name and request.file_name.strip():
            effective_file_id = rag.milvus_client.get_file_id_by_name(request.file_name.strip())
            if not effective_file_id:
                raise HTTPException(status_code=404, detail=f"未找到文件 {request.file_name}")
        
        # 初始检索
        retrieval_start = time.time()
        query_embedding = rag.embedding_model.encode([request.question])[0]
        initial_top_k = request.initial_top_k or INITIAL_RETRIEVAL_TOP_K

        if effective_file_id:
            initial_results = rag.milvus_client.search_similar_in_file(query_embedding, file_id=effective_file_id, top_k=initial_top_k)
        else:
            initial_results = rag.milvus_client.search_similar(query_embedding, top_k=initial_top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        logger.info(f"初始检索完成，找到 {len(initial_results)} 个文档")
        
        # 重排处理
        rerank_start = time.time()
        reranked_results = rag.reranker.rerank_with_scores(
            request.question, 
            initial_results, 
            top_k=request.top_k or RERANKER_TOP_K
        )
        rerank_time = (time.time() - rerank_start) * 1000
        
        logger.info(f"重排完成，返回 {len(reranked_results)} 个文档")
        
        ''' 
        rerank_results  = list(text,score)
        '''    
        rerank_results = [RerankItem(content=text, rerank_score=score, initial_score=initial_score,file_id=effective_file_id,file_name=request.file_name) for text, score,initial_score in reranked_results]
        return RerankResponse(
            question=request.question,
            total_documents=len(initial_results),
            reranked_documents=len(reranked_results),
            initial_retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=retrieval_time + rerank_time,
            results=rerank_results
        )
    except Exception as e:
        logger.error(f"重排查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"重排查询失败: {str(e)}")



@router.post("/batch", summary="批量重排查询", response_model=RerankBatchResponse , deprecated=True)
async def rerank_batch(request: RerankBatchRequest):
    """
    批量重排查询
    
    - 支持多个问题同时处理
    - 提高处理效率
    - 返回批量处理统计信息
    """
    try:
        batch_start_time = time.time()
        batch_id = str(uuid.uuid4())
        
        logger.info(f"开始批量重排 {batch_id}，问题数量: {len(request.questions)}")
        
        # 检查重排模型是否可用
        if not rag.reranker:
            raise HTTPException(status_code=503, detail="重排模型未加载或不可用")
        
        batch_results = []
        processed_count = 0
        
        for i, question in enumerate(request.questions):
            try:
                # 创建单个重排请求
                single_request = RerankRequest(
                    question=question,
                    top_k=request.top_k,
                    initial_top_k=INITIAL_RETRIEVAL_TOP_K
                )
                
                # 调用单个重排
                single_response = await rerank_single(single_request)
                batch_results.append(single_response)
                processed_count += 1
                
                logger.info(f"批量处理进度: {processed_count}/{len(request.questions)}")
                
            except Exception as e:
                logger.error(f"批量处理第 {i+1} 个问题失败: {e}")
                # 继续处理其他问题
                continue
        
        batch_time = (time.time() - batch_start_time) * 1000
        
        response = RerankBatchResponse(
            batch_id=batch_id,
            total_questions=len(request.questions),
            processed_questions=processed_count,
            batch_results=batch_results,
            batch_processing_time_ms=batch_time
        )
        
        logger.info(f"批量重排 {batch_id} 完成，处理了 {processed_count}/{len(request.questions)} 个问题")
        return response
        
    except Exception as e:
        logger.error(f"批量重排失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量重排失败: {str(e)}")

@router.get("/status", summary="获取重排服务状态")
async def get_rerank_status():
    """获取重排服务的详细状态信息"""
    try:
        status = {
            "reranker_enabled": USE_RERANKER,
            "reranker_loaded": rag.reranker is not None,
            "reranker_device": str(rag.reranker.device) if rag.reranker else None,
            "reranker_model_info": {
                "model_name": "bge-reranker-large",
                "gpu_devices": str(rag.reranker.device) if rag.reranker else None,
                "memory_optimization": True,
                "max_memory_fraction": 0.2
            },
            "embedding_model_info": {
                "model_name": "bge-large-zh",
                "device": str(rag.embedding_model.device),
                "embedding_dim": 1024
            },
            "default_parameters": {
                "reranker_top_k": RERANKER_TOP_K,
                "initial_retrieval_top_k": INITIAL_RETRIEVAL_TOP_K
            },
            "service_status": "healthy" if rag.reranker else "unavailable"
        }
        
        return status
        
    except Exception as e:
        logger.error(f"获取重排状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取重排状态失败: {str(e)}")

