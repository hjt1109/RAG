from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification
import torch
import torch.nn as nn
from loguru import logger
from ..config import RERANKER_MODEL_PATH, RERANKER_GPU_DEVICES, ENABLE_MEMORY_OPTIMIZATION, MAX_MEMORY_FRACTION, ENABLE_MEMORY_POOLING
from typing import List, Tuple
import numpy as np
import gc
import os
from typing import List, Dict, Tuple

class RerankerModel:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式，确保全局只有一个 RerankerModel 实例
        """
        if cls._instance is None:
            cls._instance = super(RerankerModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化重排模型，仅在第一次实例化时加载模型和分词器
        """
        if self._initialized:
            logger.info("RerankerModel already initialized, reusing existing instance.")
            return

        try:
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
            self.model.eval()
            
            # 默认设备为 CPU
            self.device = torch.device("cpu")
            
            # 解析配置的设备
            configured_device_str = RERANKER_GPU_DEVICES.split(',')[0].strip().lower()

            if configured_device_str == "cpu" or not torch.cuda.is_available():
                self.device = torch.device("cpu")
                logger.info("Reranker model loaded on CPU as configured or CUDA is not available.")
            else:
                try:
                    if not configured_device_str.startswith("cuda:"):
                        configured_device_str = f"cuda:{configured_device_str}" if configured_device_str.isdigit() else "cuda:0"
                    
                    device_id = int(configured_device_str.split(':')[1]) if ':' in configured_device_str else 0
                    
                    if device_id < torch.cuda.device_count():
                        if self._check_gpu_memory(device_id):
                            self.device = torch.device(f"cuda:{device_id}")
                            logger.info(f"Reranker model loaded on single GPU: {self.device}")
                        else:
                            logger.warning(f"GPU {device_id} has insufficient memory for reranker. Falling back to CPU.")
                            self.device = torch.device("cpu")
                    else:
                        logger.warning(f"Configured GPU device '{configured_device_str}' is not available. Falling back to CPU.")
                        self.device = torch.device("cpu")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing RERANKER_GPU_DEVICES '{RERANKER_GPU_DEVICES}': {e}. Falling back to CPU.")
                    self.device = torch.device("cpu")
            
            self.model = self.model.to(self.device)

            # 设置内存优化
            if ENABLE_MEMORY_OPTIMIZATION:
                self._setup_memory_optimization()
                
            self._initialized = True
            logger.info("RerankerModel initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self._initialized = False
            raise

    def _setup_memory_optimization(self):
        """设置内存优化"""
        if ENABLE_MEMORY_POOLING:
            # 启用内存池
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("启用CUDA内存池优化")
        
        # 设置内存限制 (仅对当前模型加载的GPU生效)
        if self.device.type == 'cuda':
            device_id = self.device.index
            if device_id is not None: # Ensure device_id is not None for CUDA devices
                try:
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    max_memory = int(total_memory * MAX_MEMORY_FRACTION)
                    # Note: torch.cuda.set_per_process_memory_fraction is deprecated/removed in newer PyTorch.
                    # This setting primarily serves as a guideline or requires specific environment variable setup (e.g., PYTORCH_CUDA_ALLOC_CONF)
                    # or careful batch sizing to respect memory limits.
                    logger.info(f"GPU {device_id} 内存限制设置为总内存的 {MAX_MEMORY_FRACTION*100:.1f}% (约 {max_memory / 1024**3:.1f}GB)")
                except Exception as e:
                    logger.warning(f"Failed to apply specific memory fraction for GPU {device_id}: {e}")
        else:
            logger.info("Memory optimization settings for GPU skipped as reranker model is on CPU.")


    def _check_gpu_memory(self, device_id: int) -> bool:
        """检查GPU内存是否足够"""
        try:
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            # Ensure context is on the correct device for memory query
            torch.cuda.set_device(device_id)
            allocated_memory = torch.cuda.memory_allocated(device_id)
            free_memory = total_memory - allocated_memory
            
            # 估算重排模型所需内存（约4GB for bge-reranker-large, adjust based on your model's exact size)
            required_memory = 4 * 1024**3  # Increased to 4GB for bge-reranker-large
            
            if free_memory < required_memory:
                logger.warning(f"GPU {device_id} 可用内存不足: {free_memory / 1024**3:.1f}GB < 估计所需内存 {required_memory / 1024**3:.1f}GB")
                return False
            
            return True
        except Exception as e:
            logger.error(f"检查GPU {device_id} 内存时出错: {e}")
            return False

    def _clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU内存已清理")
        

    def rerank(self, query: str, passages: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """
        对检索到的文档进行重排
        
        Args:
            query: 查询文本
            passages: 候选文档列表
            top_k: 返回前k个结果，如果为None则返回所有结果
            
        Returns:
            List[Tuple[str, float]]: 重排后的文档和分数列表，按分数降序排列
        """
        if not passages:
            return []
            
        try:
            # 构建查询-文档对
            pairs = [[query, passage] for passage in passages]
            logger.debug(f"Pairs: {pairs}")
            # 编码
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            logger.debug(f"Tokenizer output: {inputs}")
         
            # 将输入数据移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 计算相似度分数s
            with torch.no_grad():

                # outputs = self.model(**inputs)  
                scores = self.model(**inputs,return_dict=True).logits.view(-1,).float()  
                normalized_scores = torch.sigmoid(scores)
               
            # scores = torch.nn.functional.normalize(scores, p=2, dim=1)
            # 如果只有一个文档，确保scores是数组
            if len(passages) == 1:
                # If it's a scalar tensor, extract the item, otherwise squeeze and extract
                scores = [scores.item()] if scores.ndim == 0 else [scores.squeeze().item()]
            
            # 将文档和分数配对
            doc_scores = list(zip(passages, normalized_scores.tolist()))
            
            # 按分数降序排序
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 如果指定了top_k，则只返回前k个
            if top_k is not None:
                doc_scores = doc_scores[:top_k]
            
            # 清理GPU内存
            if ENABLE_MEMORY_OPTIMIZATION and self.device.type == 'cuda':
                self._clear_gpu_memory()

            logger.debug(f"Reranked {len(passages)} documents using {self.device}, returning top {len(doc_scores)}")
            return doc_scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # 发生错误时也清理内存
            if ENABLE_MEMORY_OPTIMIZATION and self.device.type == 'cuda':
                self._clear_gpu_memory()
            # 如果重排失败，返回原始顺序
            logger.warning("Reranking failed, returning passages with default scores.")
            return [(passage, 0.0) for passage in passages]

    def rerank_with_scores(self, query: str, passages_with_scores: List[Tuple[str, float]], top_k: int = None) -> List[Tuple[str, float,float]]:
        """
        对带有初始分数的文档进行重排
        
        Args:
            query: 查询文本
            passages_with_scores: 候选文档和初始分数列表
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[str, float]]: 重排后的文档和分数列表
        """
        if not passages_with_scores:
            return []
        
        # 提取文档文本
        passages = [doc for doc, _ in passages_with_scores]
        
        # 进行重排
        reranked_results = self.rerank(query, passages, top_k)
        # 如果reranked_results的text在passages_with_scores的text中，如果匹配到，则在reranked_results中添加initial_score
        score_map = dict(passages_with_scores)
        reranked_results = [((text,r_score,score_map.get(text))) for text, r_score in reranked_results] 

        return reranked_results
    
    def rerank_components(self, initial_results: Dict[str, List], top_k: int = None) -> Dict[str, List[Tuple[Dict, float, float]]]:
        """
        对结构化的初始结果进行重排，保留初始分数并添加重排分数

        Args:
            initial_results: 字典，键为查询，值为包含 (组件信息, 初始分数) 的列表
            top_k: 返回前 k 个结果，如果为 None 则返回所有结果

        Returns:
            Dict[str, List[Tuple[Dict, float, float]]]: 重排后的字典，值为 (组件信息, 初始分数, 重排分数) 的列表
        """
        reranked_results = {}
        
        try:
            for query, components in initial_results.items():
                if not components:
                    reranked_results[query] = []
                    continue
                
                # 提取组件名称作为文档内容，保留初始分数
                passages_with_scores = [(comp['组件名称'], score) for comp, score in components]
                
                # 使用 rerank_with_scores 方法进行重排
                reranked = self.rerank_with_scores(query, passages_with_scores, top_k)
                
                # 将重排结果映射回原始组件信息
                component_map = {comp['组件名称']: comp for comp, _ in components}
                reranked_components = [
                    (component_map[text], initial_score, rerank_score)
                    for text, rerank_score, initial_score in reranked
                ]
                
                reranked_results[query] = reranked_components
                
            logger.info(f"Completed reranking for {len(initial_results)} queries")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking components failed: {e}")
            if ENABLE_MEMORY_OPTIMIZATION and self.device.type == 'cuda':
                self._clear_gpu_memory()
            # 返回原始结构，添加默认重排分数 0.0
            logger.warning("Reranking components failed, returning original components with default rerank scores.")
            return {
                query: [(comp, score, 0.0) for comp, score in components]
                for query, components in initial_results.items()
            }
