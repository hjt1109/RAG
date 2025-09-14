import numpy as np
from typing import List, Dict, Any
from loguru import logger
from .milvus_utils import My_MilvusClient
from .embedding_utils import EmbeddingModel
from .reranker_utils import RerankerModel
from .graph_utils import OperationGraph
from ..config import USE_RERANKER, RERANKER_TOP_K, INITIAL_RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
import re 


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RAGPipeline:
    def __init__(self):
        self.milvus_client = My_MilvusClient(dim=1024)
        self.embedding_model = EmbeddingModel()
        
        # Initialize reranker (if enabled)
        self.reranker = None
        if USE_RERANKER:
            try:
                self.reranker = RerankerModel()
                logger.info("Reranker model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker model: {e}. Continuing without reranker.")
                self.reranker = None
        
        # Initialize graph
        self.graph = OperationGraph()
        logger.info("Operation graph initialized")
        logger.info(f"Graph initialized successfully :{self.graph}")

    def ingest_documents(self, texts: List[str], file_id: str = None, file_name: str = None):
        """Ingest documents into Milvus with optional file_id and file_name"""

        logger.debug(f"Ingesting texts: {texts}")
        try:
            embeddings = self.embedding_model.encode(texts)
            self.milvus_client.insert_documents(texts, embeddings, file_id or "", file_name or "")
            logger.info(f"Documents ingested successfully: {len(texts)} docs, file_id={file_id}, file_name={file_name}")
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise


    def is_invalid(self, results: List[str], query_emb: List[float]) -> bool:
        """Judge if RAG recall is invalid (e.g., low similarity or empty)"""
        if not results:
            logger.warning("No results retrieved from Milvus")
            return True
        sims = [cosine_similarity(query_emb, self.embedding_model.encode([r])[0]) for r in results]
        max_sim = max(sims) if sims else 0
        logger.debug(f"Max similarity score: {max_sim}")
        return max_sim < SIMILARITY_THRESHOLD

    def query(self, question: str, use_graph: bool = True) -> str:
        """Query the RAG system with optional graph enhancement, return context sequence"""
        try:
            query_embedding = self.embedding_model.encode([question])[0]
            
            # Initial retrieval
            initial_top_k = INITIAL_RETRIEVAL_TOP_K if self.reranker else 5
            search_results = self.milvus_client.search_similar(query_embedding, top_k=initial_top_k)
            
            # Extract texts and scores
            contexts = [text for text, score in search_results]
            similarity_scores = [score for text, score in search_results]
            logger.info(f"Initial retrieved contexts count: {len(contexts)}")
            
            # Rerank if enabled
            if self.reranker and contexts:
                logger.info("Starting document reranking...")
                reranked_results = self.reranker.rerank_with_scores(question, search_results, top_k=RERANKER_TOP_K)
                contexts = [text for text, score in reranked_results]
                similarity_scores = [score for text, score in reranked_results]
                logger.info(f"Reranked contexts count: {len(contexts)}")
            
            # Integrate graph if enabled
            final_contexts = contexts
            if use_graph:
                enhanced = self.graph.validate_rag_recall(contexts, question)
                if enhanced:
                    final_contexts = enhanced
                    logger.info(f"Graph-enhanced sequence: {final_contexts}")
                elif self.is_invalid(contexts, query_embedding):
                    start = self.graph.infer_start_node(question)
                    final_contexts = self.graph.generate_sequence(start)
                    logger.info(f"Fallback to graph-generated sequence: {final_contexts}")
            
            # Log final results
            num_contexts = len(final_contexts)
            for i in range(num_contexts):
                score = similarity_scores[i] if i < len(similarity_scores) else 'N/A'
                logger.info(f"Final Context {i}: {final_contexts[i]} with similarity score: {score}")
            
            # Return joined contexts as string
            return "\n".join(final_contexts) if final_contexts else "数据库中找不到相关内容"
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

    def query_in_file(self, question: str, file_id: str, use_graph: bool = True) -> str:
        """Query within a specific file with optional graph enhancement, return context sequence"""
        try:
            query_embedding = self.embedding_model.encode([question])[0]
            
            # Initial retrieval
            initial_top_k = INITIAL_RETRIEVAL_TOP_K if self.reranker else 5
            search_results = self.milvus_client.search_similar_in_file(query_embedding, file_id, top_k=initial_top_k)
            
            contexts = [text for text, score in search_results]
            similarity_scores = [score for text, score in search_results]
            logger.info(f"Initial retrieved contexts count (file_id={file_id}): {len(contexts)}")
            
            # Rerank if enabled
            if self.reranker and contexts:
                logger.info("Starting document reranking...")
                reranked_results = self.reranker.rerank_with_scores(question, search_results, top_k=RERANKER_TOP_K)
                contexts = [text for text, score in reranked_results]
                similarity_scores = [score for text, score in reranked_results]
                logger.info(f"Reranked contexts count: {len(contexts)}")
            
            # Integrate graph
            final_contexts = contexts
            if use_graph:
                enhanced = self.graph.validate_rag_recall(contexts, question)
                if enhanced:
                    final_contexts = enhanced
                    logger.info(f"Graph-enhanced sequence: {final_contexts}")
                elif self.is_invalid(contexts, query_embedding):
                    start = self.graph.infer_start_node(question)
                    final_contexts = self.graph.generate_sequence(start)
                    logger.info(f"Fallback to graph-generated sequence: {final_contexts}")
            
            num_contexts = len(final_contexts)
            for i in range(num_contexts):
                score = similarity_scores[i] if i < len(similarity_scores) else 'N/A'
                logger.info(f"Final Context {i}: {final_contexts[i]} with similarity score: {score}")
            
            # Return joined contexts as string
            return "\n".join(final_contexts) if final_contexts else "数据库中找不到相关内容"
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

    def query_by_file_name(self, question: str, file_name: str, top_k: int = 5, use_graph: bool = True) -> List[Dict[str, Any]]:
        """Query by file name with optional graph enhancement, return structured results"""
        try:
            query_embedding = self.embedding_model.encode([question])[0]
            search_results = self.milvus_client.search_similar_by_filename(query_embedding, file_name, top_k)
            
            contexts = [item['text'] for item in search_results]
            similarity_scores = [item['score'] for item in search_results]
            logger.info(f"Initial retrieved contexts count (file_name={file_name}): {len(contexts)}")
            
            # Rerank if enabled
            if self.reranker and contexts:
                logger.info("Starting document reranking...")
                reranked_results = self.reranker.rerank_with_scores(question, search_results, top_k=RERANKER_TOP_K)
                contexts = [text for text, score in reranked_results]
                similarity_scores = [score for text, score in reranked_results]
                logger.info(f"Reranked contexts count: {len(contexts)}")
            
            # Integrate graph
            final_contexts = contexts
            if use_graph:
                enhanced = self.graph.validate_rag_recall(contexts, question)
                if enhanced:
                    final_contexts = enhanced
                    logger.info(f"Graph-enhanced sequence: {final_contexts}")
                elif self.is_invalid(contexts, query_embedding):
                    start = self.graph.infer_start_node(question)
                    final_contexts = self.graph.generate_sequence(start)
                    logger.info(f"Fallback to graph-generated sequence: {final_contexts}")
            
            # Return structured response
            results = [
                {
                    "file_id": search_results[i]["file_id"] if i < len(search_results) else "",
                    "file_name": file_name,
                    "text": text,
                    "score": similarity_scores[i] if i < len(similarity_scores) else 0.0
                }
                for i, text in enumerate(final_contexts)
            ]
            logger.info(f"Final results count: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Query by file name failed: {e}")
            raise

    def query_multi_step(self, question: str,file_id: str, use_graph: bool = True) -> List[Dict[str, Any]]:
        """Process a multi-step query, performing RAG retrieval per step and chaining with graph."""
        try:
            # Parse steps from the query
            steps = re.split(r'\n+\d+[、.]', question.strip())
            step_ids = [int(m.group(1)) for m in re.finditer(r'\n+(\d+)[、.]', '\n' + question.strip())]
            steps = [step.strip() for step in steps if step.strip()]
            if not steps:
                logger.error("No valid steps found in query")
                return [{"step_id": 1, "step_text": "无效的操作步骤"}]
            if len(step_ids) != len(steps):
                logger.warning(f"Mismatch between step IDs ({len(step_ids)}) and steps ({len(steps)}), using sequential IDs")
                step_ids = list(range(1, len(steps) + 1))

            final_sequence = []
            previous_next_node = None

            for i, (step_id, step) in enumerate(zip(step_ids, steps), 1):
                logger.info(f"Processing step {step_id}: {step}")
                query_embedding = self.embedding_model.encode([step])[0]
                
                # Initial retrieval
                initial_top_k = INITIAL_RETRIEVAL_TOP_K if self.reranker else 5
                search_results = self.milvus_client.search_similar_in_file(query_embedding, file_id, top_k=initial_top_k)
                contexts = [text for text, score in search_results]
                similarity_scores = [score for text, score in search_results]
                logger.info(f"Step {step_id} - Initial retrieved contexts: {contexts}")

                # Rerank if enabled
                if self.reranker and contexts:
                    logger.info(f"Step {step_id} - Starting document reranking...")
                    reranked_results = self.reranker.rerank_with_scores(step, search_results, top_k=RERANKER_TOP_K)
                    contexts = [text for text, rerank_score, initial_score in reranked_results]
                    similarity_scores = [rerank_score for text, rerank_score, initial_score in reranked_results]
                    logger.info(f"Step {step_id} - Reranked contexts: {contexts}")

                # Graph validation
                final_contexts = contexts
                if use_graph:
                    enhanced = self.graph.validate_rag_recall(contexts, step)
                    if enhanced:
                        final_contexts = enhanced
                        logger.info(f"Step {step_id} - Graph-enhanced sequence: {final_contexts}")
                    elif self.is_invalid(contexts, query_embedding):
                        start = self.graph.infer_start_node(step)
                        final_contexts = self.graph.generate_sequence(start)
                        logger.info(f"Step {step_id} - Fallback to graph-generated sequence: {final_contexts}")
                    else:
                        final_contexts = [contexts[0]]  # Use top RAG result if valid
                        logger.info(f"Step {step_id} - Using top RAG result: {final_contexts}")

                    # Compare with previous step's next node (if applicable)
                    if previous_next_node and i > 1:
                        if previous_next_node in self.graph.G.nodes and final_contexts[0] != previous_next_node:
                            logger.warning(f"Step {step_id} - Mismatch with previous next node {previous_next_node}, using graph")
                            final_contexts = self.graph.generate_sequence(previous_next_node)
                            logger.info(f"Step {step_id} - Corrected sequence: {final_contexts}")

                # Select first valid context as current node
                current_node = final_contexts[0] if final_contexts else None
                if not current_node:
                    logger.warning(f"Step {step_id} - No valid context, skipping")
                    continue

                # Get next node for the next step
                previous_next_node = self.graph.get_next_node(current_node) if use_graph else None
                logger.debug(f"Step {step_id} - Current node: {current_node}, Next node: {previous_next_node}")

                # Add current node to final sequence if not already present for this step
                final_sequence.append({"step_id": step_id, "step_text": current_node})
                
               

            #根据step_text将final_sequence:list(dict)中的内容去重,并返回去重后的list(dict)
            seen = {}
            final_sequence = [seen.setdefault(item['step_text'], item) for item in final_sequence if item['step_text'] not in seen]
            # Log final sequence
            for item in final_sequence:
                logger.info(f"Final Sequence Step {item['step_id']}: {item['step_text']}")
            
            return final_sequence
        except Exception as e:
            logger.error(f"Multi-step query processing failed: {e}")
            return [{"step_id": 1, "step_text": f"错误：{str(e)}"}]
