import networkx as nx
import numpy as np
from ..config import COMPONENTS, EDGES, SIMILARITY_THRESHOLD
from .embedding_utils import EmbeddingModel
from loguru import logger
from typing import List, Optional

embedding_model = EmbeddingModel()

def cosine_similarity(a, b):
    """Calculate cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class OperationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        # Normalize colons in COMPONENTS
        normalized_components = [comp.replace(':', '：').strip() for comp in set(COMPONENTS)]
        for comp in normalized_components:
            self.G.add_node(comp)
        # Add edges, normalizing colons
        for from_node, to_node, prob in EDGES:
            from_node = from_node.replace(':', '：')
            to_node = to_node.replace(':', '：')
            if from_node in self.G and to_node in self.G:
                self.G.add_edge(from_node, to_node, weight=prob)
            else:
                logger.warning(f"Edge {from_node} -> {to_node} skipped, nodes not found.")
        logger.debug(f"Graph initialized with nodes: {list(self.G.nodes)}")

    def _normalize_colon(self, text: str) -> str:
        """Replace English colon (:) with Chinese colon (：) and strip whitespace."""
        return text.replace(':', '：').strip()

    def infer_start_node(self, query: str) -> str:
        """Infer starting node from query using embedding similarity."""
        query_emb = embedding_model.encode([query])[0]
        max_sim = -1
        best_node = None
        for node in self.G.nodes:
            node_text = node.replace('组件名称：', '')
            node_emb = embedding_model.encode([node_text])[0]
            sim = cosine_similarity(query_emb, node_emb)
            if sim > max_sim:
                max_sim = sim
                best_node = node
        if max_sim < SIMILARITY_THRESHOLD:
            start_nodes = [n for n in self.G.nodes if self.G.in_degree(n) == 0]
            if start_nodes:
                logger.info(f"Low similarity ({max_sim:.4f}), defaulting to start node: {start_nodes[0]}")
                return start_nodes[0]
            logger.warning("No start nodes found, using first node")
            return list(self.G.nodes)[0]
        logger.debug(f"Inferred start node: {best_node} with similarity {max_sim:.4f}")
        return best_node

    def get_next_node(self, current_node: str, min_prob: float = 0.5) -> Optional[str]:
        """Greedily select the next node with highest edge weight above min_prob."""
        current_node = self._normalize_colon(current_node)
        if current_node not in self.G:
            logger.error(f"Node {current_node} not in graph")
            return None
        neighbors = [(n, self.G[current_node][n]['weight']) for n in self.G.successors(current_node)]
        if not neighbors:
            logger.debug(f"No neighbors for {current_node}")
            return None
        next_node, prob = max(neighbors, key=lambda x: x[1])
        if prob < min_prob:
            logger.debug(f"Probability {prob:.4f} below threshold {min_prob}")
            return None
        logger.debug(f"Next node for {current_node}: {next_node} (prob: {prob:.4f})")
        return next_node

    def generate_sequence(self, start_node: str, max_length: int = 10, min_prob: float = 0.5) -> List[str]:
        """Generate high-probability sequence from start node (greedy)."""
        start_node = self._normalize_colon(start_node)
        if start_node not in self.G:
            logger.error(f"Start node {start_node} not in graph")
            raise ValueError(f"Start node {start_node} not in graph")
        path = [start_node]
        visited = {start_node}  # Track visited nodes to avoid cycles
        current = start_node
        for _ in range(max_length - 1):
            next_node = self.get_next_node(current, min_prob)
            if not next_node:
                break
            if next_node in visited:
                logger.debug(f"Cycle detected, stopping at {next_node}")
                break
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        logger.info(f"Generated sequence: {path}")
        return path

    def validate_rag_recall(self, rag_results: List[str], query: str) -> Optional[List[str]]:
        """Validate RAG recalls: check if in graph, return enhanced sequence."""
        valid_results = []
        normalized_nodes = {self._normalize_colon(node): node for node in self.G.nodes}
        
        logger.debug(f"RAG results: {rag_results}")
        logger.debug(f"Graph nodes: {list(self.G.nodes)}")
        
        for result in rag_results:
            normalized_result = self._normalize_colon(result)
            if normalized_result in normalized_nodes:
                
                full_node = normalized_nodes[normalized_result]
                
                valid_results.append(full_node)
            
            else:
                logger.debug(f"Result {result} (normalized: {normalized_result}) not in graph nodes")
        
        if not valid_results:
            logger.info("No valid RAG results, returning None")
            return None
        sequence = self.generate_sequence(valid_results[0])
        logger.info(f"Enhanced sequence from {valid_results[0]}: {sequence}")
        return sequence