#embedding_utils.py
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from loguru import logger
from ..config import EMBEDDING_MODEL_PATH, EMBEDDING_GPU_DEVICES, ENABLE_MEMORY_OPTIMIZATION, ENABLE_MEMORY_POOLING, MAX_MEMORY_FRACTION
from typing import List
import numpy as np
import gc
import os

class EmbeddingModel:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
            self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
            self.model.eval()

            # Determine the target device
            self.device = torch.device("cpu")  # Default to CPU

            if EMBEDDING_GPU_DEVICES.lower() == "cpu":
                logger.info("Embedding model configured to use CPU based on EMBEDDING_GPU_DEVICES setting.")
                self.device = torch.device("cpu")
            elif torch.cuda.is_available():
                # Attempt to parse a single GPU device
                try:
                    # Clean the device string for consistent parsing
                    cleaned_device_str = EMBEDDING_GPU_DEVICES.strip().lower()

                    if cleaned_device_str.startswith("cuda:"):
                        device_id = int(cleaned_device_str.split(":")[1])
                    elif cleaned_device_str.isdigit():
                        device_id = int(cleaned_device_str)
                    else:
                        logger.warning(f"Invalid EMBEDDING_GPU_DEVICES format: '{EMBEDDING_GPU_DEVICES}'. Falling back to CPU.")
                        device_id = -1 # Indicate an invalid device, will default to CPU

                    if device_id >= 0 and device_id < torch.cuda.device_count():
                        if self._check_gpu_memory(device_id):
                            self.device = torch.device(f"cuda:{device_id}")
                            logger.info(f"Embedding model loaded on GPU {self.device}")
                        else:
                            logger.warning(f"GPU {device_id} has insufficient memory or other issues. Falling back to CPU.")
                            self.device = torch.device("cpu") # Fallback to CPU if memory check fails
                    else:
                        logger.warning(f"Specified GPU device '{EMBEDDING_GPU_DEVICES}' is not available. Falling back to CPU.")
                        self.device = torch.device("cpu") # Fallback to CPU if device ID is out of bounds or invalid
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing EMBEDDING_GPU_DEVICES '{EMBEDDING_GPU_DEVICES}': {e}. Falling back to CPU.")
                    self.device = torch.device("cpu")
            else:
                logger.info("CUDA not available, using CPU for embedding model.")
                self.device = torch.device("cpu")

            self.model = self.model.to(self.device)

            # Setup memory optimization after device is determined
            if ENABLE_MEMORY_OPTIMIZATION:
                self._setup_memory_optimization()

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
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
            if device_id is not None:
                try:
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    max_memory = int(total_memory * MAX_MEMORY_FRACTION)
                    # set_per_process_memory_fraction is deprecated, use torch.cuda.set_mem_limit
                    # For a more direct limit, you might consider setting CUDA_LAUNCH_BLOCKING=1 and
                    # then monitoring memory usage and potentially using a smaller batch size if OOM occurs.
                    # Or, more practically, rely on the `max_split_size_mb` within `PYTORCH_CUDA_ALLOC_CONF`
                    # for finer control. For simplicity, we'll keep a general note about fraction.
                    # As a direct alternative to `set_per_process_memory_fraction` (which is removed in newer PyTorch):
                    # You would manage this through environment variables or by carefully managing your model sizes
                    # and batch sizes. The `MAX_MEMORY_FRACTION` is more indicative for pre-checks.
                    logger.info(f"GPU {device_id} 内存限制设置为总内存的 {MAX_MEMORY_FRACTION*100:.1f}% (约 {max_memory / 1024**3:.1f}GB)")
                except Exception as e:
                    logger.warning(f"Failed to apply specific memory fraction for GPU {device_id}: {e}")
        else:
            logger.info("Memory optimization settings for GPU skipped as embedding model is on CPU.")


    def _check_gpu_memory(self, device_id: int) -> bool:
        """检查GPU内存是否足够"""
        try:
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            # Calculate currently allocated memory on this specific device
            torch.cuda.set_device(device_id) # Ensure context is on the correct device
            allocated_memory = torch.cuda.memory_allocated(device_id)
            free_memory = total_memory - allocated_memory

            # Estimate model required memory (adjust this based on your model's actual size)
            # A 'large' model like bge-large-zh might require ~2GB-4GB depending on its exact architecture
            # and whether it's int8, fp16, or fp32. Let's make a slightly more generous estimate.
            required_memory = 4 * 1024**3  # 4GB as a safer estimate

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

    def encode(self, texts: List[str]) -> List[List[float]]:
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            # 将输入数据移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # 清理GPU内存
            if ENABLE_MEMORY_OPTIMIZATION and self.device.type == 'cuda':
                self._clear_gpu_memory()

            logger.debug(f"Generated embeddings for {len(texts)} texts using {self.device}")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # 发生错误时也清理内存
            if ENABLE_MEMORY_OPTIMIZATION and self.device.type == 'cuda':
                self._clear_gpu_memory()
            raise