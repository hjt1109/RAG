from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import os

from rag_pipeline import RAGPipeline
from entitys.models import ChatCompletionRequest

router = APIRouter(prefix="/v1", tags=["Chat Operations"])

# 默认模型
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "DeepSeek-R1-Distill-Qwen-32B")

# 全局RAG实例
rag = RAGPipeline()

@router.post("/chat/completions",deprecated=True)
async def chat_completions(request: ChatCompletionRequest):
    """
    接收 LobeChat/OpenAI 协议的 messages 列表，
    提取用户提问，调用 RAG，返回符合 OpenAI 格式的响应。
    """
    try:
        # 如果没传消息或为空
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # 提取最后一个用户输入（role="user"）
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        question = user_messages[-1].content.strip()

        # 调用 RAG
        answer = rag.query(question)

        # 返回 OpenAI 协议格式
        return JSONResponse({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": request.model or DEFAULT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    # "finish_reason": "stop"
                    "stream" : "true"# 流式输出

                }
            ]
        })
    except Exception as e:
        logger.exception(f"chat_completions error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 