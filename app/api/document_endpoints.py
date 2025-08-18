from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger

from excel_processor import ExcelProcessor
from rag_pipeline import RAGPipeline
from entitys.models import DocumentUploadResponse

router = APIRouter(prefix="/document", tags=["Document Operations"])

# 全局实例
excel_processor = ExcelProcessor()
rag = RAGPipeline()

@router.post("/upload", summary="上传并解析Excel文档", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传Excel文档（CSV或XLSX格式）进行解析和向量化
    
    - 支持的文件格式：CSV, XLSX, XLS
    - 自动识别表头（第一行）
    - 将表头与内容拼接成适合向量化的文本
    - 存储到Milvus向量数据库
    """
    try:
        # 处理Excel文件
        file_id, processed_texts, table_id = excel_processor.process_excel_file(file)
        
        # 将处理后的文本向量化并存储到Milvus
        rag.ingest_documents(processed_texts, file_id, file.filename)
        
        return DocumentUploadResponse(
            file_id=file_id,
            table_id=table_id,
            status_code=200,
            message="插入成功",
            processed_count=len(processed_texts)
        )
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"文档上传处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}") 