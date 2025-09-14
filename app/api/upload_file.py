from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger
from ..Utils.excel_processor import ExcelProcessor
from ..entitys.models import DocumentUploadResponse
from ..services.Vector_Insert import VectorInsert


router = APIRouter(prefix="/upload_v2", tags=["upload_file "])
excel_processor = ExcelProcessor()
vector_insert = VectorInsert()


@router.post("/file", summary="上传文件", description="上传文件", response_model=DocumentUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"upload file {file.filename}")
        file_id, processed_texts = excel_processor.process_data_component_file(file)
        vector_insert.insert_vectors(processed_texts, file_id, file.filename)
        logger.info(f"upload file {file.filename} success")
        return DocumentUploadResponse(
            file_id=file_id,
            status_code=200,
            message="插入成功",
        )
    except Exception as e:
        logger.error(f"upload file {file.filename} error: {e}")
        raise HTTPException(status_code=500, detail="上传文件失败")



#交易名称文件插入
@router.post("/transactionFile", summary="上传交易文件", description="上传交易文件", response_model=DocumentUploadResponse)
async def upload_transaction_file(file: UploadFile = File(...)):
    try:
        logger.info(f"upload transaction file {file.filename}")
        file_id, processed_texts = excel_processor.process_data_component_file(file)
        vector_insert.insert_transaction_vectors(processed_texts, file_id, file.filename)
        logger.info(f"upload transaction file {file.filename} success")
        return DocumentUploadResponse(
            file_id=file_id,
            status_code=200,
            message="插入成功",
        )
    except Exception as e:
        logger.error(f"upload transaction file {file.filename} error: {e}")
        raise HTTPException(status_code=500, detail="上传交易文件失败")

#交易名称文件插入v2形式：“把交易名称：功能描述向量化”
@router.post("/transactionFileV2", summary="上传交易文件v2", description="上传交易文件v2 把交易名称：功能描述向量化", response_model=DocumentUploadResponse)
async def upload_transaction_file_v2(file: UploadFile = File(...)):
    try:
        logger.info(f"upload transaction file v2 {file.filename}")
        file_id, processed_texts = excel_processor.process_data_component_file(file)
        vector_insert.insert_transaction_vectors_v2(processed_texts, file_id, file.filename)
        logger.info(f"upload transaction file v2 {file.filename} success")
        return DocumentUploadResponse(
            file_id=file_id,
            status_code=200,
            message="插入成功",
        )
    except Exception as e:
        logger.error(f"upload transaction file v2 {file.filename} error: {e}")
        raise HTTPException(status_code=500, detail="上传交易文件v2失败")

#交易名称文件插入v3形式：“把交易名称：功能描述向量化”
@router.post("/transactionFileV3", summary="上传交易文件v3", description="上传交易文件v3 把交易名称：功能描述向量化", response_model=DocumentUploadResponse)
async def upload_transaction_file_v3(file: UploadFile = File(...)):
    try:
        logger.info(f"upload transaction file v3 {file.filename}")
        file_id, processed_texts = excel_processor.process_data_component_file(file)
        vector_insert.insert_transaction_vectors_v3(processed_texts, file_id, file.filename)
        logger.info(f"upload transaction file v3 {file.filename} success")
        return DocumentUploadResponse(
            file_id=file_id,
            status_code=200,
            message="插入成功",
        )
    except Exception as e:
        logger.error(f"upload transaction file v3 {file.filename} error: {e}")
        raise HTTPException(status_code=500, detail="上传交易文件v3失败")

"""
上传向量化组件信息表的内容， 涉及到入参、出参的召回
"""
@router.post("/DataItemFile", summary="上传组件信息表", description="上传组件信息表，涉及到 入参、出参的召回的功能", response_model=DocumentUploadResponse)
async def upload_dataItem_file(file: UploadFile = File(...)):
    try:
        logger.info(f"upload dataItem file {file.filename}")
        file_id, processed_texts = excel_processor.process_data_component_file(file)
        vector_insert.insert_dataItem_vectors_v1(processed_texts, file_id, file.filename)
        logger.info(f"upload  {file.filename} success")
        return DocumentUploadResponse(
            file_id=file_id,
            status_code=200,
            message="插入成功",
        )
    except Exception as e:
        logger.error(f"upload dataItem file {file.filename} error: {e}")
        raise HTTPException(status_code=500, detail="上传数据组件文件失败")