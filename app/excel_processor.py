import pandas as pd
import uuid
import os
from typing import List, Dict, Tuple
from loguru import logger
from fastapi import UploadFile, HTTPException
import tempfile


class ExcelProcessor:
    """Excel文件处理器，支持CSV和XLSX格式"""
    
    def __init__(self):
        self.supported_extensions = {'.csv', '.xlsx', '.xls'}
    
    def validate_file(self, file: UploadFile) -> bool:
        """验证文件格式是否支持"""
        if not file.filename:
            return False
        
        file_extension = os.path.splitext(file.filename.lower())[1]
        return file_extension in self.supported_extensions
    
    def process_excel_file(self, file: UploadFile) -> Tuple[str, List[str], str]:
        """
        处理Excel文件，返回文件ID、处理后的文本列表和数据库表ID
        
        Args:
            file: 上传的文件
            
        Returns:
            Tuple[str, List[str], str]: (文件ID, 处理后的文本列表, 数据库表ID)
        """
        if not self.validate_file(file):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式。支持的格式: {', '.join(self.supported_extensions)}"
            )
        
        # 生成唯一文件ID
        file_id = str(uuid.uuid4())
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                # 写入上传的文件内容
                content = file.file.read()
                temp_file.write(content)
                temp_file.flush()
                
                # 根据文件扩展名读取数据
                file_extension = os.path.splitext(file.filename.lower())[1]
                
                if file_extension == '.csv':
                    # 尝试不同的编码格式
                    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
                    df = None
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(temp_file.name, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is None:
                        raise HTTPException(status_code=400, detail="无法解析CSV文件，请检查文件编码")
                else:
                    # XLSX文件
                    df = pd.read_excel(temp_file.name)
                
                # 清理临时文件
                os.unlink(temp_file.name)
                
                # 处理数据
                processed_texts = self._process_dataframe(df, file_id)
                
                # 生成数据库表ID（使用文件ID作为表ID）
                table_id = file_id
                
                logger.info(f"成功处理文件 {file.filename}，文件ID: {file_id}，生成了 {len(processed_texts)} 条记录")
                
                return file_id, processed_texts, table_id
                
        except Exception as e:
            logger.error(f"处理文件 {file.filename} 时发生错误: {e}")
            raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    
    def _process_dataframe(self, df: pd.DataFrame, file_id: str) -> List[str]:
        """
        处理DataFrame，将表头与内容拼接
        
        Args:
            df: pandas DataFrame
            file_id: 文件ID
            
        Returns:
            List[str]: 处理后的文本列表
        """
        processed_texts = []
        
        # 获取表头
        headers = df.columns.tolist()
        
        # 处理每一行数据
        for index, row in df.iterrows():
            # 跳过空行
            if row.isna().all():
                continue
            
            # 将表头与内容拼接
            row_texts = []
            for header, value in zip(headers, row):
                # 处理空值
                if pd.isna(value):
                    value = ""
                else:
                    value = str(value).strip()
                
                # 拼接表头和内容
                if value:  # 只添加非空的内容
                    row_texts.append(f"{header}: {value}")
            
            # 将一行中的所有字段拼接成一个文本
            if row_texts:
                combined_text = " | ".join(row_texts)
                # 添加文件ID作为元数据
                final_text = f"{combined_text}"
                processed_texts.append(final_text)
        
        return processed_texts