#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定文件ID的文档
支持通过文件ID删除Milvus中的文档记录
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .milvus_utils import My_MilvusClient
from loguru import logger
from typing import Dict, Any, Optional
import argparse


class FileDeleter:
    """文件删除器，用于删除指定文件ID的文档"""
    
    def __init__(self):
        """初始化Milvus客户端"""
        try:
            self.milvus_client = My_MilvusClient()
            logger.info("文件删除器初始化成功")
        except Exception as e:
            logger.error(f"初始化文件删除器失败: {e}")
            raise
    
    def delete_file_by_id(self, file_id: str) -> Dict[str, Any]:
        """
        根据文件ID删除文档
        
        Args:
            file_id: 要删除的文件ID
            
        Returns:
            Dict[str, Any]: 删除结果
        """
        try:
            logger.info(f"开始删除文件ID: {file_id}")
            
            # 首先查询该文件ID是否存在
            existing_docs = self.milvus_client.client.query(
                collection_name=self.milvus_client.collection_name,
                filter=f'file_id == "{file_id}"',
                output_fields=["id", "file_id", "file_name"]
            )
            
            if not existing_docs:
                logger.warning(f"文件ID {file_id} 不存在")
                return {
                    "success": False,
                    "message": f"文件ID {file_id} 不存在",
                    "deleted_count": 0,
                    "file_id": file_id
                }
            
            # 获取要删除的文档ID列表
            doc_ids = [doc["id"] for doc in existing_docs]
            file_name = existing_docs[0].get("file_name", "未知文件")
            
            logger.info(f"找到 {len(doc_ids)} 个文档需要删除，文件名: {file_name}")
            
            # 执行删除操作
            delete_result = self.milvus_client.client.delete(
                collection_name=self.milvus_client.collection_name,
                pks=doc_ids
            )
            
            logger.info(f"删除操作完成，删除结果: {delete_result}")
            
            return {
                "success": True,
                "message": f"成功删除文件 {file_name} (ID: {file_id})",
                "deleted_count": len(doc_ids),
                "file_id": file_id,
                "file_name": file_name,
                "delete_result": delete_result
            }
            
        except Exception as e:
            logger.error(f"删除文件ID {file_id} 时发生错误: {e}")
            return {
                "success": False,
                "message": f"删除失败: {str(e)}",
                "deleted_count": 0,
                "file_id": file_id
            }
    
    def list_all_files(self) -> Dict[str, Any]:
        """
        列出所有文件信息
        
        Returns:
            Dict[str, Any]: 文件列表信息
        """
        try:
            # 查询所有文档，按文件ID分组
            all_docs = self.milvus_client.client.query(
                collection_name=self.milvus_client.collection_name,
                output_fields=["id", "file_id", "file_name", "text"]
            )
            
            # 按文件ID分组
            files_info = {}
            for doc in all_docs:
                file_id = doc["file_id"]
                if file_id not in files_info:
                    files_info[file_id] = {
                        "file_id": file_id,
                        "file_name": doc["file_name"],
                        "doc_count": 0,
                        "sample_texts": []
                    }
                
                files_info[file_id]["doc_count"] += 1
                # 保存前3个文本作为示例
                if len(files_info[file_id]["sample_texts"]) < 3:
                    files_info[file_id]["sample_texts"].append(doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"])
            
            return {
                "success": True,
                "total_files": len(files_info),
                "files": list(files_info.values())
            }
            
        except Exception as e:
            logger.error(f"获取文件列表时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取文件列表失败: {str(e)}",
                "total_files": 0,
                "files": []
            }
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        获取指定文件ID的详细信息
        
        Args:
            file_id: 文件ID
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        try:
            docs = self.milvus_client.client.query(
                collection_name=self.milvus_client.collection_name,
                filter=f'file_id == "{file_id}"',
                output_fields=["id", "file_id", "file_name", "text"]
            )
            
            if not docs:
                return {
                    "success": False,
                    "message": f"文件ID {file_id} 不存在",
                    "file_info": None
                }
            
            file_info = {
                "file_id": file_id,
                "file_name": docs[0]["file_name"],
                "doc_count": len(docs),
                "sample_texts": [doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"] for doc in docs[:5]]
            }
            
            return {
                "success": True,
                "message": "获取文件信息成功",
                "file_info": file_info
            }
            
        except Exception as e:
            logger.error(f"获取文件信息时发生错误: {e}")
            return {
                "success": False,
                "message": f"获取文件信息失败: {str(e)}",
                "file_info": None
            }


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="删除指定文件ID的文档")
    parser.add_argument("--file-id", type=str, help="要删除的文件ID")
    parser.add_argument("--list", action="store_true", help="列出所有文件")
    parser.add_argument("--info", type=str, help="获取指定文件ID的详细信息")
    parser.add_argument("--confirm", action="store_true", help="确认删除操作")
    
    args = parser.parse_args()
    
    try:
        deleter = FileDeleter()
        
        if args.list:
            # 列出所有文件
            result = deleter.list_all_files()
            if result["success"]:
                print(f"\n找到 {result['total_files']} 个文件:")
                for file_info in result["files"]:
                    print(f"\n文件ID: {file_info['file_id']}")
                    print(f"文件名: {file_info['file_name']}")
                    print(f"文档数量: {file_info['doc_count']}")
                    print(f"示例文本: {file_info['sample_texts'][0] if file_info['sample_texts'] else '无'}")
            else:
                print(f"错误: {result['message']}")
        
        elif args.info:
            # 获取文件信息
            result = deleter.get_file_info(args.info)
            if result["success"]:
                file_info = result["file_info"]
                print(f"\n文件信息:")
                print(f"文件ID: {file_info['file_id']}")
                print(f"文件名: {file_info['file_name']}")
                print(f"文档数量: {file_info['doc_count']}")
                print(f"示例文本:")
                for i, text in enumerate(file_info['sample_texts'], 1):
                    print(f"  {i}. {text}")
            else:
                print(f"错误: {result['message']}")
        
        elif args.file_id:
            # 删除文件
            if not args.confirm:
                print(f"警告: 即将删除文件ID {args.file_id}")
                print("请使用 --confirm 参数确认删除操作")
                return
            
            result = deleter.delete_file_by_id(args.file_id)
            if result["success"]:
                print(f"✅ {result['message']}")
                print(f"删除文档数量: {result['deleted_count']}")
            else:
                print(f"❌ {result['message']}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"程序执行失败: {e}")
