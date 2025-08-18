import gradio as gr
import requests
from typing import List, Dict, Any, Optional
import os

# --- API配置 ---
# 定义后端API服务的基础URL和各个功能对应的端点
API_BASE_URL = "http://localhost:8012"
API_ENDPOINTS = {
    "upload": f"{API_BASE_URL}/document/upload",
    "rerank": f"{API_BASE_URL}/rerank/single",
    "health": f"{API_BASE_URL}/health",
    "files": f"{API_BASE_URL}/delete/files",
    "collections": f"{API_BASE_URL}/collection/list",
    "rerank_status": f"{API_BASE_URL}/rerank/status"
}

class RAGInterface:
    """
    负责与后端API进行通信的接口类。
    """
    def __init__(self):
        # 使用requests.Session可以复用TCP连接，并统一设置超时
        self.session = requests.Session()
        self.session.timeout = 30
    
    def check_api_health(self) -> Dict[str, Any]:
        """检查后端API的健康状态。"""
        try:
            response = self.session.get(API_ENDPOINTS["health"])
            if response.status_code == 200:
                return {"status": "healthy", "message": "API服务正常"}
            else:
                return {"status": "error", "message": f"API服务异常: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": f"无法连接到API服务: {str(e)}"}
    
    def upload_document(self, file) -> Dict[str, Any]:
        """上传文档到后端服务。"""
        if file is None:
            return {"success": False, "message": "请选择要上传的文件"}
        
        try:
            with open(file.name, 'rb') as f:
                files = {'file': (os.path.basename(file.name), f, 'application/octet-stream')}
                response = self.session.post(API_ENDPOINTS["upload"], files=files)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "message": f"文档上传成功！\n文件ID: {result['file_id']}\n处理记录数: {result['processed_count']}",
                }
            else:
                return {"success": False, "message": f"上传失败: {response.text}"}
        except Exception as e:
            return {"success": False, "message": f"上传异常: {str(e)}"}
    
    def rerank_query(self, question: str, top_k: int = 5, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        向后端API发起重排查询请求，直接将 `filename` 发送给后端。
        """
        if not question.strip():
            return {"success": False, "message": "请输入问题"}
        
        try:
            payload = {"question": question, "top_k": top_k}
            if filename:
                payload["file_name"] = filename
            
            response = self.session.post(API_ENDPOINTS["rerank"], json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return {"success": True, **result}
            else:
                return {"success": False, "message": f"重排查询失败: {response.text}"}
        except Exception as e:
            return {"success": False, "message": f"重排查询异常: {str(e)}"}
    
    def get_files_list(self) -> Dict[str, Any]:
        """获取已上传的全部文件列表。"""
        try:
            response = self.session.get(API_ENDPOINTS["files"])
            if response.status_code == 200:
                result = response.json()
                return {"success": True, "files": result.get('files', []), "total_files": result.get('total_files', 0)}
            else:
                return {"success": False, "message": f"获取文件列表失败: {response.text}"}
        except Exception as e:
            return {"success": False, "message": f"获取文件列表异常: {str(e)}"}

    def get_collections_list(self) -> Dict[str, Any]:
        """获取知识库列表。"""
        try:
            response = self.session.get(API_ENDPOINTS["collections"])
            if response.status_code == 200:
                result = response.json()
                return {"success": True, "collections": result.get('collections', []), "total_collections": result.get('total_collections', 0)}
            else:
                return {"success": False, "message": f"获取知识库列表失败: {response.text}"}
        except Exception as e:
            return {"success": False, "message": f"获取知识库列表异常: {str(e)}"}
    
    def get_rerank_status(self) -> Dict[str, Any]:
        """获取重排服务的状态信息。"""
        try:
            response = self.session.get(API_ENDPOINTS["rerank_status"])
            if response.status_code == 200:
                return {"success": True, "status": response.json()}
            else:
                return {"success": False, "message": f"获取重排状态失败: {response.text}"}
        except Exception as e:
            return {"success": False, "message": f"获取重排状态异常: {str(e)}"}

# --- Gradio 界面与逻辑 ---

rag_interface = RAGInterface()

def format_rerank_results(results: List[Dict]) -> str:
    """格式化重排查询结果，以便在Gradio界面上美观地显示。"""
    if not results:
        return "暂无重排结果"
    
    formatted_parts = []
    for i, item in enumerate(results, 1):
        content = item.get('content', '')
        content_preview = (content[:200] + "...") if len(content) > 200 else content
        score = item.get('rerank_score', 0)
        file_name = item.get('file_name', '未知文件')
        formatted_parts.append(f"**结果 {i}** (分数: {score:.4f}, 文件: {file_name})\n{content_preview}\n---")
    
    return "\n".join(formatted_parts)

def create_interface():
    """创建并配置Gradio应用界面。"""
    with gr.Blocks(title="RAG系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 RAG检索召回系统")
        gr.Markdown("支持文档上传和检索召回结果展示。")
        
        with gr.Row():
            status_btn = gr.Button("🔍 检查服务状态", variant="secondary")
            status_output = gr.Textbox(label="服务状态", interactive=False)
        
        with gr.Tab("📄 文档上传"):
            with gr.Row():
                with gr.Column(scale=2):
                    upload_file = gr.File(label="选择Excel/CSV文件", file_types=[".csv", ".xlsx", ".xls"])
                    upload_btn = gr.Button("📤 上传文档", variant="primary")
                with gr.Column(scale=1):
                    upload_result = gr.Textbox(label="上传结果", lines=5, interactive=False)
        
        with gr.Tab("🔍 重排查询"):
            with gr.Row():
                with gr.Column(scale=3):
                    rerank_question = gr.Textbox(label="查询问题", placeholder="请输入要查询的问题", lines=2)
                    with gr.Row():
                        top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="返回结果数量 (Top-K)")
                        # 【已修改】初始化时下拉框为空，将通过事件动态填充
                        file_dropdown = gr.Dropdown(
                            label="指定文件范围（可选）",
                            choices=[],
                            value=None,
                            allow_custom_value=True,
                            info="不选则在所有文件中搜索"
                        )
                    rerank_btn = gr.Button("🔍 开始重排查询", variant="primary")
                with gr.Column(scale=2):
                    rerank_results = gr.Markdown(label="重排结果")
                    rerank_stats = gr.Textbox(label="查询统计", lines=2, interactive=False)

        with gr.Tab("📁 管理"):
            with gr.Row():
                # 【已修改】此按钮现在会同时刷新下拉框和数据表
                refresh_files_btn = gr.Button("🔄 刷新文件列表")
                refresh_collections_btn = gr.Button("🔄 刷新知识库列表")
            with gr.Row():
                files_list = gr.Dataframe(headers=["文件ID", "文件名", "文档数量"], label="已上传文件列表", interactive=False, scale=1)
                collections_list = gr.Dataframe(headers=["知识库ID", "名称", "文档数", "当前使用"], label="知识库列表", interactive=False, scale=2)
        
        with gr.Tab("ℹ️ 系统信息"):
            rerank_status_btn = gr.Button("🔍 检查重排服务状态")
            rerank_status_output = gr.JSON(label="重排服务状态")
        
        # --- 事件处理函数 ---

        def handle_status_check():
            result = rag_interface.check_api_health()
            return f"状态: {result['status']}\n消息: {result['message']}"

        # 【新增】用于同时更新文件下拉列表和文件数据表的函数
        def refresh_all_file_components():
            """获取最新文件列表并同时更新下拉菜单和数据表。"""
            result = rag_interface.get_files_list()
            if result.get("success"):
                files_data = result.get("files", [])
                df_data = [[f.get("file_id", ""), f.get("file_name", ""), f.get("doc_count", 0)] for f in files_data]
                dropdown_choices = [f.get("file_name", "") for f in files_data]
                # 使用 gr.update 同时更新两个组件
                return gr.Dropdown(choices=dropdown_choices), gr.Dataframe(value=df_data)
            else:
                # 如果失败，返回空状态
                return gr.Dropdown(choices=[]), gr.Dataframe(value=[])

        # 【已修改】上传函数现在会触发文件列表的刷新
        def handle_doc_upload(file):
            """处理文档上传，并在成功后刷新文件列表。"""
            if not file:
                # 如果没有文件，则无需刷新，仅返回错误消息
                # 注意：返回值的数量需要和 outputs 匹配
                return "错误：未选择任何文件。", gr.Dropdown(), gr.Dataframe()
            
            upload_msg = rag_interface.upload_document(file)["message"]
            
            # 上传操作后，立即调用刷新函数来更新UI
            dropdown_update, df_update = refresh_all_file_components()
            
            return upload_msg, dropdown_update, df_update

        def handle_rerank_query(question, top_k, filename_from_ui):
            """处理重排查询，直接将文件名传递给后端。"""
            filename_to_send = filename_from_ui.strip() if filename_from_ui else None
            result = rag_interface.rerank_query(question, top_k, filename_to_send)
            
            if result.get("success"):
                formatted_results = format_rerank_results(result.get("results", []))
                stats = f"总文档数: {result.get('total_documents',0)}, 重排文档数: {result.get('reranked_documents',0)}, 耗时: {result.get('total_time_ms',0):.2f}ms"
                return formatted_results, stats
            else:
                return f"❌ 错误: {result.get('message', '未知错误')}", ""

        def handle_get_collections():
            result = rag_interface.get_collections_list()
            return [[c.get("collection_id", ""), c.get("collection_name", ""), c.get("document_count", 0), "是" if c.get("is_current") else "否"] for c in result.get("collections", [])] if result["success"] else []

        def handle_get_rerank_status():
            result = rag_interface.get_rerank_status()
            return result["status"] if result["success"] else {"error": result.get("message", "未知错误")}
        
        # --- 绑定事件 ---
        
        status_btn.click(handle_status_check, outputs=status_output)
        
        # 【已修改】上传按钮现在会更新三个组件：上传结果、文件下拉框、文件数据表
        upload_btn.click(
            handle_doc_upload, 
            inputs=upload_file, 
            outputs=[upload_result, file_dropdown, files_list]
        )
        
        rerank_btn.click(
            handle_rerank_query,
            inputs=[rerank_question, top_k_slider, file_dropdown],
            outputs=[rerank_results, rerank_stats]
        )
        
        # 【已修改】刷新文件按钮现在同时更新下拉框和数据表
        refresh_files_btn.click(
            refresh_all_file_components, 
            outputs=[file_dropdown, files_list]
        )
        
        refresh_collections_btn.click(handle_get_collections, outputs=collections_list)
        rerank_status_btn.click(handle_get_rerank_status, outputs=rerank_status_output)
        
        # 【已修改】页面加载时，除了检查状态，还立即刷新文件列表
        interface.load(handle_status_check, outputs=status_output).then(
            refresh_all_file_components, 
            outputs=[file_dropdown, files_list]
        )
    
    return interface

if __name__ == "__main__":
    app_interface = create_interface()
    app_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )