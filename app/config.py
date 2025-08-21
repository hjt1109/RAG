from dotenv import load_dotenv
import os

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19531")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "default")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/app/models/bge-large-zh")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "/app/models/bge-reranker-large")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "http://192.168.242.193:8100/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "DeepSeek-R1-Distill-Qwen-32B")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")  # Add mirror endpoint

# GPU配置 - 按照用户需求
EMBEDDING_GPU_DEVICES = os.getenv("EMBEDDING_GPU_DEVICES", "cpu")  # embedding模型使用单GPU 1
RERANKER_GPU_DEVICES = os.getenv("RERANKER_GPU_DEVICES", "cpu")  # 重排模型使用多GPU 1,2,3

# GPU内存优化配置
ENABLE_MEMORY_OPTIMIZATION = os.getenv("ENABLE_MEMORY_OPTIMIZATION", "true").lower() == "true"
MAX_MEMORY_FRACTION = float(os.getenv("MAX_MEMORY_FRACTION", "0.2"))  # 限制使用30%内存，避免OOM
ENABLE_MEMORY_POOLING = os.getenv("ENABLE_MEMORY_POOLING", "true").lower() == "true"

# 重排配置
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"  # 是否启用重排
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))  # 重排后返回的文档数量
INITIAL_RETRIEVAL_TOP_K = int(os.getenv("INITIAL_RETRIEVAL_TOP_K", "10"))  # 初始检索的文档数量







# ... 原有配置 ...

# 图节点：组件名称列表（去重）
COMPONENTS = [
    "组件名称: 账户详情查询",
    "组件名称: 现金存款",
    "组件名称: 个人活期账户销户",
    "组件名称: 查询账户币种",
    "组件名称: 查询证件类型",
    "组件名称: 生成身份证号码",
    "组件名称: 查询账户编号",
    "组件名称: 账户信息查询",
    "组件名称: 登录",
    "组件名称: 提任务",
    "组件名称: 查询测管用户",
    "组件名称: 查询正常的个人活期存款账户编号",
    "组件名称: 查询零存整取账户编号",
    "组件名称: 查询教育储蓄账户编号",
    "组件名称: 查询二类账户编号",
    "组件名称: 查询三类账户编号",
    "组件名称: 查询冻结的个人活期存款账户编号",
    "组件名称: 查询挂失的个人活期存款账户编号",
    "组件名称: 查询销户的个人活期存款账户编号",
    "组件名称: 查询个人支票账户编号",
    "组件名称: 查询正常的个人保证金活期账户编号"
]

# 图边：(from_node, to_node, prob) 元组列表，从日志统计得到
# 示例：实际用pandas从日志计算转移概率
EDGES = [
    ("组件名称: 登录", "组件名称: 查询账户编号", 0.8),
    ("组件名称: 登录", "组件名称: 提任务", 0.2),
    ("组件名称: 查询账户编号", "组件名称: 账户详情查询", 0.7),
    ("组件名称: 查询账户编号", "组件名称: 账户信息查询", 0.3),
    ("组件名称: 账户详情查询", "组件名称: 现金存款", 0.6),
    ("组件名称: 账户详情查询", "组件名称: 个人活期账户销户", 0.4),
    ("组件名称: 现金存款", "组件名称: 账户详情查询", 0.9),  # 存款后验证
    ("组件名称: 个人活期账户销户", "组件名称: 查询销户的个人活期存款账户编号", 0.5),
    ("组件名称: 提任务", "组件名称: 查询测管用户", 0.7),
    ("组件名称: 查询测管用户", "组件名称: 生成身份证号码", 0.4),
    ("组件名称: 生成身份证号码", "组件名称: 查询证件类型", 0.6),
    ("组件名称: 查询证件类型", "组件名称: 查询账户币种", 0.5),
    ("组件名称: 查询账户币种", "组件名称: 查询正常的个人活期存款账户编号", 0.8),
    ("组件名称: 查询正常的个人活期存款账户编号", "组件名称: 查询零存整取账户编号", 0.3),
    ("组件名称: 查询零存整取账户编号", "组件名称: 查询教育储蓄账户编号", 0.4),
    ("组件名称: 查询教育储蓄账户编号", "组件名称: 查询二类账户编号", 0.5),
    ("组件名称: 查询二类账户编号", "组件名称: 查询三类账户编号", 0.6),
    ("组件名称: 查询三类账户编号", "组件名称: 查询冻结的个人活期存款账户编号", 0.2),
    ("组件名称: 查询冻结的个人活期存款账户编号", "组件名称: 查询挂失的个人活期存款账户编号", 0.3),
    ("组件名称: 查询挂失的个人活期存款账户编号", "组件名称: 查询销户的个人活期存款账户编号", 0.4),
    ("组件名称: 查询销户的个人活期存款账户编号", "组件名称: 查询个人支票账户编号", 0.5),
    ("组件名称: 查询个人支票账户编号", "组件名称: 查询正常的个人保证金活期账户编号", 0.6)
]

# 相似度阈值
SIMILARITY_THRESHOLD = 0.7  # 用于判断RAG召回有效性