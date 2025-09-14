import re
from typing import Optional, List, Tuple


def questionParse(q: Optional[str]) -> Tuple[List[str], List[str]]:
    """
    将问题q解析为数据要求和数据项
    输入示例:
        q = "数据要求：有效的个人活期存款账号4，状态为正常，数据项：账号A"
    输出示例:
        (["有效的个人活期存款账号4", "状态为正常"], ["账号A"])
    """
    if not q:
        return [], []
    
    # 用正则提取 "数据要求：" 和 "数据项："
    match = re.search(r"数据要求[:：](.*)数据项[:：](.*)", q)
    if not match:
        return [], []
    
    requirements = match.group(1).strip()
    items = match.group(2).strip()
    
    # 按顿号、逗号、空格等分隔
    req_list = [r.strip() for r in re.split(r"[，,、;；]", requirements) if r.strip()]
    item_list = [i.strip() for i in re.split(r"[，,、;；]", items) if i.strip()]
    
    return req_list, item_list