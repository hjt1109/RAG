from app.Utils.Mutil_Retrieval import Multi_Retrieval_withfile_id



results = Multi_Retrieval_withfile_id(components = ['存款账户信息查询', '个人现金存款'], system_name = '核心系统', file_id = 'bc8ac5be-bb97-4b30-a34b-7c5a85a9286f', filter_score = 0.8, top_k = 5)
print(f"results: {results}")