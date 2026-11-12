from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    model_name_or_path=r"D:\ai_models\modelscope_cache\models\BAAI\bge-reranker-large",
    device="cuda",      # GPU 加速
    use_fp16=True       # 半精度推理
)

# 计算相关性得分
pairs = [
    ("什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"),
    ("什么是万用表？", "今天天气很好")
]
scores = reranker.compute_score(sentence_pairs=pairs)
print(scores)
# 输出: [7.8984375, -9.4765625] 高分 = 高相关