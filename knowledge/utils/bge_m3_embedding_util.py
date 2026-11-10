from pymilvus.model.hybrid import BGEM3EmbeddingFunction


def get_beg_m3_embedding_model():
    """获取 BGE-M3 嵌入模型实例"""
    return BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",
        device="gpu",
        use_fp16=False,  # 是否使用半精度
    )


def generate_hybrid_embeddings(embedding_model: BGEM3EmbeddingFunction, documents):
    """
    同时生成稠密和稀疏向量

    Args:
        embedding_model: BGE-M3 模型实例
        documents: 文本列表

    Returns:
        {
            "dense": [[0,1, 0.2, ...], ...],     # 稠密向量
            "sparse": [{token_id: weight}, ...]  # 稀疏向量
        }
    """

    embedding_result = embedding_model.encode_documents(documents)

    # 稀疏向量
    processed_sparse = []
    csr_array = embedding_result['sparse']
    for i in range(len(documents)):
        start = csr_array.indptr[i]
        end = csr_array.indptr[i + 1]
        token_ids = csr_array.indices[start:end].tolist()
        weights = csr_array.data[start:end].tolist()
        sparse_vector = dict(zip(token_ids, weights))
        processed_sparse.append(sparse_vector)

    return {
        "dense": [d.tolist() for d in embedding_result["dense"]],
        "sparse": processed_sparse
    }
