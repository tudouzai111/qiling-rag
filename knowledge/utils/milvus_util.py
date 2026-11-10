import logging
from logging import INFO

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

logging.basicConfig(level=INFO)
logger = logging.getLogger(__name__)

def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5,
):
    """
    创建混合检索请求

    Args:
        dense_vector: 稠密向量 [0.1, 0.2, ...]
        sparse_vector: 稀疏向量 {token_id: weight}
        limit: 每路检索返回数量

    Returns:
        [dense_req, sparse_req] 检索请求列表
    """
    # 默认参数
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}  # 稠密用余弦
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}  # 稀疏用内积

    # 创建稠密向量检索请求
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",  # Collection 中的稠密向量字段
        param=dense_params,
        expr=expr,  # 过滤表达式（可选）
        limit=limit,
    )

    # 创建稀疏向量检索请求
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",  # Collection 中的稀疏向量字段
        param=sparse_params,
        expr=expr,
        limit=limit,
    )

    return [dense_req, sparse_req]


def execute_hybrid_search_query(
    milvus_client: MilvusClient,
    collection_name,
    search_requests,
    ranker_weights=(0.5, 0.5),
    norm_score=True,
    limit=5,
    output_fields=None,
    search_params=None
):
    """
    执行混合检索

    Args:
        milvus_client: Milvus 客户端
        collection_name: 集合名称
        search_requests: 检索请求 [dense_req, sparse_req]
        ranker_weights: 融合权重 (稠密权重, 稀疏权重)
        norm_score: 是否归一化分数
        limit: 最终返回数量
        output_fields: 返回字段，显式指定entity的字段, 如果为None, entity只包含主键
        search_params: 搜索参数

    Returns:
        检索结果hits: [[hit1, hit2, ...], ...]
        一个hits代表一个查询向量的查询结果
        hit:
            [
                {
                    "id": 1001,
                    "distance": 0.85,
                    "entity": {
                        "title": "Milvus 混合检索指南",
                        "category": "技术文档"
                    }
                },
                {
                    "id": 2045,
                    "distance": 0.72,
                    "entity": {
                        "title": "向量数据库对比",
                        "category": "评测"
                    }
                }
            ]
    """
    try:
        # 1. 创建融合排序器
        rerank = WeightedRanker(
            ranker_weights[0],  # 稠密向量权重
            ranker_weights[1],  # 稀疏向量权重
            norm_score=norm_score,
        )

        # 默认输出字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 2. 执行混合检索
        results = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,  # 检索请求列表
            ranker=rerank,  # 融合排序器
            limit=limit,
            output_fields=output_fields,
            search_requests=search_params
        )

        total_hits = sum(len(hits) for hits in results) if results else 0
        logger.info(f"Milvus混合搜索完成，共处理 {len(results) if results else 0} 个查询，总计找到 {total_hits} 个结果")
        return results
    except Exception as e:
        raise RuntimeError(f"执行Milvus混合搜索失败(collection={collection_name}): {e}") from e
