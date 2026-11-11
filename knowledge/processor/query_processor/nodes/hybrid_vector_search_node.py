import json

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    item_name_filter,
)


class HybridVectorSearchNode(BaseNode):
    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 获取嵌入模型客户端
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败：{e}")
            return state

        # 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败：{e}")
            return state

        # 嵌入、检索查询问题
        try:
            embed_query_vector = generate_bge_m3_hybrid_vectors(
                model=bge_m3_client, embedding_documents=[rewritten_query]
            )
        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}嵌入获取失败：{e}")
            return state

        # 混合向量检索
        try:
            expr, expr_params = item_name_filter(item_names)
            # 创建混合向量
            hybrid_search_request = create_hybrid_search_requests(
                dense_vector=embed_query_vector.get("dense")[0],
                sparse_vector=embed_query_vector.get("sparse")[0],
                expr=expr,
                expr_params=expr_params,
                limit=5,
            )
            hybrid_search_response = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_request,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["chunk_id", "content", "item_name", "title"],
            )
            if not hybrid_search_response or not hybrid_search_response[0]:
                return state

            state["embedding_chunks"] = hybrid_search_response[0]
            return state
        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}混合向量检索失败：{e}")
            return state

    def _validate_state(self, state: QueryGraphState) -> tuple[str, list[str]]:
        # 用户的问题(LLM重写后的)
        rewritten_query = state.get("rewritten_query")
        # 获取商品名列表
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name, field_name="rewritten_query", expected_type=str
            )
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name, field_name="item_names", expected_type=list
            )

        return rewritten_query, item_names


if __name__ == "__main__":
    from knowledge.processor.query_processor.state import get_default_state

    # 用真实流程的状态结构做底（embedding_chunks等字段有默认值，0命中时不会崩）
    state = get_default_state()
    state["rewritten_query"] = "H3C LA2608室内无线网关怎么使用，教教我"
    # 注意：item_names必须是库中item_name字段的完整精确值（expr是 item_name in [...] 精确匹配）
    state["item_names"] = ["H3C LA2608-GM/GU 室内无线网关"]

    vector_search = HybridVectorSearchNode()
    result = vector_search.process(state)

    chunks = result.get("embedding_chunks") or []
    print(f"检索到 {len(chunks)} 条结果")
    for r in chunks:
        print(json.dumps(r, ensure_ascii=False, indent=2))
