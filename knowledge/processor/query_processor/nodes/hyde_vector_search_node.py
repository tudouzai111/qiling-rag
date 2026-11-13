from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    item_name_filter,
)


class HyDeVectorSearchNode(BaseNode):
    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> dict:
        # 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 利用LLM生成原始问题对应的假设性答案（解决跨域不对称）
        hy_document = self._generate_hy_document(rewritten_query, item_names)
        if hy_document is None:
            return {}

        # 获取嵌入模型客户端
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败：{e}")
            return {}

        # 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败：{e}")
            return {}

        # 嵌入假设性文档
        try:
            embed_hy_vector = generate_bge_m3_hybrid_vectors(
                bge_m3_client, [f"{rewritten_query}\n{hy_document}"]
            )
        except Exception as e:
            self.logger.error(f"假设性文档{hy_document}嵌入失败：{e}")
            return {}

            # 混合向量检索
        try:
            expr, expr_params = item_name_filter(item_names)
            # 创建混合向量
            hybrid_search_request = create_hybrid_search_requests(
                dense_vector=embed_hy_vector.get("dense")[0],
                sparse_vector=embed_hy_vector.get("sparse")[0],
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
                return {}

            return {"hyde_embedding_chunks": hybrid_search_response[0]}
        except Exception as e:
            self.logger.error(
                f"原始问题{rewritten_query}对应的假设性文档{hy_document}混合向量检索失败：{e}"
            )
            return {}

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

    def _generate_hy_document(
        self, rewritten_query: str, item_names: list[str]
    ) -> str | None:
        """
        生成假设性文档

        Args:
            rewritten_query:

        Returns:

        """

        # 获取 LLM 客户端
        try:
            llm_client = AIClients.get_openai_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"获取LLM客户端失败：{e}")
            return None

        # 获取提示词
        system_prompt = f"您是一位{item_names}方面的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names, rewritten_query=rewritten_query
        )

        # 调用
        try:
            llm_response = llm_client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as e:
            self.logger.error(f"LLM生成{item_names}的假设性文档失败：{e}")
            return None
        if not llm_response.content.strip():
            return None

        return llm_response.content.strip()


if __name__ == "__main__":
    state = {
        "rewritten_query": "H3C LA2608室内无线网关怎么使用，教教我",
        "item_names": ["H3C LA2608-GM/GU 室内无线网关"],
    }

    node = HyDeVectorSearchNode()
    result = node.process(state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n[HyDe 检索结果]：{len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(
            f"  [{i}] chunk_id={entity.get('chunk_id')} "
            f"item_name={entity.get('item_name')} "
            f"distance={chunk.get('distance', 'N/A')}"
        )
        content = entity.get("content", "")
        print(f"      内容：{content[:80]}...")
    print("-" * 60)
    print("测试完成")
