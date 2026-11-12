from typing import Any

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState


class RrfMergeNode(BaseNode):
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取本地检索的两路结果
        embedding_chunks = state.get("embedding_chunks", [])
        hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])

        # 定义两路检索结果和对应路的权重映射表
        search_result_weight = {
            "embedding_search_chunks": (
                self._validate_search_result(embedding_chunks),
                1.0,
            ),
            "hyde_embedding_search_chunks": (
                self._validate_search_result(hyde_embedding_chunks),
                1.0,
            ),
        }
        rrf_inputs = list(search_result_weight.values())
        # 用RRF计算两路文档的分数
        merged_rrf_results: list[tuple[dict[str, Any], float]] = self._merge_rrf_docs(
            rrf_inputs, self.config.rrf_k, self.config.rrf_max_results
        )

        # RRF融合得到的是 (chunk对象, 分数) 列表：分数只服务于排序，排完序就完成使命了。
        # 这里像老师版一样只把chunk对象存入state——下游reranker拿到纯dict列表，
        # 可以直接 for chunk in state['rrf_chunks']: chunk.get('content')
        state['rrf_chunks'] = [chunk for chunk, _score in merged_rrf_results]

        # RRF得分打进日志便于观察（若连分数一起存进state，下游契约就乱了）
        score_preview = [f"{chunk.get('chunk_id')}={score:.5f}" for chunk, score in merged_rrf_results]
        self.logger.info(f"RRF融合完成，共{len(score_preview)}条（按总分降序）：{score_preview}")

        return state

    @staticmethod
    def _merge_rrf_docs(
        rrf_inputs: list[tuple[list[dict[str, Any]], float | int]],
        rrf_k: int,
        rrf_max_results: int,
    ) -> list[tuple[dict[str, Any], float | int]]:
        """
        RRF计算多路检索返回的文档得分
        Args:
            rrf_inputs: 多路检索的文档以及对应的权重
            rrf_k: 平滑参数
            rrf_max_results: 最大返回个数

        Returns:
            tuple[dict, float]: 多路检索文档对象, 经过RRF计算之后的文档得分
        """
        chunk_score: dict[Any, float | int] = {}
        chunk_data: dict[Any, dict[str, Any]] = {}

        for search_result, weight in rrf_inputs:
            for rank, entity in enumerate(search_result, 1):
                chunk_id = entity.get("chunk_id")
                if not chunk_id:
                    continue

                chunk_score[chunk_id] = chunk_score.get(
                    chunk_id, float(0)
                ) + weight / (rrf_k + rank)
                chunk_data.setdefault(chunk_id, entity)  # 根据 key 去重

        rrf_result = sorted(
            [
                # chunk_score中的每个key都经过setdefault写入chunk_data，必然存在，用下标取值
                # （避免.get()把dict | None类型引入推导式，导致与返回类型注解不符）
                (chunk_data[chunk_id], score)
                for chunk_id, score in chunk_score.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return rrf_result[:rrf_max_results] if rrf_max_results else rrf_result

    def _validate_search_result(
        self, search_result: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """校验Milvus检索结果"""
        if not search_result:
            return []

        search_results = []
        for res in search_result:
            if not res or not isinstance(res, dict):
                continue

            entity = res.get("entity")
            if not entity or not isinstance(entity, dict):
                continue

            search_results.append(entity)

        return search_results


if __name__ == "__main__":
    # BaseNode的self.logger需要在根logger配置好handler后才能输出INFO日志
    RrfMergeNode.setup_logging()

    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfMergeNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")
