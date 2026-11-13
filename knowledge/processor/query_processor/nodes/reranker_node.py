import math
from typing import Any

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    name = "reranker_node"

    @staticmethod
    def _sigmoid(score: float) -> float:
        """
        sigmoid 归一化 1 / (1 + e⁻ᶻ)
        为什么不用 min-max (x - min) / (max - min) ?
        归一化后只是把分数线性拉伸到 0-1，依赖最小最大值，并没有解决问题
        Reranker 是交叉编码器，sigmoid 刚好可以把 Reranker 输出 Logit 映射到 0-1，更适合重排序分数归一化
        """
        return 1.0 / (1.0 + math.exp(-score))

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取用户的问题
        user_query = state.get("rewritten_query") or state.get("original_query")

        # 获取两路检索结果（本地检索、网络检索）
        reranker_outputs: list[dict[str, Any]] = self._collect_rerank_inputs(state)

        # 利用Reranker精排
        # 为什么不把网络检索也加到rrf中？
        # 因为rrf是把不同检索方式检索同一个向量数据库，而网络检索是在网络检索，就算放到rrf里也只能单路排序
        refine_docs: list[dict[str, Any]] = self._refine_rank(
            user_query, reranker_outputs
        )

        # 动态top_k：
        # top_k过大：LLM上下文过长，可能会截取无关上下文
        # top_k过小：LLM检索上下文信息不全面
        reranked_docs = self._cliff_cutoff(
            refine_docs, self.config.rerank_min_top_k, self.config.rerank_max_top_k
        )
        state["reranked_docs"] = reranked_docs

        return state

    def _collect_rerank_inputs(self, state: QueryGraphState) -> list[dict[str, Any]]:
        """获取两路检索结果"""
        docs = []
        # 获取本地检索结果
        rrf_chunks = state.get("rrf_chunks", [])
        for chunk in rrf_chunks:
            if not chunk or not isinstance(chunk, dict):
                continue

            content = chunk.get("content", "")
            if not content:
                continue
            title = chunk.get("title", "")
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            # 格式化文档
            formated_local_doc = self._format_doc(
                content=content, chunk_id=chunk_id, title=title, source="local"
            )
            self.logger.info(f"获取本地搜索结果：{len(formated_local_doc)} 个")
            docs.append(formated_local_doc)

        # 获取远程检索结果
        web_search_docs = state.get("web_search_docs", [])
        for doc in web_search_docs:
            if not doc or not isinstance(doc, dict):
                continue

            content = doc.get("snippet", "")
            title = doc.get("title", "")
            url = doc.get("url", "")

            # 格式化文档
            formated_web_doc = self._format_doc(
                content=content, title=title, url=url, source="web"
            )
            self.logger.info(f"获取远程联网搜索结果：{len(formated_web_doc)} 个")
            docs.append(formated_web_doc)

        self.logger.info(f"获取Reranker搜索结果：{len(docs)} 个")
        return docs

    @staticmethod
    def _format_doc(
        content: str,
        chunk_id: int | None = None,
        title: str = "",
        url: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        """格式化本地或远程检索到的文档"""
        return {
            "content": content,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "source": source,
        }

    def _refine_rank(
        self, user_query: str, reranker_outputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        用Reranker模型进行打分和排序（精排）

        Args:
            user_query: 用户的查询
            reranker_outputs: 本地和远程的检索结果

        Returns:
            文档和分数
        """

        try:
            reranker_client = AIClients.get_bge_reranker_client()
        except ConnectionError as e:
            self.logger.error(f"获取BGE-Reranker重排序模型失败：{e}")
            return []

        # 构建Q->D pairs
        query_doc_pairs = [(user_query, d.get("content", "")) for d in reranker_outputs]

        try:
            # 计算：BGE-Reranker模型计算出来的得分是没有大小限制的
            reranker_scores = reranker_client.compute_score(
                sentence_pairs=query_doc_pairs
            )
            doc_score = [
                # 对分数做归一化处理
                {**d, "score": self._sigmoid(float(s))}
                for d, s in zip(reranker_outputs, reranker_scores)
            ]
            # 排序
            sorted_doc_score = sorted(doc_score, key=lambda x: x["score"], reverse=True)
            return sorted_doc_score
        except Exception as e:
            self.logger.error(f"获取BGE-Reranker重排序模型计算失败：{e}")
            return [{**d, "score": None} for d in reranker_outputs]

    def _cliff_cutoff(
        self,
        refine_docs: list[dict[str, Any]],
        rerank_min_top_k: int,
        rerank_max_top_k: int,
    ) -> list[dict[str, Any]]:
        """
        动态top_k机制截取文档个数：自动决定保留多少篇文档送给 LLM 生成答案，而不是用一个固定的 top_k 一刀切。
        从头开始寻找最大断崖点

        Args:
            refine_docs: 精排后的文档
            rerank_min_top_k: 最少返回文档数
            rerank_max_top_k: 最多返回文档数

        Returns:

        """
        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        cutoff_idx = upper_bound
        max_gap = 0
        for i in range(0, upper_bound - 1):
            current_doc_score = refine_docs[i].get('score')
            next_doc_score = refine_docs[i + 1].get('score')
            if not current_doc_score or not next_doc_score:
                continue

            gap = current_doc_score - next_doc_score
            if gap >= 0.15 and gap > max_gap:
                max_gap = gap
                cutoff_idx = i + 1
                self.logger.info(f"位置{cutoff_idx}发生了断崖")

        # 至少保留lower_bound个
        cutoff_idx = max(cutoff_idx, lower_bound)
        cutoff_docs = refine_docs[:cutoff_idx]

        return cutoff_docs




if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。",
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。",
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。",
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。",
            },
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
