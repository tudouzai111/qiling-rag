import json
import logging
import re
from builtins import Exception
from json import JSONDecodeError
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.config import get_config
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)
from knowledge.utils.mongo_history_util import get_recent_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _ItemNameExtractor:
    def extract_item_name(self, original_query, history_context) -> dict[str, Any]:
        """
        提取商品名
        Args:
            original_query: 用户原始查询
            history_context: 历史对话上下文

        Returns:

        """
        llm_result = {"item_names": [], "rewritten_query": original_query}

        # 获取LLM
        try:
            llm_client = AIClients.get_openai_llm_client()
        except ConnectionError as e:
            logger.error(f"LLM客户端获取失败：{e}")
            return llm_result

        # 获取商品名提取的提示词
        # 系统提示词
        item_name_system_prompt = "您是一位商品名提取专家，请从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        # 用户提示词
        item_name_user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context.strip()
            if history_context
            else "暂无历史上下文",
            query=original_query,
        )

        # 调用LLM
        llm_response = llm_client.invoke(
            [
                SystemMessage(content=item_name_system_prompt),
                HumanMessage(content=item_name_user_prompt),
            ]
        )
        llm_response_content = llm_response.content
        if not llm_response_content:
            return llm_result

        # 清洗和解析
        parsed_result: dict[str, Any] = self._clean_and_parse(llm_response_content)

        llm_result["item_names"] = parsed_result.get("item_names")
        llm_result["rewritten_query"] = (
            parsed_result.get("rewritten_query")
            if parsed_result.get("rewritten_query")
            else original_query
        )
        return llm_result

    @staticmethod
    def _clean_and_parse(llm_response_content: str) -> dict[str, Any]:
        """
        清洗以及解析LLM的结果
        Args:
            llm_response_content: LLM的输出

        Returns:

        """
        # 去除json代码块围栏标记```xxx```，虽然在ai_clients.py的_create_openai_llm_client中已经配置了输出结构，这里再一次去除兜底
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 解析
        try:
            # 反序列化
            llm_content_obj: dict[str, Any] = json.loads(content)
            # 判断类型以及清洗
            raw_item_names = llm_content_obj.get("item_names")
            if not isinstance(raw_item_names, list):
                item_names = []
            else:
                item_names = [
                    item_name.strip()
                    for item_name in raw_item_names
                    if isinstance(item_name, str) and item_name.strip()
                ]

            raw_rewritten_query = llm_content_obj.get("rewritten_query")
            if not isinstance(raw_rewritten_query, str):
                rewritten_query = []
            else:
                rewritten_query = raw_rewritten_query.strip()

            return {"item_names": item_names, "rewritten_query": rewritten_query}
        except JSONDecodeError as e:
            logger.error(f"LLM输出结果 {llm_response_content} 反序列化失败：{e}")
            raise JSONDecodeError(msg=e.msg, doc=e.doc, pos=e.pos)


class _ItemNameAligner:
    def __init__(self) -> None:
        self._config = get_config()

    def search_and_align(self, item_names: list[str]) -> tuple[list[str], list[str]]:
        """
        检索向量数据库并和向量数据库的商品名对齐 最终返回确定的商品名列表或者模糊的商品名列表

        Args:
            item_names: LLM提取的商品名列表

        Returns:

        """
        # 混合检索向量数据库
        search_result = self._search_vector(item_names)
        if not search_result:
            return [], []

        # 对齐
        confirmed, options = self._align(search_result)

        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _search_vector(self, item_names: list[str]) -> list[dict[str, Any]]:
        """
        对LLM提取的商品名列表进行向量检索
        Args:
            item_names: LLM提取的商品名列表

        Returns:
            {"extracted_name": "LLM提取的商品名", "matches": [{向量库中查询到的文档1}, {向量库中查询到的文档12}]},
        """
        search_result = []

        # 获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"Milvus客户端获取失败: {e}")
            return search_result
        # 获取BGE-M3模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"BGE-M3模型获取失败: {e}")
            return search_result

        # 商品名列表混合向量化
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(
                model=bge_m3_client, embedding_documents=item_names
            )
        except Exception as e:
            logger.error(f"商品列表{item_names}生成混合向量失败：{e}")
            return search_result

        # 混合向量检索
        for idx, item_name in enumerate(item_names):
            hybrid_search_requests = create_hybrid_search_requests(
                hybrid_vector_result["dense"][idx], hybrid_vector_result["sparse"][idx]
            )

            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self._config.item_name_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["item_name"],
            )

            # {
            #     "id": 1001,
            #     "distance": 0.85,
            #     "entity": {"item_name": "Milvus 混合检索指南", "category": "技术文档"},
            # },
            matches = [
                {
                    "score": item_search_res["distance"],
                    "item_name": item_search_res["entity"]["item_name"],
                }
                for item_search_res in (
                    hybrid_search_result[0] if hybrid_search_result else []
                )
            ]
            search_result.append({"extracted_name": item_name, "matches": matches})

        return search_result

    def _align(
            self, search_result: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """
        容器：
            confirmed：用于下游检索
            options：用于用户手动确认
        对齐规则：
            向量数据库中查到的商品名分数大于 0.7 放到 confirmed
            向量数据库中查到的商品名分数小于等于 0.7 大于 0.45 放到 options
            向量数据库中查到的商品名分数小于等于 0.45 都不放
        疑问：
            置信度 0.7 0.45 怎么来的？

        Args:
            search_result: 向量数据库检索到的结果

        Returns:
            confirmed options 列表中的商品名
        """
        confirmed = []
        options = []

        for item_search_result in search_result:
            # 商品名
            llm_extract_item_name = item_search_result.get("extracted_name")
            # 搜索结果
            item_name_matches: list[dict[str, Any]] = item_search_result.get("matches", [])
            # 根据分数降序排序
            item_name_matches_sorted = sorted(
                item_name_matches, key=lambda x: x["score"], reverse=True
            )
            # 收集高置信度
            high = [
                h
                for h in item_name_matches_sorted
                if h.get("score", 0) > self._config.item_name_high_confidence
            ]
            if high:
                # 从向量数据库中搜索到的商品名和LLM提取到的商品名是否相等
                extract = next(
                    (h for h in high if h.get("item_name") == llm_extract_item_name),
                    None,
                )
                # 高精度匹配命中
                if extract:
                    picked = extract.get('item_name')
                    if picked not in confirmed:  # 去重
                        confirmed.append(picked)
                # 唯一高置信度
                elif len(high) == 1:
                    picked = high[0]['item_name']
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 有多个分数值高的商品匹配
                else:
                    top_score = high[0]['score']
                    # 如果第一个和第二个分数差距较大，只取第一个
                    if top_score - high[1]['score'] >= self._config.item_name_score_gap:
                        picked = high[0]['item_name']
                        if picked not in confirmed:
                            confirmed.append(picked)
                    else:
                        for h in high[:self._config.item_name_max_options]:
                            picked = h.get('item_name')
                            if picked not in options and picked not in confirmed:
                                options.append(picked)
            else:
                # 中置信度
                mid = [
                    m
                    for m in item_name_matches_sorted
                    if m['score'] >= self._config.item_name_mid_confidence
                       and m.get('item_name') not in options
                       and m.get('item_name') not in confirmed
                ]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get('item_name'))  # 一次循环内部只留三个可选
        # 最终只留三个可选的商品名
        return confirmed, options[:self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: list[str], search_result: list[dict[str, Any]]):
        # 构建 商品名->最高分数 映射
        item_name_score = {}
        for search_res in search_result:
            matches = search_res.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        if not item_name_score:
            return confirmed

        # 取出分数最大的作为基准
        max_score = max(item_name_score.values())
        return [
            name
            for name, score in item_name_score.items()
            if max_score - score <= self._config.item_name_score_gap
        ]


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取用户的原始问题
        original_query = state.get("original_query", "")

        # 获取历史对话
        session_id = state.get('session_id')
        # 限制10条消息，5轮对话 TODO MongoDB相关代码还没看
        history_context = get_recent_messages(session_id=session_id, limit=10)
        formatted_history = []
        for history in history_context:
            role = history.get('role', '')
            text = history.get('text', '')
            formatted_context = f"角色：{role}, 内容：{text}"
            formatted_history.append(formatted_context)
        formatted_history_str = " ".join(formatted_history)

        # 利用LLM进行商品名提取和查询重写
        llm_result: dict[str, Any] = self._extractor.extract_item_name(
            original_query, formatted_history_str
        )
        item_names = llm_result.get("item_names", "")
        rewritten_query = llm_result.get("rewritten_query", "")
        if item_names:
            confirmed, options = self._aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []

        # 决策
        self._decide(confirmed, options, state, rewritten_query, item_names)

        # 更新内存state中的history历史上下文
        state['history'] = formatted_history

        return state

    @staticmethod
    def _decide(
            confirmed: list[str],
            options: list[str],
            state: QueryGraphState,
            rewritten_query: str,
            item_names: list[str],
    ):
        """
        根据 confirmed options 来判断是继续检索还是返回用户提示信息

        Args:
            confirmed: 已经确认的商品名列表
            options: 模糊的商品名列表
            state: 查询状态
            rewritten_query: 重写后的问题
            item_names: LLM提取到的商品名列表

        Returns:

        """
        if confirmed:
            state["item_names"] = confirmed  # 对齐后的商品名
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}"
            )
        else:
            state["answer"] = (
                "抱歉：我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            )


if __name__ == "__main__":
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        "original_query": "我准备使用H3C LA2608室内无线网关和华为擎云 L420x，但是我不会用，你教我吧"
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)
