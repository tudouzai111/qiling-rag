from typing import Any

from langchain_openai import ChatOpenAI

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 判断流式输出
        is_stream = state.get("is_stream")
        # 获取任务id
        task_id = state.get("task_id")
        # 判断是否已有答案
        if state.get("answer"):
            # 直接推送答案
            self._push_exist_answer(task_id, is_stream, state)
            is_streamed = False
        else:
            # 组装提示词
            prompt = self._build_prompt(state)
            state["prompt"] = prompt

            # 调用LLM
            self._generate_answer(prompt, task_id, state)
            is_streamed = is_stream

        # 保存历史对话
        self._save_history(state)

        # 关闭sse通道
        if is_stream:
            # 已经执行过流式输出，关闭
            if is_streamed:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.FINAL,
                    data={},
                )
            # 执行流式输出，关闭
            else:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.FINAL,
                    data={"answer": state.get("answer")},
                )

        return state

    def _push_exist_answer(self, task_id: str, is_stream: bool, state: QueryGraphState):
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get("answer"))

    def _build_prompt(self, state: QueryGraphState) -> str:
        """构建提示词"""
        # 获取提示词最大长度
        max_context_chars = self.config.max_context_chars

        # 获取用户原始问题
        user_query = state.get("rewritten_query")

        # 获取商品名列表
        item_names = state.get("item_names", [])

        # 构建检索的上下文
        retrieval_context = state.get("reranked_docs", [])
        formatted_context, usage_chars = self._format_retrieval_context(
            retrieval_context, max_context_chars
        )

        # 构建历史上下文
        # 从内存中获取历史上下文
        chat_history_context = state.get("history") or []
        formatted_history = self._format_chat_history(chat_history_context, usage_chars)

        # 格式化提示词模板
        return ANSWER_PROMPT.format(
            context=formatted_context or "暂无检索到的上下文",
            history=formatted_history or "暂无历史上下文",
            item_names=",".join(item_names),
            question=user_query,
        )

    def _format_retrieval_context(
        self, retrieval_context: list[dict[str, Any]], max_context_chars: int
    ) -> tuple[str, int]:
        """
        格式化检索到的上下文

        Args:
            retrieval_context: 检索到的上下文
            max_context_chars: 上下文最大长度

        Returns:

        """
        formatted_lines = []
        usage = 0
        for idx, context in enumerate(retrieval_context):
            content = context.get("content", "")
            if not content:
                continue

            # 拼接元数据列表
            metadata_content = [f"文档[{idx}]"]
            for meta_field, template in [
                ("chunk_id", "[chunk_id={}]"),
                ("title", "[title={}]"),
                ("source", "[source={}]"),
                ("url", "[url={}]"),
            ]:
                field_value = str(context.get(meta_field, "")).strip()
                if field_value:
                    metadata_content.append(template.format(field_value))

            # 获取得分
            soc_score = context.get("score", float(0))
            if soc_score is not None:
                metadata_content.append(f"[score={float(soc_score):.6f}]")

            # 构建完整行数据（内容+元数据）
            formatted_line = " ".join(metadata_content) + "\n" + content

            # 计算行与行之间的字符数(\n\n)
            sep_chars = 2 if formatted_line else 0
            total_length = sep_chars + len(formatted_line)

            # 计算总长度
            if usage + total_length > max_context_chars:
                break
            else:
                formatted_lines.append(formatted_line)
                usage += total_length

        return "\n\n".join(formatted_lines), max_context_chars - usage

    def _generate_answer(self, prompt: str, task_id: str, state: QueryGraphState):
        try:
            llm_client = AIClients.get_openai_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"LLM客户端获取失败：{e}")
            return

        if state.get("is_stream"):
            # 流式调用LLM生成回答，写入sse队列
            state["answer"] = self._stream_llm(task_id, prompt, llm_client)
        else:
            # 非流式调用LLM生成回答
            state["answer"] = self._invoke_llm(prompt, llm_client)
            # 写入任务结果队列
            set_task_result(task_id=task_id, key="answer", value=state["answer"])

    def _invoke_llm(self, prompt: str, llm_client: ChatOpenAI) -> str:
        """非流式LLM调用"""
        try:
            # 同步调用
            llm_res = llm_client.invoke(prompt)

            llm_content = getattr(llm_res, "content", "") or ""
            if not llm_content:
                return "LLM同步调用回答失败"

            return llm_content
        except Exception as e:
            self.logger.error(f"LLM同步调用回答失败：{e}")
            return "LLM同步调用回答失败"

    def _stream_llm(self, task_id: str, prompt: str, llm_client: ChatOpenAI):
        """流式LLM调用"""
        # 全量数据
        accelerate_delta = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    push_sse_event(
                        task_id=task_id,
                        event=SSEEvent.DELTA,
                        data={"delta": delta_text},
                    )
                    accelerate_delta += delta_text
        except Exception as e:
            self.logger.error(f"LLM流式调用回答失败：{e}")
            return "LLM流式调用回答失败"

        return accelerate_delta

    def _save_history(self, state: QueryGraphState):
        """
        保存历史对话
        存储到MongoDB的kb001库的chat_message表中
        """
        session_id = state.get("session_id")
        user_query = state.get("original_query")
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names") or []
        try:
            # 保存用户角色的信息
            save_chat_message(
                session_id=session_id,
                role="user",
                text=user_query,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            # 保存AI角色的信息
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state.get("answer"),
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
        except Exception as e:
            self.logger.error(f"保存历史对话到MongoDB中失败：{e}")

    def _format_chat_history(
        self, chat_history_context: list[dict[str, Any]], usage_chars: int
    ):
        """
        格式化历史上下文

        Args:
            chat_history_context: 历史上下文
            usage_chars: 可用上下文容量

        Returns:

        """

        formatted_lines = []
        used_chars = 0
        role_map = {"user": "用户", "assistant": "AI助手"}
        for msg in chat_history_context:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"
            # 计算分割符长度
            seperator_usage = 1 if formatted_line else 0
            # 计算总长度
            total_usage = seperator_usage + len(formatted_line)
            if used_chars + total_usage > usage_chars:
                break

            formatted_lines.append(formatted_line)
            usage_chars += total_usage

        return '\n'.join(formatted_lines)