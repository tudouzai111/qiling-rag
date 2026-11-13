import re
import urllib.request
from typing import Any

from langchain_openai import ChatOpenAI

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import set_task_result

# markdown图链：URL部分允许空格（MinIO对象名含空格，如"H3C LA2608... 用户手册"），捕获到右括号为止
_MD_IMG_RE = re.compile(r'!\[[^\]\n]*\]\(([^)\n]+)\)', re.IGNORECASE)
# 裸URL（无空格场景，如联网搜索结果）
_BARE_IMG_RE = re.compile(r'https?://[^\s\)\]"\']+\.(?:jpg|jpeg|png|gif|webp)', re.IGNORECASE)


def _normalize_image_url(raw: str) -> str | None:
    """规范化图片URL：校验扩展名 + 空格编码为%20（MinIO对象名含空格）"""
    u = raw.strip()
    if not re.search(r'\.(?:jpg|jpeg|png|gif|webp)$', u, re.IGNORECASE):
        return None
    if ' ' in u:
        u = u.replace(' ', '%20')
    return u


def _filter_icon_images(urls: list[str], min_bytes: int = 4096, timeout: int = 3) -> list[str]:
    """过滤疑似图标的极小图片。

    设备手册PDF里混有装饰性小图标（约2KB，如"配置记录"铅笔图标），
    真实图表通常远大于此。HEAD探测Content-Length，小于阈值丢弃；
    探测失败时保守保留（宁多勿错）。
    """
    kept: list[str] = []
    for u in urls:
        try:
            req = urllib.request.Request(u, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                size = int(resp.headers.get("Content-Length") or 0)
            if size >= min_bytes:
                kept.append(u)
        except Exception:
            kept.append(u)
    return kept


def _is_image_line(line: str) -> bool:
    """判断整行是否为图片URL（LLM抄写产生的散落行）"""
    s = line.strip()
    if not re.search(r'\.(?:jpg|jpeg|png|gif|webp)$', s, re.IGNORECASE):
        return False
    return s.startswith('http') or '/' in s


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 判断流式输出
        is_stream = state.get("is_stream")
        # 获取任务id
        task_id = state.get("task_id")
        # 判断是否已有答案
        if state.get("answer"):
            # 已有答案（澄清/道歉等拦截路径），直接落任务结果
            self._push_exist_answer(task_id, is_stream, state)
        else:
            # 组装提示词
            prompt = self._build_prompt(state)
            state["prompt"] = prompt

            # 调用LLM
            self._generate_answer(prompt, task_id, state)

        # 整理答案中的图片：从切片原文收集URL，剔除LLM抄写的散落行，统一追加【图片】区块
        # 放在保存历史之前，MongoDB里存的也是整理后的文本
        images = self._polish_answer(state)

        # 非流式：用整理后的答案覆盖任务结果（_generate_answer写入的是整理前的文本）
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get("answer", ""))

        # 保存历史对话
        self._save_history(state)

        # 关闭sse通道（FINAL统一携带整理后的答案与图片列表，前端以服务端答案为准）
        if is_stream:
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.FINAL,
                data={"answer": state.get("answer"), "images": images},
            )

        return state

    def _push_exist_answer(self, task_id: str, is_stream: bool, state: QueryGraphState):
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get("answer"))

    def _collect_image_urls(self, state: QueryGraphState) -> list[str]:
        """从检索上下文收集图片URL（markdown图链/裸URL；空格统一编码为%20）"""
        urls: list[str] = []
        for doc in state.get("reranked_docs", []):
            content = doc.get("content", "") or ""
            for raw in _MD_IMG_RE.findall(content):
                norm = _normalize_image_url(raw)
                if norm:
                    urls.append(norm)
            for raw in _BARE_IMG_RE.findall(content):
                norm = _normalize_image_url(raw)
                if norm:
                    urls.append(norm)
        # 去重并保持出现顺序
        seen: set[str] = set()
        return [u for u in urls if not (u in seen or seen.add(u))]

    def _polish_answer(self, state: QueryGraphState) -> list[str]:
        """剔除LLM抄写的散落图片行，统一追加规范【图片】区块；返回图片列表"""
        images = self._collect_image_urls(state)
        images = _filter_icon_images(images)
        if not images:
            return []
        answer = state.get("answer") or ""
        kept = [ln for ln in answer.split("\n") if not _is_image_line(ln)]
        state["answer"] = "\n".join(kept).rstrip() + "\n【图片】\n" + "\n".join(images)
        return images

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
            soc_score = context.get("score")
            if soc_score is not None:
                metadata_content.append(f"[score={float(soc_score):.6f}]")

            # 构建完整行数据（内容+元数据）
            formatted_line = " ".join(metadata_content) + "\n" + content

            # 计算行与行之间的字符数(\n\n)
            sep_chars = 2 if formatted_lines else 0
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
            state["answer"] = "LLM客户端获取失败"
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
            seperator_usage = 1 if formatted_lines else 0
            # 计算总长度
            total_usage = seperator_usage + len(formatted_line)
            if used_chars + total_usage > usage_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return '\n'.join(formatted_lines)