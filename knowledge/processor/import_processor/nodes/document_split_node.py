import re
from typing import Tuple, Dict, Any, List

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.state import ImportGraphState


class DocumentSplitNode(BaseNode):
    """
    文档切分节点

    处理流程：
        读取md文件内容
        按照markdown标题进行章节切分
        对超长章节进行二次切分
        合并过短的相邻章节
        组装最终content = title + body
    """

    name = 'document_split_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分入口节点
        """

        # 参数校验
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state)

        # 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        """
        参数校验并返回参数

        Args:
            state: 导入图状态，包含 md_content 和 file_title

        Returns:
            Tuple[str, str, int, int]:
                md_content: 统一换行符后的 Markdown 内容
                file_title: 文件标题
                max_content_length: 切片最大长度配置
                min_content_length: 切片最小长度配置

        Raises:
            ValueError: 切片长度参数校验失败时抛出

        """

        config = self.config

        self.log_step('step1', '切分文档的参数校验以及获取...')

        # 获取 md_content
        md_content = state.get('md_content', '')

        # 统一换行符
        if md_content:
            md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

        # 获取文件标题
        file_title = state.get('file_title', '')

        # 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f'切片长度参数校验失败')

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        根据标题切分（1-6个#）

        Args:
            md_content: 需要切分的md
            file_title: 文档标题

        Returns:
            List[Dict]: 切分的多个章节列表，每个章节格式：{
                "title": "# 第一章",
                "body": "正文内容...",
                "file_title": "万用表",
                "parent_title": "# 第一章"
            }
        """
        self.log_step('step2', '根据标题进行切分...')

        in_fence = False  # 是否在代码块内
        body_lines = []  # 正文内容
        sections: List[Dict[str, Any]] = []  # 章节内容
        current_level = 0
        current_title = ""  # 当前内容的标题
        hierarchy = [""] * 7  # 存储所有标题内容，用来确认section的父标题，1-6对应6级标题
        md_lines = md_content.split('\n')  # 根据 \n 切分
        # 标题正则：匹配行首的1-6个#号和标题的文本内容
        # (): 捕获组：产生三个，group(0)：匹配整个文本，group(1): 1-6个#号，group(2): 标题的文本内容
        heading_re = re.compile(r'^\s*(#{1,6})\s+(.+)')

        def _flush() -> sections:
            """
            将当前匹配到的标题封装到section对象
            section: {
                "body": "收集到的所有行",
                "title": "当前标题",
                "parent_title": "当前内容的父标题",
                "file_title": "文档标题"
            }

            Returns:
                sections

            """

            # 处理内容行
            body = '\n'.join(body_lines)
            parent_title = ""  # 没有祖先时（如一级标题/文档前言）留空
            for i in range(current_level - 1, 0, -1):
                if hierarchy[i]:
                    parent_title = hierarchy[i]
                    break

            sections.append({
                "body": body,
                "title": current_title if current_title else file_title,
                "parent_title": parent_title,
                "file_title": file_title
            })

        # 遍历切分后的md_lines
        for md_line in md_lines:
            # 检查代码块（``` 或 ~~~）因为代码块中的 # 不能识别为标题
            if md_line.startswith('```') or md_line.startswith('~~~'):
                in_fence = not in_fence  # 因为代码块是成对出现的，所以直接取反，出了代码块区域时刚好in_fence = False

            # 正则匹配，如果是代码块内部就不需要
            match = heading_re.match(md_line) if not in_fence else None

            # 如果是标题
            if match:
                # 将当前匹配到的标题行打包
                _flush()
                # 当前标题就是当前行
                current_title = md_line
                current_level = len(match.group(1))  # 当前标题层级
                hierarchy[current_level] = current_title

                for i in range(current_level + 1, 7):
                    hierarchy[i] = ""

                body_lines = []
            # 如果不是标题或者是代码块
            else:
                body_lines.append(md_line)

        _flush()
        return sections
