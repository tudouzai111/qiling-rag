import json
import os
import re
from typing import Tuple, Dict, Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


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

        config = self.config

        # 参数校验
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)

        # 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

        # 二次切分或合并
        final_sections = self._split_and_merge(sections, config.max_content_length, config.min_content_length)

        # 组装chunk对象
        final_chunks = self._assemble_chunks(final_sections)

        # 备份测试
        self._backup_chunks(final_chunks, state)

        # 更新state chunks
        state['chunks'] = final_chunks

        return state

    def _validate_state(self, state: ImportGraphState, config: ImportConfig) -> Tuple[str, str, int, int]:
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

        def _flush() -> None:
            """
            将当前匹配到的标题封装到section对象
            section: {
                "body": "收集到的所有行",
                "title": "当前标题",
                "parent_title": "当前内容的父标题",
                "file_title": "文档标题"
            }

            Returns:
                None（副作用：把当前章节 append 进 sections）

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

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) \
            -> List[Dict[str, Any]]:
        """
        切分较大的章节（section）以及合并较小的章节

        Args:
            sections: 所有经过一次切分后的章节
            max_content_length: 最大内容长度（content: title + \n\n + body）如果 section 中的内容长度超过，需要进行切割，尽量不要过多切割 section
            min_content_length: 最小内容长度，如果 section 中的长度不足，就需要进行合并，尽量不要过多合并 section

        Returns:
            List[Dict[str, Any]]
            先切分后合并的章节列表

        """

        # 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_section(section, max_content_length))

        # 合并
        final_sections = self._merger_section(current_sections, min_content_length)

        return final_sections

    def _split_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """
        切分章节内容

        Args:
            section: 当前章节内容
            max_content_length: 最大章节长度阈值

        Returns:
            List[Dict[str, Any]]
        """

        # 获取 section 中的属性
        # section: {
        #     "body": "收集到的所有行",
        #     "title": "当前标题",
        #     "parent_title": "当前内容的父标题",
        #     "file_title": "文档标题"
        # }
        body = section.get('body')
        title = section.get('title')
        parent_title = section.get('parent_title')
        file_title = section.get('file_title')

        # 如果标题很长
        if len(title) >= 80:
            title = title[:80]

        # 对表格标签做处理
        body = MarkdownTableLinearizer.process(body)
        section['body'] = body

        # 判断是否切分
        # 标题前缀
        title_prefix = f'{title}\n\n'

        # 获取标题前缀加 body 长度
        total_length = len(title_prefix) + len(body)

        # 判断长度是否大于阈值
        if total_length <= max_content_length:
            return [section]

        body_length = max_content_length - len(title_prefix)
        # 如果标题还是很长
        if body_length <= 0:
            return [section]

        # 切分
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,  # chunk 块的大小
            chunk_overlap=0,  # 块与块之间重叠的字符数
            separators=["\n\n", "\n", "，", "。", "？", "！", "；", ",", ".", "?", "!", ";", " ", ""],  # 分隔符
        )

        sections = text_spliter.split_text(body)

        # 如果没有切分直接返回
        if len(sections) == 1:
            return [section]

        sub_sections = []
        for index, section in enumerate(sections):
            sub_sections.append({
                'body': section,
                'title': f'{title}_{index + 1}',
                'parent_title': parent_title,
                'file_title': file_title
            })

        return sub_sections

    def _merger_section(self, sections: List[Dict[str, Any]], min_content_length: int) -> List[Dict[str, Any]]:
        """
        合并章节

        Args:
            sections: 二次切分后的 sections
            min_content_length: section 的内容最小长度阈值

        Returns:
            List[Dict[str, Any]]
            合并之后的 section 列表

        """

        # 空列表守卫：没有可合并的对象时直接原样返回，避免 sections[0] 越界
        if len(sections) <= 1:
            return sections

        current_section = sections[0]
        final_sections = []

        for next_section in sections[1:]:
            # 如果当前 section 和上一个 section 是同一个父标题 并且 满足长度小于最小内容长度阈值
            if (current_section['parent_title'] == next_section['parent_title']) and \
                    len(current_section.get('body')) < min_content_length:
                # 合并
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )

                # 合并后当前标题回退为父标题；顶级章节没有父标题时用文档名兜底，避免 title 变成空串
                current_section['title'] = current_section['parent_title'] or current_section['file_title']
            else:
                final_sections.append(current_section)
                current_section = next_section

        # 最后一个 current_section 没遍历到，再加一下
        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        组装 chunks

        Args:
            final_sections:

        Returns:

        """
        final_chunks = []
        for section in final_sections:
            body = section.get('body')
            title = section.get('title')
            parent_title = section.get('parent_title')
            file_title = section.get('file_title')

            # strip 掉首尾空白，避免 content 开头出现连续换行（正文自带的空行）
            content = f"{title.strip()}\n\n{body.strip()}"

            final_chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title
            })
        self.logger.info(f"最终切割后能够进入到嵌入节点的 chunk 个数：{len(final_chunks)}")
        return final_chunks

    def _backup_chunks(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get('file_dir', '')
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, 'chunks.json')

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f'备份失败：{e}')


if __name__ == '__main__':
    setup_logging()
    document_split_node = DocumentSplitNode()

    md_path = r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test\hybrid_auto\test_new.md"
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    init_state = {
        'md_content': md_content,        # ← md_to_img_node 的产出（内容，不是路径）
        'file_title': 'test',            # ← entry_node 的产出
        'file_dir': r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test",
    }
    result = document_split_node.process(init_state)

    print('chunk 个数：', len(result['chunks']))
    print(json.dumps(result['chunks'][:3], ensure_ascii=False, indent=2))
    print('备份文件：',
          r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test\chunks.json")
