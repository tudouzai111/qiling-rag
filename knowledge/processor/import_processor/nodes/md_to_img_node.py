import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Deque

from minio import Minio
from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """
    图片上下文信息
    """
    head: str  # 图片标题
    pre_text: str  # 图片上文内容
    post_text: str  # 图片下文内容


@dataclass
class ImageInfo:
    """
    图片的完整信息
    图片名字：作为存储图片摘要的字典容器key
    图片地址：vlm要用，minio要用
    图片上下文信息：vlm要用
    """
    name: str  # 图片名字
    path: str  # 图片地址
    image_context: ImageContext  # 图片上下文信息


class _MdFileHandler:
    """
    读取md内容、md_path、图片目录
    备份新的md_content，方便测试
    """

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state) -> Tuple[str, Path, Path]:
        """
        读取md内容，路径，图片目录

        Args:
            state: 上一个节点更新后的state

        Returns:
            Tuple[str, Path, Path]

            md_content, md_path_obj, img_dir_obj

            md文件内容
            md文件路径
            图片目录路径
        """
        md_path = state.get('md_path', '')
        if not md_path:
            raise ValidationError('解析的文件不存在', node_name=self.node_name)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            FileProcessingError('解析的文件路径不存在', node_name=self.node_name)

        try:
            with open(md_path_obj, 'r', encoding='utf_8') as f:
                md_content = f.read()
        except IOError as e:
            self.logger.error(f'MD文件: {md_path_obj.name} 打开失败')
            raise FileProcessingError('文件打开失败', node_name=self.node_name)

        img_dir_obj = md_path_obj.parent / 'images'

        return md_content, md_path_obj, img_dir_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        """
        备份原始 Markdown 文件，生成带 _new 后缀的新文件。

        用于测试或预览场景：将处理后的 Markdown 内容写入到与源文件同目录的备份文件中，
        保留原始文件不变。

        Args:
            md_path_obj (Path): 原始 Markdown 文件的 Path 对象。
            new_md_content (str): 要写入备份文件的新 Markdown 内容。

        Returns:
            str: 备份文件的绝对路径（字符串形式）。

        """

        # 设定备份的md文件名 xxx.md 为 xxx_new.md
        # with_name: 只修改路径中的“文件名”部分，保留前面的“目录路径”不变，然后返回一个新的 Path 对象。
        new_file_path = md_path_obj.with_name(f'{md_path_obj.stem}_new{md_path_obj.suffix}')

        # 创建新md文件并写入内容
        try:
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(new_md_content)
            self.logger.info(f'文件已备份至：{new_file_path}')
        except IOError as e:
            self.logger.error(f'写入新文件失败 {new_file_path}：{e}')
            raise FileProcessingError(f'文件备份写入失败：{e}', node_name='me_img_node')

        return str(new_file_path)


class _ImgScanner:
    """
    根据图片目录，得到该目录下有效的图片文件
    到md文件中定位图片的位置
    获取该图片在md中的上下文内容
    最终组装所有图片的上下文内容List
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, img_content_length: int, image_extensions: Set[str]) \
            -> List[ImageInfo]:
        """
        扫描图片目录，遍历每个图片文件，去md文件中获取到上下文信息（ImageContext），封装所有图片的完整信息列表（ImageInfo）

        上文信息：标题加上文内容
        下文信息：下文内容

        Args:
            img_dir_obj: 图片目录路径
            md_content: md文件内容
            img_content_length: 图片上下文长度
            image_extensions: 图片文件扩展名

        Returns:
            List[ImageInfo]: 所有图片的完整信息列表

        """

        img_info_list: List[ImageInfo] = []

        # 遍历图片目录下的每个文件
        for img_path in img_dir_obj.iterdir():
            # 过滤子目录
            if img_path.is_dir():
                self.logger.info(f'过滤子目录: {img_path}')
                continue

            # 过滤非图片文件
            if img_path.suffix not in image_extensions:
                self.logger.info(f'过滤非图片文件: {img_path}')
                continue

            # 获取图片上下文
            ctx = self._get_img_context(img_path.name, md_content, img_content_length)

            if not ctx:
                self.logger.info(f'过滤图片上下文为空: {img_path}')
                continue

            # 封装ImageInfo
            img_info_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                image_context=ctx
            ))

        self.logger.info(f'扫描到{len(img_info_list)}张有效图片')

        return img_info_list

    def _get_img_context(self, img_name: str, md_content: str, img_content_length: int) -> Optional[ImageContext]:
        """
        根据图片名字，从md文件中获取到图片的上下文信息（ImageContext）

        Args:
            img_name: 图片名字
            md_content: md文件内容
            img_content_length: 图片上下文长度

        Returns:
            ImageContext: 图片上下文信息

        """

        # 图片格式正则
        # ![](images\xxx.jpg "图片描述")
        # . 任意字符
        # * 任意数量，0次或无数次
        # ? 单独使用时，表示前面的字符出现 0 次或 1 次
        # 但 *? 是 非贪婪匹配，表示前面的字符出现 0 次或多次，但尽可能少匹配
        # 所以 .*? 应理解为“任意字符，0 次或多次，非贪婪
        # escape (a.png) 转义特殊字符
        # 例如：a.png 转义为 a\.png
        pattern = re.compile(r'!\[.*?\]\(.*?' + re.escape(img_name) + r'.*?\)')

        # 按行切割md_content
        md_lines = md_content.split('\n')
        for md_idx, md_line in enumerate(md_lines):
            # 当前行不是图片行，跳过
            if not pattern.search(md_line):
                continue

            # 上文标题索引作为起始索引：标题加上文内容，找到最近的一级标题位置
            head, prev_index = self._find_heading_up(md_lines, md_idx)
            # 获取上文标题的下一行到当前行的所有行
            pre_lines = md_lines[prev_index + 1:md_idx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction='front')

            # 下文标题索引作为结束索引
            next_index = self._find_heading_down(md_lines, md_idx)
            next_lines = md_lines[md_idx + 1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length, direction='back')

            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context,
            )

        return None

    def _find_heading_up(self, md_lines: List[str], md_idx: int) -> Tuple[str, int]:
        """
        找到最近的一级标题位置，作为上文标题索引
        上文标题：标题加上文内容

        Args:
            md_lines: md文件内容按行切割后的列表
            md_idx: 当前行索引

        Returns:
            Tuple[str, int]: 上文标题内容和上文标题索引
        """
        for i in range(md_idx - 1, -1, -1):
            # 匹配句首如果是1-6个#加一个或多个空白字符则为标题
            if re.match(r'^#{1,6}\s+', md_lines[i]):
                return md_lines[i], i

        return '', -1

    def _find_heading_down(self, md_lines: List[str], md_idx: int) -> int:
        """
        找到最近的一级标题位置，作为下文标题索引
        下文标题：下文内容

        Args:
            md_lines: md文件内容按行切割后的列表
            md_idx: 当前行索引

        Returns:
            int: 下文标题索引
        """
        for i in range(md_idx + 1, len(md_lines)):
            if re.match(r'^#{1, 6}\s+', md_lines[i]):
                return i

        return len(md_lines)

    def _extract_limited_context(self, lines: List[str], img_content_length: int, direction: str) -> str:
        """
        从lines中提取有限上下文长度的文本
        采用段落方式截取，最后根据段落的字符数是否达到上下文长度选择留取
        段落定义：按\n切分
            切分后碰到了空字符串，是一个段落
            如果碰到了图片行，是一个段落

        Args:
            lines: 上文内容列表
            img_content_length: 图片上下文长度
            direction: 方向：front或back

        Returns:
            str: 上下文内容

        """

        paragraphs = []
        current_paragraph = []

        for line in lines:
            # 是空白行
            is_blank_line = not line.strip()
            # 是图片行
            is_img_line = re.match(r'!\[.*?\]\(.*?\)$', line.strip())
            # 如果当前行是段落的最后一行
            if is_blank_line or is_img_line:
                if current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        # 处理最后一个剩余段落
        if current_paragraph:
            paragraphs.append('\n'.join(current_paragraph))

        # 根据上下文方向反转段落，如果是上文就反转实现从下往上
        if direction == 'front':
            paragraphs.reverse()

        # 判断长度，选择最后留下哪些段落
        total = 0
        selected_paragraphs = []
        for paragraph in paragraphs:
            # 如果当前段落使总长度超过上下文长度，跳出循环
            if total + len(paragraph) > img_content_length and selected_paragraphs:
                break
            selected_paragraphs.append(paragraph)
            total += len(paragraph)

        # 还原反转
        if direction == 'front':
            selected_paragraphs.reverse()

        return ''.join(selected_paragraphs)


class _VLMSummarizer:
    """
    根据每个图片信息以及上下文信息，生成对应图片的摘要信息
    """

    def __init__(self, logger: Logger, max_requests: int):
        self.logger = logger
        self.max_requests = max_requests

    def summary_all(self, document_name: str, img_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        根据每个图片信息以及上下文信息，生成对应图片的摘要信息

        Args:
            document_name: 文档名称
            img_info_list: 图片信息列表
            vl_model: 模型名称

        Returns:
            Dict[str, str]: 图片摘要字典，key为图片名称，value为摘要
        """

        summaries = {}
        # 获取VLM客户端
        try:
            vlm_client = AIClients.get_openai_vlm_client()
        except Exception as e:
            for img_info in img_info_list:
                summaries[img_info.name] = '暂无摘要'
            return summaries

        # 调用VLM为每一张图片生成摘要
        #
        request_timestamps: Deque[float] = deque()
        for img_info in img_info_list:
            # 滑动窗口算法限流
            self._enforce_rate_limit(request_timestamps, self.max_requests)
            summaries[img_info.name] = self._summary_one(document_name, img_info, vlm_client, vl_model)
        self.logger.info(f'生成{len(summaries)}图片摘要')

        return summaries

    def _summary_one(self, document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        根据图片标题名已经图片信息，生成对应图片的摘要信息

        Args:
            document_name: 图片标题名
            img_info: 图片信息
            vlm_client: VLM客户端
            vl_model: 模型名称

        Returns:
            str: 图片摘要
        """

        # 构造VLM需要的上下文：标题名，上文内容，下文内容
        parts = [
            p
            for p in (
                img_info.image_context.head,
                img_info.image_context.pre_text,
                img_info.image_context.post_text
            ) if p
        ]

        # 构建最终上下文
        final_context = '\n'.join(parts) if parts else '暂无上下文'

        # 根据图片地址获取图片内容（二进制字节流）---文本协议认识（base64编码）--->解码(utf-8)--->字符串---根据字符串解码还原图片内容
        try:
            with open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f'读取图片文件内容失败 {img_info.path}: {e}')
            return '暂无图片描述'

        # 用VLM客户端调用VLM模型
        try:
            response = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：\"{final_context}\"\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # data url
                                # data:image指定数据为jpeg格式的图片
                                # base64,{}  指定编码方式为base64
                                # 这样就可以把图片转换为字符串放进json中传输
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        }
                    ],
                },
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f'图片摘要生成失败 {img_info.path}: {e}')
            return '暂无图片描述'

    def _enforce_rate_limit(self, request_timestamps: Deque[float], max_requests: int, window: int = 60):
        """
        滑动窗口限流器（Sliding Window Rate Limiter）

        基于双端队列实现滑动窗口算法，控制单位时间内的请求频率。
        当请求超过限制时，会阻塞等待直到最老的请求滑出窗口。

        Args:
            request_timestamps (Deque[float]): 请求时间戳队列，存储最近请求的时间戳（秒级浮点数）。
                要求按时间顺序排列，最老的请求在队列头部（left）。
            max_requests (int): 窗口内允许的最大请求数。若 <= 0 则直接放行。
            window (int, optional): 滑动窗口的时间长度，单位秒。默认为 60 秒。

        Returns:
            None: 该方法无返回值。超限时会阻塞当前线程。

        """
        # 如果窗口内允许的最大请求数 <= 0 直接放行
        if max_requests <= 0:
            return

        # 获取当前时间
        now = time.time()

        # 清除所有已完成并且时间也满足window的请求
        while request_timestamps and now - request_timestamps[0] >= window:
            request_timestamps.popleft()

        # 如果队列已经满了
        if len(request_timestamps) >= max_requests:
            # 那就阻塞等待
            sleep_time = window - (now - request_timestamps[0])
            if sleep_time > 0:
                self.logger.info(f'达到限流器速率限制（每{window}秒{max_requests}次），暂停{sleep_time:.2f}秒')
                time.sleep(sleep_time)

            # 阻塞结束后获取新时间
            now = time.time()
            # 清除所有过期的请求
            while request_timestamps and now - request_timestamps[0] >= window:
                request_timestamps.popleft()

        # 当前请求时间入队
        request_timestamps.append(now)


class _ImageUploader:
    """
    将本地图片上传到minio，得到该图片在MinIO中可访问的URL
    替换md中的摘要和图片地址
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self, object_dir_name: str, md_content: str, img_info_list: List[ImageInfo],
                           summaries: Dict[str, str], minio_url: str, minio_bucket_name: str) -> str:
        """
        上传图片到MinIO并更新md中的图片地址以及摘要

        Args:
            object_dir_name: minio对象存储目录名称，用于构建MinIO中的存储路径（如文档名）
            md_content: 原始Markdown文件内容
            img_info_list: 图片信息列表，包含每张图片的本地路径、名称和上下文信息
            summaries: 图片摘要字典，key为图片名称，value为VLM生成的标题/描述
            minio_url: MinIO服务地址，用于拼接完整的图片访问URL
            minio_bucket_name: MinIO存储桶名称，指定图片上传的目标桶

        Returns:
            str: 更新后的Markdown内容字符串，图片地址已替换为MinIO URL，并插入了摘要信息

        """

        # 上传图片到MinIO
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_url, minio_bucket_name)

        # 更新
        md_content = self._updata_md_content(md_content, summaries, remote_urls)

        return md_content

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_url: str,
                    minio_bucket_name: str) -> Dict[str, str]:

        remote_urls = {}  # key: 图片的名字  value: 图片的远程url

        # 获取MinIO客户端
        try:
            minio_client: Minio = StorageClients().get_minio_client()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls

        # 遍历上传图片
        for img_info in img_info_list:
            object_name = f'{object_dir_name}/{img_info.name}'
            try:
                minio_client.fput_object(minio_bucket_name, object_name, img_info.path)
                self.logger.info(f'上传图片到MinIO成功：{img_info.name}')
                # 图片路径为  http://localhost:9000/桶名/对象名
                remote_urls[img_info.name] = f'{minio_url}/{minio_bucket_name}/{object_name}'
            except Exception as e:
                self.logger.error(f'上传图片到MinIO失败：{img_info.name}')
                remote_urls[img_info.name] = img_info.path

        self.logger.info(f'获取到远程的图片地址：{len(remote_urls)}个')
        return remote_urls

    def _updata_md_content(self, md_content: str, summaries: Dict[str, str], remote_urls: Dict[str, str]) -> str:
        """
        更新Markdown中的图片描述和图片URL为MinIO远程地址

        Args:
            md_content: 原始Markdown文件内容
            summaries: 图片摘要字典，key为图片名称，value为VLM生成的摘要
            remote_urls: 远程URL字典，key为图片名称，value为MinIO中的访问URL

        Returns:
            str: 更新后的Markdown内容，图片路径已替换为MinIO URL，并包含摘要描述
        """

        # 用正则找图片：![摘要](远程图片URL)
        # 捕获组：group(0) 所有匹配的内容
        # group(1) 第一个匹配的内容，也就是摘要
        # group(2) 第二个匹配的内容，也就是URL
        pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

        def replacer(match: re.Match) -> str:
            """
                ![摘要](MinIO远程图片地址)
            """

            for img_name, img_summary in summaries.items():
                # md中的图片URL
                origin_img_path = match.group(2)
                # 如果summaries中的img_name对应匹配到的图片
                img_name_in_md = Path(origin_img_path).name
                if img_name == img_name_in_md:
                    return f'![{img_summary}]({remote_urls[img_name]})'
            return match.group(0)
        return pattern.sub(replacer, md_content)


class MDToImgNode(BaseNode):
    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(self.logger, self.name)
        self._img_scanner = _ImgScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.max_requests)
        self._image_uploader = _ImageUploader(self.logger)

    name = 'md_to_img_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑

        Args:
            state:

        Returns:

        """

        config = self.config

        # 读取md内容、路径以及图片的目录
        self.log_step('step1', '读取md内容、路径以及图片的目录')
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        # md中没有图片则直接返回md_content
        if not img_dir_obj.exists():
            state['md_content'] = md_content
            return state

        # 根据图片目录和md内容，得到所有图片的上下文信息
        self.log_step('step2', '扫描图片目录，得到所有图片的上下文信息')
        img_info_list: List[ImageInfo] = self._img_scanner.scan_imgs_dir(
            img_dir_obj,
            md_content,
            config.img_content_length,  # 图片上下文最大长度
            config.image_extensions  # 图片文件后缀
        )

        # 根据图片生成摘要
        # 图片摘要字典，key为图片名称，value为摘要
        # path的几个参数
        # print(p.name)  # 输出: 风景照.jpg  (包含后缀)
        # print(p.stem)  # 输出: 风景照     (不包含后缀)
        # print(p.suffix)  # 输出: .jpg       (单独拿后缀)
        self.log_step('step3', '根据图片路径和上下文信息通过大模型生成图片摘要')
        summaries: Dict[str, str] = self._vlm_summarizer.summary_all(md_path_obj.stem, img_info_list, config.vl_model)

        # 将图片上传到MinIO并更新md中的图片地址以及摘要
        self.log_step('step4', '将图片上传到MinIO并更新md中的图片地址以及摘要')
        new_md_content = self._image_uploader.upload_and_replace(
            md_path_obj.stem,
            md_content,
            img_info_list,
            summaries,
            config.get_minio_base_url(),
            config.minio_bucket
        )

        # 备份md文件用于测试
        self.log_step('step5', '将md文件备份一份用于测试')
        self._md_file_handler.backup(md_path_obj, new_md_content)

        state['md_content'] = new_md_content

        return state


if __name__ == '__main__':
    setup_logging()
    md_to_img_node = MDToImgNode()
    init_state: ImportGraphState = {
        'md_path': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test\hybrid_auto\test.md'
    }
    md_to_img_node.process(init_state)
