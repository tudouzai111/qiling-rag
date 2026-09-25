import json
import subprocess
import time
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import FileProcessingError, ValidationError, \
    PdfConversionError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = 'pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口：将 PDF 文件转换为 Markdown 文件

        Args:
            state: 导入流程图状态

        Returns:

        """

        # 接收pdf的路径path，用mineru将pdf转换为markdown文件
        import_file_path_obj, file_dir_obj = self._validate_state_inputs_path(state)

        # 执行mineru解析
        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError('MinerU解析PDF失败', self.name)

        # 获取解析后的markdown文件路径
        md_path = self._get_md_paths(import_file_path_obj, file_dir_obj)

        state['md_path'] = md_path

        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> tuple[Path, Path]:
        """
        验证导入流程图状态中的路径信息是否有效

        Args:
            state: 导入流程图状态

        Returns:
            tuple[Path, Path]: 导入文件路径对象和文件目录对象

        """

        self.log_step("step1", "验证导入流程图状态中的路径信息是否有效")

        # 获取输入的pdf文件path
        import_file_path = state.get('import_file_path', '')

        # 获取解析后的输出目录
        file_dir = state.get('file_dir', '')

        # 判断是否为空
        if not import_file_path:
            raise ValidationError('解析的文件不存在', self.name)

        # 标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)

        # 判断是否是一个真实的路径
        if not import_file_path_obj.exists():
            raise FileProcessingError('解析的文件路径不存在', self.name)

        # 判断输出目录是否为空
        if not file_dir:
            # 默认为输入文件所在目录
            file_dir = import_file_path_obj.parent

        # 标准化输出目录
        file_dir_obj = Path(file_dir)
        self.logger.info(f'上传文件的路径：{import_file_path}')
        self.logger.info(f'输出目录：{file_dir}')

        # 返回输出文件以及输出目录的标准path
        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        """
        执行mineru解析

        Args:
            import_file_path_obj: 解析文件的path路径
            file_dir_obj: 解析文件输出目录

        Returns:
            状态：0 或 非0
            0: 解析成功
            非0: 解析失败
        """

        # mineru解析命令：mineru -p input_path -o output_dir --source=local

        # 子进程执行cmd命令
        # 定义cmd
        cmd = [
            'mineru',
            '-p',
            str(import_file_path_obj),
            '-o',
            str(file_dir_obj),
            '--source', 'local',
        ]

        # 执行cmd命令并获取所有日志
        start_time = time.time()
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,  # 正确日志
            stderr=subprocess.STDOUT,  # 错误日志
            text=True,  # 输出文本字符串
            encoding='utf-8',
            errors='replace',  # 替换异常字符
            bufsize=1  # 实时输出，按行输出
        )

        # 实时打印日志
        for line in proc.stdout:
            self.logger.info(f'MinerU解析产生的日志：{line}')

        # 主线程等待子进程执行完成
        processed_result = proc.wait()
        end_time = time.time()

        if processed_result == 0:
            self.logger.info(f'MinerU解析PDF成功，耗时：{end_time - start_time:.2f}秒')
        else:
            self.logger.error(f'MinerU解析PDF失败')

        return processed_result

    def _get_md_paths(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        获取转换结果路径

        MinerU 输出目录结构:
        file_dir_path/
          └── 文件名/
               └── hybrid_auto/
                    ├── 文件名.md
                    └── images/

        Args:
            import_file_path_obj: 解析文件的path路径
            file_dir_obj: 解析文件输出目录

        Returns:
            str: 解析后的markdown文件路径

        """

        file_name = import_file_path_obj.stem

        return str(file_dir_obj / file_name / 'hybrid_auto' / f'{file_name}.md')


# 测试
if __name__ == '__main__':
    setup_logging()
    # 构建节点实例
    pdf_to_md_node = PdfToMdNode()

    # 构建节点状态
    init_state = {
        'import_file_path': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test.pdf',
        'file_dir': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir',
    }

    # 调用process
    result = pdf_to_md_node.process(init_state)

    # 序列化
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)

