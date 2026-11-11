import json
import subprocess
import time
from pathlib import Path

import psutil

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import FileProcessingError, ValidationError, \
    PdfConversionError
from knowledge.processor.import_processor.state import ImportGraphState

# mineru本地API服务的进程特征（cli以 `python -m mineru.cli.fast_api` 拉起子进程）
_MINERU_SERVER_CMD_MARKER = "mineru.cli.fast_api"


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

    def _cleanup_orphan_mineru_servers(self):
        """
        清理遗留的mineru本地API服务进程，防止内存被孤儿进程逐渐吃光

        背景：mineru CLI每次解析都会拉起一个 `python -m mineru.cli.fast_api` 子进程，
        正常路径会随CLI退出而关停，但失败/异常路径可能遗留孤儿进程（父进程已死、
        子进程仍持有数GB内存）。本机内存紧张时，遗留进程会导致下一次解析
        在分配内存时直接OOM（numpy Unable to allocate）。
        仅清理"父进程已退出"的孤儿，不会误杀并行导入时正在使用的服务。
        """
        killed = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
            try:
                info = proc.info
                # 按参数列表精确匹配：服务进程的cmdline形如
                # [python.exe, '-m', 'mineru.cli.fast_api', '--port', ...]
                # 命令行文本里"提到"该字符串的无关进程不会被匹配
                if _MINERU_SERVER_CMD_MARKER not in (info['cmdline'] or []):
                    continue
                # 父进程（mineru CLI）仍存活 → 是并行导入正在使用的服务，跳过
                if info['ppid'] and psutil.pid_exists(info['ppid']):
                    continue
                psutil.Process(info['pid']).kill()
                killed.append(info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if killed:
            self.logger.warning(f'已清理遗留的MinerU本地服务进程：{killed}')

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

        # 先清理上次可能遗留的孤儿服务进程，释放内存
        self._cleanup_orphan_mineru_servers()

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
        # 可选：低显存机器通过 MINERU_BACKEND 切换解析后端（如 vlm-auto-engine / pipeline）
        if self.config.mineru_backend:
            cmd += ['-b', self.config.mineru_backend]

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

        # 无论成功失败，清理本次可能遗留的孤儿服务进程
        self._cleanup_orphan_mineru_servers()

        if processed_result == 0:
            self.logger.info(f'MinerU解析PDF成功，耗时：{end_time - start_time:.2f}秒')
        else:
            self.logger.error(f'MinerU解析PDF失败')

        return processed_result

    def _get_md_paths(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        获取转换结果路径

        MinerU 输出目录结构（二级子目录名取决于解析后端，不写死）:
        file_dir_path/
          └── 文件名/
               ├── hybrid_auto/          （默认后端）
               ├── vlm_auto_engine/      （vlm-auto-engine 后端）
               └── pipeline/             （pipeline 后端）
                    ├── 文件名.md
                    └── images/

        Args:
            import_file_path_obj: 解析文件的path路径
            file_dir_obj: 解析文件输出目录

        Returns:
            str: 解析后的markdown文件路径

        Raises:
            PdfConversionError: 未找到MinerU输出的markdown文件
        """

        file_name = import_file_path_obj.stem
        file_root = file_dir_obj / file_name

        # 在所有后端子目录中查找markdown产物
        candidates = sorted(file_root.glob(f"*/{file_name}.md"))

        # 多个命中时优先默认后端 hybrid_auto
        for candidate in candidates:
            if candidate.parent.name == "hybrid_auto":
                return str(candidate)

        if candidates:
            return str(candidates[0])

        raise PdfConversionError(
            message=f"未找到MinerU输出的markdown文件：{file_root}", node_name=self.name
        )


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

