import logging
import os.path
import shutil
import time
import uuid
from datetime import datetime

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_node_duration,
    add_running_task,
    update_task_status,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UploadService:
    """文件上传"""

    def run_import_graph(self, task_id: str, import_file_path: str, file_dir: str):
        """运行导入的图谱"""

        # 更新任务状态
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        # 定义运行graph流程的状态
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir,
        }
        try:
            for event in import_app.stream(graph_state):
                for node_name, state in event.items():
                    logger.info(f"当前正在执行的节点：{node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"[{task_id}] 执行导入过程中出现异常：{e}")
            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self, file: UploadFile):
        """文件上传处理"""

        # 任务id：8个随机数
        task_id = uuid.uuid4().hex[:8]
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        # 临时目录路径
        base_dir = self.get_base_dir()
        file_dir = os.path.join(base_dir, task_id)
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 保存文件到 MinIO 中
        self.sve_upload_file_to_minio(import_file_path, file.filename)
        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)

        return task_id, import_file_path, file_dir

    @staticmethod
    def get_base_dir():
        # %Y: 四位  %y: 两位
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    @staticmethod
    def save_upload_file_to_local(file: UploadFile, file_dir: str):
        """保存文件到临时目录"""

        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)
        try:
            with open(import_file_path, "wb") as f:
                # 分批写入文件
                shutil.copyfileobj(file.file, f)
        except OSError as e:
            logger.info(f"{file.filename} 文件写入临时目录失败：{e}")
            raise FileProcessingError(
                message=f"{file.filename} 文件写入临时目录失败：{e}"
            )
        return import_file_path

    def sve_upload_file_to_minio(self, import_file_path: str, filename: str):
        """保存文件到minio"""

        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"minio 客户端获取失败：{e}")
            return
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"{filename} 上传到 MinIO 失败：{e}")
