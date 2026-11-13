import os.path

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_upload_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.service.upload_service import UploadService
from knowledge.schema.upload_schema import TaskStatusResponse, UploadResponse
from knowledge.utils.task_util import get_task_info


def create_app():
    app = FastAPI(description="掌柜智库导入的应用", version="v1.0")

    # 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # 为True时，其他三个不能为*
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件 import.html
    page_dir = get_front_page_dir()
    if page_dir and os.path.exists(page_dir):
        # StaticFiles: 对外提供 page_dir 目录下的静态资源，访问入口是 /front/...
        app.mount("/front", StaticFiles(directory=page_dir))
    register_router(app)
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def hello_world():
        return {"flag": "success"}

    # 上传
    @app.post("/upload", response_model=UploadResponse)
    def upload_endpoint(
        file: UploadFile,
        background_tasks: BackgroundTasks,
        upload_service: UploadService = Depends(get_upload_file_service),
    ):
        """处理文件的上传"""

        # 将上传的文件写入本地目录临时存储并上传到MinIO持久化
        task_id, import_file_path, file_dir = upload_service.process_upload_file(file)
        # 运行整个导入的图谱
        background_tasks.add_task(
            upload_service.run_import_graph, task_id, import_file_path, file_dir
        )
        return UploadResponse(message=f"{file.filename}文件上传成功", task_id=task_id)

    # 查询上传任务的状态
    @app.get("/status/{task_id}")
    def get_task_status_endpoint(task_id: str):
        """查询上传任务的状态，1.5秒轮询一次"""

        task_info = get_task_info(task_id)

        return TaskStatusResponse(**task_info)


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="localhost", port=8000, log_level="info")
