import asyncio
import logging
import os
from typing import Union

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import (
    QueryRequest,
    QueryResponse,
    StreamSubmitResponse,
)
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


logger = logging.getLogger(__name__)


def prewarm_local_models():
    """启动时在主线程预热本地模型（BGE-M3 / BGE-Reranker）。

    这两个模型是进程内两次 GB 级内存分配；若推迟到首个请求时才懒加载，
    遇上系统内存紧张（导入服务驻留 BGE-M3、浏览器占用高等）时，torch 原生层
    分配失败不走 Python 异常而是直接段错误（SIGSEGV，日志戛然而止、无堆栈）。
    服务启动时内存最空闲，此时一次性装好，运行期不再发生大块分配。
    """
    try:
        from knowledge.utils.client.ai_clients import AIClients

        AIClients.get_bge_m3_client()
        AIClients.get_bge_reranker_client()
        logger.info("本地模型预热完成（BGE-M3 / BGE-Reranker）")
    except Exception as e:
        # 预热失败不阻塞启动，退化为原来的懒加载行为
        logger.error(f"本地模型预热失败（首次查询时会再尝试加载）：{e}")


def create_app():
    """
    创建FastAPI实例
    Returns:
    FastAPI实例

    """

    # 1. 实例化
    app = FastAPI(description="器灵查询的应用", version="v1.0")

    # 2. 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ← 和 credentials=True 冲突
        allow_credentials=False,  # 自定义cookies Authorization  tsl客户端证书信息
        allow_methods=[
            "*"
        ],  # ← 和 credentials=True 冲突  GET(获取资源)  POST(新增)  DELETE（删除） PUT（修改）
        allow_headers=[
            "*"
        ],  # ← 和 credentials=True 冲突   自定义的头字段 token  content-type:application/json
    )

    # 3. 挂载静态文件（qiling.html / forge.html）
    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir))

    # 4. 注册路由
    register_router(app)

    # 5. 预热本地模型（主线程、内存最空闲时机）
    prewarm_local_models()

    # 6. 返回fastapi实例
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def hello_world():
        return {"flag": "success"}

    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ) -> Union[
        StreamSubmitResponse, QueryResponse
    ]:  # FastAPI:自动将json格式的字符串反序列化为指定的约束schema
        """
        处理查询请求
        Args:
            request: 前端发送的请求数据（服务端用对象接收）
            background_tasks： FastAPI提供的后台任务对象
            service： 查询业务组件对象


        Returns:

        """

        # 1. 获取session_id
        session_id = request.session_id or service.generate_session_id()

        # 2. 获取任务id
        task_id = service.generate_task_id()

        # 3. 如果是流式调用
        # 3.1 流式调用
        if request.is_stream:
            # a. sse队列创建出来(容器中任务id和队列的对应关系也有了)
            create_sse_queue(task_id=task_id)

            # b. 利用fastapi的background启动的线程执行查询任务
            background_tasks.add_task(
                service.run_query_graph,
                session_id=session_id,
                task_id=task_id,
                query=request.query,
                is_stream=request.is_stream,
            )
            return StreamSubmitResponse(
                message="查询请求已经提交", session_id=session_id, task_id=task_id
            )

        # 3.2 非流式调用(直接用当前线程运行查询流程 不启动一个新线程执行--->合理?。不合理：不是只有流式慢 非流式也慢 )
        else:
            # a. 运行查询的pineline
            # 获取当前事件循环对象(uvicorn)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                service.run_query_graph,
                session_id,
                task_id,
                request.query,
                request.is_stream,
            )
            # b. 从任务结果队列中获取答案
            answer = service.get_task_result(task_id)
            # c. 返回查询响应结果对象
            return QueryResponse(
                message="查询请求已经处理完了", session_id=session_id, answer=answer
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """
        返回sse协议要的数据包：流式+yield使用 最佳搭配
        1. 如何返回（直接返回）利用生成器yield返回(未来给的数据包通过yield返回)
        2. 返回注意事项："event:自定义\ndata:自定义\n\n"
        Args:
            task_id: 任务id
            StreamingResponse:将后端组合的sse协议格式的数据 返回给前端

        Returns:

        """
        return StreamingResponse(
            content=sse_generator(task_id, request), media_type="text/event-stream"
        )

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str,
        limit: int = 50,
        service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)
