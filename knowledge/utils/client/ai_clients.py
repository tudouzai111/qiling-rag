import threading
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()


class AIClients(BaseClientManager):
    """AI 模型类客户端：ALIYUN(VLM)"""

    _aliyun_client: Optional[OpenAI] = None
    _aliyun_lock = threading.Lock()

    # ── ALIYUN / VLM ──

    @classmethod
    def get_aliyun(cls) -> OpenAI:
        return cls._get_or_create("_aliyun_client", cls._aliyun_lock, cls._create_aliyun)

    @classmethod
    def _create_aliyun(cls) -> OpenAI:
        try:
            api_key = cls._require_env("ALIYUN_API_KEY")
            base_url = cls._require_env("ALIYUN_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"ALIYUN 客户端初始化成功 (base_url={base_url})")
            return client

        # raise不带参数时，会重新抛出当前正在处理的异常
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ALIYUN 客户端创建失败: {e}")
            raise ConnectionError(f"ALIYUN 连接失败: {e}") from e
