import threading
from typing import Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_vlm_client: Optional[OpenAI] = None
    _openai_vlm_lock = threading.Lock()

    _openai_llm_client: Optional[ChatOpenAI] = None
    _openai_llm_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    @classmethod
    def get_openai_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_vlm_client", cls._openai_vlm_lock, cls._create_openai_vlm_client)

    @classmethod
    def _create_openai_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("VLM_API_KEY")
            base_url = cls._require_env("VLM_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"VLM 客户端初始化成功 (base_url={base_url})")
            return client

        # raise不带参数时，会重新抛出当前正在处理的异常
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"VLM 客户端创建失败: {e}")
            raise ConnectionError(f"VLM 连接失败: {e}") from e

    @classmethod
    def get_openai_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        return cls._get_or_create("_openai_llm_client", cls._openai_llm_lock,
                                  lambda: cls._create_openai_llm_client(response_format))

    @classmethod
    def _create_openai_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        try:
            model_name = cls._require_env("LLM_DEFAULT_MODEL")
            temperature = cls._require_env("LLM_DEFAULT_TEMPERATURE")
            api_key = cls._require_env("LLM_API_KEY")
            base_url = cls._require_env("LLM_API_BASE")

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            client = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=base_url,
                model_kwargs=model_kwargs
            )
            logger.info(f"LLM 客户端初始化成功 (base_url={base_url})")
            return client

        # raise不带参数时，会重新抛出当前正在处理的异常
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"LLM 客户端创建失败: {e}")
            raise ConnectionError(f"LLM 连接失败: {e}") from e

    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env('BGE_M3_PATH')
            device = cls._require_env('BGE_DEVICE')
            fp16 = cls._require_env('BGE_FP16')

            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16
            )

            return bge_m3_ef

        # raise不带参数时，会重新抛出当前正在处理的异常
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE M3 客户端创建失败: {e}")
            raise ConnectionError(f"BGE M3 客户端创建失败: {e}") from e


if __name__ == '__main__':
    llm_client: ChatOpenAI = AIClients.get_openai_llm_client()
    llm_res = llm_client.invoke("请您给我讲一个笑话，要求输出格式是一个json")
    llm_result = llm_res.content

    print(llm_result)
