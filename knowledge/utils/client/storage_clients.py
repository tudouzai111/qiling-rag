import logging
import threading

from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger(__name__)


from dotenv import load_dotenv
from minio import Minio
from pymilvus import MilvusClient

from knowledge.utils.client.base import BaseClientManager

load_dotenv()


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus、MongoDB"""

    _minio_client: Minio | None = None
    _minio_lock = threading.Lock()

    _milvus_client: MilvusClient | None = None
    _milvus_lock = threading.Lock()

    _mongo_client: Database | None = None
    _mongo_lock = threading.Lock()


    # ── MinIO ──

    @classmethod
    def get_minio_client(cls) -> Minio:
        return cls._get_or_create("_minio_client", cls._minio_lock, cls._create_minio_client)

    @classmethod
    def _create_minio_client(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' 已存在")

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except OSError:
            raise
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e

    # ── Milvus ──

    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus_client)

    @classmethod
    def _create_milvus_client(cls) -> MilvusClient:
        try:
            url = cls._require_env('MILVUS_URL')

            client = MilvusClient(url=url)

            return client

        except OSError:
            raise
        except Exception as e:
            logger.error(f"MilvusClient 客户端创建失败: {e}")
            raise ConnectionError(f"MilvusClient 连接失败: {e}") from e

    # ── MongoDB ──

    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create("_mongo_db", cls._mongo_lock, cls._create_mongo_db)

    @classmethod
    def _create_mongo_db(cls) -> Database:
        try:
            mongo_url = cls._require_env("MONGO_URL")
            db_name = cls._require_env("MONGO_DB_NAME")

            # 1. 实例化客户端
            client = MongoClient(mongo_url)

            # 2. 根据客户端获取数据库对象
            db = client[db_name]

            logger.info(f"MongoDB 客户端初始化成功 (db={db_name})")

            # 3. 返回数据库对象
            return db
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MongoDB 客户端创建失败: {e}")
            raise ConnectionError(f"MongoDB 连接失败: {e}") from e


if __name__ == '__main__':
    print(StorageClients.get_milvus_client())