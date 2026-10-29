import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import MilvusError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class _ScalarFieldSpc:
    field_name: str
    datatype: DataType
    max_length: int | None = None


_SCALAR_FIELDS: Sequence[_ScalarFieldSpc] = (
    _ScalarFieldSpc(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _ScalarFieldSpc(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _ScalarFieldSpc(
        field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535
    ),
    _ScalarFieldSpc(
        field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
    ),
    _ScalarFieldSpc(
        field_name="item_name", datatype=DataType.VARCHAR, max_length=65535
    ),
)


class _MilvusSchemaBuilder:
    """
    MilvusSchema字段约束
    """

    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        """创建schema"""
        # 动态字段，在静态字段的基础上再添加额外的字段
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        # 添加主键字段的约束
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        # 添加向量字段的约束
        # 稠密向量
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim
        )
        # 稀疏向量
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        # 添加标量字段的约束
        # "content": "",
        # "title": "test",
        # "parent_title": "",
        # "file_title": "test",
        # "item_name": "H3C LA2608-GM/GU 室内无线网关"
        for spec in _SCALAR_FIELDS:
            kwargs: dict = {"field_name": spec.field_name, "datatype": spec.datatype}
            if spec.max_length:
                kwargs["max_length"] = spec.max_length
            schema.add_field(**kwargs)

        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build_index(milvus_client: MilvusClient):
        index = milvus_client.prepare_index_params()
        # 稠密向量
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            # Milvus 计算出来的稠密向量已经进行了归一化处理，COSINE和IP对稠密向量的排序结果是等价的，但显式使用COSINE更安全，换了不归一化的模型也不用改参数
        )
        # 稀疏向量
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP",  # 只有 IP 和 BM25
        )

        return index


class _MilvusInserter:
    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert_rows(self, data: list[dict[str, Any]]):
        inserted_res = self._milvus_client.insert(
            collection_name=self._collection_name, data=data
        )
        chunk_ids = inserted_res.get("ids")
        for chunk_id, chunk in zip(chunk_ids, data):
            chunk["chunk_id"] = chunk_id


class ImportMilvusNode(BaseNode):
    """
    角色：门面模式（充当门面）

    """

    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 校验
        validated_chunks, dim = self._validate_state(state)
        # 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus 客户端创建失败：{e}")
            raise MilvusError(
                message=f"Milvus 客户端创建失败: {e}", node_name=self.name
            )
        # 获取chunks集合
        chunks_collection = self.config.chunks_collection
        # 创建集合
        self._create_chunks_collection(chunks_collection, milvus_client, dim)
        # 插入数据
        inserter = _MilvusInserter(milvus_client, chunks_collection)
        inserter.insert_rows(validated_chunks)
        return state

    def _validate_state(
        self, state: ImportGraphState
    ) -> tuple[list[dict[str, Any]], int]:
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(
                message="待入库的chunks为空或类型无效", node_name=self.name
            )

        validated_chunks = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"chunks[{idx}] 类型无效：{type(chunk).__name__}，期望：dict",
                    node_name=self.name,
                )
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{idx}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError(
                message="所有 chunk 均无有效向量，无法入库", node_name=self.name
            )

        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")

        return validated_chunks, dim

    def _create_chunks_collection(
        self, chunks_collection: str, milvus_client: MilvusClient, dim: int
    ):
        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"{chunks_collection} 集合已存在 跳过创建集合")
            return
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
        index_params = _MilvusIndexBuilder.build_index(milvus_client)
        milvus_client.create_collection(
            collection_name=chunks_collection, schema=schema, index_params=index_params
        )


if __name__ == "__main__":
    setup_logging()

    temp_dir = Path(
        r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到input_path: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    state: ImportGraphState = {"chunks": content["chunks"]}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至：{output_path}")
