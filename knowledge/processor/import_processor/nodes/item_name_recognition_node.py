import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType, MilvusClient

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import (
    LLMError,
    StateFieldError,
    ValidationError,
)
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.prompts.import_prompt import (
    ITEM_NAME_SYSTEM_PROMPT,
    ITEM_NAME_USER_PROMPT_TEMPLATE,
)
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        1. 构建商品名识别的LLM上下文
        2. 调用LLM模型进行商品名识别
        3. 将LLM提取到的商品名进行嵌入
        4. 存储到Milvus（MYSQL模糊查询不能考虑语义）
        """

        # 参数校验
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = (
            self._validate_state(state)
        )

        # 构建上下文
        item_name_context = self._prepare_llm_context(chunks, item_name_chunk_k)

        # 调用 LLM 模型，提取商品名
        item_name = self._recognition_item_name(item_name_context, file_title)

        # 向量化
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)

        # 回填：更新 LLM 提取的 item_name
        self._fill_item_name(state, chunks, item_name)

        return state

    def _validate_state(self, state: ImportGraphState) -> tuple[str, list, int, int]:
        # 获取文档标题：商品具体型号兜底
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(
                node_name=self.name, field_name="file_title", expected_type=str
            )

        # 获取 chunks 给 LLM 作为上下文信息
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name, field_name="chunks", expected_type=list
            )

        # 获取 item_name_chunk_k(3) item_name_chunk_size(2500)
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="商品名识别的辅助切片数不合法")
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="商品名识别的辅助切片长度不合法")

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_llm_context(self, chunks: list[dict], item_name_chunk_k: int) -> str:
        """
        准备商品名识别的上下文

        Args:
            chunks: 文档所有切分块列表
            item_name_chunk_k: 准备使用的切分块数

        Returns:
            上下文信息
        """
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue

            # 获取 chunk 的 content
            content = chunk.get("content")
            splice_context = f"【切片】- {index} - {content}"
            final_context.append(splice_context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_name_context: str, file_title: str) -> str:
        # 调用 LLM 客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_openai_llm_client(
                response_format=False
            )
        except ConnectionError as e:
            self.logger.error(
                f"OpenAI 的 LLM 客户端创建失败，降级使用文件标题 {file_title} 作为商品名：{e!s}"
            )
            return file_title

        # 调用 LLM 模型
        # 获取商品名识别的 LLM系统提示词
        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        # 获取商品名识别的 LLM用户提示词
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
            file_title=file_title, context=item_name_context
        )

        # 调用
        try:
            llm_response = llm_client.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

            llm_result = llm_response.content.strip("")
            if not llm_result or llm_result == "UNKNOWN":
                self.logger.error(
                    f"LLM 提取商品名失败，降级使用文件标题 {file_title} 作为商品名兜底"
                )
                return file_title

            self.logger.info(f"LLM 为文档 {file_title} 提取的商品名：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(
                f"LLM 提取商品名失败，降级使用文件标题 {file_title} 作为商品名：{e!s}"
            )
            raise LLMError(message=str(e))

    def _embedding_item_name(
        self, item_name: str
    ) -> tuple[list | None, dict[str, Any] | None]:
        """
        商品名嵌入

        Args:
            item_name: 商品名

        Returns:
            稠密向量，稀疏向量
        """

        # 获取嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE M3 模型客户端创建失败：{e}")
            return None, None

        # 计算稠密向量和稀疏向量
        vector_result = bge_m3_client.encode_documents(documents=[item_name])
        # 输出示例
        # 稠密向量
        # [-0.0019397735595703125, 0.00955963134765625, -0.03594970703125]
        # 稀疏向量
        # < Compressed Sparse Row sparse array of dtype 'float64' with 7 stored elements and shape (2, 250002) >
        # Coords       Values
        # (0, 6)       0.1510009765625
        # (0, 28309)   0.30078125
        try:
            # 获取向量
            # 稠密向量
            dense_vector = vector_result.get("dense")[0].tolist()

            # 稀疏向量矩阵csr
            sparse_csr = vector_result.get("sparse")
            # 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 获取token_id
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            # 获取weight
            weight = sparse_csr.data[start_index:end_index].tolist()
            # 构建字典结构
            sparse_vector = dict(zip(token_id, weight))

            self.logger.info(f"计算出来的稠密向量的维度 {len(dense_vector)}")

            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE M3 嵌入模型计算 {item_name} 向量失败：{e!s}")
            return None, None

    def _insert_milvus(
        self,
        dense_vector: list,
        sparse_vector: dict[str, Any],
        file_title: str,
        item_name: str,
    ):
        """
        将LLM识别到的商品名保存到Milvus数据库中
        {
            "dense_vector": "",
            "sparse_vector": "",
            "file_title": "",
            "item_name": ""
        }

        Args:
            dense_vector: 商品名稠密向量
            sparse_vector: 商品名稀疏向量
            file_title: 文档名
            item_name: 商品名

        Returns:

        """

        # 判断稠密向量和稀疏向量是否都存在
        if not dense_vector or not sparse_vector:
            return

        # 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus 客户端创建失败：{e!s}")
            return

        # Milvus 三大核心概念 集合：Collection[集合名 字段约束：schema 索引]
        # 集合名：表名类似于归纳数据的容器，逻辑概念
        # 约束：类似于MySQL字段的长度、字段的类型
        # 索引：类似于MySQL中的索引【索引的类型比较多 B+树 Hash】Milvus的索引类型也很多
        # 索引的本质就是算法（图、数、Hash）目的：提高检索查询效率，Milvus中不管是稠密向量还是稀疏向量的索引都是为了能够快速找到和问题相似的向量
        # 使用Milvus的规范：1、创建集合（约束、索引）2、插入数据 3、查询/检索
        try:
            # 创建Milvus集合（幂等性校验）
            item_name_collection_name = self.config.item_name_collection
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(
                    milvus_client, item_name_collection_name
                )

            # 先删除该文档的旧数据
            milvus_client.delete(
                collection_name=item_name_collection_name,
                filter=f"file_title == '{file_title}'",
            )

            # 构建数据行
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }

            # 插入数据
            inserted_result = milvus_client.insert(
                collection_name=item_name_collection_name, data=[item_name_data_row]
            )
            self.logger.info(
                f"插入的结果：{inserted_result}, 主键值：{inserted_result.get('ids')}"
            )
        except Exception as e:
            self.logger.error(f"商品名 {item_name} 插入失败：{e!s}")

    def _create_item_name_collection(
        self, milvus_client: MilvusClient, item_name_collection_name: str
    ):
        """
        创建商品名集合

        Args:
            milvus_client: 集合的名字
            item_name_collection_name: milvus客户端

        Returns:

        """

        # 创建schema约束
        schema = milvus_client.create_schema()
        # 创建主键字段约束
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            max_length=10,
            is_primary=True,
            auto_id=True,
        )
        # 创建标量字段约束
        # {
        #     "dense_vector": "",
        #     "sparse_vector": "",
        #     "file_title": "",
        #     "item_name": ""
        # }
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=65535
        )
        # 创建向量字段约束
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 创建索引
        index_params = milvus_client.prepare_index_params()
        # 为向量字段建立索引
        # 稠密向量
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            # Milvus 计算出来的稠密向量已经进行了归一化处理，COSINE和IP对稠密向量的排序结果是等价的，但显式使用COSINE更安全，换了不归一化的模型也不用改参数
        )
        # 稀疏向量
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP",  # 只有 IP 和 BM25
        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=item_name_collection_name,
            schema=schema,
            index_params=index_params,
        )

        self.logger.info(f"创建 {item_name_collection_name} 集合成功")

    def _fill_item_name(
        self, state: ImportGraphState, chunks: list[dict], item_name: str
    ):
        """
        回填 item_name
        1. 回填给 chunk 方便下游模型使用
        2. 回填给 state 方便其他节点使用

        Args:
            state:
            item_name:

        Returns:

        """

        for chunk in chunks:
            chunk["item_name"] = item_name

        state["item_name"] = item_name


if __name__ == "__main__":
    setup_logging()

    chunk_json_path = r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\chunks.json"
    chunk_item_name_json_path = r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\chunks_item_name.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

        state = {"file_title": "H3C 无线网关手册", "chunks": chunk_content}

        node = ItemNameRecognitionNode()

        final_state = node.process(state)

        print(f"商品名：{final_state.get('item_name')}")
        print(f"chunks数量：{len(final_state.get('chunks', []))}")
        print(f"首个chunk是否含item_name：{'item_name' in final_state['chunks'][0]}")

        # # 验证：把刚插进去的行查出来
        # milvus_client = StorageClients.get_milvus_client()
        # # 先加载集合到内存（query/search 的前置条件）
        # milvus_client.load_collection(collection_name="kb_item_names")
        # # insert 到可查询有短暂延迟，等一下保证刚插的行可见
        # import time
        #
        # time.sleep(3)
        #
        # # 用标量过滤查（注意：Milvus filter 里字符串值必须用【单引号】，双引号会匹配不到）
        # filtered_rows = milvus_client.query(
        #     collection_name="kb_item_names",
        #     filter="file_title == 'H3C 无线网关手册'",
        #     output_fields=["file_title", "item_name", "pk"],
        #     limit=100,
        # )
        # print(f"filter 查到 {len(filtered_rows)} 行：")
        # for row in filtered_rows:
        #     print(repr(row))
        #
        # # 对照：不加过滤的全表，用于确认总数
        # all_rows = milvus_client.query(
        #     collection_name="kb_item_names",
        #     filter="",
        #     output_fields=["file_title", "item_name", "pk"],
        #     limit=100,
        # )
        # print(f"全表共 {len(all_rows)} 行（重复插入会累积，用于观察幂等问题）：")
        # for row in all_rows:
        #     print(repr(row))

    with open(chunk_item_name_json_path, "w", encoding="utf-8") as f:
        json.dump(chunk_content, f, ensure_ascii=False, indent=4)
