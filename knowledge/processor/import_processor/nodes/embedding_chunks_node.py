import json
from pathlib import Path
from typing import Any

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import (
    EmbeddingError,
    StateFieldError,
    ValidationError,
)
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.log_step("step1", "校验chunks的数据结构")
        validated_chunks = self._validated_state(state)

        self.log_step("step2", "获取 BEG-M3 嵌入模型")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BEG-M3 模型创建失败：{e}")
            raise EmbeddingError(
                message=f"BEG-M3 模型创建失败：{e}", node_name=self.name
            )

        self.log_step("step3", "批量嵌入")
        # 获取批量阈值
        batch_size = self.config.embedding_batch_size
        # 获取 chunks 的总数
        total = len(validated_chunks)
        final_chunks = []
        for index in range(0, total, batch_size):
            batch_chunks = validated_chunks[index : index + batch_size]
            batch_end = index + len(batch_chunks)
            self.logger.info(f"嵌入批次 [{index + 1}-{batch_end}] / {total}")
            current_chunks = self._embed_batch_chunks(batch_chunks, embed_model)
            final_chunks.extend(current_chunks)
        state["chunks"] = final_chunks
        return state

    def _validated_state(self, state: ImportGraphState):
        chunks = state.get("chunks", "")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name, field_name="chunks", expected_type=list
            )

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"chunk[{idx}] 的类型 {type(chunk.__name__)} 不符合预期 dict",
                    node_name=self.name,
                )

        return chunks

    def _embed_batch_chunks(
        self, batch_chunks: list[dict[str, Any]], embed_model: BGEM3EmbeddingFunction
    ) -> list[dict[str, Any]]:
        """
        嵌入一批 chunks
        """

        embedding_documents = [
            f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}"
            for chunk in batch_chunks
        ]
        try:
            embed_vector = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入chunks失败：{e}", node_name=self.name)
        if not embed_vector:
            raise EmbeddingError(message="嵌入chunks结果不存在")

        sparse_csr = embed_vector.get("sparse")
        for idx, chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = embed_vector.get("dense", [])[idx].tolist()
            chunk["sparse_vector"] = self._extract_sparse_vector(sparse_csr, idx)
        return batch_chunks

    def _extract_sparse_vector(self, sparse_csr, idx: int):

        # 从行索引中获取当前chunk的起始索引和结束索引
        start_index = sparse_csr.indptr[idx]
        end_index = sparse_csr.indptr[idx + 1]
        # 获取token_id
        token_id = sparse_csr.indices[start_index:end_index].tolist()
        # 获取weight
        weight = sparse_csr.data[start_index:end_index].tolist()

        return dict(zip(token_id, weight))


if __name__ == "__main__":
    setup_logging()

    base_dir = Path(
        r"D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    )
    input_path = base_dir / "chunks_item_name.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入chunks.json文件：{input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至：\n{output_path}")
