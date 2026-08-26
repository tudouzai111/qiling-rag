from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name = 'entry_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        根据上传的文件后缀，修改 state 中的 is_md_read_enabled 和 is_pdf_read_enabled

        Args:
            state (ImportGraphState): 当前图状态对象，包含上传文件路径及相关配置。

        Returns:
            ImportGraphState: 更新后的状态对象，读取开关字段已被设置。

        """

        # 导入文件路径
        import_file_path = state.get('import_file_path', '')
        # 导入(出)文件目录
        file_dir = state.get('file_dir', '')
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str)

        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)
        if not import_file_path_obj:
            raise StateFieldError(node_name=self.name, field_name='import_file_path_obj', expected_type=Path)
        if not file_dir_obj:
            raise StateFieldError(node_name=self.name, field_name='file_dir_obj', expected_type=Path)

        # 获取文件后缀
        if import_file_path_obj.suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = str(import_file_path_obj)
        else:
            self.logger.error(f'该文件的后缀格式不支持: {import_file_path_obj.suffix}')
            raise ValidationError(f'该文件的后缀格式不支持: {import_file_path_obj.suffix}', node_name=self.name)

        # 获取上传文件的标题，更新到 state 中
        state['file_title'] = import_file_path_obj.stem

        return state


if __name__ == '__main__':
    entry_node = EntryNode()
    init_state = {
        'import_file_path': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test.pdf',
        'file_dir': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir'
    }
    result = entry_node(init_state)
    import json

    print(json.dumps(result, ensure_ascii=False, indent=4))
