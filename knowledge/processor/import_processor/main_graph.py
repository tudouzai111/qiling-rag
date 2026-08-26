import json

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.md_to_img_node import MDToImgNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState


def import_router(state: ImportGraphState):
    """
    根据 state 中的文件是 pdf 还是 md 决定下一个到达哪个节点
    """
    if state.get('is_pdf_read_enabled'):
        return 'pdf_to_md_node'
    if state.get('is_md_read_enabled'):
        return 'md_to_md_node'
    return '__end__'


def import_graph() -> CompiledStateGraph:
    # 定义运行时图状态
    work_flow = StateGraph(state_schema=ImportGraphState)  # type:ignore

    # 添加入口节点
    work_flow.set_entry_point('entry_node')

    # 定义其他节点名和节点实例的映射表
    node_name_obj = {
        'entry_node': EntryNode(),
        'pdf_to_md_node': PdfToMdNode(),
        'md_to_img_node': MDToImgNode()
    }

    # 添加节点
    for node_name, node_obj in node_name_obj.items():
        work_flow.add_node(node_name, node_obj)

    # 添加条件边
    work_flow.add_conditional_edges('entry_node', import_router, {
        'pdf_to_md_node': 'pdf_to_md_node',  # key: 路由的返回值，value: 节点名
        'md_to_img_node': 'md_to_img_node',
        '__end__': '__end__'
    })

    # 添加业务边
    work_flow.add_edge('pdf_to_md_node', 'md_to_img_node')
    work_flow.add_edge('md_to_img_node', '__end__')

    # 编译图
    compiled_state_graph = work_flow.compile()
    return compiled_state_graph


import_app = import_graph()

# 测试
def run_import_graph():
    # 定义运行graph流程的状态
    graph_state = {
        'import_file_path': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\test.pdf',
        'file_dir': r'D:\Project\PycharmProjects\shopkeeper_brain\knowledge\processor\import_processor\temp_dir'
    }
    for event in import_app.stream(graph_state):
        final_state = {}
        for node_name, state in event.items():
            print(f'当前正在执行的节点：{node_name}')
            print(f'当前正在执行的节点状态：{state}')
            final_state = state
    return final_state


if __name__ == '__main__':
    setup_logging()
    final_state = run_import_graph()
    print(json.dumps(final_state, ensure_ascii=False, indent=4))

    # 整个执行的状态图
    print(import_app.get_graph().print_ascii())
