import asyncio
import json
from json import JSONDecodeError
from typing import Any

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


class WebMcpSearchNode(BaseNode):
    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 执行MCP服务
        web_search_results = asyncio.run(self._execute_mcp_server(rewritten_query))
        if not web_search_results:
            return state

        state['web_search_docs'] = web_search_results

        return state

    def _validate_state(self, state: QueryGraphState) -> tuple[str, list[str]]:
        # 用户的问题(LLM重写后的)
        rewritten_query = state.get("rewritten_query")
        # 获取商品名列表
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name, field_name="rewritten_query", expected_type=str
            )
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name, field_name="item_names", expected_type=list
            )

        return rewritten_query, item_names

    async def _execute_mcp_server(self, rewritten_query: str) -> list[dict[str, Any]]:
        """
        执行MCP服务

        Args:
            rewritten_query:

        Returns:

        """
        # 定义MCP客户端（StreamableHttp）
        # api文档：https://openai.github.io/openai-agents-python/mcp/#2-streamable-http-mcp-servers
        # MCP地址：https://bailian.console.aliyun.com/cn-beijing/store/mcp/detail/WebSearch
        # 如果类里定义了__enter__和__exit__就可以用with，异步是__aenter__和__aexit__
        async with MCPServerStreamableHttp(
            name="联网搜索",  # MCP客户端名字，自定义
            params={
                "url": self.config.mcp_dashscope_base_url,
                "headers": {"Authorization": f"Bearer {self.config.aliyun_api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,  # 缓存MCP服务下的工具列表
            max_retry_attempts=3,
        ) as mcp_client:
            # 调用工具
            web_search_result = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": rewritten_query, "count": 3},
            )
            # 解析数据
            # content=[TextContent(type='text', text='{
            #     "pages": [
            #         {
            #             "snippet": "太原今日天气 (原标题:太原今日天气) 中国天气网讯 今天是11月10日星期二,一起来看天气。 今晨6时,太原晴,气温-7℃,东北风3-4级,相对湿度91%。 预计,今天白天晴,最高气温12.4℃,微风,今天夜间晴,最低气温-6.2℃,微风。 天气信息就是这么多啦~想看更多内容,请访问中国天气网移动网站:https://e.weather.com.cn!",
            #             "hostname": "网易",
            #             "hostlogo": "https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=1534926245,1016405979&fm=195&app=88&f=JPEG?w=200&h=200",
            #             "title": "太原今日天气",
            #             "url": "https://www.163.com/news/article/LBBSV27700019TLK.html",
            #         },
            #         ......
            #     ],
            #     "request_id": "6b852eae-ccc5-9835-9ab9-e288b7a62f66",
            #     "status": 0,
            # }
            text_content = web_search_result.content[0]
            if not text_content:
                return []

            text_content_text = text_content.text
            if not text_content_text:
                return []

            try:
                text_content_obj: dict[str, Any] = json.loads(text_content_text)

                pages = text_content_obj.get("pages", [])
                if not pages:
                    return []

                web_search_results = []
                for page in pages:
                    web_search_results.append(
                        {
                            "snippet": page.get("snippet", "").strip(),
                            "title": page.get("title", "").strip(),
                            "url": page.get("url", "").strip(),
                        }
                    )
                return web_search_results
            except JSONDecodeError as e:
                self.logger.error(f"MCP服务 bailian_web_search 检索失败：{e}")
                return []


if __name__ == "__main__":
    state = {
        "rewritten_query": "今天太原天气如何?",
        "item_names": ["H3C LA2608-GM/GU 室内无线网关"],
    }

    node = WebMcpSearchNode()
    node.process(state)
