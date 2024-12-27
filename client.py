import asyncio
import sys
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # 加载环境变量

class MCPClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.anthropic: Anthropic = Anthropic()
        self.stdio: Any = None
        self.write: Any = None

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器
        
        Args:
            server_script_path: 服务器脚本路径 (.py 或 .js)
        """
        try:
            print(f"\n正在连接到服务器脚本: {server_script_path}")
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("服务器脚本必须是 .py 或 .js 文件")
                
            command = "python" if is_python else "node"
            print(f"使用命令: {command} {server_script_path}")
            
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )
            
            print("正在创建 stdio_client...")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            print("stdio_client 创建成功")
            
            self.stdio, self.write = stdio_transport
            print("获取到 stdio 和 write 流")
            
            print("正在创建 ClientSession...")
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            print("ClientSession 创建成功")
            
            print("正在初始化会话...")
            await self.session.initialize()
            print("会话初始化完成")
            
            # 列出可用工具
            print("正在获取可用工具列表...")
            response = await self.session.list_tools()
            tools = response.tools
            print("\n已连接到服务器，可用工具:", [tool.name for tool in tools])
            
        except Exception as e:
            print(f"连接服务器时发生错误: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            raise

    async def process_query(self, query: str) -> str:
        """处理查询，使用 Claude 和可用工具"""
        try:
            print(f"\n开始处理查询: {query}")
            messages = [
                {
                    "role": "user",
                    "content": query
                }
            ]

            response = await self.session.list_tools()
            available_tools = [{ 
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]
            print(f"可用工具列表: {[tool['name'] for tool in available_tools]}")

            try:
                print("正在调用 Claude API...")
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )
                print("Claude API 调用成功")
            except Exception as e:
                print(f"Claude API 调用失败: {str(e)}")
                raise

            # 处理响应和工具调用
            tool_results = []
            final_text = []

            print("\n开始处理 Claude 响应...")
            for content in response.content:
                if content.type == 'text':
                    print("收到文本响应")
                    final_text.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    print(f"\n执行工具调用: {tool_name}")
                    print(f"工具参数: {tool_args}")
                    
                    try:
                        # 执行工具调用
                        print(f"开始执行工具 {tool_name}")
                        result = await self.session.call_tool(tool_name, tool_args)
                        print(f"工具 {tool_name} 调用结果: {result}")
                        tool_results.append({"call": tool_name, "result": result})
                        final_text.append(f"[调用工具 {tool_name}，参数 {tool_args}]")
                    except Exception as e:
                        print(f"工具调用失败: {str(e)}")
                        print(f"错误类型: {type(e).__name__}")
                        import traceback
                        print(f"错误堆栈: {traceback.format_exc()}")
                        raise

                    # 继续与工具结果的对话
                    if hasattr(content, 'text') and content.text:
                        messages.append({
                          "role": "assistant",
                          "content": content.text
                        })
                    messages.append({
                        "role": "user", 
                        "content": result.content
                    })

                    try:
                        print("\n继续与 Claude 对话...")
                        response = self.anthropic.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1000,
                            messages=messages,
                        )
                        print("Claude 响应成功")
                    except Exception as e:
                        print(f"获取 Claude 响应失败: {str(e)}")
                        raise

                    final_text.append(response.content[0].text)

            print("\n查询处理完成")
            return "\n".join(final_text)
            
        except Exception as e:
            print(f"\n处理查询时发生错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return f"处理查询时发生错误: {str(e)}"

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动！")
        print("输入您的查询或输入 'quit' 退出。")
        
        while True:
            try:
                query = input("\n查询: ").strip()
                
                if query.lower() == 'quit':
                    print("正在退出...")
                    break
                    
                if not query:
                    print("\n请输入有效的查询内容。")
                    continue
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except KeyboardInterrupt:
                print("\n接收到中断信号")
                return
            except Exception as e:
                print(f"\n错误: {str(e)}")
                print("如需退出，请输入 'quit' 或按 Ctrl+C")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("用法: python client.py <服务器脚本路径>")
        sys.exit(1)
    
    print(f"服务器脚本路径: {sys.argv[1]}")

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在清理资源...")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        try:
            await client.cleanup()
            print("资源清理完成，程序退出")
        except Exception as e:
            print(f"清理资源时发生错误: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已终止")
    except Exception as e:
        print(f"程序异常退出: {str(e)}")
