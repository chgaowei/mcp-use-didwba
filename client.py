import asyncio
import sys
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.anthropic: Anthropic = Anthropic()
        self.stdio: Any = None
        self.write: Any = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server
        
        Args:
            server_script_path: Server script path (.py or .js)
        """
        print(f"\nConnecting to server script: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        print(f"Using command: {command} {server_script_path}")
        
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        print("Creating stdio_client...")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        print("stdio_client created successfully")
        
        self.stdio, self.write = stdio_transport
        print("Obtained stdio and write streams")
        
        print("Creating ClientSession...")
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        print("ClientSession created successfully")
        
        print("Initializing session...")
        await self.session.initialize()
        print("Session initialization complete")
        
        # List available tools
        print("Getting available tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server, available tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process query using Claude and available tools"""
        print(f"\nProcessing query: {query}")
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
        print(f"Available tools: {[tool['name'] for tool in available_tools]}")

        try:
            print("Calling Claude API...")
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )
            print("Claude API call successful")
        except Exception as e:
            print(f"Claude API call failed: {str(e)}")
            raise

        # Process response and tool calls
        tool_results = []
        final_text = []

        print("\nProcessing Claude response...")
        for content in response.content:
            if content.type == 'text':
                print("Received text response")
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                print(f"\nExecuting tool call: {tool_name}")
                print(f"Tool arguments: {tool_args}")
                
                print(f"Starting tool execution: {tool_name}")
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"Tool {tool_name} execution result: {result}")
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Called tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                        "role": "assistant",
                        "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                print("\nContinuing conversation with Claude...")
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )
                print("Claude response received")

                final_text.append(response.content[0].text)

        print("\nQuery processing complete")
        return "\n".join(final_text)
            

    async def chat_loop(self):
        """Run interactive chat loop"""
        print("\nMCP client started!")
        print("Enter your query or type 'quit' to exit.")
        
        while True:

            query = input("\nQuery: ").strip()
            
            if query.lower() == 'quit':
                print("Exiting...")
                break
                
            if not query:
                print("\nPlease enter a valid query.")
                continue
                
            response = await self.process_query(query)
            print("\n" + response)
                

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_script_path>")
        sys.exit(1)
    
    print(f"Server script path: {sys.argv[1]}")

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\nInterrupt received, cleaning up resources...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        await client.cleanup()
        print("Resource cleanup complete, program exiting")

if __name__ == "__main__":
    asyncio.run(main())
