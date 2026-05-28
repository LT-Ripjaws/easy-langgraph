# Model Context Protocol (MCP) with LangGraph

MCP stands for **Model Context Protocol**. It is an open standard for connecting AI applications to external systems such as tools, APIs, databases, files, and reusable prompts.

In this project, MCP is best understood as the next step after `10_Tools`:

```text
10_Tools: local Python tools inside the same LangGraph app
11_MCPS: tools/resources/prompts exposed by external MCP servers
```

---

## 1. Why MCP Exists

Without MCP, every AI app needs custom integration code for every external system:

```text
LangGraph app -> custom database code
LangGraph app -> custom search code
LangGraph app -> custom file code
LangGraph app -> custom calendar code
```

MCP standardizes that connection:

```text
AI host/client -> MCP protocol -> MCP server -> external system
```

This means the same MCP server can be reused by different AI clients, such as a LangGraph app, Claude Desktop, an IDE, or another MCP-compatible assistant.

---

## 2. Core MCP Architecture

MCP uses a host-client-server architecture.

```text
Host app
 |
 | creates and manages
 v
MCP client
 |
 | JSON-RPC over stdio or HTTP
 v
MCP server
 |
 | exposes capabilities
 v
Tools, resources, prompts, APIs, files, databases
```

### Host

The host is the AI application the user interacts with.

Examples:

- Claude Desktop
- ChatGPT
- Cursor
- A LangGraph app
- A custom Streamlit chatbot

The host controls the user interface, permissions, model calls, and how much context is shared.

### Client

The MCP client lives inside the host application. It connects to one MCP server and handles protocol messages.

In LangGraph/LangChain projects, `MultiServerMCPClient` can create connections to one or more MCP servers.

### Server

The MCP server is a separate process or remote service that exposes capabilities.

A server might expose:

- Database queries
- File search
- Browser automation
- Weather APIs
- GitHub operations
- Company-specific internal APIs

---

## 3. MCP Capabilities

MCP servers can expose three major capability types.

## Tools

Tools are executable functions that a model can request.

Examples:

- `search_web(query)`
- `query_database(sql)`
- `create_ticket(title, description)`
- `calculate_total(price, quantity)`

Tools are **model-controlled**. The LLM can decide to call a tool when it needs an action or external data.

## Resources

Resources are readable context items exposed by a server.

Examples:

- A local file
- A database schema
- A project README
- An API response
- A document from a knowledge base

Resources are usually **application-controlled**. The host decides when to include them as context.

## Prompts

Prompts are reusable prompt templates exposed by a server.

Examples:

- `summarize_document`
- `code_review`
- `generate_test_plan`

Prompts are usually **user-controlled**. A UI might expose them as slash commands or selectable templates.

---

## 4. How MCP Works Internally

At a high level:

1. The host starts or connects to an MCP server.
2. The client and server initialize a session.
3. They negotiate protocol version and capabilities.
4. The client lists available tools, resources, or prompts.
5. The LLM decides whether to call a tool.
6. The client sends a JSON-RPC request to the server.
7. The server executes the operation and returns structured content.
8. The host passes the result back into the conversation or graph state.

For a tool call, the protocol shape is conceptually:

```text
Client -> tools/list -> Server returns tool names and schemas
Client -> tools/call -> Server executes selected tool
Server -> result -> Client converts result for the AI app
```

---

## 5. MCP vs LangChain Tools

The `10_Tools` folder uses local LangChain tools:

```python
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    ...
```

That function lives inside the same Python app as the graph.

MCP tools live behind a protocol boundary:

```text
LangGraph app -> MCP client -> MCP server -> actual function
```

| Feature | Local LangChain Tool | MCP Tool |
|---------|----------------------|----------|
| Location | Same Python process | Separate server/process/service |
| Reuse | Usually app-specific | Reusable across MCP clients |
| Discovery | Tool list is coded locally | Client asks server for tools |
| Communication | Direct Python call | JSON-RPC over stdio or HTTP |
| Best for | Simple local app logic | Shared integrations and external systems |
| Security boundary | Same app runtime | Server boundary plus host permissions |
| Resources/prompts | Not the main abstraction | First-class MCP primitives |

Short version:

```text
Tool = function the model can call
MCP = protocol for discovering and calling tools, reading resources, and loading prompts from external servers
```

---

## 6. How MCP Fits into LangGraph

In LangGraph, MCP tools eventually become normal LangChain-compatible tools.

The flow is:

```text
MCP server exposes tools
        |
        v
MultiServerMCPClient loads tools
        |
        v
LangChain-compatible tool objects
        |
        v
llm.bind_tools(tools)
        |
        v
ToolNode executes tool calls inside LangGraph
```

So the graph pattern is almost the same as `10_Tools`; only the source of the tools changes.

---

## 7. Example MCP Server

A simple MCP math server could look like this:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Important details:

- `FastMCP("math")` creates the server.
- `@mcp.tool()` exposes a function as an MCP tool.
- Type hints and docstrings help create the tool schema.
- `stdio` means the client launches the server as a subprocess.

For stdio MCP servers, do not print normal logs to stdout. stdout is reserved for JSON-RPC protocol messages. Use stderr or a logging library instead.

---

## 8. Example LangGraph Client

This is the LangGraph side that connects to the MCP server and uses its tools.

```python
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)

client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["math_server.py"],
        }
    }
)

tools = await client.get_tools()
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

bot = graph.compile()

response = bot.invoke({
    "messages": [HumanMessage(content="What is 12 * 9?")]
})

print(response["messages"][-1].content)
```

This looks very similar to the local tools graph. The difference is that `tools` came from an MCP server instead of being defined directly in the notebook.

---

## 9. Transports

MCP supports different ways for clients and servers to communicate.

### stdio

```text
Client launches server subprocess
Client writes JSON-RPC to stdin
Server writes JSON-RPC to stdout
```

Best for:

- Local tools
- Desktop apps
- Simple development
- File-system or local project integrations

Example:

```python
{
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["math_server.py"],
    }
}
```

### Streamable HTTP

```text
Client connects to remote MCP endpoint over HTTP
Server handles one or more clients
```

Best for:

- Remote services
- Deployed APIs
- Multi-user systems
- Authentication with headers

Example:

```python
{
    "weather": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
        "headers": {
            "Authorization": "Bearer YOUR_TOKEN"
        },
    }
}
```

---

## 10. Why Use MCP Instead of Only Tools?

Use local tools when:

- The tool is small and specific to one notebook
- You do not need reuse outside the current app
- You want the simplest possible setup

Use MCP when:

- Multiple AI apps should reuse the same capability
- The integration should run as a separate service
- You want a standard interface for tools, resources, and prompts
- The external system has its own permissions or lifecycle
- You want to plug into existing MCP-compatible clients

Example progression:

```text
Local calculator tool
-> MCP math server
-> Same math server used by LangGraph, Claude Desktop, and an IDE
```

---

## 11. Security and Safety

MCP makes integrations easier, but tools can still take real actions.

Good practices:

- Show users which tools are available.
- Ask for confirmation before destructive or sensitive actions.
- Validate all tool inputs on the server.
- Sanitize tool outputs before passing them to the model.
- Add timeouts for slow tools.
- Use authentication for remote HTTP servers.
- Keep server permissions narrow.
- Avoid exposing secrets in tool descriptions or results.
- Log tool usage for debugging and auditing.

Important design idea:

```text
The host controls what context is shared.
The server should not see the full conversation unless the host deliberately sends it.
```

This separation is one of MCP's main benefits.

---

## 12. Common Pitfalls

### Pitfall 1: Treating MCP as a Model

MCP is not an LLM and does not generate text by itself.

It is a protocol for connecting an AI app to external capabilities.

### Pitfall 2: Forgetting the Tool Loop

In LangGraph, a practical tool-calling agent usually needs:

```python
graph.add_edge("tools", "chat_node")
```

Without this, the graph may stop at the tool result instead of letting the LLM turn that result into a final answer.

### Pitfall 3: Logging to stdout in stdio Servers

For stdio servers, stdout is used for protocol messages.

Use stderr or file logging for debug output.

### Pitfall 4: Giving Tools Too Much Power

Do not expose broad tools like `run_shell_command` or `delete_file` unless you have strong permission checks and user confirmation.

### Pitfall 5: Confusing Resources with Tools

Resources provide context.

Tools perform actions.

Prompts provide reusable instructions.

---

## 13. How This Connects to the Project

The learning path is:

```text
6_Basic_Chatbot
-> chatbot with message memory

7_Persistance
-> checkpointing and thread state

8_Langgraph_Chatbot
-> Streamlit UI around the graph

9_Langgraph_Database
-> SQLite-backed persistence

10_Tools
-> local tool calling with ToolNode

11_MCPS
-> external tool/resource/prompt servers via MCP
```

MCP becomes especially useful once your project grows beyond one notebook. It lets you separate "agent logic" from "external system integration".

---

## 14. Minimal Implementation Checklist

To add a real MCP example to this folder later:

- [ ] Create an MCP server file, such as `math_server.py`
- [ ] Define tools with `@mcp.tool()`
- [ ] Run the server with `transport="stdio"` or HTTP
- [ ] Install `langchain-mcp-adapters`
- [ ] Load tools with `MultiServerMCPClient`
- [ ] Bind tools to the LLM with `llm.bind_tools(tools)`
- [ ] Add `ToolNode(tools)` to the LangGraph graph
- [ ] Add `tools_condition`
- [ ] Add `tools -> chat_node` loop edge
- [ ] Test with a question that requires the MCP tool

---

## 15. References

- MCP introduction: https://modelcontextprotocol.io/docs/getting-started/intro
- MCP architecture: https://modelcontextprotocol.io/specification/2025-06-18/architecture
- MCP tools: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- MCP resources: https://modelcontextprotocol.io/specification/2025-06-18/server/resources
- MCP prompts: https://modelcontextprotocol.io/specification/2025-06-18/server/prompts
- MCP transports: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
- LangChain MCP adapters: https://docs.langchain.com/oss/python/langchain/mcp

---

## Summary

MCP is a standard connection layer for AI apps.

Local tools answer the question:

```text
How can this graph call this Python function?
```

MCP answers the bigger question:

```text
How can any compatible AI app discover and use external capabilities safely and consistently?
```

In LangGraph, MCP tools are loaded through a client, converted into LangChain-compatible tools, and then used with the same `ToolNode` and `tools_condition` pattern you already saw in `10_Tools`.
