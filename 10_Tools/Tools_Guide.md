# LangGraph Tools Guide

This folder introduces tool calling in LangGraph.

The notebook shows how an LLM can decide to call external functions instead of answering only from its own model knowledge.

## File

```text
10_Tools/
|-- Tooling.ipynb
```

## What This Notebook Builds

The notebook creates a small tool-enabled chatbot with:

- Gemini as the chat model
- A prebuilt DuckDuckGo search tool
- A custom calculator tool
- `llm.bind_tools(...)` to expose tools to the model
- `ToolNode` to execute selected tools
- `tools_condition` to route tool calls

## Tools Used

### Prebuilt Search Tool

```python
search_tool = DuckDuckGoSearchRun(region="us-en")
```

This gives the model a search function for live web-style lookups.

### Custom Calculator Tool

```python
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Performs basic arithmetic operations on two numbers."""
```

The `@tool` decorator converts a normal Python function into a LangChain tool. The function name, arguments, type hints, and docstring help the LLM understand when and how to call it.

Supported operations:

- `add`
- `subtract`
- `multiply`
- `divide`

## Binding Tools to the LLM

```python
tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)
```

`bind_tools` does not execute tools by itself. It teaches the model that these tools exist and allows the model to return tool-call requests.

## State

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The graph stores the conversation in `messages`. `add_messages` appends new human, AI, and tool messages without overwriting earlier messages.

## Graph Nodes

### Chat Node

```python
def chat_node(state: ChatState) -> str:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

This node asks the LLM what to do next. The model can either answer directly or request a tool call.

### Tool Node

```python
tool_node = ToolNode(tools)
```

`ToolNode` reads tool-call requests from the latest AI message, executes the matching tool, and appends the result as a tool message.

## Routing

```python
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
```

`tools_condition` checks the latest AI message:

- If the model requested a tool, route to `tools`
- If no tool is needed, end the graph

## Important Note

The notebook currently does not add this edge:

```python
graph.add_edge("tools", "chat_node")
```

Without that edge, the graph can stop after executing the tool. That means the final message may be the raw tool result, not a polished assistant response.

For a full ReAct-style agent loop, use:

```python
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
```

Then the flow becomes:

```text
User -> chat_node -> tools -> chat_node -> final answer
```

## Example

```python
response = bot.invoke({
    "messages": [HumanMessage(content="what is 123*45?")]
})
print(response["messages"][-1].content)
```

The model should choose the calculator tool because exact arithmetic is better handled by code than by the LLM.

## Summary

Tools let a LangGraph app move from "text generation only" to "reason plus act".

Core pattern:

```text
LLM decides -> ToolNode executes -> result is added to messages -> LLM continues
```

Use tools when the model needs external data, exact computation, API calls, database access, file operations, or any action that should be performed by code.
