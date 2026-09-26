# LangGraph Tools Guide
### From LLM Answers to Tool-Using Agents

---

## Part 1: What Are Tools in an LLM Application?

> **A tool is a normal piece of code that an LLM is allowed to request when it needs help doing something outside pure text generation.**

An LLM is very good at language: explaining, summarizing, reasoning, drafting, classifying, and planning. But many real applications need more than language. They need to search, calculate, query databases, call APIs, read files, send emails, or perform actions in the outside world.

That is where tools come in.

The important idea is:

```text
The LLM decides what should be done.
The tool does the actual work.
LangGraph controls the loop between them.
```

Without tools, a chatbot can only answer from its model knowledge. With tools, the chatbot can act like an agent: it can inspect the user request, decide whether an external function is needed, call that function, read the result, and then respond with a better final answer.

---

## Part 2: Why Tools Matter

LLMs should not be trusted to do everything from memory. Some tasks are better handled by deterministic code or external systems.

### Example 1: Exact Math

If the user asks:

```text
what is 123 * 45?
```

The model might be able to answer, but exact arithmetic should be handled by code. A calculator tool gives a reliable result.

### Example 2: Fresh Information

If the user asks about something recent, the model's training data may be outdated. A search tool can fetch newer information.

### Example 3: External Actions

If the user asks an assistant to create a ticket, send a message, update a database row, or fetch an order status, the model needs access to application code or APIs.

### The Mental Model

```text
Plain chatbot:

User question -> LLM -> Answer

Tool-using chatbot:

User question -> LLM decides -> Tool executes -> Tool result -> LLM writes answer
```

The LLM is not magically executing code. It is producing a structured request that says, "Call this tool with these arguments." LangGraph then executes the tool and puts the result back into the conversation.

---

## Part 3: What This Notebook Builds

The notebook in this folder builds a small LangGraph app that can use tools.

```text
15_Tools/
|-- Tooling.ipynb
```

It uses:

- Gemini as the chat model
- A prebuilt DuckDuckGo search tool
- A custom calculator tool
- `llm.bind_tools(...)` to expose tools to the model
- `ToolNode` to execute selected tools
- `tools_condition` to route between the LLM and tools
- `add_messages` to preserve conversation history in graph state

The final result is a tool-enabled chatbot graph.

---

## Part 4: The Big Picture Flow

The tool-calling workflow has three main jobs:

1. The chat node asks the LLM what to do.
2. The model either answers directly or requests a tool call.
3. If a tool was requested, LangGraph runs the tool and stores the result.

```text
                 User Message
                      |
                      v
              +---------------+
              |   chat_node   |
              |  LLM decides  |
              +-------+-------+
                      |
          +-----------+-----------+
          |                       |
          v                       v
   No tool needed          Tool call requested
          |                       |
          v                       v
        END               +---------------+
                          |     tools     |
                          |   ToolNode    |
                          +-------+-------+
                                  |
                                  v
                             Tool result
```

For a full agent loop, the graph should continue from `tools` back to `chat_node`:

```text
User -> chat_node -> tools -> chat_node -> final answer
```

This second trip to `chat_node` matters because the raw tool result is not always the best final user-facing answer. The LLM usually needs to read the tool output and explain it clearly.

---

## Part 5: Imports - The Building Blocks

The notebook begins by importing the pieces needed to build a LangGraph tool workflow.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
```

The most important imports are:

| Import | Purpose |
|---|---|
| `StateGraph` | Creates the LangGraph workflow |
| `START`, `END` | Mark where execution begins and finishes |
| `BaseMessage`, `HumanMessage` | Represent chat messages |
| `add_messages` | Appends messages to state instead of overwriting them |
| `ToolNode` | Executes tool calls requested by the LLM |
| `tools_condition` | Routes to tools if the LLM requested a tool call |
| `DuckDuckGoSearchRun` | A prebuilt search tool |
| `@tool` | Converts a Python function into a LangChain tool |

---

## Part 6: The LLM

The notebook uses Gemini through LangChain:

```python
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.9
)
```

The model is the reasoning layer. It reads the conversation and decides whether it can answer directly or whether it needs one of the available tools.

The `temperature` controls randomness. A higher value like `0.9` can make responses more varied. For tool-heavy production workflows, lower temperatures are often preferred because tool selection should usually be stable and predictable.

---

## Part 7: Tool Type 1 - A Prebuilt Search Tool

The first tool is a ready-made search tool:

```python
search_tool = DuckDuckGoSearchRun(region="us-en")
```

This gives the LLM access to a search function. If the model sees a question that likely needs current or external information, it can request this tool instead of answering only from memory.

### When a Search Tool Is Useful

- Current events
- Company or product lookups
- Documentation lookup
- Quick factual research
- Questions where the model's built-in knowledge may be stale

### Key Point

The search tool is still just code. The model does not browse by itself. It asks for the search tool, LangGraph executes the tool, and the search result is returned as a message.

---

## Part 8: Tool Type 2 - A Custom Calculator Tool

The second tool is written manually in the notebook:

```python
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Performs basic arithmetic operations on two numbers."""
    if operation == "add":
        result = first_num + second_num
    elif operation == "subtract":
        result = first_num - second_num
    elif operation == "multiply":
        result = first_num * second_num
    elif operation == "divide":
        if second_num == 0:
            return {"error": "Cannot divide by zero."}
        result = first_num / second_num
    else:
        return {
            "error": "Invalid operation. Supported operations are add, subtract, multiply, divide."
        }

    return {"result": result}
```

This is a regular Python function with one extra decorator: `@tool`.

### What `@tool` Does

The `@tool` decorator converts the function into a tool object that LangChain-compatible models can understand.

The LLM uses four pieces of information:

| Part | Why It Matters |
|---|---|
| Function name | Tells the model what the tool is called |
| Argument names | Tells the model what inputs the tool expects |
| Type hints | Tells the model the expected argument types |
| Docstring | Tells the model when and why to use the tool |

For this function:

```python
def calculator(first_num: float, second_num: float, operation: str) -> dict:
```

The model learns that it can call a tool named `calculator` with:

- `first_num`: a number
- `second_num`: a number
- `operation`: a string such as `"add"` or `"multiply"`

### Supported Operations

```text
add       -> first_num + second_num
subtract  -> first_num - second_num
multiply  -> first_num * second_num
divide    -> first_num / second_num
```

### Why Return a Dictionary?

The tool returns structured data:

```python
return {"result": result}
```

Structured results are easier for the LLM and the rest of the application to interpret than loose text. They also make error handling clearer:

```python
return {"error": "Cannot divide by zero."}
```

---

## Part 9: Binding Tools to the LLM

After creating tools, the notebook puts them in a list:

```python
tools = [search_tool, calculator]
```

Then it binds them to the model:

```python
llm_with_tools = llm.bind_tools(tools)
```

This line is very important.

> `bind_tools` does not execute tools. It only tells the model which tools exist and what schemas they use.

After binding, the model can respond in two ways:

1. A normal assistant answer.
2. A tool-call request.

### Normal Answer

If no tool is needed, the model can simply answer:

```text
The capital of France is Paris.
```

### Tool-Call Request

If a tool is needed, the model can produce a structured tool call that means:

```text
Call calculator with:
- first_num = 123
- second_num = 45
- operation = "multiply"
```

The model chooses the tool and arguments. LangGraph executes the tool.

---

## Part 10: State - Where the Conversation Lives

The graph state is simple:

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

This says the graph has one state field: `messages`.

The `messages` list stores the full conversation:

- Human messages
- AI messages
- Tool-call requests
- Tool results

### Why `add_messages` Is Used

Without a reducer, each node update would overwrite the old value. But chat history should grow over time.

`add_messages` tells LangGraph:

```text
When a node returns new messages, append them to the existing conversation.
```

So the state evolves like this:

```text
Initial state:
messages = [HumanMessage("what is 123*45?")]

After chat_node:
messages = [
  HumanMessage("what is 123*45?"),
  AIMessage(tool_call=calculator(...))
]

After ToolNode:
messages = [
  HumanMessage("what is 123*45?"),
  AIMessage(tool_call=calculator(...)),
  ToolMessage({"result": 5535})
]
```

This accumulated message history is what allows the LLM to see what happened earlier in the graph.

---

## Part 11: Node 1 - The Chat Node

The chat node is where the LLM is called:

```python
def chat_node(state: ChatState) -> dict:
    """Node that handles the chat interaction with the LLM and tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

The node does three things:

1. Reads the current conversation from `state["messages"]`.
2. Sends those messages to the tool-bound LLM.
3. Returns the model response as a new message.

### What Can the Response Contain?

The response can be:

- A final natural-language answer
- A tool-call request

The chat node itself does not execute tools. It only asks the LLM what should happen next.

### Small Type Note

In the notebook, the function is annotated as:

```python
def chat_node(state: ChatState) -> str:
```

Conceptually, the node returns a dictionary update:

```python
return {"messages": [response]}
```

So `-> dict` is the clearer annotation for study purposes.

---

## Part 12: Node 2 - The Tool Node

The second node is created with LangGraph's prebuilt `ToolNode`:

```python
tool_node = ToolNode(tools)
```

`ToolNode` is a ready-made node that knows how to:

1. Look at the latest AI message.
2. Check whether it contains tool calls.
3. Match the requested tool name to one of the available tools.
4. Execute the tool with the arguments chosen by the LLM.
5. Add the tool result back into `messages`.

This saves you from manually parsing tool calls and dispatching functions.

### ToolNode Mental Model

```text
AIMessage says: call calculator(first_num=123, second_num=45, operation="multiply")
        |
        v
ToolNode finds calculator
        |
        v
calculator(123, 45, "multiply")
        |
        v
ToolMessage({"result": 5535})
```

The result becomes part of the graph state, so the LLM can use it in the next step.

---

## Part 13: Building the Graph

The notebook creates a `StateGraph` using `ChatState`:

```python
graph = StateGraph(ChatState)
```

Then it adds two nodes:

```python
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
```

At this point, LangGraph knows which functions exist, but it does not yet know the execution order. That is what edges are for.

---

## Part 14: Routing with `tools_condition`

The graph starts at `chat_node`:

```python
graph.add_edge(START, "chat_node")
```

After `chat_node`, the graph uses a conditional edge:

```python
graph.add_conditional_edges("chat_node", tools_condition)
```

`tools_condition` is a prebuilt routing function. It checks the latest AI message and decides what should happen next.

### The Routing Logic

```text
If the latest AI message contains tool calls:
    route to "tools"

If the latest AI message does not contain tool calls:
    route to END
```

So the graph can branch:

```text
chat_node
   |
   +-- tool call exists -> tools
   |
   +-- no tool call ----> END
```

This is the same core LangGraph idea from conditional workflows: a function reads state and decides the next node.

---

## Part 15: The Important Missing Edge

The notebook currently has:

```python
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
```

This is enough to route from the LLM to the tool node. But it does not add the edge that sends the tool result back to the LLM:

```python
graph.add_edge("tools", "chat_node")
```

### Why This Edge Matters

Without this edge, execution can stop after the tool runs. The final message may be the raw tool result rather than a polished assistant answer.

With this edge, the graph becomes a ReAct-style loop:

```text
Reason -> Act -> Observe -> Reason again -> Answer
```

In LangGraph form:

```python
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
```

And visually:

```text
             +-------------+
             |  chat_node  |
             +------+------+
                    |
          tools_condition
                    |
        +-----------+-----------+
        |                       |
        v                       v
      tools                    END
        |
        +-----------> chat_node
```

This loop continues until the model no longer requests a tool. Then `tools_condition` routes to `END`.

---

## Part 16: Compiling the Graph

After nodes and edges are defined, the graph is compiled:

```python
bot = graph.compile()
```

Compiling turns the graph blueprint into a runnable application.

You can think of the difference like this:

| Object | Meaning |
|---|---|
| `graph` | The design or blueprint |
| `bot` | The compiled runnable app |

Once compiled, you can call:

```python
bot.invoke(...)
```

---

## Part 17: Running the Tool-Using Chatbot

The notebook invokes the graph with a human message:

```python
response = bot.invoke({
    "messages": [HumanMessage(content="what is 123*45?")]
})

print(response["messages"][-1].content)
```

The expected reasoning is:

```text
User asks for exact multiplication.
LLM sees that calculator is available.
LLM requests calculator.
ToolNode executes calculator.
Calculator returns 5535.
The graph stores the result in messages.
```

If the `tools -> chat_node` edge is included, the LLM then reads the tool result and produces a final answer like:

```text
123 * 45 = 5535.
```

---

## Part 18: Tool Calling vs Direct Answering

Tool use should not happen for every message. A good tool-enabled agent decides when tools are useful and when they are unnecessary.

| User Request | Best Behavior |
|---|---|
| "Hello" | Answer directly |
| "Explain what a graph node is" | Answer directly |
| "What is 123 * 45?" | Use calculator |
| "Search for current LangGraph docs" | Use search |
| "Create a support ticket" | Use an API/tool |
| "What did we just discuss?" | Use conversation state, not an external tool |

The model's job is to choose. LangGraph's job is to make that choice executable and controllable.

---

## Part 19: Common Beginner Confusions

### Confusion 1: Does `bind_tools` run the tools?

No. `bind_tools` only exposes tool schemas to the model.

```text
bind_tools = "Here are the tools you may request."
ToolNode   = "I will actually run the requested tool."
```

### Confusion 2: Does the LLM know the Python function body?

Not directly. The LLM mainly sees the tool name, argument schema, and description/docstring. That is why good names, type hints, and docstrings matter.

### Confusion 3: Why do we need `ToolNode`?

Because the model only requests a tool call. Something in your program must safely execute that tool call. `ToolNode` is the prebuilt LangGraph node that handles this.

### Confusion 4: Why do we need to return to `chat_node` after tools?

Because the tool result is usually an observation, not the final answer. The LLM needs one more turn to interpret the observation for the user.

### Confusion 5: Are tools always safe?

No. Tools can perform real actions. A calculator is low risk. A tool that sends emails, updates databases, or makes purchases needs validation, permissions, and often human approval.

---

## Part 20: Safety and Design Guidelines

When designing tools, keep them small, clear, and predictable.

### Good Tool Design

- Give the tool a clear name.
- Use precise argument names.
- Add type hints.
- Write a docstring that explains when to use the tool.
- Return structured data.
- Validate inputs inside the tool.
- Return useful error messages.

### Riskier Tool Design

- Vague tool names like `do_task`
- Unclear arguments like `data` or `input`
- No docstring
- No input validation
- Tools that perform irreversible actions without confirmation
- Tools that expose secrets or sensitive system behavior

### A Practical Rule

```text
If a tool can change the outside world, treat it as a serious action.
```

For high-impact actions, combine tools with human-in-the-loop approval.

---

## Summary

Tools are what let a LangGraph app move from "only generate text" to "reason and act."

The core pattern is:

```text
LLM reads messages
    |
    v
LLM decides whether a tool is needed
    |
    +-- no tool needed -> final answer
    |
    +-- tool needed ----> ToolNode executes tool
                              |
                              v
                        result added to messages
                              |
                              v
                        LLM reads result and answers
```

The key pieces are:

| Piece | Role |
|---|---|
| `@tool` | Converts a Python function into a tool |
| Prebuilt tools | Ready-made external capabilities like search |
| `bind_tools` | Tells the LLM which tools it may request |
| `ToolNode` | Executes requested tools |
| `tools_condition` | Routes to tools or ends the graph |
| `add_messages` | Preserves conversation and tool results in state |
| `graph.add_edge("tools", "chat_node")` | Lets the LLM turn tool results into final answers |

Once you understand this loop, tool-using agents become much less mysterious:

```text
The model reasons.
The graph routes.
The tool acts.
The state remembers.
The model responds.
```
