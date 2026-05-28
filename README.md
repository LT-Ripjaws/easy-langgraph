# Easy-Langgraph

![LangGraph workflow banner](assets/easy-langgraph-banner.png)

`easy-langgraph` is a hands-on learning repository for building agentic workflows with
LangGraph. I created it while learning about Langgraph myself as practice.
It should be a nice beginner's tutorial to anyone interested in Langgraph.
It starts with the mental model behind graphs, state, nodes, and edges, then
walks through practical notebooks and guides for sequential workflows, parallel
execution, conditional routing, persistence, chatbots, tools, MCP, human-in-the-loop
flows, subgraphs, and memory.

## What You'll Learn

- How LangGraph represents an LLM workflow as a stateful graph.
- How to design state with `TypedDict`, reducers, and message history.
- How to build sequential, parallel, conditional, and iterative workflows.
- How to add persistence with checkpointers and `thread_id` session config.
- How to build chatbot-style apps with Streamlit and Gemini.
- How to connect tools, databases, MCP servers, human review, and memory.

## Repository Map

| Folder | Focus |
| --- | --- |
| `01_Required_Concepts/` | Core ideas: Generative AI vs. Agentic AI, LangChain vs. LangGraph, and LangGraph fundamentals. |
| `02_Sequential_Workflows/` | Linear graph flows and prompt chaining examples. |
| `03_Parallel_Workflows/` | Fan-out/fan-in workflows that process independent tasks in parallel. |
| `04_Conditional_Workflows/` | Conditional edges, routing, and evaluator-style branching. |
| `05_Iterative_Workflow/` | Generate, evaluate, and refine loops. |
| `06_Basic_Chatbot/` | A basic persistent chatbot notebook using message state. |
| `07_Persistance/` | Persistence concepts, checkpointers, threads, storage backends, and recovery patterns. |
| `08_Langgraph_Chatbot/` | Streamlit chatbot app backed by a LangGraph graph and in-memory checkpointing. |
| `09_Langgraph_Database/` | SQLite-backed chatbot persistence example. |
| `10_Tools/` | Tool-calling agents with search, calculator tools, `ToolNode`, and routing. |
| `11_MCPS/` | Model Context Protocol concepts and integration notes. |
| `12_Human_in_the_loop/` | Interrupts, approvals, review steps, and human-controlled graph execution. |
| `13_Subgraphs/` | Reusable graph composition with subgraphs. |
| `14_Short_term_memory/` | Thread-scoped memory and conversation state. |
| `15_Long_term_memory/` | Cross-thread durable memory with namespaces, keys, and stores. |

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

There is no pinned requirements file yet, so install the packages used across the
notebooks and Python examples:

```bash
pip install --upgrade pip
pip install jupyter ipykernel python-dotenv streamlit requests pydantic
pip install langgraph langchain-core langchain-google-genai langchain-community duckduckgo-search
pip install langchain-mcp-adapters mcp langgraph-checkpoint-sqlite
```

Some guide files mention optional production backends such as Postgres or Redis. Install
those packages only when you are running those specific examples.

### 3. Configure your API key

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and add your Gemini API key:

```env
GOOGLE_API_KEY='your_gemini_api_key_here'
```

Never commit `.env`. It is already ignored by Git.

## Running The Examples

Start Jupyter and open any notebook:

```bash
jupyter notebook
```

Run the Streamlit chatbot:

```bash
streamlit run 08_Langgraph_Chatbot/frontend.py
```

Run the SQLite persistence demo:

```bash
python 09_Langgraph_Database/langgraph_database_backend.py
```

## Common LangGraph Pattern Used Here

Most examples follow the same shape:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    output: str

def node_name(state: State) -> dict:
    return {"output": state["input"].upper()}

graph = StateGraph(State)
graph.add_node("node_name", node_name)
graph.add_edge(START, "node_name")
graph.add_edge("node_name", END)

app = graph.compile()
result = app.invoke({"input": "hello"})
```

For chatbot examples, message history usually uses `add_messages`:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

## Troubleshooting

If Gemini calls fail, confirm that `.env` exists and contains `GOOGLE_API_KEY`.

If Jupyter cannot see installed packages, make sure the notebook is using the Python
environment where you installed the dependencies.

If `SqliteSaver` is missing, install the SQLite checkpoint package:

```bash
pip install langgraph-checkpoint-sqlite
```

## Useful Links

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- LangChain documentation: https://python.langchain.com/
- Gemini API docs: https://ai.google.dev/gemini-api/docs

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
