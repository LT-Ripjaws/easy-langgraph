# Easy-Langgraph

![Graph of connected nodes](assets/easy-langgraph-banner.jpg)

`easy-langgraph` is a hands-on learning repository for building agentic workflows with
LangGraph. I created it while learning about Langgraph myself as practice.
It should be a nice beginner's tutorial to anyone interested in Langgraph.
It starts with the mental model behind graphs, state, nodes, and edges, then
walks through practical notebooks and guides for workflows, streaming, persistence,
chatbots, memory, tools, agents, multi-agent systems, RAG, and finally how to make a
graph reliable and deploy it.

Every lesson folder from `02` onward has a `*_Guide.md` that explains the concept step by step (`01` holds the background reading). Most folders also
have a notebook or Python script you can run. All examples use Google Gemini.

The notebooks in folders 07, 10, 16, 20, 21, 23 and 24 are saved without outputs. Run them
to see the results.

## What You'll Learn

- How LangGraph represents an LLM workflow as a stateful graph.
- How to design state with `TypedDict`, reducers, and message history.
- How to build sequential, parallel, conditional, and iterative workflows, and route with `Command`.
- How to stream graph progress and LLM tokens as they are produced.
- How to add persistence with checkpointers and `thread_id`, and rewind a run with time travel.
- How to build chatbot apps with Streamlit, SQLite, and short- and long-term memory.
- How to give agents tools, use the prebuilt `create_agent`, and connect MCP servers.
- How to add human review, split work across subgraphs and multiple agents, and build agentic RAG.
- How to handle failures with retries, error handlers, timeouts, and caching.
- How to write workflows with the Functional API, trace them, and deploy them with `langgraph dev`.

## Repository Map

### Part 1: Foundations and workflow patterns

| Folder | Focus |
| --- | --- |
| `01_Required_Concepts/` | Core ideas: Generative AI vs. Agentic AI, LangChain vs. LangGraph, and LangGraph fundamentals. |
| `02_Sequential_Workflows/` | Linear graph flows and prompt chaining examples. |
| `03_Parallel_Workflows/` | Fan-out/fan-in workflows that process independent tasks in parallel. |
| `04_Conditional_Workflows/` | Conditional edges, routing, and evaluator-style branching. |
| `05_Iterative_Workflow/` | Generate, evaluate, and refine loops. |
| `06_Command_Routing/` | Routing and updating state in one step with `Command(update=..., goto=...)`. |
| `07_Streaming/` | Streaming with `stream_mode`: values, updates, LLM tokens (`messages`), and custom progress events. |

### Part 2: Chatbots, persistence, and memory

| Folder | Focus |
| --- | --- |
| `08_Basic_Chatbot/` | A basic persistent chatbot notebook using message state. |
| `09_Persistence/` | Persistence concepts, checkpointers, threads, storage backends, and recovery patterns. |
| `10_Time_Travel/` | Inspecting checkpoint history, replaying from an old checkpoint, and forking with `update_state`. |
| `11_Langgraph_Chatbot/` | Streamlit chatbot app backed by a LangGraph graph and in-memory checkpointing. |
| `12_Langgraph_Database/` | SQLite-backed chatbot persistence example. |
| `13_Short_term_memory/` | Thread-scoped memory and conversation state. |
| `14_Long_term_memory/` | Cross-thread durable memory with namespaces, keys, and stores. |

### Part 3: Tools and agents

| Folder | Focus |
| --- | --- |
| `15_Tools/` | Tool-calling agents with search, calculator tools, `ToolNode`, and routing. |
| `16_Prebuilt_Agents/` | LangChain 1.0 `create_agent`: tools, structured output, memory, runtime context, and middleware. |
| `17_MCP/` | Model Context Protocol concepts and integration notes. |
| `18_Human_in_the_loop/` | Interrupts, approvals, review steps, and human-controlled graph execution. |
| `19_Subgraphs/` | Reusable graph composition with subgraphs. |
| `20_Multi_Agent/` | Supervisor and handoff patterns for splitting work across several agents. |
| `21_Agentic_RAG/` | Retrieval-augmented generation where the agent decides when to retrieve, grades documents, and rewrites the question. |

### Part 4: Production

| Folder | Focus |
| --- | --- |
| `22_Fault_Tolerance/` | Retry policies, node error handlers, timeouts, recursion limits, and node caching. |
| `23_Functional_API/` | Writing workflows with `@entrypoint` and `@task` instead of a graph, including interrupts. |
| `24_Observability_and_Deployment/` | Graph visualization, LangSmith tracing, and serving a graph with `langgraph.json` and `langgraph dev`. |

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

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The versions this repository was tested with are noted in `requirements.txt`. Some guide
files mention optional production backends such as Postgres or Redis. Install those
packages only when you are running those specific examples.

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
streamlit run 11_Langgraph_Chatbot/frontend.py
```

Run the SQLite persistence demo:

```bash
python 12_Langgraph_Database/langgraph_database_backend.py
```

Serve a graph locally with the LangGraph dev server (see the guide in folder 24):

```bash
cd 24_Observability_and_Deployment/app
langgraph dev
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

If you get `429 RESOURCE_EXHAUSTED`, you have hit the Gemini free-tier daily limit. At the
time of writing the free tier allows about 20 requests per day for each model, and a notebook
can use 5 to 15 of them. The limit resets once a day. You can switch the `model=` name to
another Gemini model, which has its own daily limit, or use a paid API key.

If Jupyter cannot see installed packages, make sure the notebook is using the Python
environment where you installed the dependencies.

If `SqliteSaver` is missing, install the SQLite checkpoint package:

```bash
pip install langgraph-checkpoint-sqlite
```

If `create_agent` cannot be imported, you have an old version of LangChain. It was added
in LangChain 1.0 and replaces `langgraph.prebuilt.create_react_agent`:

```bash
pip install -U "langchain>=1.0"
```

## Useful Links

- LangGraph documentation: https://docs.langchain.com/oss/python/langgraph/overview
- LangChain documentation: https://docs.langchain.com/oss/python/langchain/overview
- LangSmith documentation: https://docs.langchain.com/langsmith/home
- Gemini API docs: https://ai.google.dev/gemini-api/docs

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
