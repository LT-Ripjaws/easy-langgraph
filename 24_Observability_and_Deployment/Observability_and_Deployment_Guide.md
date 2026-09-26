# LangGraph Observability and Deployment Guide
### Seeing What a Graph Does, and Running It as a Service

---

## Part 1: Two Separate Problems

Once a graph works, two questions come up that earlier lessons do not answer:

1. **Observability**: while developing, how do you see what a graph actually did on a given run, without adding `print` statements everywhere?
2. **Deployment**: once it works, how do you run it as a long-lived service other applications can call, instead of only inside a notebook?

This guide covers both. Part 2 through Part 6 are about observability, using the notebook in this folder. Part 7 onward is about deployment, using the `app/` folder next to this guide.

```text
24_Observability_and_Deployment/
|-- 1_visualize_and_trace.ipynb
|-- Observability_and_Deployment_Guide.md
|-- app/
    |-- agent.py
    |-- langgraph.json
    |-- README.md
```

---

## Part 2: Drawing the Graph

A compiled graph can describe its own structure. `app.get_graph()` returns that structure as a `Graph` object; `.draw_mermaid()` turns it into [Mermaid](https://mermaid.js.org/) diagram syntax, as plain text:

```python
print(app.get_graph().draw_mermaid())
```

```text
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([<p>__start__</p>]):::first
    chat_node(chat_node)
    tools(tools)
    __end__([<p>__end__</p>]):::last
    __start__ --> chat_node;
    tools --> chat_node;
    chat_node -.-> tools;
    chat_node -.-> __end__;
```

This text renders as a diagram automatically on GitHub, in most Markdown viewers (including this one), and in the [Mermaid Live Editor](https://mermaid.live). No network call is needed to produce it: the diagram is generated entirely from the graph's own node and edge definitions.

`.draw_mermaid_png()` renders the same diagram to a PNG image instead of text, but it does so by calling a hosted rendering service over the network by default. If there is no network access, or that service is unreachable, it fails; the Mermaid text form always works.

---

## Part 3: Inspecting a Run With `stream_mode`

`invoke()` only gives you the final state. To see what happened *during* a run, use `stream()` with a stream mode that reports intermediate events.

### `stream_mode="tasks"`

Yields one event when a task (a node execution) starts, and one when it finishes:

```python
for event in app.stream(initial_state, stream_mode="tasks"):
    print(event)
```

```text
{'id': '...', 'name': 'chat_node', 'input': {...}, 'triggers': ('branch:to:chat_node',)}
{'id': '...', 'name': 'chat_node', 'error': None, 'result': {...}, 'interrupts': []}
```

### `stream_mode="debug"`

A superset of `"tasks"` that also includes checkpoint bookkeeping events:

```python
for event in app.stream(initial_state, stream_mode="debug"):
    print(event["type"], "-", event["payload"].get("name", ""))
```

```text
task - chat_node
task_result - chat_node
```

`"tasks"` is narrower and easier to read when you only care about what each node did. `"debug"` is closer to what a full trace (see Part 4) shows, including checkpoint-level detail.

### Other Stream Modes

This repo has already used `stream_mode="values"` (full state after each step) and the default `"updates"` (just what changed) in earlier lessons. `"tasks"` and `"debug"` are lower-level: they show individual task execution, not just state changes.

---

## Part 4: LangSmith Tracing

Stream modes are useful while a notebook cell is open in front of you. For anything longer-running, or for looking back at a run after the fact, LangSmith tracing captures every run automatically and shows it in a web UI: every node's input and output, how long each step took, and any errors, without changing any graph code.

### Turning It On

Set these before the process that runs the graph starts (for example, in `.env`, read by `load_dotenv()`):

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your LangSmith API key>
LANGSMITH_PROJECT=<a project name, optional>
```

With `LANGSMITH_TRACING=true` and a valid key set, every `invoke`/`stream`/`ainvoke`/`astream` call is traced from then on. No code in the graph itself needs to change.

### It Is Optional

Nothing in this repo, including this notebook, requires a LangSmith account or API key. The notebook's cells run and print their own output whether or not tracing is on. Treat tracing as something you turn on when you want the extra visibility, not a dependency.

### What Tracing Gives You Beyond `stream_mode`

- A persistent record of past runs, not just the one currently streaming in front of you.
- A visual timeline of node execution, including time spent, without writing any code to print it.
- The ability to inspect a run days later, or share a link to a specific run with someone else.
- **Evaluation**: LangSmith can run a graph against a saved dataset of inputs and expected outputs, and score the results, which is a separate feature from tracing built on the same platform.

---

## Part 5: A Note on Naming (as of October 2025)

LangChain renamed two products in October 2025:

```text
LangGraph Platform   -> LangSmith Deployment
LangGraph Studio     -> LangSmith Studio
```

The underlying tools did not change: `langgraph.json`, `langgraph dev`, `langgraph build`, and `langgraph up` all still work the same way. Only the product names (and some URLs, like the Studio UI) changed. Documentation and blog posts written before the rename may still refer to "LangGraph Platform" or "LangGraph Studio"; treat those as the same thing under a new name.

---

## Part 6: What This Notebook Does Not Cover

Building the graph itself (state, nodes, tools, edges) is not new material here; it reuses the tool-calling chatbot from `15_Tools`. If any of that looks unfamiliar, `15_Tools/Tools_Guide.md` covers it in full.

---

## Part 7: Packaging a Graph for Deployment

The `app/` folder next to this guide is a minimal, deployable version of a LangGraph app: a Python module that exports a compiled graph, plus a config file describing how to run it.

### `agent.py`

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
# ... tools, state, chat_node, ToolNode, edges ...

builder = StateGraph(ChatState)
# ... add_node / add_edge calls ...

graph = builder.compile()
```

The only requirement for deployment is that some module-level variable holds a **compiled** graph. `langgraph.json` (below) tells the CLI which variable, in which file. Note that `agent.py` does not compile the graph with its own checkpointer: the dev server and LangSmith Deployment attach persistence around the graph automatically, so a deployed graph is usually compiled without one.

### `langgraph.json`

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": "../../.env"
}
```

| Field | Meaning |
|---|---|
| `dependencies` | What to install. `"."` means "this directory is itself a local package" (the CLI generates minimal package metadata if there is no `pyproject.toml`). Third-party packages the graph imports, such as `langchain-google-genai`, must be listed here too, or be in a `requirements.txt`/`pyproject.toml`; otherwise `langgraph build` produces an image that fails on import. |
| `graphs` | A name (`"agent"`) mapped to `<file>:<variable>`. This is the name used in the API (`/assistants/search`, the SDK, LangSmith Studio) and can differ from the file or variable name. |
| `env` | Path to a `.env` file, relative to `langgraph.json`. Here it points at the repo's root `.env` (two directories up), so this app reuses the same `GOOGLE_API_KEY` as every other lesson instead of needing its own copy. |

---

## Part 8: Running It Locally — `langgraph dev`

From the `app/` folder:

```bash
langgraph dev --no-browser --port 2024
```

This starts an in-memory API server (no Docker needed) with hot reloading: editing `agent.py` and saving restarts the graph automatically. `--no-browser` skips auto-opening LangSmith Studio in a browser, which matters when running this on a headless machine or inside a script.

### Checking It Started

```bash
curl http://127.0.0.1:2024/ok
# {"ok":true}

curl -X POST http://127.0.0.1:2024/assistants/search -H "Content-Type: application/json" -d "{}"
# [{"assistant_id": "...", "graph_id": "agent", "name": "agent", ...}]
```

`/assistants/search` lists every graph the server loaded from `langgraph.json`; seeing `"graph_id": "agent"` confirms `agent.py` imported cleanly and `graph` compiled without errors.

### Calling It From Studio

The URL printed on startup (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`) opens LangSmith Studio pointed at the local server, where you can send messages to the graph and see the same kind of step-by-step trace described in Part 3, in a UI instead of notebook output.

### Calling It From Code — `langgraph-sdk`

`langgraph-sdk` is installed automatically as a dependency of `langgraph-cli`. It talks to a running server (local or deployed) over HTTP:

```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")
result = await client.runs.wait(
    None,               # thread_id: None creates a new stateless run
    "agent",             # the graph name from langgraph.json
    input={"messages": [{"role": "human", "content": "hello"}]},
)
print(result["messages"][-1]["content"])
```

`client.runs.wait(...)` runs the graph and waits for the final result, similar to `invoke()` on a local compiled graph. `client.runs.stream(..., stream_mode="tasks")` mirrors `stream()`, accepting the same stream modes covered in Part 3.

`langgraph_sdk` has both an async client (`get_client`, used above with `await`) and a sync client (`get_sync_client`), for code that is not already running inside an event loop:

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://127.0.0.1:2024")
result = client.runs.wait(None, "agent", input={"messages": [{"role": "human", "content": "hello"}]})
```

---

## Part 9: Building for Production — `langgraph build` / `langgraph up`

```bash
langgraph build -t my-agent-image
```

Builds a Docker image containing the graph and its dependencies, using the same `langgraph.json`. This is a heavier, production-shaped artifact compared to the in-memory `langgraph dev` server: it needs `dependencies` to resolve to installable packages (not just a local dev environment that already has everything installed), and it is meant to run unattended.

```bash
langgraph up
```

Runs that image locally with Docker Compose, useful for testing the production build path before deploying it for real. Actual deployment (a hosted, managed version of this) goes through LangSmith Deployment; `langgraph build`/`up` are what a self-hosted deployment, or a deployment platform expecting a Docker image, would use.

---

## Common Beginner Confusions

### Confusion 1: Do I need a LangSmith account to run any of this?

No. `langgraph dev`, the notebook, and `draw_mermaid()` all work with no LangSmith account. LangSmith tracing (Part 4) and LangSmith Deployment (Part 7 onward, for hosted deployment) are the two pieces that need one; running `langgraph dev` locally does not.

### Confusion 2: Why did `draw_mermaid_png()` fail?

It needs network access to a hosted Mermaid rendering service by default. `draw_mermaid()` (text) needs no network and always works; paste its output into the Mermaid Live Editor if you want an image without that dependency.

### Confusion 3: Is `stream_mode="debug"` the same as LangSmith tracing?

They show similar information (what ran, in what order, with what result) but `stream_mode="debug"` is local to the one call you are streaming, printed to your own code. LangSmith tracing is a persistent, hosted record of every run, viewable later, and does not require you to be actively streaming the run yourself.

### Confusion 4: What's the difference between `langgraph dev` and `langgraph up`?

`langgraph dev` runs the graph in-process, in the same Python environment you already have, with hot reloading; it is meant for active development. `langgraph up` runs a Docker image built by `langgraph build`; it is closer to how the graph would actually run once deployed.

### Confusion 5: My `langgraph.json` references `../../.env` — is that safe to commit?

Yes: it is a *path* to the `.env` file, not the file's contents. The actual `.env` file (with real secret values) should never be committed; only `.env.example` (with placeholder values) belongs in version control. `langgraph.json` pointing at a relative path is just configuration.

### Confusion 6: The renamed "LangSmith Deployment" and "LangSmith Studio" — did the CLI commands change too?

No. Only the product names (and some marketing/UI surfaces) changed in October 2025. `langgraph dev`, `langgraph build`, `langgraph up`, and `langgraph.json` are unchanged.

---

## Summary

- `app.get_graph().draw_mermaid()` gives a text diagram of a compiled graph with no network dependency; `.draw_mermaid_png()` needs one.
- `stream_mode="tasks"` and `stream_mode="debug"` show step-by-step execution of a single run, useful while developing.
- LangSmith tracing (three environment variables, no code changes) gives a persistent, hosted record of every run, plus evaluation tooling; it is optional everywhere in this repo.
- LangGraph Platform and LangGraph Studio were renamed to LangSmith Deployment and LangSmith Studio in October 2025; the CLI and config format did not change.
- A deployable app needs a module-level compiled graph (`agent.py`) and a `langgraph.json` naming it.
- `langgraph dev` runs it locally with hot reloading for development; `langgraph build`/`langgraph up` produce and run a production-shaped Docker image; `langgraph-sdk` calls a running server (local or deployed) from code.
