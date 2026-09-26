# LangGraph Fault Tolerance Guide
### Keeping a Graph Running When Something Goes Wrong

---

## Part 1: What Can Go Wrong in a Graph

A LangGraph app is a program made of nodes and edges. Like any program, it can fail in a few different ways:

- A node raises an exception because of a transient problem, like a dropped connection.
- A node raises an exception because of a real bug or bad input.
- A node hangs and never returns, like a slow API call that never comes back.
- A conditional edge loops forever instead of reaching `END`.
- The whole process crashes (the machine restarts, the container gets killed) partway through a run.
- A node is slow and expensive, and gets called again with the exact same input.

LangGraph has a specific tool for each of these:

```text
Transient failure         -> RetryPolicy
Failure after retries     -> error_handler
Node hangs                -> timeout / TimeoutPolicy
Infinite loop              -> recursion_limit
Process crash              -> checkpointer (durable execution)
Repeated expensive input   -> cache_policy + InMemoryCache
```

This guide covers each one. The code matches `1_fault_tolerance.ipynb` in this folder.

---

## Part 2: What This Notebook Builds

```text
22_Fault_Tolerance/
|-- 1_fault_tolerance.ipynb
|-- Fault_Tolerance_Guide.md
```

The notebook builds several small, separate graphs, each demonstrating one fault-tolerance feature with plain Python nodes. Plain Python nodes are used (instead of LLM nodes) so the failure and recovery behavior is easy to see without waiting on an API call. One section shows how the same `retry_policy` argument applies to a node that calls an LLM, since that is the more common real-world case.

Requires `langgraph>=1.2` for `error_handler`, `timeout`/`TimeoutPolicy`, and `set_node_defaults`.

---

## Part 3: Retrying a Flaky Node

Some failures are not really failures of your code. A network call can drop. A downstream service can be briefly overloaded. The right response is usually "try again," not "give up."

```python
from langgraph.types import RetryPolicy

graph.add_node(
    "flaky_node",
    flaky_node,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.01, retry_on=ConnectionError),
)
```

### RetryPolicy Fields

| Field | Meaning |
|---|---|
| `max_attempts` | Total number of tries, including the first one. Default is 3. |
| `initial_interval` | Seconds to wait before the first retry. Default is 0.5. |
| `backoff_factor` | Multiplier applied to the wait time after each retry. Default is 2.0 (exponential backoff). |
| `max_interval` | Upper bound on the wait time between retries. Default is 128.0 seconds. |
| `jitter` | Whether to randomize the wait time slightly, to avoid many retries firing at the exact same moment. Default is `True`. |
| `retry_on` | Which exception type(s) to retry on, or a function that takes the exception and returns `True`/`False`. |

### Why `retry_on` Matters

Without `retry_on`, LangGraph uses a default policy that retries on most exceptions except a short list of programming errors (like `ValueError`, `TypeError`). Setting `retry_on` explicitly is safer for beginners: it makes clear which failures are expected to be transient, and everything else fails immediately instead of being retried and delaying the real error.

```text
retry_on=ConnectionError
    -> only ConnectionError triggers a retry
    -> a ValueError from a real bug fails immediately, as it should
```

### What the Notebook Shows

The notebook's `flaky_node` raises `ConnectionError` on its first two calls and succeeds on the third. With `RetryPolicy(max_attempts=5, retry_on=ConnectionError)`, the output shows:

```text
attempt 1
attempt 2
attempt 3
{'n': 1}
```

Three attempts happened inside a single `invoke()` call. The caller never sees the two failures.

---

## Part 4: A Retry Policy on an LLM Node

LLM API calls fail for the same reasons any network call fails: rate limits, timeouts, transient server errors. `retry_policy` works exactly the same way on a node that calls an LLM as it does on a plain Python node:

```python
from langgraph.types import RetryPolicy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

class ChatState(TypedDict):
    messages: list[BaseMessage]

def llm_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

llm_graph = (
    StateGraph(ChatState)
    .add_node("llm_node", llm_node, retry_policy=RetryPolicy(max_attempts=3))
    .add_edge(START, "llm_node")
    .add_edge("llm_node", END)
    .compile()
)
```

Here the default `retry_on` behavior is left in place, since LLM client libraries usually raise their own specific exception types for rate limits and transient errors rather than a plain `ConnectionError`. Wrapping the LLM call with a retry policy is common enough in production that it is worth doing by default for any node that makes a real network call.

---

## Part 5: Handling a Node That Keeps Failing

Retries only help with failures that go away on their own. If a node is still failing after all its retry attempts (or has no retry policy at all), the failure needs somewhere to go. `error_handler` is that place.

```python
from langgraph.errors import NodeError
from langgraph.types import Command

def handle_boom(state: TaskState, error: NodeError) -> Command:
    print(f"node '{error.node}' failed with: {error.error!r}")
    return Command(update={"status": "recovered"}, goto=END)

graph.add_node("boom_node", boom_node, error_handler=handle_boom)
```

### What the Handler Receives

`error_handler` receives two arguments. The second one must be named exactly `error` and annotated `NodeError` (`def handler(state, error: NodeError)`); LangGraph uses the name and the annotation to know where to put the error, and a plain `def handler(state, error)` raises a `TypeError`:

- `state`: the graph state at the point of failure, same as any node.
- `error`: a `NodeError` object with two fields, `error.node` (the name of the node that failed) and `error.error` (the exception instance itself).

### What the Handler Returns

The handler returns a `Command`, the same object used for dynamic routing elsewhere in LangGraph:

```text
Command(update={...}, goto=...)
    update -> state changes to apply, same shape as a normal node's return value
    goto   -> which node (or END) to go to next
```

This means a handler can not only stop the failure from crashing the run, it can steer the graph to a fallback path, a "please try again" node, or straight to `END` with an error status recorded in state.

### Important Boundaries

- The handler runs **after** retries are exhausted, not instead of them. If both `retry_policy` and `error_handler` are set on the same node, retries happen first.
- If the handler itself raises, the run fails for real. A handler is meant to be simple and reliable.
- `error_handler` set as a default via `set_node_defaults` does not apply to error-handler nodes themselves, so a handler never accidentally catches its own failures.

---

## Part 6: Timing Out a Slow Node

A node that hangs is worse than one that fails fast, because nothing tells you it is stuck. `timeout` cancels a node after a fixed number of seconds.

```python
from langgraph.types import TimeoutPolicy
from langgraph.errors import NodeTimeoutError

async def slow_node(state: SlowState) -> dict:
    await asyncio.sleep(1.0)
    return {"n": state["n"] + 1}

graph.add_node("slow_node", slow_node, timeout=TimeoutPolicy(run_timeout=0.2))
```

### Async Only

Node timeouts only work on **async** nodes (`async def`), run with `ainvoke`/`astream`/`ainvoke` instead of `invoke`/`stream`. A synchronous node cannot be safely interrupted mid-execution from the outside, so LangGraph does not support `timeout` on it. In a Jupyter notebook, `await app.ainvoke(...)` can be called directly at the top level of a cell; you do not need `asyncio.run(...)` there.

### `timeout=` Shorthand vs `TimeoutPolicy`

```python
add_node("slow_node", slow_node, timeout=0.2)
# is shorthand for:
add_node("slow_node", slow_node, timeout=TimeoutPolicy(run_timeout=0.2))
```

`TimeoutPolicy` gives more control:

| Field | Meaning |
|---|---|
| `run_timeout` | Maximum total seconds the node is allowed to run. |
| `idle_timeout` | Maximum seconds allowed between "heartbeats" (progress signals), useful for long-running streaming nodes that are still making progress. |
| `refresh_on` | How the idle timer resets; `"auto"` (default) or `"heartbeat"`. |

### What Happens on Timeout

When a node exceeds its timeout, LangGraph raises `NodeTimeoutError` (from `langgraph.errors`). If the node also has a `retry_policy`, a timeout counts as a failure and can be retried like any other exception, depending on `retry_on`.

```text
caught NodeTimeoutError: Node 'slow_node' exceeded its run timeout of 0.200s (elapsed: 0.219s).
```

---

## Part 7: Stopping Infinite Loops

A conditional edge that never routes to `END` will loop forever unless something stops it. `recursion_limit` is that stop switch, set per run in the config rather than on the graph itself.

```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke({"n": 0}, {"recursion_limit": 5})
except GraphRecursionError as e:
    print(f"caught GraphRecursionError: {e}")
```

### Why It's a Run Config, Not a Graph Setting

`recursion_limit` lives in the config passed to `invoke`/`stream`, not in `add_node` or `compile`. This matters because different callers of the same compiled graph might want different limits: a quick interactive test might use a low limit to fail fast, while a production batch job might need a much higher one for a graph that legitimately takes many steps.

### What Counts as One Step

Each super-step (one round where LangGraph runs all nodes that are ready to run) counts as one step toward the limit. A simple `a -> b -> a -> b -> ...` loop hits the limit quickly; a large graph that fans out to many nodes per super-step can still hit it after relatively few rounds if the limit is set low.

---

## Part 8: Caching a Node's Result

Some nodes are slow but deterministic: the same input always produces the same output, and recomputing it wastes time (and possibly money, for an API call). Node-level caching skips re-running the node when it sees a cached result for the same input.

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

graph.add_node("slow_lookup", slow_lookup, cache_policy=CachePolicy(ttl=60))
compiled = graph.compile(cache=InMemoryCache())
```

Two things are required together:

1. `cache_policy=CachePolicy(ttl=...)` on the specific node that should be cached. `ttl` is how long (in seconds) a cached result stays valid; `None` means it never expires on its own.
2. `cache=InMemoryCache()` on `compile()`. This is where cached results are actually stored. `InMemoryCache` keeps them in process memory, so they do not survive a restart; LangGraph also supports other cache backends for persistent caching across restarts.

### What Gets Cached

The cache key is based on the node's input (the part of state it reads) by default. Two calls with the same input hit the same cache entry; different input runs the node fresh, same as normal.

```text
First call,  n=1  -> node runs,  0.30s, node ran 1 time
Second call, n=1  -> cache hit,  0.00s, node still ran 1 time total
```

### When Not to Cache

Do not cache a node that has side effects meant to happen every time (sending an email, writing to a database) or one whose output should change even for the same input (a node that calls `random` or reads the current time). Caching is for pure, expensive lookups.

---

## Part 9: Setting Defaults for Every Node

Repeating the same `retry_policy` or `timeout` on every `add_node` call becomes noisy once a graph has more than a couple of nodes. `set_node_defaults` sets a policy once for the whole graph.

```python
graph = (
    StateGraph(State)
    .set_node_defaults(retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError))
    .add_node("a", node_a)
    .add_node("b", node_b, retry_policy=custom_retry)  # overrides the default for this node only
    .add_edge(START, "a")
    .compile()
)
```

### Rules Worth Knowing

- A value passed directly to `add_node` always overrides the default for that node.
- Defaults are applied at `compile()` time, not immediately when `set_node_defaults` is called.
- `retry_policy` and `timeout` defaults apply to every node, including error-handler nodes.
- `cache_policy` and `error_handler` defaults apply only to regular nodes. Caching an error handler's result is unsafe (the whole point of a handler is to react to a fresh failure), and a default error handler is never applied to another error handler, so handlers cannot accidentally catch their own failures.
- Defaults set on a parent graph are **not** inherited by subgraphs (see `19_Subgraphs`); each subgraph needs its own defaults if it wants them.

---

## Part 10: Durable Execution With Checkpointers

Everything above handles a failure **inside** a single run: a node throws, times out, or a loop runs too long, but the process itself keeps running the whole time. A checkpointer handles the case where the process itself stops, for example a crash, a deployment restart, or a container getting killed.

`09_Persistence` introduces checkpointers for giving a chatbot conversation memory across turns. The exact same mechanism gives durable execution:

```text
Normal run, no checkpointer:
    Node A runs -> Node B runs -> process crashes -> everything is lost,
    must start over from Node A

Run with a checkpointer:
    Node A runs -> checkpoint saved -> Node B runs -> checkpoint saved -> process crashes
    -> restart, call invoke(None, config) with the same thread_id
    -> resumes from the last checkpoint, Node A is not re-run
```

Concretely:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "job-42"}}
app.invoke(initial_state, config)   # crashes partway through

# after restarting the process:
app.invoke(None, config)            # None = continue from the last checkpoint
```

Every time a node finishes, LangGraph writes a checkpoint (keyed by `thread_id`) before moving to the next node. Calling `invoke(None, config)` with the same `thread_id` picks up from the most recent checkpoint. The input must be `None`: passing the original input again starts a new run from `START`. `InMemorySaver` only persists for the life of the Python process, which is fine for local development; a production deployment uses a durable backend (such as a database-backed checkpointer) so the checkpoint survives the process actually restarting.

### Why This Matters Together With Retries and Timeouts

Retries and timeouts are about a single node failing while the process keeps running. A checkpointer is about the process not running at all for a while. A production graph typically wants both: retries/timeouts so a single bad API call does not fail the whole run, and a checkpointer so a full crash does not lose all the work already done.

---

## Common Beginner Confusions

### Confusion 1: Does `retry_policy` retry the whole graph?

No. It retries a single node. The rest of the graph's already-completed nodes are not re-run.

### Confusion 2: Is `error_handler` the same as a `try`/`except` inside the node?

No. `try`/`except` inside a node is still useful for expected, recoverable conditions you can handle locally. `error_handler` is for failures that escape the node entirely (an uncaught exception), after any retries have been exhausted. Use both where they make sense: catch what you can locally, and let `error_handler` be the safety net for what you cannot.

### Confusion 3: Why doesn't `timeout` work on my synchronous node?

Node timeouts require an `async def` node run with `ainvoke`/`astream`. A synchronous function call cannot be safely cancelled from the outside once it has started; only cooperative, `await`-based code can be.

### Confusion 4: If I set `recursion_limit=5`, does my graph only get 5 nodes?

No. It limits super-steps, not nodes, and a limit of N allows fewer than N super-steps. A graph with 5 sequential nodes and no loops raises `GraphRecursionError` at `recursion_limit=5` and runs at 6. The default limit is large enough for normal graphs, so in practice it only matters for graphs that loop.

### Confusion 5: Does caching remember results forever?

Only if `ttl=None`. Otherwise, a cached result expires after `ttl` seconds and the node runs fresh again the next time it is called.

### Confusion 6: Is a checkpointer the same thing as caching?

No. A checkpointer saves the state of an in-progress *run* so it can resume after a crash. A cache saves the *result of a specific node call* so it does not have to be recomputed. They solve different problems and are often used together.

---

## Summary

LangGraph gives each kind of failure its own tool, so a graph does not need one giant `try`/`except` wrapped around everything:

```text
RetryPolicy      -> retry a node on specific, expected exceptions
error_handler    -> react to a node failing for good, after retries
timeout / TimeoutPolicy -> stop a hung async node
recursion_limit  -> stop an infinite loop, per run
CachePolicy + InMemoryCache -> skip re-running an expensive, deterministic node
set_node_defaults -> apply any of the above to every node in a graph at once
checkpointer     -> survive the whole process crashing, by resuming from the last checkpoint
```

Together, these turn "one bad API call crashes my whole app" into "one bad API call gets retried, and if it truly fails, the graph reacts to it on purpose."
