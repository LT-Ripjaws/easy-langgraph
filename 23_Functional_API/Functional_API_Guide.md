# LangGraph Functional API Guide
### Writing a Workflow as Plain Python Instead of a Graph

---

## Part 1: Two Ways to Build the Same Thing

Every earlier lesson in this repo uses the Graph API: define a state shape, write node functions, wire them together with `add_node`/`add_edge`, compile, run. That is a good fit when a workflow branches, loops, or has many possible paths between steps.

LangGraph also has a Functional API: `@task` and `@entrypoint`. Instead of nodes and edges, you write a function that calls other functions, the way you would write any Python script. It compiles down to the exact same runtime as the Graph API: same checkpointer support, same `interrupt`/resume mechanism, same streaming.

```text
Graph API:                          Functional API:

StateGraph(State)                   @task
  .add_node("a", node_a)            def step_one(x): ...
  .add_node("b", node_b)
  .add_edge(START, "a")             @task
  .add_edge("a", "b")               def step_two(x): ...
  .add_edge("b", END)
  .compile()                        @entrypoint(checkpointer=...)
                                     def workflow(x):
                                         y = step_one(x).result()
                                         return step_two(y).result()
```

Neither one replaces the other. This guide covers when the Functional API is the better fit, and the rules that make it work correctly.

---

## Part 2: What This Notebook Builds

```text
23_Functional_API/
|-- 1_functional_api.ipynb
|-- Functional_API_Guide.md
```

The notebook builds a small essay-writing workflow:

1. Write an outline (one LLM call, as a `@task`).
2. Write every section of the outline in parallel (several LLM calls, each a `@task`, all started before any of them are awaited).
3. Pause with `interrupt()` for a human to approve the draft.
4. Resume with `Command(resume=...)` and return the finished essay with `entrypoint.final`.

It also includes a short, LLM-free demonstration of the determinism rule (Part 6 below), since that rule is easy to state and easy to get wrong.

Requires `langgraph>=1.2`. Imports used: `from langgraph.func import entrypoint, task`, `from langgraph.checkpoint.memory import InMemorySaver`, `from langgraph.types import interrupt, Command`.

---

## Part 3: `@task` — A Unit of Work That Returns a Future

```python
from langgraph.func import task

@task
def write_outline(topic: str) -> str:
    response = llm.invoke(f"List 3 short section titles for an essay about '{topic}'.")
    return response.content
```

Calling `write_outline("cats")` does **not** run the function and hand back a string. It schedules the work and returns a future-like object immediately. You get the actual value by calling `.result()`:

```python
future = write_outline("cats")   # scheduled, not finished yet
outline_text = future.result()   # blocks until it finishes, then returns the string
```

This is a deliberate difference from a normal function call. It is also the entire mechanism behind running things in parallel.

---

## Part 4: Parallel Tasks — Call First, Read Results After

```python
@task
def write_section(topic: str, section_title: str) -> str:
    response = llm.invoke(f"Write a short paragraph about '{section_title}' for an essay on '{topic}'.")
    return response.content

# Call every task first...
futures = [write_section(topic, title) for title in section_titles]

# ...then read every result. All three LLM calls above run concurrently.
sections = [f.result() for f in futures]
```

There is no special "parallel" keyword or decorator argument. The pattern is just: don't call `.result()` until after you have started everything you want to run at the same time.

```text
Sequential (slow):
    write_section(a).result()  -> waits
    write_section(b).result()  -> waits
    write_section(c).result()  -> waits
    total time ~ a + b + c

Parallel (fast):
    fa = write_section(a)      -> scheduled
    fb = write_section(b)      -> scheduled
    fc = write_section(c)      -> scheduled
    fa.result(), fb.result(), fc.result()
    total time ~ max(a, b, c)
```

---

## Part 5: `@entrypoint` — The Workflow Function

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def essay_workflow(topic: str):
    outline_text = write_outline(topic).result()
    ...
```

`@entrypoint` marks the function LangGraph actually runs as a workflow. A few things follow from that:

- It needs a `checkpointer`, the same as a graph that wants memory or `interrupt` support. `InMemorySaver` works for local development; it does not persist across process restarts.
- It takes a `config` with a `thread_id`, exactly like `StateGraph.compile(checkpointer=...)` does:

```python
config = {"configurable": {"thread_id": "essay-1"}}
essay_workflow.invoke("the benefits of plain language", config)
```

- It supports `.invoke(...)` and `.stream(...)`, the same as a compiled graph.

---

## Part 6: `interrupt()` and `Command(resume=...)`

`18_Human_in_the_loop` introduces `interrupt()` for the Graph API. It works the same way inside an entrypoint:

```python
from langgraph.types import interrupt, Command

@entrypoint(checkpointer=InMemorySaver())
def essay_workflow(topic: str):
    ...
    approved = interrupt({"sections": sections})
    if not approved:
        return {"approved": False}
    ...
```

Calling `essay_workflow.stream(topic, config)` runs up to the `interrupt()` call and stops, yielding a final `{"__interrupt__": (...)}` chunk with the value passed to `interrupt()`. Whatever is driving the workflow (a human reviewing output, a UI waiting for a click) reads that value and decides what to send back.

```python
result = essay_workflow.invoke(Command(resume=True), config)
```

`Command(resume=True)` sends `True` back as the return value of the `interrupt()` call, and the entrypoint continues from there.

---

## Part 7: `entrypoint.final` — Returning One Value, Saving Another

Normally, whatever an entrypoint returns is both the value handed back to the caller and the value saved as that thread's state. `entrypoint.final` lets you split those two:

```python
@entrypoint(checkpointer=InMemorySaver())
def my_workflow(number: int, *, previous: int = None):
    previous = previous or 0
    return entrypoint.final(value=previous, save=2 * number)
```

- `value` is returned to whoever called `.invoke()`/`.stream()` right now.
- `save` becomes the `previous` argument the *next* time this same `thread_id` is invoked.

```text
my_workflow.invoke(3, config)  -> returns 0   (previous was None the first time)
my_workflow.invoke(1, config)  -> returns 6   (previous was 3 * 2 = 6, saved last time)
```

In the essay workflow, `entrypoint.final(value={"approved": True, "essay": essay}, save=sections)` returns the finished essay to the caller while saving the individual section drafts as the thread's state, in case a later step in a larger workflow wanted to reuse them without regenerating them.

---

## Part 8: The Determinism Rule

This is the rule that makes the Functional API safe to use with `interrupt()` and checkpointers:

> On resume, LangGraph re-runs the entrypoint's Python body from the beginning. Only `@task` results are cached and skipped on replay.

That means:

```text
Inside @entrypoint, directly:
    time.time(), random.random(), an API call, a database write
    -> runs AGAIN every time the entrypoint resumes
    -> can produce a different value than it did the first time
    -> can duplicate a side effect (e.g. sending an email twice)

Inside a @task:
    same operations
    -> the FIRST run's result is cached
    -> a resume returns the cached result instead of re-executing
```

**The rule: side effects and non-deterministic calls belong inside `@task`, never directly in the `@entrypoint` body.**

The notebook demonstrates this with plain counters instead of an LLM:

```python
counters = {"entry_runs": 0, "task_runs": 0}

@task
def counted_task(x: int) -> int:
    counters["task_runs"] += 1
    return x * 2

@entrypoint(checkpointer=InMemorySaver())
def demo_workflow(x: int):
    counters["entry_runs"] += 1
    doubled = counted_task(x).result()
    interrupt("pause for review")
    return doubled
```

After streaming once and then resuming with `Command(resume=True)`, `counters["entry_runs"]` is `2` (the entrypoint body ran twice: once up to the interrupt, once again on resume) while `counters["task_runs"]` stays at `1` (the task's result was cached and not recomputed). The essay workflow relies on exactly this: the three parallel `write_section` calls are not repeated when the workflow resumes after the human approval step.

---

## Part 9: Tasks Support the Same Fault-Tolerance Policies as Nodes

`22_Fault_Tolerance` covers `RetryPolicy`, `CachePolicy`, and `timeout` for graph nodes. `@task` accepts the same three arguments, because a task is running on the same underlying execution engine as a node:

```python
from langgraph.types import RetryPolicy, CachePolicy

@task(retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError))
def call_flaky_api(x: int) -> int:
    ...

@task(cache_policy=CachePolicy(ttl=60))
def expensive_lookup(key: str) -> dict:
    ...
```

As with graph nodes, a `cache_policy` only takes effect when a cache is attached, here on
the entrypoint: `@entrypoint(checkpointer=..., cache=InMemoryCache())`
(`from langgraph.cache.memory import InMemoryCache`).

There is no separate `error_handler` for tasks; a task that fails after its retries are exhausted raises normally inside the entrypoint, where a regular `try`/`except` around `.result()` can handle it.

---

## Part 10: Graph API vs Functional API

| | Graph API (`StateGraph`) | Functional API (`entrypoint`/`task`) |
|---|---|---|
| Unit of work | A node function, registered with `add_node` | A `@task` function, called directly like any function |
| Control flow | Edges and conditional edges, wired explicitly | Plain Python: `if`, `for`, calling functions in order |
| Shared state | A typed state object every node reads and returns updates to | Local variables inside the entrypoint function |
| Parallel work | Multiple nodes reached from the same edge (see `03_Parallel_Workflows`) | Call several tasks before reading any `.result()` |
| Visualizing the flow | `get_graph().draw_mermaid()` (see `24_Observability_and_Deployment`) | No diagram; the flow is the Python source itself |
| Human-in-the-loop | `interrupt()` inside a node | `interrupt()` inside the entrypoint |
| Memory / durability | `compile(checkpointer=...)` | `entrypoint(checkpointer=...)` |
| Best fit | Workflows with branching, loops, or many possible paths | Workflows that read like a normal script, with a few parallel steps |

Both compile to the same underlying LangGraph runtime, so switching between them later (or mixing them, by calling a compiled graph from inside a task) does not mean starting over.

---

## Common Beginner Confusions

### Confusion 1: Is a `@task` the same as a node?

Similar in spirit (a named unit of work), but a task is called directly like a function and returns a future. A node is registered separately with `add_node` and is invoked automatically by the graph's edges.

### Confusion 2: Does calling a `@task` function block?

No. Calling it schedules the work and returns immediately. `.result()` is what blocks, waiting for the value.

### Confusion 3: If I forget `.result()`, does the task still run?

If nothing in the entrypoint ever calls `.result()` on a future, LangGraph still needs the task to complete before the run can finish, but you will not get its return value inside your own code. Call `.result()` wherever you actually need the value.

### Confusion 4: Why does my `@entrypoint` function seem to "run twice" when I add logging?

Because it does, by design, whenever there's an `interrupt()` (or any resume) in it. See Part 8. Logging or side effects placed directly in the entrypoint body will appear to happen more than once; move them into a `@task` if they should only happen once.

### Confusion 5: Do I need a `checkpointer` for every entrypoint?

Only if you need `interrupt()`, memory across calls, or `entrypoint.final`'s `previous` argument. A one-shot entrypoint with no pausing and no memory can skip it, the same way a compiled graph can skip a checkpointer if it does not need memory.

---

## Summary

- `@task` marks a function as a schedulable unit of work; calling it returns a future, and `.result()` gets the value.
- Calling several tasks before reading any `.result()` runs them in parallel.
- `@entrypoint(checkpointer=...)` marks the workflow function; it takes a `thread_id` in its config like a compiled graph does.
- `interrupt(value)` pauses the workflow and surfaces `value`; `Command(resume=...)` sends a value back in and continues.
- `entrypoint.final(value=..., save=...)` returns one value now while saving a different value as the thread's state for next time.
- The entrypoint body re-runs from the top on every resume; only `@task` results are cached. Keep side effects and non-deterministic calls inside `@task`.
- The Functional API and Graph API are two ways to describe the same underlying workflow; pick whichever one reads more naturally for the shape of the problem.
