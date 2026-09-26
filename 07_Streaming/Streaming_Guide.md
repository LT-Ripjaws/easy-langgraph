# LangGraph Streaming Guide
### Watching a Graph Run Instead of Waiting for It

---

## Part 1: `invoke` vs. `stream`

Every earlier notebook in this repo calls `graph.invoke(...)`. `invoke` runs the
whole graph and hands back the final state once everything is done:

```python
result = bot.invoke({"messages": [HumanMessage(content="hi")]})
# nothing happens until the entire run finishes
```

`graph.stream(...)` runs the same graph, but instead of waiting silently, it
yields something after every step while the run is still in progress:

```python
for chunk in bot.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)
# prints something after each node finishes, before the whole run is done
```

What exactly gets yielded, and how often, depends on `stream_mode`. This
notebook works through the modes a beginner is most likely to need:
`"values"`, `"updates"`, `"messages"`, `"custom"`, and passing more than one
mode at once.

---

## Part 2: What This Notebook Builds

```text
07_Streaming/
|-- 1_streaming.ipynb
|-- Streaming_Guide.md
```

A two-node graph: `outline_step` (no LLM call, just builds a short instruction
string) followed by `write_poem` (one LLM call that writes a short poem from
that instruction). The same compiled graph is streamed five times, once per
mode being demonstrated, so each section can be read on its own.

---

## Part 3: The Graph

```python
class PoemState(TypedDict):
    topic: str
    outline: str
    poem: str


def outline_step(state: PoemState) -> dict:
    writer = get_stream_writer()
    writer({"progress": f"outlining a poem about {state['topic']}..."})
    return {"outline": f"A cheerful 3-line poem about {state['topic']}."}


def write_poem(state: PoemState) -> dict:
    writer = get_stream_writer()
    writer({"progress": "writing the poem..."})

    response = llm.invoke(state["outline"])
    return {"poem": response.content}
```

```python
poem_graph = StateGraph(PoemState)
poem_graph.add_node("outline_step", outline_step)
poem_graph.add_node("write_poem", write_poem)
poem_graph.add_edge(START, "outline_step")
poem_graph.add_edge("outline_step", "write_poem")
poem_graph.add_edge("write_poem", END)

poem_app = poem_graph.compile()
```

```text
START -> outline_step -> write_poem -> END
```

`outline_step` needs no LLM call at all — it just formats a short instruction
string. Only `write_poem` costs an API call. Both nodes call
`get_stream_writer()` and send a short progress message; that call has nothing
to do with graph state and is explained in Part 7.

---

## Part 4: `stream_mode="values"`

```python
for chunk in poem_app.stream({"topic": "cats", "outline": "", "poem": ""}, stream_mode="values"):
    print(chunk)
```

Each yielded item is the **complete state dict** as it stood right after a
node finished. With a two-node graph, this yields three times: once for the
input itself, once after `outline_step`, once after `write_poem`.

```text
{'topic': 'cats', 'outline': '', 'poem': ''}
{'topic': 'cats', 'outline': 'A cheerful 3-line poem about cats.', 'poem': ''}
{'topic': 'cats', 'outline': 'A cheerful 3-line poem about cats.', 'poem': '<the poem>'}
```

This mode is easiest to reason about because you always see the whole picture,
but it repeats fields that did not change on every yield.

---

## Part 5: `stream_mode="updates"`

```python
for chunk in poem_app.stream({"topic": "robots", "outline": "", "poem": ""}, stream_mode="updates"):
    print(chunk)
```

Each yielded item is a dict keyed by node name, containing only what that node
returned:

```text
{'outline_step': {'outline': 'A cheerful 3-line poem about robots.'}}
{'write_poem': {'poem': '<the poem>'}}
```

This is usually the more useful default for logging or debugging a run,
because it tells you exactly which node produced which change, without
repeating the parts of state that stayed the same.

---

## Part 6: `stream_mode="messages"` — Token-by-Token Output

```python
for message_chunk, metadata in poem_app.stream(
    {"topic": "dogs", "outline": "", "poem": ""}, stream_mode="messages"
):
    if metadata["langgraph_node"] == "write_poem" and message_chunk.content:
        print(message_chunk.content, end="", flush=True)
print()
```

This mode streams the LLM's output as it is generated, piece by piece, instead
of waiting for the whole response. Each yielded item is a **tuple**:

```text
(message_chunk, metadata)
```

| Part | What it is |
|---|---|
| `message_chunk` | A partial `AIMessageChunk`; `.content` is the next piece of text |
| `metadata` | A dict describing where the chunk came from |

`metadata["langgraph_node"]` names the node that produced the chunk. In this
notebook only `write_poem` calls the LLM, so the `if` check above is always
true for the chunks that matter — but the check is not decoration. In a graph
with more than one LLM-calling node (a classifier node and a writer node, say),
every chunk from every LLM call arrives on the same stream, and
`metadata["langgraph_node"]` is how you tell them apart and print only the one
you actually want to show the user.

`11_Langgraph_Chatbot/frontend.py` already uses this exact mode in a running
Streamlit app:

```python
ai_message = st.write_stream(
    message_chunk.content for message_chunk, metadata in bot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode='messages'
    )
)
```

That app has only one LLM-calling node, so it does not filter by
`langgraph_node` — every chunk on the stream is the one it wants. This
notebook's version adds the filter because the pattern is worth seeing once,
even in a graph where it happens to always be true.

---

## Part 7: `stream_mode="custom"` — Progress From Inside a Node

```python
def outline_step(state: PoemState) -> dict:
    writer = get_stream_writer()
    writer({"progress": f"outlining a poem about {state['topic']}..."})
    return {"outline": f"A cheerful 3-line poem about {state['topic']}."}
```

`get_stream_writer()` returns a function. Calling it sends whatever you pass —
here a small dict — straight to anyone consuming the stream with
`stream_mode="custom"`. It is a side channel:

```text
It does NOT change graph state.
It does NOT appear in "values" or "updates" output.
It only shows up when something is streaming with stream_mode="custom".
```

```python
for chunk in poem_app.stream({"topic": "the ocean", "outline": "", "poem": ""}, stream_mode="custom"):
    print(chunk)
```

```text
{'progress': 'outlining a poem about the ocean...'}
{'progress': 'writing the poem...'}
```

This is the mechanism for a node to say "I'm 30% done" or "searching the web
now" to a UI, independent of whatever data it eventually returns into state.
Both nodes call `get_stream_writer()` here, including `outline_step`, which
makes no LLM call at all — custom events do not require an LLM in the node.

---

## Part 8: Multiple Modes at Once

```python
for mode, chunk in poem_app.stream(
    {"topic": "pizza", "outline": "", "poem": ""}, stream_mode=["updates", "custom"]
):
    print(mode, "->", chunk)
```

Passing a **list** to `stream_mode` merges the requested streams into one.
Each yielded item becomes a `(mode, chunk)` tuple instead of just `chunk`, so
you can tell which stream a given item came from:

```text
custom -> {'progress': 'outlining a poem about pizza...'}
updates -> {'outline_step': {'outline': 'A cheerful 3-line poem about pizza.'}}
custom -> {'progress': 'writing the poem...'}
updates -> {'write_poem': {'poem': '<the poem>'}}
```

This is the usual way to build a real UI: `"custom"` drives a progress
indicator while `"updates"` (or `"messages"`) drives the actual content.

---

## Part 9: A Note on `graph.stream_events` (LangGraph 1.2)

LangGraph 1.2 also ships a higher-level streaming method,
`graph.stream_events(input, version="v3")`. It is marked experimental (it prints a
`LangChainBetaWarning`), so its details may still change. Instead of matching on
`stream_mode` strings and unpacking tuples yourself, it returns typed
projections you can iterate directly, such as `stream.messages` for
token-by-token output or `stream.values` for full-state snapshots — the same
underlying data as `stream_mode`, organized so you do not have to branch on a
mode string. `stream_mode` itself also accepts two more values not covered in
this notebook: `"checkpoints"` (yields the checkpoint saved after each step; the
checkpoints themselves are explained in `10_Time_Travel`) and `"tasks"` (yields task start/finish events for
each node, useful for tracing parallel branches), plus `"debug"` (a verbose
mode that includes internal execution detail). This notebook sticks to
`"values"`, `"updates"`, `"messages"`, and `"custom"` because they cover what a
beginner needs first; reach for `stream_events` or the extra modes once the
basics here feel routine.

---

## Part 10: Common Beginner Confusions

### Confusion 1: Does `stream_mode="messages"` only work with chat models?

It streams whatever the underlying LLM client supports streaming. For a chat
model like `ChatGoogleGenerativeAI`, that means token-by-token (or small
chunk-by-chunk) text as `AIMessageChunk` objects.

### Confusion 2: Why is `"messages"` a tuple and not just the chunk?

Because a graph can have more than one node calling an LLM, or more than one
LLM call inside the same node. The `metadata` half of the tuple is what lets
you tell chunks apart — `metadata["langgraph_node"]` is the one field most
beginners need.

### Confusion 3: Does `stream_mode="custom"` need `get_stream_writer()` in
every node?

No. Only nodes that call `writer(...)` produce custom events. A node that
never calls `get_stream_writer()` simply contributes nothing to a
`stream_mode="custom"` run.

### Confusion 4: Can I use `stream_mode="custom"` without an LLM at all?

Yes — `outline_step` in this notebook proves it. Custom events are just a
function call inside a node; nothing about them requires an LLM to be
involved.

### Confusion 5: If I pass a list to `stream_mode`, do I still get separate
loops for each mode?

No — one loop, one stream. Each item is tagged with which mode produced it,
`(mode, chunk)`, instead of getting two separate `for` loops.

---

## Summary

`graph.stream(...)` lets you observe a run while it happens, instead of only
seeing the final state.

```text
stream_mode="values"    -> full state after each step
stream_mode="updates"   -> only what changed, per node
stream_mode="messages"  -> LLM output, token by token, as (chunk, metadata)
stream_mode="custom"    -> whatever a node sends with get_stream_writer()
stream_mode=[a, b]      -> both streams merged, as (mode, chunk)
```

| Piece | Role |
|---|---|
| `stream_mode="values"` | See the whole state at every step |
| `stream_mode="updates"` | See only what each node changed |
| `stream_mode="messages"` | Stream LLM tokens as they are generated |
| `metadata["langgraph_node"]` | Tells which node a message chunk came from |
| `get_stream_writer()` | Sends a custom progress event from inside a node |
| `stream_mode="custom"` | Reads back whatever `get_stream_writer()` sent |
| `stream_mode=[...]` | Merges several modes into one `(mode, chunk)` stream |

Pick the mode that matches what you are trying to show: `"updates"` for logs,
`"messages"` for a typing-style chat UI, `"custom"` for progress indicators,
and a list of modes when a UI needs more than one of these at once.
