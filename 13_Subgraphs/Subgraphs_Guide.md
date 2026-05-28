# Subgraphs in LangGraph

This folder shows how to build a small graph, compile it, and use it as a node inside another graph.

## Files

```text
13_Subgraphs/
|-- Subgraphs.ipynb
|-- Subgraphs_Guide.md
```

## What Is a Subgraph?

A subgraph is a LangGraph graph used inside another LangGraph graph.

Instead of making one large workflow:

```text
START -> clean_notes -> make_bullets -> write_report -> END
```

You can group part of the logic:

```text
Parent graph:
START -> notes_subgraph -> write_report -> END

notes_subgraph:
START -> clean_notes -> make_bullets -> END
```

This keeps larger workflows easier to read and reuse.

## What the Notebook Builds

The notebook creates:

1. A subgraph that cleans raw notes and turns them into bullet points.
2. A parent graph that calls the subgraph and then writes a final report.

## Shared State

Both the parent graph and subgraph use the same state schema:

```python
class ReportState(TypedDict, total=False):
    topic: str
    raw_notes: str
    cleaned_notes: str
    bullet_points: list[str]
    final_report: str
```

Because they share state keys, the parent can pass `raw_notes` into the subgraph, and the subgraph can return `bullet_points` for the parent to use.

## Building the Subgraph

```python
notes_builder = StateGraph(ReportState)

notes_builder.add_node("clean_notes", clean_notes)
notes_builder.add_node("make_bullets", make_bullets)

notes_builder.add_edge(START, "clean_notes")
notes_builder.add_edge("clean_notes", "make_bullets")
notes_builder.add_edge("make_bullets", END)

notes_subgraph = notes_builder.compile()
```

The compiled `notes_subgraph` behaves like a node from the parent graph's point of view.

## Using the Subgraph in a Parent Graph

```python
parent_builder = StateGraph(ReportState)

parent_builder.add_node("notes_subgraph", notes_subgraph)
parent_builder.add_node("write_report", write_report)

parent_builder.add_edge(START, "notes_subgraph")
parent_builder.add_edge("notes_subgraph", "write_report")
parent_builder.add_edge("write_report", END)

report_graph = parent_builder.compile()
```

The parent does not need to know the internal details of the subgraph. It only sees the state updates that come out of it.

## Why Use Subgraphs?

Use subgraphs when:

- A workflow section has multiple steps.
- You want to reuse the same mini-workflow in multiple places.
- A parent graph is becoming hard to read.
- You want separate teams or files to own different graph sections.
- You are building multi-agent systems where each agent has its own graph.

## Persistence Note

Subgraphs can interact with checkpointing.

Common modes:

| Subgraph compile option | Behavior |
|-------------------------|----------|
| `checkpointer=None` | Default per-invocation behavior; good for most subgraph calls |
| `checkpointer=True` | Keep subgraph memory across calls in the same thread |
| `checkpointer=False` | Run like a plain function call with no checkpointing |

For simple deterministic demos, compiling without a checkpointer is usually enough.

## Common Pitfalls

### No Shared State Keys

If the parent and subgraph use different state schemas, they need a wrapper node to translate input and output.

### Too Much Hidden Logic

Subgraphs help organize complexity, but a deeply nested graph can become hard to debug. Keep names clear.

### Checkpoint Confusion

If you call the same checkpointed subgraph multiple times in one node, you can run into checkpoint namespace conflicts. Use stateless subgraphs or separate graph calls when needed.

## References

- LangGraph subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- LangGraph graph API: https://docs.langchain.com/oss/python/langgraph/graph-api

## Summary

Core pattern:

```text
Build child StateGraph -> compile child graph -> add child graph as parent node
```

Subgraphs are a clean way to turn complex workflows into smaller, reusable graph modules.
