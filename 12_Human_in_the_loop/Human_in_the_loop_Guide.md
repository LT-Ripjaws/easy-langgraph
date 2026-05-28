# Human-in-the-Loop in LangGraph

This folder shows how to pause a LangGraph workflow, wait for human input, and then resume from the same point.

## Files

```text
12_Human_in_the_loop/
|-- Human_in_the_loop.ipynb
|-- Human_in_the_loop_Guide.md
```

## What the Notebook Builds

The notebook creates a tiny approval workflow:

```text
START -> create_draft -> human_review -> finalize -> END
```

The workflow:

1. Creates a simple draft message.
2. Pauses at `human_review`.
3. Waits for a human approval or rejection.
4. Resumes and produces the final result.

## Key Concepts

### `interrupt()`

```python
review = interrupt({
    "question": "Approve this draft?",
    "draft": state["draft"],
})
```

`interrupt()` pauses graph execution and returns its payload to the caller. The graph cannot continue until it is resumed.

### `Command(resume=...)`

```python
app.invoke(
    Command(resume={"approved": True, "feedback": "Looks good."}),
    config=config,
)
```

The resume value becomes the return value of `interrupt()` inside the paused node.

### Checkpointer

```python
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
```

Human-in-the-loop needs checkpointing because LangGraph must save where execution paused.

### Thread ID

```python
config = {"configurable": {"thread_id": "hitl-demo-1"}}
```

The same `thread_id` must be used when starting and resuming the graph.

## Why This Matters

Human-in-the-loop is useful when an AI workflow should not continue automatically.

Use it for:

- Approving emails before sending
- Reviewing tool calls before execution
- Editing generated text before publishing
- Confirming database updates
- Checking sensitive or high-impact decisions

## Important Notes

- `interrupt()` payloads should be JSON-serializable.
- Use a durable checkpointer, such as SQLite or PostgreSQL, for production.
- Code before an `interrupt()` can run again on resume, so avoid non-idempotent side effects before the pause.
- The demo uses `MemorySaver`, so pause state is lost if the Python process restarts.

## References

- LangGraph interrupts: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/
- LangChain HITL docs: https://docs.langchain.com/oss/python/langgraph/human-in-the-loop

## Summary

Core pattern:

```text
Run graph -> interrupt -> human decision -> Command(resume=...) -> continue graph
```

This turns a normal graph into an interactive workflow where humans can approve, reject, or edit important steps.
