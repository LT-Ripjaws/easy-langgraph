# Short-Term Memory in LangGraph

Short-term memory in LangGraph means memory scoped to one conversation thread. It lets a graph remember what happened earlier in the same session, such as previous user messages, assistant replies, tool results, uploaded files, retrieved documents, intermediate decisions, or temporary task state.

In one sentence:

```text
Short-term memory = graph state saved and restored by a checkpointer for one thread_id.
```

---

## 1. Why Short-Term Memory Exists

LLM calls are naturally stateless. If you call a model twice, the second call does not automatically know what happened in the first call.

Without short-term memory:

```text
Turn 1: User says "My name is Bob"
Turn 2: User asks "What is my name?"
Model sees only turn 2, so it does not know.
```

With short-term memory:

```text
Turn 1 is saved into thread state.
Turn 2 loads the same thread state.
The model sees the prior conversation and can answer.
```

LangGraph solves this by saving graph state after graph steps and loading that state again when the same `thread_id` is used.

---

## 2. Core Idea

Short-term memory has four moving parts:

| Part | Meaning |
|------|---------|
| State | The data your graph carries between nodes |
| Reducers | Rules for merging state updates, such as appending messages |
| Checkpointer | Persistence layer that saves state snapshots |
| Thread ID | The session key used to load the right memory |

The common chatbot version looks like this:

```text
messages field in state
        +
add_messages reducer
        +
checkpointer
        +
thread_id
        =
multi-turn conversation memory
```

---

## 3. What Is Stored

Short-term memory is not limited to chat messages. It can store anything that belongs to the current thread.

Common examples:

- Conversation history
- Tool call results
- Current plan
- Retrieved documents
- Uploaded file references
- Temporary form values
- Human approval state
- Generated drafts
- Agent scratchpad
- Routing decisions
- Error recovery metadata

The important rule is scope:

```text
If the data belongs only to this conversation or task thread, it belongs in short-term memory.
```

If the data should be shared across different conversations, it is probably long-term memory instead.

---

## 4. How LangGraph Implements It

LangGraph short-term memory is implemented through checkpointing.

When you compile a graph with a checkpointer, LangGraph saves state snapshots during execution. The snapshots are organized by `thread_id`.

High-level flow:

```text
1. User invokes graph with input and thread_id.
2. Checkpointer loads the latest state for that thread.
3. Graph nodes run using the restored state.
4. Nodes return partial state updates.
5. Reducers merge updates into the state.
6. Checkpointer saves the new state.
7. The next call with the same thread_id resumes from that state.
```

This is why the same graph can support many separate conversations:

```text
thread_id = "alice" -> Alice's state
thread_id = "bob"   -> Bob's state
thread_id = "task7" -> Task 7's state
```

Each thread has its own checkpoint history.

---

## 5. The Role of State

State is the shape of memory.

For a chatbot, the most important state key is usually `messages`.

Conceptually:

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `messages` field stores the current conversation history. The `add_messages` reducer tells LangGraph to append new messages instead of replacing the entire list.

Without a reducer, this can happen:

```text
old messages -> overwritten by new message
```

With `add_messages`, this happens:

```text
old messages + new user message + new assistant message
```

Reducers matter because LangGraph nodes normally return partial updates, not the full state.

---

## 6. The Role of the Checkpointer

The checkpointer is the storage mechanism for short-term memory.

Common checkpointers:

| Checkpointer | Use Case |
|--------------|----------|
| `InMemorySaver` / `MemorySaver` | Local demos and experiments |
| `SqliteSaver` | Local persistence across restarts |
| `PostgresSaver` | Production, multi-user apps |
| `RedisSaver` | Fast distributed systems |

The current LangGraph docs use `InMemorySaver` for the in-memory checkpointer. Some older examples and this project use `MemorySaver`. The concept is the same: memory is kept in process memory and lost when the process restarts.

Production short-term memory should use a durable database-backed checkpointer.

---

## 7. The Role of `thread_id`

The `thread_id` is the lookup key for short-term memory.

Example shape:

```python
config = {"configurable": {"thread_id": "user-123-session-1"}}
```

LangGraph uses this ID to find the right checkpoints.

Same thread ID:

```text
continues the same conversation
```

Different thread ID:

```text
starts or resumes a different conversation
```

Important:

- Always pass a stable `thread_id` when using a checkpointer.
- Do not hardcode one thread ID for every user.
- Use separate thread IDs for separate conversations.
- Store thread IDs in your UI, database, or session system.

---

## 8. Checkpoints and History

A checkpoint is a saved snapshot of the graph state.

LangGraph saves checkpoints at graph step boundaries. In a sequential graph:

```text
START -> node_a -> node_b -> END
```

LangGraph can save separate snapshots around the input step, `node_a`, and `node_b`.

This enables:

- Conversation memory
- Human-in-the-loop resume
- Time travel debugging
- Recovery after failures
- State inspection

Useful operations:

| Operation | Purpose |
|-----------|---------|
| `graph.get_state(config)` | See the latest state for a thread |
| `graph.get_state_history(config)` | See historical checkpoints |
| `graph.update_state(config, values)` | Create a new checkpoint with edited state |
| `checkpointer.delete_thread(thread_id)` | Delete all checkpoints for a thread |

Short-term memory is therefore not just "chat history." It is a checkpointed state timeline.

---

## 9. How It Works in Chatbots

A typical chatbot graph uses this loop:

```text
User sends a new message
        |
        v
Graph loads prior messages for thread_id
        |
        v
Chat node calls model with message history
        |
        v
Model returns assistant message
        |
        v
add_messages appends response
        |
        v
Checkpointer saves updated state
```

The UI does not need to resend the entire conversation every time. It can send only the newest message and the `thread_id`.

LangGraph reconstructs the thread state from the checkpoint.

---

## 10. Short-Term Memory vs Long-Term Memory

| Question | Short-Term Memory | Long-Term Memory |
|----------|-------------------|------------------|
| Scope | One thread | Across threads |
| Storage API | Checkpointer | Store |
| Main key | `thread_id` | Namespace and key |
| Typical data | Conversation history, temporary state | User facts, preferences, learned instructions |
| Automatically part of graph state | Yes | No, must be read/written intentionally |
| Example | "Earlier in this chat, user said X" | "This user prefers concise answers" |

Short-term memory answers:

```text
What has happened in this conversation?
```

Long-term memory answers:

```text
What should I remember about this user or app across conversations?
```

---

## 11. Managing Long Conversations

Short-term memory can grow too large.

If every message is saved forever and passed to the LLM, problems appear:

- Context window overflow
- Higher cost
- Slower responses
- Worse model focus
- Stale information distracting the model
- Tool call message ordering issues

LangGraph gives you several strategies.

## Trim Messages

Trimming means sending only part of the message history to the model while still keeping the graph state.

Common strategies:

- Keep the last N messages
- Keep messages under a token budget
- Keep the latest user turn and necessary tool result pairs
- Keep system instructions and recent conversation

This is simple and fast, but old details may become invisible to the model.

## Delete Messages

Deleting means removing messages from the graph state itself.

LangGraph supports `RemoveMessage` when the state uses a compatible message reducer such as `add_messages`.

Use deletion when:

- The history is too large
- You need to remove sensitive content
- You want to reset a conversation
- You want permanent state cleanup

Be careful with deletion. Some model providers expect valid message order, and tool calls usually need matching tool results.

## Summarize Messages

Summarization compresses older messages into a shorter summary.

Pattern:

```text
older messages -> summary field
recent messages remain in messages
```

This keeps important context without passing every old message to the LLM.

Summarization is useful when:

- Conversations are long
- Older details still matter
- You want continuity without huge token costs
- You need a compact running memory of the thread

Tradeoff: summaries can lose nuance or introduce errors, so they should be reviewed or regenerated carefully for high-stakes workflows.

## Custom Filtering

You can write your own logic to select memory:

- Keep only messages related to the current topic
- Keep only decisions and facts
- Remove chit-chat
- Keep all tool outputs but summarize assistant text
- Keep documents by reference instead of raw content

This becomes a context engineering problem: choose exactly what the model needs for the next step.

---

## 12. Short-Term Memory with Human-in-the-Loop

Human-in-the-loop workflows depend on short-term memory.

When a graph calls `interrupt()`, LangGraph pauses execution and saves the checkpoint. Later, the graph resumes from the same thread using the saved state.

Without a checkpointer:

```text
the graph cannot reliably pause and resume
```

With a checkpointer:

```text
pause -> save state -> wait for human -> resume with same thread_id
```

This is why the HITL example in `18_Human_in_the_loop` compiles with a checkpointer.

---

## 13. Short-Term Memory with Subgraphs

If a parent graph is compiled with a checkpointer, LangGraph can propagate checkpointing to subgraphs.

This matters when:

- A subgraph needs to pause with `interrupt()`
- You want to inspect subgraph state
- A subgraph needs durable execution
- A multi-agent graph delegates tasks to child graphs

For most simple subgraphs, keep subgraph memory per invocation. Use per-thread subgraph memory only when the subgraph itself needs to remember across calls.

---

## 14. Production Design

For production, think about memory as infrastructure.

## Backend Choice

Use durable checkpointers:

- SQLite for local single-machine prototypes
- Postgres for production multi-user apps
- Redis for fast distributed use cases

In-memory checkpointers are not enough for production because memory disappears on process restart.

## Retention

Define how long thread memory should live.

Questions:

- Should inactive conversations expire?
- Can users delete their conversation history?
- How much checkpoint history do you keep?
- Do you need all historical checkpoints or only the latest state?

## Privacy and Security

Short-term memory may contain sensitive data.

Practices:

- Avoid storing secrets directly in state.
- Store large files externally and keep references in state.
- Encrypt persisted checkpoint data where needed.
- Implement per-user thread authorization.
- Add deletion workflows.
- Be careful with logs and traces.

## Migrations

Database-backed checkpointers often require setup or migrations before use. Run those as a deployment step rather than relying on ad hoc runtime setup.

---

## 15. Common Pitfalls

## Pitfall 1: No `thread_id`

If there is no `thread_id`, the checkpointer does not know which thread to load or save.

## Pitfall 2: One Thread for Every User

Hardcoding `thread_id = "1"` for every user mixes conversations together.

## Pitfall 3: Assuming In-Memory Means Durable

`InMemorySaver` and `MemorySaver` are useful for demos, but they lose data when the process exits.

## Pitfall 4: Returning Full State from Every Node

Nodes should usually return only the fields they update. Let reducers merge state.

## Pitfall 5: Letting Messages Grow Forever

Long message history increases cost and can hurt model quality. Use trimming, deletion, summarization, or filtering.

## Pitfall 6: Deleting Tool Messages Incorrectly

If an assistant tool call remains but the matching tool result is deleted, some providers may reject the message history.

## Pitfall 7: Confusing UI State with Graph State

A frontend may store visible messages in session state, but LangGraph memory lives in checkpoints. They should be kept in sync intentionally.

---

## 16. Practical Mental Model

Think of short-term memory like a saved game file for one conversation.

```text
thread_id = save slot
state = game world
checkpoint = saved snapshot
checkpointer = save system
```

Every time the graph runs, it loads the saved slot, advances the world, and saves again.

---

## 17. Summary

Short-term memory in LangGraph is:

- Thread-scoped
- Stored in graph state
- Saved by checkpointers
- Retrieved with `thread_id`
- Useful for multi-turn conversations and resumable workflows
- Managed with trimming, deletion, summarization, and checkpoint tools

Core pattern:

```text
State + checkpointer + thread_id = short-term memory
```

Use it when the graph needs to remember what happened inside one conversation or task.

---

## References

- LangGraph memory guide: https://docs.langchain.com/oss/python/langgraph/add-memory
- LangGraph persistence guide: https://docs.langchain.com/oss/python/langgraph/persistence
- Memory conceptual overview: https://docs.langchain.com/oss/python/concepts/memory
