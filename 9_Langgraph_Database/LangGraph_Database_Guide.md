# LangGraph Database Persistence Guide

This folder shows how to persist LangGraph chatbot memory in a SQLite database instead of only keeping it in RAM.

## What This Example Builds

`langgraph_database_backend.py` creates a simple stateful chatbot with:

- Gemini as the chat model
- LangGraph state managed through `add_messages`
- SQLite as the checkpoint storage
- A fixed `thread_id` for testing conversation memory

## Key File

```text
9_Langgraph_Database/
|-- langgraph_database_backend.py
```

## Important Code Pieces

### State

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` makes LangGraph append new chat messages to the existing conversation history.

### SQLite Connection

```python
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

This creates a local `chatbot.db` file and uses it as the LangGraph checkpoint store.

Unlike `MemorySaver`, SQLite persistence can survive Python process restarts because the conversation state is saved to disk.

### Compiled Bot

```python
bot = graph.compile(checkpointer=checkpointer)
```

The checkpointer is what makes the graph persistent.

## How Thread Memory Works

```python
CONFIG = {'configurable': {'thread_id': '1'}}
```

The `thread_id` tells LangGraph which saved conversation to load.

If you reuse the same `thread_id`, the bot can remember earlier messages. If you use a new `thread_id`, LangGraph starts a separate conversation.

## Running the Example

From the `9_Langgraph_Database` folder:

```bash
python langgraph_database_backend.py
```

The script sends one test message:

```python
HumanMessage(content="Hi i am Ripjaws, how are you?")
```

Then it prints the full returned graph state.

## Notes

- `chatbot.db` is generated automatically when the script runs.
- Use unique `thread_id` values for different users or chat sessions.
- Avoid keeping test invocations at the bottom of the file if this backend will be imported by a frontend.
- SQLite is good for local demos and small apps; use PostgreSQL for production multi-user systems.

## Summary

This example upgrades the previous in-memory chatbot by replacing `MemorySaver` with `SqliteSaver`.

Core idea:

```text
HumanMessage + thread_id -> LangGraph -> SQLite checkpoint -> persistent chat memory
```
