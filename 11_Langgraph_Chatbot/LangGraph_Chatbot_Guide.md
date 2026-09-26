# LangGraph Chatbot with Streamlit: Complete Guide

**Last Updated:** 2026-05-25  
**Difficulty:** Beginner to Intermediate  
**Prerequisites:** Basic LangGraph state, messages, checkpointing, and Python functions

---

## Table of Contents

1. [What This Folder Builds](#1-what-this-folder-builds)
2. [Project Files](#2-project-files)
3. [Architecture Overview](#3-architecture-overview)
4. [Backend: The LangGraph Bot](#4-backend-the-langgraph-bot)
5. [Frontend: The Streamlit Chat UI](#5-frontend-the-streamlit-chat-ui)
6. [How Memory Works](#6-how-memory-works)
7. [How Streaming Works](#7-how-streaming-works)
8. [Running the App](#8-running-the-app)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Improvement Ideas](#10-improvement-ideas)
11. [Summary](#11-summary)

---

## 1. What This Folder Builds

The `11_Langgraph_Chatbot` folder builds a full chatbot app with:

- A LangGraph backend that manages conversation state
- A Gemini chat model that generates responses
- `MemorySaver` checkpointing for per-thread chat history
- A Streamlit frontend with chat bubbles, sidebar threads, and response streaming

This is the next step after the basic chatbot example. Instead of only running a chatbot in a notebook or terminal, this folder wraps the LangGraph bot in a small user interface.

---

## 2. Project Files

```text
11_Langgraph_Chatbot/
|-- backend.py    # LangGraph state, node, graph, checkpointer, compiled bot
|-- frontend.py   # Streamlit UI, chat sessions, sidebar, streaming output
```

### `backend.py`

Responsible for the AI workflow:

- Loads environment variables with `load_dotenv()`
- Creates the Gemini LLM
- Defines chatbot state with `add_messages`
- Builds a one-node LangGraph graph
- Compiles the graph with `MemorySaver`
- Exposes the compiled `bot`

### `frontend.py`

Responsible for the user experience:

- Creates and stores thread IDs
- Tracks visible chat history in `st.session_state`
- Shows previous conversations in the sidebar
- Sends user messages to the LangGraph bot
- Streams assistant responses into the Streamlit chat window

---

## 3. Architecture Overview

```text
User
 |
 v
Streamlit UI in frontend.py
 |
 | sends HumanMessage + thread_id
 v
Compiled LangGraph bot from backend.py
 |
 | loads previous state for this thread
 v
chat_node
 |
 | sends full message history to Gemini
 v
Gemini response
 |
 | add_messages appends response
 v
MemorySaver checkpoint
 |
 v
Streamlit streams response back to user
```

The important idea is that the frontend does not manually build the full conversation context for the model. It sends the latest user message plus a `thread_id`. LangGraph uses the checkpointer to recover the existing state for that thread and then appends the new messages.

---

## 4. Backend: The LangGraph Bot

The backend is intentionally small. It defines the reusable graph and exports a compiled `bot`.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9)
```

### Chat State

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The state contains one field: `messages`.

The `add_messages` reducer is critical. It tells LangGraph to append new messages to the existing history instead of replacing the whole list each time.

Without `add_messages`, every new turn could overwrite the old conversation history.

### Chat Node

```python
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}
```

The node receives the current state, sends the full conversation history to the LLM, and returns only the new assistant message.

This is a partial state update. LangGraph combines it with the existing state by using the reducer configured on `messages`.

### Graph Construction

```python
checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

bot = graph.compile(checkpointer=checkpointer)
```

The graph has one simple path:

```text
START -> chat_node -> END
```

Even though the graph is small, it is stateful because it is compiled with `MemorySaver`.

---

## 5. Frontend: The Streamlit Chat UI

The frontend imports the compiled bot:

```python
import streamlit as st
from backend import bot
from langchain_core.messages import HumanMessage
import uuid
```

### Thread IDs

Each conversation needs its own `thread_id`.

```python
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id
```

The `thread_id` is passed into LangGraph config:

```python
CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
```

Each unique `thread_id` gets a separate checkpoint history. That is how the sidebar can switch between conversations without mixing messages.

### Resetting Chat

```python
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
```

When the user clicks **New Chat**, the app:

- Generates a new thread ID
- Clears the visible message history
- Saves the new active thread
- Adds it to the sidebar list

### Loading a Previous Conversation

```python
def load_conversation(thread_id):
    state = bot.get_state(config={'configurable': {'thread_id': thread_id}}).values
    return state.get('messages', [])
```

`bot.get_state(...)` asks LangGraph for the saved state of a specific thread.

The returned messages are LangChain message objects, so the UI converts them into Streamlit-friendly dictionaries:

```python
temp_messages = []
for message in messages:
    if isinstance(message, HumanMessage):
        role = 'user'
    else:
        role = 'assistant'
    temp_messages.append({'role': role, 'content': message.content})
```

### Session State

Streamlit reruns the script after interactions, so the app stores important values in `st.session_state`:

```python
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
```

These keys serve different roles:

| Key | Purpose |
|-----|---------|
| `message_history` | Messages currently displayed in the UI |
| `thread_id` | Active LangGraph conversation thread |
| `chat_threads` | List of previous thread IDs for the sidebar |

---

## 6. How Memory Works

There are two memory layers in this app.

### 1. Streamlit Session Memory

`st.session_state['message_history']` stores messages for display.

This is UI memory. It helps Streamlit redraw the chat window after every rerun.

### 2. LangGraph Checkpoint Memory

`MemorySaver` stores the actual LangGraph state by `thread_id`.

This is graph memory. It lets the bot remember previous turns when the same thread is invoked again.

```python
bot.stream(
    {'messages': [HumanMessage(content=user_input)]},
    config=CONFIG,
    stream_mode='messages'
)
```

The user only sends the newest message. LangGraph loads the old messages from the checkpoint, appends the new user message, runs the node, and saves the new assistant response.

### Important Limitation

`MemorySaver` stores data in RAM. Conversation history is lost when the Python or Streamlit process restarts.

For production or long-lived apps, use a persistent checkpointer such as SQLite or PostgreSQL.

---

## 7. How Streaming Works

The app streams the assistant response into the chat window:

```python
with st.chat_message('assistant'):
    ai_message = st.write_stream(
        message_chunk.content
        for message_chunk, metadata in bot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        )
    )
```

### What `stream_mode='messages'` Does

`stream_mode='messages'` makes LangGraph yield message chunks as the model produces them.

Each streamed item is unpacked as:

```python
message_chunk, metadata
```

The app passes only `message_chunk.content` into `st.write_stream`, which creates the live typing effect.

After streaming finishes, Streamlit returns the full assistant message as `ai_message`, and the app stores it in UI history:

```python
st.session_state['message_history'].append({
    'role': 'assistant',
    'content': ai_message
})
```

---

## 8. Running the App

### 1. Install Dependencies

From the repository root:

```bash
pip install streamlit langgraph langchain-core langchain-google-genai python-dotenv
```

### 2. Set Environment Variables

Create a `.env` file in the repository root with your Google API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

The backend loads this with:

```python
load_dotenv()
```

### 3. Start Streamlit

From inside the `11_Langgraph_Chatbot` folder:

```bash
streamlit run frontend.py
```

Or from the repository root:

```bash
streamlit run 11_Langgraph_Chatbot/frontend.py
```

---

## 9. Common Pitfalls

### Pitfall 1: Forgetting the Thread ID

```python
# Wrong: no thread_id means no stable conversation memory
bot.invoke({'messages': [HumanMessage(content='Hello')]})
```

```python
# Correct
config = {'configurable': {'thread_id': 'chat_1'}}
bot.invoke({'messages': [HumanMessage(content='Hello')]}, config=config)
```

The checkpointer needs a thread ID to know which conversation to load and update.

### Pitfall 2: Confusing UI History with Graph History

`message_history` is for displaying messages in Streamlit.

The LangGraph checkpoint is the real source of model memory.

If the UI history is cleared but the same `thread_id` is reused, the model can still remember earlier messages.

### Pitfall 3: Expecting `MemorySaver` to Survive Restarts

`MemorySaver` is in-memory only.

It is good for learning, testing, and demos. It is not enough for a deployed chatbot that must keep conversations after restart.

### Pitfall 4: Returning the Full State from `chat_node`

```python
# Avoid this
def chat_node(state):
    response = llm.invoke(state['messages'])
    state['messages'].append(response)
    return state
```

```python
# Prefer this
def chat_node(state):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}
```

Nodes should return only the state fields they update.

### Pitfall 5: Using a Fixed Thread for Everyone

```python
# Bad for multi-user apps
CONFIG = {'configurable': {'thread_id': '1'}}
```

Every user would share the same conversation. Generate unique thread IDs for separate conversations.

---

## 10. Improvement Ideas

### Use String Thread IDs

The current app stores UUID objects as thread IDs. For portability, especially with persistent checkpointers, returning strings is often safer:

```python
def generate_thread_id():
    return str(uuid.uuid4())
```

### Add Persistent Storage

Replace `MemorySaver` with SQLite for local persistence:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("chatbot.sqlite")
bot = graph.compile(checkpointer=checkpointer)
```

Use PostgreSQL for production multi-user deployments.

### Add Conversation Titles

The sidebar currently displays raw thread IDs:

```python
st.sidebar.button(f'Chat Thread: {str(thread_id)}')
```

A better UI could generate a short title from the first user message.

### Add Error Handling

Wrap bot execution so the UI handles API failures gracefully:

```python
try:
    stream = bot.stream(..., stream_mode='messages')
except Exception as error:
    st.error("The assistant could not respond. Please try again.")
```

### Add a System Prompt

You can guide the assistant behavior by adding a `SystemMessage` before sending messages to the model, or by injecting one inside `chat_node`.

```python
from langchain_core.messages import SystemMessage

def chat_node(state: ChatState):
    system = SystemMessage(content="You are a helpful LangGraph tutor.")
    response = llm.invoke([system] + state['messages'])
    return {'messages': [response]}
```

---

## 11. Summary

This folder demonstrates how to turn a LangGraph chatbot into a usable Streamlit application.

Key concepts:

1. `backend.py` owns the LangGraph workflow.
2. `frontend.py` owns the Streamlit user interface.
3. `add_messages` keeps conversation messages from being overwritten.
4. `MemorySaver` stores state by `thread_id`.
5. `st.session_state` keeps the UI stable across Streamlit reruns.
6. `bot.get_state(...)` reloads previous thread history.
7. `bot.stream(..., stream_mode='messages')` powers live assistant output.

The core pattern is simple but powerful:

```text
HumanMessage + thread_id -> LangGraph checkpoint -> LLM response -> saved state -> streamed UI
```

Once this pattern is clear, you can extend it with persistent storage, tool calling, better conversation management, authentication, or deployment-ready error handling.

---

*This guide is part of the easy-langgraph project.*
