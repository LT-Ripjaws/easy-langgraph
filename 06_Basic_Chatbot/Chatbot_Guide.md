# Building Conversational AI with LangGraph: Chatbot Fundamentals

## The Foundation of Stateful AI Interactions

A conversational chatbot represents one of the most fundamental yet powerful applications of LangGraph. Unlike stateless prompt-response systems, LangGraph enables **persistent multi-turn conversations** where the AI remembers context, maintains coherence, and builds rapport across extended interactions.

This guide covers the essential patterns for building production-ready chatbots with LangGraph's memory management system.

> **Industry Insight:** According to recent implementations documented in Towards AI and Medium's data science community, proper memory management is the #1 factor distinguishing production-ready chatbots from prototypes. The patterns in this guide reflect current best practices from the LangGraph community (2025-2026).

---

## Core Concepts: Why LangGraph for Chatbots?

### The Stateless Problem

Traditional LLM calls are stateless — each request is independent:

```python
# Stateless approach (problematic for conversations)
response1 = llm.invoke("My name is Ripjaws")  # AI learns name
response2 = llm.invoke("What's my name?")      # AI has no idea
```

Without memory, every interaction starts from scratch.

### The LangGraph Solution

LangGraph solves this with **checkpoint-based state persistence**:

```python
# Stateful approach with LangGraph
config = {'configurable': {'thread_id': 'session_123'}}
response1 = bot.invoke({'messages': [HumanMessage("My name is Ripjaws")]}, config)
response2 = bot.invoke({'messages': [HumanMessage("What's my name?")]}, config)
# AI correctly responds: "Your name is Ripjaws!"
```

The conversation history persists across invocations via the `thread_id`.

---

## Real-World Example: Complete Chatbot Implementation

Here's a production-ready pattern combining all core concepts:

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define state with message accumulator
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 2. Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9)

# 3. Create node that processes conversation
def chat_node(state: ChatState) -> dict:
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}  # Partial update only

# 4. Build and compile with memory
graph = StateGraph(ChatState)
graph.add_node('chat', chat_node)
graph.add_edge(START, 'chat')
graph.add_edge('chat', END)

checkpointer = MemorySaver()
bot = graph.compile(checkpointer=checkpointer)

# 5. Conversation loop with session management
thread_id = 'session_1'
config = {'configurable': {'thread_id': thread_id}}

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        break
    
    response = bot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=config
    )
    
    print("AI:", response['messages'][-1].content)
```

**Key Takeaways from this implementation:**
- `add_messages` reducer ensures messages accumulate, not overwrite
- `MemorySaver()` enables checkpoint-based persistence
- `thread_id` isolates conversation sessions
- Nodes return **partial updates** (only changed fields)

---

## Architecture of a LangGraph Chatbot

### High-Level Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                          │
│  (Input loop, thread_id management, response display)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   LangGraph Graph                        │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  chat_node   │ →  │  add_messages │ ← MemorySaver    │
│  │  (LLM call)  │    │  (reducer)    │   (checkpoint)   │
│  └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 Persistence Layer                        │
│        (In-memory RAM or external database)              │
└─────────────────────────────────────────────────────────┘
```

### The Three Pillars

1. **State Management** — `add_messages` reducer for conversation history
2. **Checkpointing** — `MemorySaver` for persistence across turns
3. **Thread Isolation** — `thread_id` for multi-user support

---

## Step-by-Step Implementation

### 1. Define State with Message Reducer

The state defines what data flows through your graph. For chatbots, you need conversation history:

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """State for conversational chatbot"""
    messages: Annotated[List[BaseMessage], add_messages]
```

**Why `add_messages` instead of `operator.add`:**

| Feature | `operator.add` | `add_messages` |
|---------|---------------|----------------|
| Message merging | Simple concatenation | Smart merge by ID |
| Duplicate handling | Keeps duplicates | Updates by ID |
| LangGraph optimization | Generic | Optimized for messages |
| Message formatting | No | Supports OpenAI format |

**How `add_messages` works:**

```python
from langchain_core.messages import HumanMessage, AIMessage

# Scenario 1: New messages (append)
msg1 = [HumanMessage(content="Hello", id="1")]
msg2 = [AIMessage(content="Hi!", id="2")]
result = add_messages(msg1, msg2)
# Result: [HumanMessage("Hello"), AIMessage("Hi!")]

# Scenario 2: Same ID (update/overwrite)
msg1 = [HumanMessage(content="Hello", id="1")]
msg2 = [HumanMessage(content="Hello again", id="1")]
result = add_messages(msg1, msg2)
# Result: [HumanMessage("Hello again")]  ← Updated, not duplicated
```

This smart merging prevents duplicate messages while maintaining full conversation history.

---

### 2. Create the Chat Node

The node processes user input and generates responses:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9)

def chat_node(state: ChatState) -> dict:
    """
    Process conversation and generate response.
    
    Args:
        state: Current conversation state with message history
        
    Returns:
        dict with new message to append to history
    """
    # Extract full conversation history
    messages = state['messages']
    
    # Invoke LLM with entire conversation context
    response = llm.invoke(messages)
    
    # Return ONLY the new message (partial update)
    # add_messages reducer will append it to history
    return {'messages': [response]}
```

**Key Design Decisions:**

1. **Full history to LLM** — The LLM sees the entire conversation, not just the last message
2. **Partial state update** — Node returns only new data, not the full state
3. **Single responsibility** — Node only handles LLM invocation, not state management

---

### 3. Build and Compile with Memory

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Create the graph
graph = StateGraph(ChatState)

# Add the chat node
graph.add_node('chat_node', chat_node)

# Define flow: START → chat_node → END
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Initialize memory checkpoint
checkpointer = MemorySaver()

# Compile with checkpointing enabled
bot = graph.compile(checkpointer=checkpointer)
```

**Why MemorySaver is critical:**

Without `checkpointer`:
- Each `bot.invoke()` call is stateless
- Conversation history resets every time
- No multi-turn conversation possible

With `checkpointer`:
- State persists across invocations
- `thread_id` identifies unique sessions
- Full conversation context maintained

---

### 4. Client Layer: Conversation Loop

The client layer manages user interaction and session configuration:

```python
from langchain_core.messages import HumanMessage

# Thread identifier (session ID)
# Each unique thread_id = separate conversation
seed = 'user_session_1'

while True:
    # Get user input
    user_message = input('Type here: ')
    print('User:', user_message)
    
    # Check for exit commands
    if user_message.strip().lower() in ['quit', 'q', 'exit', 'bye']:
        break
    
    # Configure session with thread_id
    # This is CRITICAL for memory persistence
    config = {'configurable': {'thread_id': seed}}
    
    # Invoke bot with user message
    response = bot.invoke(
        {'messages': [HumanMessage(content=user_message)]},
        config=config
    )
    
    # Extract latest response from conversation history
    # messages[-1] = most recent message (AI's response)
    print('AI:', response['messages'][-1].content)
```

**Understanding `thread_id`:**

```python
# Session 1: User discusses programming
config1 = {'configurable': {'thread_id': 'session_1'}}
bot.invoke({'messages': [HumanMessage("I love Python")]}, config1)
# AI remembers: User likes Python

# Session 2: Completely different conversation
config2 = {'configurable': {'thread_id': 'session_2'}}
bot.invoke({'messages': [HumanMessage("I love Python")]}, config2)
# AI has no context — new conversation

# Session 1 again: Continues previous discussion
bot.invoke({'messages': [HumanMessage("What's my favorite language?")]}, config1)
# AI responds: "Your favorite language is Python!"
```

---

## Memory Deep Dive: How Checkpointing Works

### The Checkpoint Mechanism

When you compile with `MemorySaver()`, LangGraph automatically:

1. **Before execution** — Load state from checkpoint using `thread_id`
2. **During execution** — Process nodes and update state
3. **After execution** — Save updated state back to checkpoint

```
┌──────────────────────────────────────────────────────────┐
│  bot.invoke(input, config={'thread_id': 'abc123'})       │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  1. Load checkpoint for 'abc123'                         │
│     → Retrieves: {messages: [msg1, msg2, ...]}           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  2. Execute graph nodes                                  │
│     chat_node: LLM processes full history + new input    │
│     add_messages: Appends response to history            │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  3. Save checkpoint for 'abc123'                         │
│     → Stores: {messages: [msg1, msg2, ..., new_response]}│
└──────────────────────────────────────────────────────────┘
```

### Memory Storage Options

**MemorySaver (In-Memory)**
```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```
- ✅ Fast, no setup required
- ✅ Perfect for development and demos
- ❌ Data lost when process restarts
- ❌ Not suitable for production

**PostgreSQL (Production-Ready)**
```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver(connection_string)
```
- ✅ Persistent across restarts
- ✅ Scalable for production
- ❌ Requires database setup

**SQLite (Lightweight Persistence)**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver(conn)
```
- ✅ Persistent, file-based
- ✅ No external database needed
- ⚠️ Single-user limitation

---

## Advanced Patterns

### Pattern 1: Multi-User Support

Use unique `thread_id` values to support multiple concurrent users:

```python
def get_user_config(user_id: str) -> dict:
    """Generate config for specific user"""
    return {'configurable': {'thread_id': f'user_{user_id}'}}

# User A's conversation
config_a = get_user_config('alice')
response_a = bot.invoke({'messages': [HumanMessage("Hi")]}, config_a)

# User B's conversation (completely isolated)
config_b = get_user_config('bob')
response_b = bot.invoke({'messages': [HumanMessage("Hi")]}, config_b)
```

### Pattern 2: Conversation Reset

Clear conversation history by starting a new `thread_id`:

```python
def start_new_conversation(user_id: str) -> dict:
    """Generate fresh thread_id for new conversation"""
    import uuid
    new_thread_id = f'user_{user_id}_session_{uuid.uuid4()}'
    return {'configurable': {'thread_id': new_thread_id}}

# User requests new conversation
config = start_new_conversation('alice')
# Previous history is abandoned, fresh start
```

### Pattern 3: System Message Injection

Add persistent system instructions to guide AI behavior:

```python
def create_chatbot_with_system(system_prompt: str):
    """Initialize chat with system message"""
    
    system_message = HumanMessage(content=system_prompt)
    
    config = {'configurable': {'thread_id': 'session_1'}}
    
    # Prime the conversation with system message
    bot.invoke(
        {'messages': [system_message]},
        config=config
    )
    
    return config

# Usage
config = create_chatbot_with_system(
    "You are a helpful coding assistant. "
    "Always provide code examples with explanations."
)

# Now chat with context
response = bot.invoke(
    {'messages': [HumanMessage("How do I reverse a list?")]},
    config=config
)
```

### Pattern 4: Response Streaming

Stream AI responses for better UX:

```python
config = {'configurable': {'thread_id': 'session_1'}}

# Stream the response
for chunk in bot.stream(
    {'messages': [HumanMessage("Tell me a story")]},
    config=config
):
    if 'chat_node' in chunk:
        print(chunk['chat_node']['messages'][-1].content, end='', flush=True)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting thread_id

```python
# ❌ WRONG: No thread_id = no memory
response = bot.invoke({'messages': [HumanMessage("Hi")]})

# ✅ CORRECT: Include thread_id for persistence
config = {'configurable': {'thread_id': 'session_1'}}
response = bot.invoke({'messages': [HumanMessage("Hi")]}, config=config)
```

**Symptom:** Chatbot doesn't remember previous messages  
**Fix:** Always pass `config={'configurable': {'thread_id': '...'}}`

---

### Pitfall 2: Using Wrong Reducer

```python
# ❌ WRONG: Using operator.add for messages
class State(TypedDict):
    messages: Annotated[list, operator.add]

# ✅ CORRECT: Using add_messages for messages
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Symptom:** Duplicate messages, incorrect history  
**Fix:** Always use `add_messages` for message lists

---

### Pitfall 3: Returning Full State from Node

```python
# ❌ WRONG: Returning entire state
def chat_node(state: ChatState):
    response = llm.invoke(state['messages'])
    state['messages'].append(response)
    return state  # Can cause issues with parallel execution

# ✅ CORRECT: Return only changes
def chat_node(state: ChatState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}  # Partial update
```

**Symptom:** Unexpected behavior in complex graphs  
**Fix:** Nodes should return only changed fields

---

### Pitfall 4: Not Clearing Output Cells

When working in Jupyter notebooks:

```python
# ❌ WRONG: Committing notebook with outputs
# - Contains API responses
# - May include sensitive data
# - Makes diffs noisy

# ✅ CORRECT: Clear outputs before commit
# In Jupyter: Cell → All Output → Clear
# Or: nbconvert --clear-output
```

---

## Production Considerations

### 1. Rate Limiting

Implement rate limiting to prevent API abuse:

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)
    
    def check_rate_limit(self, user_id: str) -> bool:
        now = datetime.now()
        # Remove old requests
        self.requests[user_id] = [
            req for req in self.requests[user_id]
            if now - req < self.window
        ]
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        # Record new request
        self.requests[user_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# Usage in chat loop
if not rate_limiter.check_rate_limit(user_id):
    print("Rate limit exceeded. Please wait.")
else:
    response = bot.invoke(...)
```

### 2. Error Handling

Handle API failures gracefully:

```python
from langchain_core.exceptions import LangChainException

def safe_invoke(user_message: str, config: dict):
    try:
        response = bot.invoke(
            {'messages': [HumanMessage(content=user_message)]},
            config=config
        )
        return response['messages'][-1].content
    except LangChainException as e:
        return f"Sorry, I encountered an error: {str(e)}"
    except Exception as e:
        return "An unexpected error occurred. Please try again."
```

### 3. Input Validation

Sanitize and validate user input:

```python
import re

def validate_input(text: str) -> tuple[bool, str]:
    """Validate user input"""
    if not text or not text.strip():
        return False, "Empty message"
    
    if len(text) > 4000:
        return False, "Message too long (max 4000 characters)"
    
    # Check for potentially harmful patterns
    if re.search(r'<script.*?>', text, re.IGNORECASE):
        return False, "Invalid characters in message"
    
    return True, "Valid"

# Usage
is_valid, message = validate_input(user_message)
if not is_valid:
    print(f"Error: {message}")
else:
    response = bot.invoke(...)
```

---

## Debugging Tips

### 1. Visualize the Graph

```python
from IPython.display import Image, display

# Generate graph visualization
display(Image(bot.get_graph().draw_mermaid_png()))

# Or view as Mermaid diagram
print(bot.get_graph().draw_mermaid())
```

### 2. Inspect State

```python
# View full state after invocation
response = bot.invoke(input_data, config=config)
print("Full state:", response)

# View just the messages
for msg in response['messages']:
    print(f"{type(msg).__name__}: {msg.content}")
```

### 3. Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now LangGraph will log detailed execution info
```

---

## Key Takeaways

1. **`add_messages` reducer** — Smart message management that handles duplicates by ID
2. **`MemorySaver` checkpointing** — Enables persistent state across invocations
3. **`thread_id` isolation** — Each unique ID = separate conversation session
4. **Partial state updates** — Nodes return only changed fields
5. **Full history to LLM** — LLM sees entire conversation for context
6. **Client layer separation** — Input loop is separate from graph logic

---

## Practice Exercises

1. **Basic Chatbot** — Implement the conversation loop from `1_basic_chatbot.ipynb`
2. **Multi-Session Test** — Run two conversations with different `thread_id` values and verify isolation
3. **System Prompt** — Add a system message that makes the bot act as a specific character
4. **Error Handling** — Add try-catch blocks to handle API failures gracefully
5. **Input Validation** — Implement message length limits and content filtering

---

## Next Steps

After mastering basic chatbots:

- **Conditional Workflows** (`4_Conditional_Workflows/`) — Add routing and decision-making
- **Tool Integration** — Connect chatbot to external APIs and databases
- **Human-in-the-Loop** — Add approval steps for critical actions
- **Advanced Memory** — Use PostgreSQL or Redis for production persistence

---

*For more advanced patterns, see `3_LangGraph_Core_Concepts.md` in the Required Concepts folder.*
