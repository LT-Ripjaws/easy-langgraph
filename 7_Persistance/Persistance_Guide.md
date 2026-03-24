# LangGraph Persistence: Complete Guide

**Last Updated:** 2026-03-24  
**Difficulty:** Beginner to Intermediate  
**Prerequisites:** Basic understanding of LangGraph graphs, nodes, and state

---

## Table of Contents

1. [What is Persistence?](#1-what-is-persistence)
2. [Why Persistence Matters](#2-why-persistence-matters)
3. [Checkpointers: The Heart of Persistence](#3-checkpointers-the-heart-of-persistence)
4. [Threads: Session Management](#4-threads-session-management)
5. [Short-Term vs Long-Term Memory](#5-short-term-vs-long-term-memory)
6. [Fault Tolerance](#6-fault-tolerance)
7. [Human-in-the-Loop](#7-human-in-the-loop)
8. [Time Travel](#8-time-travel)
9. [Storage Backends](#9-storage-backends)
10. [Production Best Practices](#10-production-best-practices)
11. [Code Examples](#11-code-examples)

---

## 1. What is Persistence?

**Persistence** in LangGraph means saving the state of your workflow at every step so you can:
- Resume after a crash without losing progress
- Pause for human approval and continue later
- Switch between different conversation sessions
- Debug by rewinding to any previous point
- Maintain memory across multiple interactions

### Simple Analogy

Think of persistence like a **video game save system**:
- 🎮 **Without persistence**: If your computer crashes, you lose everything and restart from level 1
- 💾 **With persistence**: You save at each checkpoint. Crash? Resume from your last save point

### In LangGraph Terms

```python
# Without persistence - state lost after crash
result = graph.invoke({"messages": ["hello"]})

# With persistence - state saved at every step
graph = StateGraph(State).compile(checkpointer=MemorySaver())
result = graph.invoke({"messages": ["hello"]}, config={"configurable": {"thread_id": "123"}})
```

---

## 2. Why Persistence Matters

### The Problem: AI Agents Are Unreliable

- LLM calls are **slow** (seconds to minutes)
- LLM calls are **flaky** (timeouts, rate limits, errors)
- Agents run for **long periods** (hours or days)
- **More steps = higher chance of failure**

### Without Persistence

```
Step 1 → Step 2 → Step 3 → Step 4 → 💥 CRASH at Step 5

Result: Start over from Step 1 😭
```

### With Persistence

```
Step 1 → Step 2 → Step 3 → Step 4 → 💥 CRASH at Step 5
                    ↑ Resume from here!
                    
Result: Only redo Steps 4-5, not 1-3 😊
```

### Real-World Benefits

| Benefit | Description |
|---------|-------------|
| **Cost Savings** | Don't re-pay for LLM calls you already made |
| **Time Savings** | Don't wait for computations you already did |
| **Better UX** | Users don't lose their conversation history |
| **Debugging** | Inspect any previous state to find bugs |
| **Compliance** | Audit trail of every decision made |

---

## 3. Checkpointers: The Heart of Persistence

### What is a Checkpointer?

A **checkpointer** is a component that saves snapshots of your graph's state at every step. Think of it as an automatic save system.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Graph Execution with Checkpointing                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  START → [Node A] → 💾 Checkpoint 1                     │
│                   ↓                                     │
│              [Node B] → 💾 Checkpoint 2                 │
│                   ↓                                     │
│              [Node C] → 💾 Checkpoint 3                 │
│                   ↓                                     │
│                 END                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Every Checkpoint Stores

1. **State values** - All variables in your TypedDict
2. **Messages** - Complete conversation history
3. **Metadata** - Timestamps, step numbers, thread IDs
4. **Version info** - Which version of each piece of data

### Checkpointer Types

| Checkpointer | Storage | Use Case |
|--------------|---------|----------|
| `MemorySaver` | RAM | Development, testing, demos |
| `SqliteSaver` | SQLite file | Single-machine apps, moderate data |
| `PostgresSaver` | PostgreSQL | Production, scaling, multi-user |
| `RedisSaver` | Redis | Fast access, distributed systems |

---

## 4. Threads: Session Management

### What is a Thread?

A **thread** is an isolated session or conversation. Each thread has its own:
- Conversation history
- State checkpoints
- Memory

### Why Threads Matter

```
User A (thread_id: "alice")     User B (thread_id: "bob")
        ↓                              ↓
  ┌───────────┐                  ┌───────────┐
  │ Alice's   │                  │ Bob's     │
  │ messages  │                  │ messages  │
  │ Alice's   │                  │ Bob's     │
  │ state     │                  │ state     │
  └───────────┘                  └───────────┘
        ↓                              ↓
  Completely separate!           Completely separate!
```

### Thread ID Best Practices

```python
# ✅ Good: Use unique identifiers
config = {"configurable": {"thread_id": "user_123_session_456"}}
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# ❌ Bad: Hardcoding or sharing thread IDs
config = {"configurable": {"thread_id": "1"}}  # Will mix with other users!
```

### Common Thread Patterns

| Pattern | Example | Use Case |
|---------|---------|----------|
| **Per User** | `thread_id: user_id` | Each user has one ongoing conversation |
| **Per Session** | `thread_id: session_id` | Each conversation session is separate |
| **Per Task** | `thread_id: task_id` | Each task/request is isolated |

---

## 5. Short-Term vs Long-Term Memory

This is **critical** to understand correctly!

### Short-Term Memory (In-Thread)

**What it is:** Memory that exists within a single thread/session

**How it works:**
- Stored in the **state** that flows between nodes
- Automatically saved by **checkpointers** at each step
- Scoped to one `thread_id`
- Lost when thread is deleted

**Example:**
```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # Short-term: conversation history
    current_topic: str                        # Short-term: current discussion topic

# This persists across turns WITHIN the same thread
# But NOT across different threads
```

**Use for:**
- Conversation history
- Current session context
- Temporary working data

### Long-Term Memory (Cross-Thread)

**What it is:** Memory that persists across multiple threads/sessions

**How it works:**
- Stored in **external storage** (database, vector store, file system)
- Uses **LangGraph Store** or custom storage
- Shared across different `thread_id` values
- Persists even after threads are deleted

**Example:**
```python
# Store user preferences that persist across ALL conversations
store = {"user_123": {"preferences": {"tone": "friendly", "language": "en"}}}

# Retrieve in any thread
def load_user_preferences(state, user_id):
    return store.get(user_id, {})
```

**Use for:**
- User preferences
- Learned information about users
- Skills/knowledge gained over time
- Personalization data

### Key Differences

| Aspect | Short-Term | Long-Term |
|--------|------------|-----------|
| **Scope** | Single thread | Across all threads |
| **Storage** | Checkpointer | External store (DB, Redis, etc.) |
| **Lifetime** | Thread lifetime | Indefinite |
| **Access** | Automatic | Manual read/write |
| **Example** | Chat history | User preferences |

### Visual Comparison

```
┌────────────────────────────────────────────────────────┐
│                    SHORT-TERM MEMORY                   │
├────────────────────────────────────────────────────────┤
│  Thread A → Checkpointer → State (messages, context)   │
│  Thread B → Checkpointer → State (messages, context)   │
│  (Separate, isolated, deleted with thread)             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    LONG-TERM MEMORY                    │
├────────────────────────────────────────────────────────┤
│  Thread A ──┐                                          │
│             ├→ External Store → User preferences, facts│
│  Thread B ──┤     (Database/Redis/File)                │
│  Thread C ──┘                                          │
│  (Shared across all threads)                           │
└────────────────────────────────────────────────────────┘
```

---

## 6. Fault Tolerance

### What is Fault Tolerance?

**Fault tolerance** means your workflow can recover from failures without losing progress.

### How LangGraph Achieves It

1. **Automatic checkpointing** at every step
2. **O(1) recovery** - Only fetches latest checkpoint (doesn't replay history)
3. **Deterministic execution** - Same input = same output (enables safe retries)
4. **Task queue** (LangGraph Platform) - Manages retries and fair scheduling

### Failure Scenarios Handled

| Failure Type | How Persistence Helps |
|--------------|----------------------|
| **LLM Timeout** | Resume from last successful step, don't redo previous work |
| **Server Crash** | Restart on any machine, load from checkpoint |
| **Network Error** | Retry failed step with preserved context |
| **Rate Limiting** | Pause and resume when limits reset |
| **Memory Exhaustion** | Offload state to disk/database |

### Recovery Flow

```
┌─────────────────────────────────────────────────────────┐
│  Fault Tolerance Workflow                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Execute Node A → 💾 Checkpoint saved               │
│  2. Execute Node B → 💾 Checkpoint saved               │
│  3. Execute Node C → 💥 ERROR!                         │
│                                                         │
│  4. Load last checkpoint (after Node B)                │
│  5. Retry Node C with preserved state                  │
│  6. Continue to Node D → END                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Performance Impact

- **Checkpoint overhead:** ~10-50ms per step (worth it!)
- **Recovery time:** O(1) - constant time regardless of history length
- **Storage cost:** ~1-10KB per checkpoint (depends on state size)

---

## 7. Human-in-the-Loop

### What is Human-in-the-Loop (HITL)?

**HITL** allows humans to pause, review, approve, or modify AI decisions before continuing.

### Why It Matters

Some decisions are too important for AI alone:
- 🏦 **Financial:** Loan approvals, large transfers
- ⚕️ **Medical:** Diagnosis suggestions, treatment plans
- ⚖️ **Legal:** Contract analysis, compliance decisions
- 📰 **Content:** Publishing sensitive information
- 🔒 **Security:** Access grants, account actions

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Human-in-the-Loop Workflow                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [AI Analysis] → [INTERRUPT] → [Human Review]          │
│                       ↑              ↓                  │
│                  Pauses here   Approve/Reject/         │
│                  waits for     Modify                  │
│                  human                                 │
│                       ↓                                  │
│              [Resume from interrupt] → [END]           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Methods

#### Method 1: `interrupt_before` (Recommended)

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    decision: str
    approved: bool

def analyze(state):
    return {"decision": "approve_loan"}

def human_review(state):
    # This node waits for human input
    print(f"Review: {state['decision']}")
    return {"approved": True}

graph = StateGraph(State)
graph.add_node("analyze", analyze)
graph.add_node("human_review", human_review)

# ⚡ INTERRUPT before human_review node
graph.add_edge(START, "analyze")
graph.add_edge("analyze", "human_review")
graph.add_edge("human_review", END)

app = graph.compile(checkpointer=MemorySaver(), 
                    interrupt_before=["human_review"])

# First run - pauses before human_review
config = {"configurable": {"thread_id": "123"}}
result = app.invoke({"decision": ""}, config)
# ⏸️ Execution pauses, waiting for human

# Human reviews and approves (in real app, via UI)
# Then resume...
result = app.invoke(None, config)  # Continues from interrupt
```

#### Method 2: `interrupt()` Function

```python
from langgraph.types import interrupt

def request_approval(state):
    # Manually trigger interrupt
    approval = interrupt({
        "question": "Approve this action?",
        "context": state["decision"]
    })
    return {"approved": approval}
```

### Key HITL Concepts

| Concept | Description |
|---------|-------------|
| **Interrupt Point** | Where execution pauses (set via `interrupt_before`) |
| **Pending State** | Saved checkpoint waiting for human input |
| **Resume** | Continue execution after human provides input |
| **Timeout** | Set maximum wait time for human response |
| **Escalation** | Auto-approve/reject if no response within timeout |

### Best Practices

✅ **Do:**
- Place interrupts at critical decision points only
- Provide clear context for human reviewers
- Implement timeout/escalation rules
- Log all human decisions for audit trails

❌ **Don't:**
- Interrupt at every step (annoys humans)
- Leave interrupts waiting indefinitely
- Skip checkpointing before interrupts
- Allow concurrent modifications to same thread

---

## 8. Time Travel

### What is Time Travel?

**Time travel** lets you rewind execution to any previous checkpoint, modify state, and re-execute forward.

### Why It's Powerful

```
Traditional Retry:
  Run 1-2-3-4-5-6-7-8-9-10 → Error at 10
  Run 1-2-3-4-5-6-7-8-9-10 (fixed) → Wasted 1-9!

Time Travel:
  Run 1-2-3-4-5-6-7-8-9-10 → Error at 10
  Rewind to checkpoint 9 → Fix state → Run 9-10 → Efficient!
```

### Use Cases

| Use Case | Description |
|----------|-------------|
| **Debugging** | Rewind to see what went wrong at any step |
| **What-If Analysis** | Try different decisions at key points |
| **User Corrections** | User says "actually, I meant X" - rewind and fix |
| **Branching Scenarios** | Explore multiple execution paths |
| **Training/Fine-tuning** | Collect alternative execution traces |

### How to Time Travel

```python
from langgraph.checkpoint import MemorySaver

# Setup with checkpointer
app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "123"}}

# Run workflow
app.invoke({"messages": ["hello"]}, config)

# Get all checkpoints (execution history)
checkpoints = list(app.get_state_history(config))

# Inspect checkpoint at step 5
checkpoint_5 = checkpoints[4]  # 0-indexed
print(checkpoint_5.values)  # See state at that point

# Rewind to checkpoint 5 and modify
new_state = checkpoint_5.values
new_state["messages"].append({"role": "user", "content": "correction"})

# Re-run from that point forward
app.invoke(new_state, config)  # Continues from checkpoint 5
```

### Time Travel + HITL = Superpower

Combine time travel with human-in-the-loop:

```
1. AI executes to interrupt point
2. Human reviews state
3. Human modifies state (time travel edit)
4. AI resumes with corrected state
5. Continue execution
```

This enables **collaborative AI** where humans guide AI decisions in real-time.

### Visual Example

```
┌─────────────────────────────────────────────────────────┐
│  Time Travel Workflow                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Original:  START → A → B → C → D → 💥 Error           │
│                                                         │
│  Time Travel:                                           │
│    1. List checkpoints: [A, B, C, D]                   │
│    2. Rewind to C                                       │
│    3. Modify state at C                                 │
│    4. Re-execute: C → D → END ✓                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Storage Backends

### Choosing the Right Backend

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **MemorySaver** | Development, testing | Fast, no setup | Lost on restart |
| **SqliteSaver** | Single-user apps, prototyping | File-based, no server | Single-writer |
| **PostgresSaver** | Production, multi-user | Scalable, reliable | Requires DB setup |
| **RedisSaver** | High-performance, distributed | Fast, pub/sub support | In-memory (optional persistence) |

### MemorySaver (In-Memory)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ✅ Fast, no configuration
# ❌ Lost when Python process ends
```

**Use for:** Development, testing, demos, temporary workflows

### SqliteSaver (File-Based)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Saves to file on disk
checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
app = graph.compile(checkpointer=checkpointer)

# ✅ Persists across restarts, no server needed
# ❌ Single-writer, slower than memory
```

**Use for:** Desktop apps, prototypes, moderate-scale applications

### PostgresSaver (Database)

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)
app = graph.compile(checkpointer=checkpointer)

# ✅ Scalable, multi-writer, reliable
# ❌ Requires PostgreSQL setup
```

**Use for:** Production systems, multi-user applications, enterprise

### RedisSaver (In-Memory Database)

```python
from langgraph.checkpoint.redis import RedisSaver

checkpointer = RedisSaver.from_conn_string(
    "redis://localhost:6379"
)
app = graph.compile(checkpointer=checkpointer)

# ✅ Very fast, distributed support
# ❌ Requires Redis, may need persistence config
```

**Use for:** High-performance systems, distributed architectures

### Storage Comparison

| Metric | Memory | SQLite | PostgreSQL | Redis |
|--------|--------|--------|------------|-------|
| **Speed** | ⚡⚡⚡ | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| **Persistence** | ❌ | ✅ | ✅ | ⚠️ (configurable) |
| **Scalability** | Low | Medium | High | High |
| **Setup Complexity** | None | Low | Medium | Medium |
| **Concurrent Writes** | No | No | Yes | Yes |

---

## 10. Production Best Practices

### ✅ Do These

#### 1. Always Use Checkpointing in Production

```python
# ❌ Bad - No persistence
app = graph.compile()

# ✅ Good - With checkpointer
app = graph.compile(checkpointer=PostgresSaver(...))
```

#### 2. Use Unique Thread IDs Per User/Session

```python
import uuid

# ✅ Good
thread_id = f"user_{user_id}_{uuid.uuid4()}"
config = {"configurable": {"thread_id": thread_id}}
```

#### 3. Implement Checkpoint Retention Policies

```python
# PostgreSQL example - keep last 30 days
# Configure in your database:
# DELETE FROM checkpoints WHERE created_at < NOW() - INTERVAL '30 days';
```

#### 4. Add Timeouts to Human-in-the-Loop

```python
# Auto-approve after 24 hours if no human response
if time_since_interrupt > timedelta(hours=24):
    auto_approve()
```

#### 5. Monitor Checkpoint Storage

```python
# Track checkpoint count and size
checkpoint_count = db.query("SELECT COUNT(*) FROM checkpoints")
checkpoint_size = db.query("SELECT SUM(size) FROM checkpoints")

# Alert if growing too fast
if checkpoint_count > 10000:
    alert("High checkpoint count - consider pruning")
```

#### 6. Use TypedDict/Pydantic for State Validation

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    messages: list = Field(default_factory=list)
    thread_id: str = Field(..., min_length=1)
    
# Prevents invalid state modifications during time travel
```

#### 7. Implement Row-Level Security for Multi-Tenant Apps

```sql
-- PostgreSQL example
CREATE POLICY tenant_isolation ON checkpoints
    USING (tenant_id = current_setting('app.current_tenant'));
```

### ❌ Avoid These

#### 1. Don't Hardcode Thread IDs

```python
# ❌ Terrible - All users share same thread!
config = {"configurable": {"thread_id": "1"}}
```

#### 2. Don't Store Sensitive Data in Checkpoints

```python
# ❌ Bad - API keys in checkpoints
state = {"api_key": "sk-123...", "messages": [...]}

# ✅ Good - Store references only
state = {"api_key_ref": "vault://keys/123", "messages": [...]}
```

#### 3. Don't Interrupt Too Frequently

```python
# ❌ Bad - Interrupts at every node
interrupt_before=["node1", "node2", "node3", "node4", "node5"]

# ✅ Good - Interrupt only at critical decisions
interrupt_before=["final_approval"]
```

#### 4. Don't Forget to Clear Old Checkpoints

```python
# ❌ Bad - Infinite growth
# Never deleting old checkpoints

# ✅ Good - Regular cleanup
def cleanup_old_checkpoints():
    db.execute("DELETE FROM checkpoints WHERE age < -30 days")
```

### Production Checklist

Before deploying with persistence:

- [ ] Checkpointer configured (not in-memory for production)
- [ ] Thread ID generation is unique per user/session
- [ ] Checkpoint retention policy defined
- [ ] Storage monitoring/alerting set up
- [ ] Backup strategy for checkpoint database
- [ ] Security: Row-level isolation for multi-tenant
- [ ] HITL timeouts configured
- [ ] Audit logging enabled for compliance
- [ ] Recovery testing performed (simulate crashes)

---

## 11. Code Examples

### Example 1: Basic Chatbot with Memory

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# State definition
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Node: Respond to user
def chatbot(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile with checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Use with thread_id for persistence
config = {"configurable": {"thread_id": "user_123"}}

# First message
result1 = app.invoke(
    {"messages": [{"role": "user", "content": "Hi, I'm Noah"}]},
    config
)
print(result1["messages"][-1].content)  # "Hello Noah!"

# Second message - remembers first!
result2 = app.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config
)
print(result2["messages"][-1].content)  # "Your name is Noah!"
```

### Example 2: Multi-User Chat with SQLite

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Persistent storage across restarts
checkpointer = SqliteSaver.from_conn_string("chat_history.sqlite")
app = graph.compile(checkpointer=checkpointer)

# User A's conversation
config_a = {"configurable": {"thread_id": "alice"}}
app.invoke({"messages": [{"role": "user", "content": "I love pizza"}]}, config_a)

# User B's conversation - completely separate
config_b = {"configurable": {"thread_id": "bob"}}
app.invoke({"messages": [{"role": "user", "content": "I hate pizza"}]}, config_b)

# Restart Python, reload from SQLite
checkpointer2 = SqliteSaver.from_conn_string("chat_history.sqlite")
app2 = graph.compile(checkpointer=checkpointer2)

# Alice's history preserved!
result = app2.invoke(
    {"messages": [{"role": "user", "content": "What food do I like?"}]},
    config_a
)
# Response: "You love pizza!"
```

### Example 3: Human-in-the-Loop Approval

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    request: str
    analysis: str
    approved: bool

def analyze_request(state):
    # AI analyzes the request
    return {"analysis": "This is a high-risk action"}

def human_approval(state):
    # This node will be interrupted
    print(f"Human review needed: {state['analysis']}")
    # In real app, wait for UI input
    return {"approved": True}

def execute_action(state):
    if state["approved"]:
        return {"result": "Action executed"}
    return {"result": "Action rejected"}

# Build graph
graph = StateGraph(ApprovalState)
graph.add_node("analyze", analyze_request)
graph.add_node("human_approval", human_approval)
graph.add_node("execute", execute_action)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "human_approval")
graph.add_edge("human_approval", "execute")
graph.add_edge("execute", END)

# Compile with interrupt before human_approval
checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"]  # ⚡ Pause here
)

config = {"configurable": {"thread_id": "approval_123"}}

# First run - pauses before human_approval
result = app.invoke({"request": "Delete database"}, config)
# ⏸️ Execution paused, waiting for human

# Human reviews (via UI, API, etc.) and approves
# Resume execution
result = app.invoke(None, config)
# ✅ Continues from human_approval node
```

### Example 4: Time Travel Debugging

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "debug_session"}}

# Run workflow
app.invoke({"input": "test"}, config)

# Get full execution history
history = list(app.get_state_history(config))

print(f"Total checkpoints: {len(history)}")

# Inspect each checkpoint
for i, checkpoint in enumerate(history):
    print(f"\n=== Checkpoint {i} ===")
    print(f"Created: {checkpoint.metadata['created_at']}")
    print(f"State: {checkpoint.values}")

# Rewind to checkpoint 3 and modify
checkpoint_3 = history[2]  # 0-indexed
modified_state = checkpoint_3.values.copy()
modified_state["input"] = "modified_test"

# Re-run from checkpoint 3
app.invoke(modified_state, config)
```

### Example 5: Long-Term Memory with External Store

```python
import json
from pathlib import Path

# Simple file-based long-term memory
MEMORY_FILE = "long_term_memory.json"

def load_memory(user_id: str) -> dict:
    if Path(MEMORY_FILE).exists():
        all_memories = json.load(open(MEMORY_FILE))
        return all_memories.get(user_id, {})
    return {}

def save_memory(user_id: str, memory: dict):
    all_memories = {}
    if Path(MEMORY_FILE).exists():
        all_memories = json.load(open(MEMORY_FILE))
    all_memories[user_id] = memory
    json.dump(all_memories, open(MEMORY_FILE, "w"))

# In your graph node
def personalize_response(state):
    user_id = state["user_id"]
    
    # Load long-term memory
    prefs = load_memory(user_id)
    
    # Use in response
    tone = prefs.get("tone", "professional")
    response = f"Hello! (Using {tone} tone as preferred)"
    
    # Update memory based on conversation
    if "casual" in state["messages"][-1]["content"]:
        prefs["tone"] = "casual"
        save_memory(user_id, prefs)
    
    return {"response": response}
```

### Example 6: Production-Ready Setup with PostgreSQL

```python
from langgraph.checkpoint.postgres import PostgresSaver
import os

# Production database connection
DB_URL = os.getenv("LANGGRAPH_DB_URL", 
    "postgresql://user:pass@localhost:5432/langgraph")

checkpointer = PostgresSaver.from_conn_string(DB_URL)
app = graph.compile(checkpointer=checkpointer)

# Thread management
def create_thread(user_id: str) -> str:
    """Create a new thread for a user"""
    import uuid
    thread_id = f"{user_id}_{uuid.uuid4()}"
    
    # Initialize thread metadata in DB
    checkpointer.put(
        {"configurable": {"thread_id": thread_id}},
        {"created_by": user_id, "created_at": datetime.now()}
    )
    
    return thread_id

def get_thread_history(thread_id: str) -> list:
    """Get full execution history for a thread"""
    config = {"configurable": {"thread_id": thread_id}}
    return list(app.get_state_history(config))

def cleanup_old_threads(days: int = 30):
    """Delete threads older than specified days"""
    # Implement based on your DB schema
    pass
```

---

## Quick Reference Cheat Sheet

### State Reducers

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Overwrite (latest value wins)
    current_draft: str
    
    # Append to list
    results: Annotated[list, operator.add]
    
    # Smart message handling (handles deletions)
    messages: Annotated[list, add_messages]
```

### Checkpointer Setup

```python
# Development
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# Single-user app
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")

# Production
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(DB_URL)
```

### Thread Configuration

```python
config = {
    "configurable": {
        "thread_id": "unique_session_id"
    }
}
result = app.invoke(state, config)
```

### Human-in-the-Loop

```python
# Interrupt before specific nodes
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_name"]
)

# Resume after human input
result = app.invoke(None, config)
```

### Time Travel

```python
# Get history
history = list(app.get_state_history(config))

# Rewind and modify
checkpoint = history[n]
modified = checkpoint.values.copy()
# ... modify state ...

# Re-execute from checkpoint
app.invoke(modified, config)
```

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| **All users share same thread** | Generate unique thread_id per user/session |
| **Checkpoints grow infinitely** | Implement retention policy (delete old checkpoints) |
| **Sensitive data in checkpoints** | Store references, not actual secrets |
| **Too many interrupts** | Only interrupt at critical decision points |
| **Forgot to compile with checkpointer** | Always pass `checkpointer=` to `compile()` |
| **Not handling concurrent modifications** | Use optimistic locking or version numbers |
| **MemorySaver in production** | Use SQLite/PostgreSQL for persistence |

---

## When to Use What

### Choose MemorySaver when:
- ✅ Developing locally
- ✅ Running tests
- ✅ Quick demos
- ❌ NOT for production

### Choose SqliteSaver when:
- ✅ Single-user desktop app
- ✅ Prototyping with persistence
- ✅ Moderate data volume
- ❌ NOT for high-concurrency

### Choose PostgresSaver when:
- ✅ Production systems
- ✅ Multi-user applications
- ✅ Need reliability & scaling
- ✅ Compliance requirements

### Choose RedisSaver when:
- ✅ High-performance needs
- ✅ Distributed architecture
- ✅ Real-time applications
- ✅ Already using Redis

---

## Summary

**Persistence in LangGraph enables:**

1. ✅ **Fault tolerance** - Recover from failures without losing progress
2. ✅ **Memory** - Maintain context across multiple interactions
3. ✅ **Threads** - Isolate different users/sessions
4. ✅ **Human-in-the-loop** - Pause for human approval
5. ✅ **Time travel** - Rewind and modify execution
6. ✅ **Debugging** - Inspect any previous state
7. ✅ **Compliance** - Audit trail of all decisions

**Key Concepts:**

- **Checkpointer**: Saves state at every step
- **Thread**: Isolated session with its own history
- **Short-term memory**: Within-thread state (automatic)
- **Long-term memory**: Cross-thread storage (manual)
- **Interrupt**: Pause point for human input
- **Time travel**: Rewind to any checkpoint

**Production Checklist:**

- [ ] Use persistent checkpointer (not MemorySaver)
- [ ] Unique thread IDs per user/session
- [ ] Checkpoint retention policy
- [ ] Monitoring and alerting
- [ ] Security and isolation
- [ ] Backup strategy

---

## Additional Resources

- **Official LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **Checkpointing Guide:** https://langchain-ai.github.io/langgraph/how-tos/persistence/
- **Human-in-the-Loop:** https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
- **This Project's Chatbot Example:** See `6_Basic_Chatbot/1_basic_chatbot.ipynb`

---

*This guide is part of the easy-langgraph project.*
