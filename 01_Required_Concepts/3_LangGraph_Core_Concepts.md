# LangGraph Core Concepts
### A Beginner's Complete Guide — From LLM Workflows to Graphs, State, Reducers, and Agentic Patterns

---

## Part 1: What Is LangGraph Doing, Fundamentally?

> **LangGraph is an orchestration framework.** Given an LLM-powered workflow to execute, it first represents that workflow as a **graph**, and then executes the graph in a controlled, stateful manner.

Everything else — nodes, edges, state, reducers — is just the machinery that makes this representation and execution possible.

### Why a Graph?

Real-world LLM tasks are rarely a straight line. They involve:
- **Decisions** — *if the output is poor, try again; otherwise, proceed*
- **Branches** — *if it's a billing question, route here; if technical, route there*
- **Parallel work** — *analyse three documents simultaneously, then combine*
- **Loops** — *search → read → evaluate → search again until satisfied*
- **Long-running memory** — *remember what happened 10 steps ago*

A simple list of sequential steps can't represent this. A **graph** can — because a graph naturally models nodes (tasks), edges (connections), branches (conditional paths), and cycles (loops).

```
Simple list (LangChain):    A → B → C → D → END   (no loops, no branches)

Graph (LangGraph):          A → B → C → D → END
                                  ↑       |
                                  └───────┘ (loop back if condition not met)
                                      ↓
                                      E → F  (alternate branch)
```

---

## Part 2: The 5 Foundational LLM Workflow Patterns

Before LangGraph's technical machinery makes sense, you need to understand the **five fundamental patterns** that all LLM workflows are built from. These were formalised by Anthropic in their "Building Effective Agents" research and are the backbone of everything LangGraph is designed to express.

---

### Pattern 1: Prompt Chaining

**The idea:** Break a complex task into a sequence of smaller LLM calls, where each step's output becomes the next step's input.

**Analogy:** An assembly line. Each station does one focused job and passes the product forward.

```
User Input
    ↓
Step 1: LLM writes an outline
    ↓
[GATE: Does the outline meet criteria?] — if No → stop or retry
    ↓
Step 2: LLM expands each section
    ↓
Step 3: LLM polishes the language
    ↓
Final Output
```

The **gate** is a programmatic check you can insert between steps — a guardrail that validates output before proceeding.

**When to use it:**
- Tasks that decompose cleanly into ordered subtasks
- When you want each LLM call to be simpler and more focused
- When you need validation checkpoints between steps
- The trade-off: slightly slower (sequential), but higher accuracy

**Real examples:**
- Generate marketing copy → translate it → adjust tone for the audience
- Write an essay outline → check the outline → write the full essay from the outline
- Extract data → validate format → write a report

---

### Pattern 2: Routing

**The idea:** Classify an incoming input and direct it to the most appropriate specialised handler, rather than using one general prompt for everything.

**Analogy:** A hospital triage nurse. They assess your condition and route you to the right specialist — not everyone sees the same doctor.

```
Incoming Request
    ↓
ROUTER (LLM classifier)
    ├─→ Billing questions    → Billing Agent
    ├─→ Technical issues     → Tech Support Agent
    ├─→ General questions    → FAQ Agent
    └─→ Escalations          → Human Agent
```

**Why not just one prompt for everything?** Optimising a prompt for one type of input often hurts performance on others. Specialised sub-prompts for each category perform far better.

**When to use it:**
- You have clearly distinct categories of inputs requiring different handling
- Customer support (billing vs. technical vs. general)
- Coding assistants (debugging vs. explanation vs. documentation)
- Content moderation (safe vs. unsafe vs. borderline)

**Two types of classifiers:**
- **LLM-based router** — the LLM reads the input and decides the category (flexible, handles nuance)
- **Traditional classifier** — a trained ML model makes the routing decision (faster, cheaper at scale)

---

### Pattern 3: Parallelization

**The idea:** Split a task into independent sub-tasks that can all run **at the same time**, then aggregate the results.

**Analogy:** A team of researchers all working simultaneously on different chapters of the same report, then handing their chapters to an editor who assembles the final document.

```
              Input
               ↓
    ┌──────────────────────┐
    │      SPLITTER        │
    └──────┬───────┬───────┘
           ↓       ↓       ↓
       Agent A  Agent B  Agent C   ← all run in parallel
       (Topic 1)(Topic 2)(Topic 3)
           ↓       ↓       ↓
    ┌──────────────────────┐
    │    AGGREGATOR/       │
    │    SYNTHESIZER       │
    └──────────────────────┘
               ↓
           Final Output
```

**Three aggregation strategies:**
- **Concatenation** — join outputs together (e.g., append chapter summaries)
- **Voting / Majority rule** — run the same task N times, pick the most common answer (boosts confidence)
- **Synthesizer** — a dedicated LLM takes all outputs and creates a coherent combined response

**When to use it:**
- Independent subtasks (no dependencies between them)
- Large documents needing multi-aspect analysis (grammar, logic, style — all in parallel)
- When you need higher confidence through multiple independent attempts
- Speed-critical workflows where tasks can run concurrently

**LangGraph implementation:** Uses the `Send()` API to dynamically dispatch multiple parallel worker nodes.

---

### Pattern 4: Orchestrator–Worker

**The idea:** A central **orchestrator** LLM dynamically analyses an incoming task, decides what subtasks are needed (at runtime — not pre-defined), delegates them to **worker** LLMs, and synthesises the results.

**Analogy:** A project manager who reads a brief, figures out which specialists are needed, assigns tasks, and compiles the final deliverable.

```
         Complex Task Input
               ↓
    ┌─────────────────────┐
    │    ORCHESTRATOR     │  ← LLM that plans and coordinates
    │  (decomposes task,  │
    │  assigns workers)   │
    └──────┬──────┬───────┘
           ↓      ↓      ↓
       Worker  Worker  Worker   ← each does a specialised subtask
         A       B       C
           ↓      ↓      ↓
    ┌─────────────────────┐
    │    SYNTHESIZER      │  ← combines all worker outputs
    └─────────────────────┘
               ↓
           Final Result
```

**Key difference from Parallelization:**
Parallelization uses a **fixed, pre-defined** set of subtasks. Orchestrator-Worker uses **dynamic, runtime-determined** subtasks — the orchestrator decides what workers to use based on the specific input.

| | Parallelization | Orchestrator-Worker |
|---|---|---|
| Subtasks defined | Before execution | During execution |
| Flexibility | Low (fixed) | High (dynamic) |
| Use case | Uniform, predictable splits | Complex, variable tasks |

**When to use it:**
- Complex coding tasks (the orchestrator reads the problem and determines which files to change)
- Research workflows (orchestrator searches, finds new leads, decides next search directions)
- Tasks where the exact steps can't be predicted until the input is seen

**Failure point to watch:** The orchestrator becomes a bottleneck. Poor task decomposition cascades into poor worker output. Always monitor orchestrator reasoning quality.

---

### Pattern 5: Evaluator–Optimizer

**The idea:** One LLM **generates** an output; another LLM **evaluates** it against defined criteria and gives feedback; the generator uses the feedback to improve; this loop continues until quality is sufficient or a maximum number of iterations is reached.

**Analogy:** A writer and an editor. The writer drafts; the editor critiques; the writer revises; repeat until published.

```
         Input / Task
               ↓
    ┌─────────────────────┐
    │     GENERATOR       │  ← LLM that produces output
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │     EVALUATOR       │  ← LLM that scores + critiques
    └──────────┬──────────┘
               ↓
    ┌─────────────────────────────────┐
    │  Is quality good enough?        │
    │  → Yes: send to user            │
    │  → No: send feedback to         │
    │         Generator → loop again  │
    └─────────────────────────────────┘
```

**Components:**
1. **Generator** — produces the initial draft or solution
2. **Evaluator** — assesses against predefined criteria (accuracy, readability, logic, tone) and produces a score + improvement plan
3. **Stop condition** — a quality threshold OR a maximum iteration count (CRITICAL: always set a max to prevent infinite loops)

**Advanced version:** Multiple specialised evaluators — one for logic, one for readability, one for syntax — each checking a different dimension simultaneously.

**When to use it:**
- Literary translation (nuances the first pass might miss)
- Code generation (generate → security review → fix → review again)
- Complex research reports requiring multiple rounds of analysis
- Any task where first-pass LLM output is insufficient and iterative refinement adds clear value

---

### The 5 Patterns at a Glance

```
PROMPT CHAINING    →  Step A → Gate → Step B → Step C  (sequential pipeline)
ROUTING            →  Input → Classifier → Agent A or B or C  (smart if-else)
PARALLELIZATION    →  Input → [A, B, C in parallel] → Aggregator  (fan-out/fan-in)
ORCHESTRATOR-WORKER→  Input → Orchestrator → [Dynamic Workers] → Synthesizer
EVALUATOR-OPTIMIZER→  Generate → Evaluate → (loop) → Final Output
```

> **Anthropic's guiding principle:** Start with the simplest solution. Use prompt chaining before routing. Use routing before parallelization. Only add complexity when it demonstrably improves outcomes.

---

## Part 3: LangGraph Core Concept 1 — The Graph

Now we translate these workflow patterns into LangGraph's actual structure.

### StateGraph: The Foundation

Every LangGraph application starts with a `StateGraph`. This is the object you use to define your entire workflow.

```python
from langgraph.graph import StateGraph, END, START

graph = StateGraph(MyState)  # We'll define MyState in Part 4
```

Think of `StateGraph` as the **blueprint** for your workflow. You populate it by adding nodes and edges.

### The Two Virtual Nodes: START and END

Every graph has two special built-in nodes:

- **START** — the entry point. Execution begins at the node(s) connected to START.
- **END** — the termination point. When a node routes to END, the graph run is complete.

```python
graph.add_edge(START, "first_node")   # execution begins here
graph.add_edge("last_node", END)      # execution ends here
```

---

## Part 4: LangGraph Core Concept 2 — Nodes

### What Is a Node?

> A **node** is simply a Python function that receives the current state, performs some work, and returns updates to the state.

Nodes are where the **actual work** happens. Each node represents one logical step in your workflow — an LLM call, a tool invocation, a database lookup, a formatting step, or any custom logic.

```python
def research_node(state: MyState) -> dict:
    # 1. Read from state
    question = state["question"]

    # 2. Do some work
    result = search_the_web(question)

    # 3. Return updates to state (not the whole state — just what changed)
    return {"search_results": result}
```

Three things to notice:
1. The node receives the **entire state** as input
2. It does its work (LLM call, tool use, computation)
3. It returns only the **parts of the state it changed** — a partial update

### Types of Nodes

| Node Type | Description | Example |
|---|---|---|
| **LLM node** | Calls an LLM and stores the response | `generate_draft_node` |
| **Tool node** | Executes an external tool (web search, API) | `web_search_node` |
| **Logic node** | Performs conditional checks or data transformation | `validate_output_node` |
| **Human node** | Pauses for human input or approval | `human_review_node` |
| **Sub-graph node** | Embeds an entire LangGraph sub-graph as a single node | Complex reusable workflows |

### Adding Nodes to the Graph

```python
graph.add_node("research", research_node)
graph.add_node("draft",    draft_writing_node)
graph.add_node("evaluate", evaluate_quality_node)
```

---

## Part 5: LangGraph Core Concept 3 — Edges

> **Edges** define the flow of control between nodes. They answer the question: *after this node finishes, what happens next?*

Edges come in two flavours:

### Type 1: Regular (Unconditional) Edge

A fixed, always-taken connection between two nodes. After Node A, always go to Node B. No conditions.

```python
graph.add_edge("research", "draft")   # always: after research, go to draft
graph.add_edge("draft", "evaluate")   # always: after draft, go to evaluate
```

```
research → draft → evaluate
```

### Type 2: Conditional Edge

A dynamic connection that routes to different nodes depending on a **routing function**; a function that reads the current state and returns the name of the next node.

```python
def should_continue(state: MyState) -> str:
    if state["quality_score"] >= 8:
        return "done"        # go to END
    elif state["retries"] >= 3:
        return "fallback"    # too many retries, go to fallback node
    else:
        return "retry"       # loop back to generate

graph.add_conditional_edges(
    "evaluate",           # from this node
    should_continue,      # call this routing function
    {
        "done":     END,
        "retry":    "draft",      # loop back!
        "fallback": "fallback_node"
    }
)
```

This is how LangGraph implements **routing** (Pattern 2) and **evaluator-optimizer loops** (Pattern 5).

### The Send API: Dynamic Parallel Edges

For parallelization (Pattern 3) and orchestrator-worker (Pattern 4), LangGraph provides a special `Send` object that creates **dynamic edges at runtime**:

```python
from langgraph.types import Send

def assign_parallel_workers(state: MyState):
    # Dynamically create one worker per section
    return [Send("worker_node", {"section": s}) for s in state["sections"]]

graph.add_conditional_edges("orchestrator", assign_parallel_workers)
```

This fans out to as many worker nodes as there are items — decided at runtime, not at design time.

---

## Part 6: LangGraph Core Concept 4 — State

This is the most important concept in all of LangGraph. Everything revolves around state.

### What Is State?

> **State** is a shared data object (like a whiteboard) that all nodes in the graph can read from and write to. It represents the complete memory of the workflow at any moment in time.

State is defined as a `TypedDict`: a Python dictionary with typed fields.

```python
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    # The user's original question
    question: str

    # Accumulated web search results
    search_results: Annotated[list, operator.add]

    # The draft answer
    draft: str

    # The evaluator's feedback
    feedback: str

    # Quality score (0-10)
    quality_score: int

    # Number of retry attempts
    retries: int

    # The final polished answer
    final_answer: str
```

### The Two Key Properties of State

**1. Shared across all nodes**
Every single node in your graph has access to the complete state. When a node is executed, the first thing it receives is the entire current state. Nothing is hidden from any node.

**2. Mutable — it evolves as execution progresses**
Each node can update any part of the state. The state is the living record of everything the workflow has done, learned, and decided. By the time you reach the final node, the state contains every intermediate result from every prior step.

```
START → Node A reads state, adds "search_results"
              ↓
        Node B reads state (has search_results now), adds "draft"
              ↓
        Node C reads state (has both), adds "feedback" and "quality_score"
              ↓
        Node D reads state (has all), produces "final_answer"
              ↓
END → final state contains everything
```

### State Scope: Short-Term vs Long-Term Memory

| Memory Type | Scope | Where Stored | Use Case |
|---|---|---|---|
| **Short-term** | Current execution / thread | In-graph state object | Conversation history, intermediate outputs |
| **Long-term** | Across sessions | External DB / vector store | User preferences, past decisions, accumulated knowledge |

Short-term memory flows through the state automatically. Long-term memory requires external stores that nodes query and update explicitly.

---

## Part 7: LangGraph Core Concept 5 — Reducers

### The Problem Reducers Solve

Imagine two parallel nodes both try to update the same state field at the same time. Without a rule for how to handle this, you'd get a conflict where one update would blindly overwrite the other.

Or imagine a node that generates a new message in a conversation. Do you want the new message to **replace** the conversation history, or **append to** it?

This is exactly what **reducers** solve.

> **A reducer** is a function attached to a state field that defines *how* updates from nodes are merged into the existing value for that field.

### The Three Reducer Behaviours

#### Behaviour 1: Overwrite (Default)
If no reducer is specified, any update simply **replaces** the existing value. Fine for scalar fields that should always hold the latest value.

```python
class State(TypedDict):
    draft: str          # no reducer → latest value overwrites previous
    quality_score: int  # no reducer → latest score overwrites previous
```

```
Current state: {"draft": "first version"}
Node update:   {"draft": "improved version"}
Result:        {"draft": "improved version"}  ← replaced
```

#### Behaviour 2: Append (operator.add)
Use Python's `operator.add` reducer to **append** new items to a list rather than replace the whole list. This is the standard pattern for accumulating messages or results.

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]   # append, don't replace
```

```
Current state: {"messages": ["Hello", "How can I help?"]}
Node update:   {"messages": ["I need help with billing"]}
Result:        {"messages": ["Hello", "How can I help?", "I need help with billing"]}  ← appended
```

This is also how **parallel nodes** can all safely contribute results to the same list; each node's output gets appended without overwriting each other.

#### Behaviour 3: Custom Reducer Function
You can write any Python function as a reducer for complete control:

```python
def merge_dicts(existing: dict, update: dict) -> dict:
    """Merge two dictionaries, keeping existing keys and adding new ones."""
    return {**existing, **update}

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]   # custom merge logic
```

### The Special Case: add_messages

LangGraph provides a pre-built reducer specifically for **conversation history** — `add_messages`. It not only appends new messages but also handles message deletion by ID, making it the recommended way to manage chat history:

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]   # smart message management
```

### Reducers Summary Table

| Reducer | Behaviour | Use For |
|---|---|---|
| None (default) | Overwrite | Single values: status, scores, current draft |
| `operator.add` | Append to list | Accumulating results, search outputs |
| `add_messages` | Smart message append | Conversation history |
| Custom function | Any logic you define | Dictionaries, custom aggregation |

---

## Part 8: LangGraph Core Concept 6 — The Execution Model (Pregel)

### Inspiration: Google's Pregel

LangGraph's execution engine is directly inspired by **Google's Pregel** system — a framework originally designed for large-scale graph processing.

### How Execution Works: Super-Steps

When you run a LangGraph workflow, execution proceeds in discrete **super-steps**:

> A **super-step** is one full iteration over the active nodes in the graph.

Here's the sequence:

```
1. All nodes begin in an INACTIVE state.

2. Graph execution starts — the START node sends the initial state to
   the first connected node(s), making them ACTIVE.

3. Each active node:
   a. Receives the current state as a message
   b. Executes its function
   c. Returns state updates
   d. Sends messages along its outgoing edges to the next node(s)
   e. Returns to INACTIVE

4. The next set of nodes becomes ACTIVE (they've received messages).
   They execute — this is the next super-step.

5. This continues until a node routes to END, completing the run.
```

### Parallel Nodes = Same Super-Step

**Nodes that run in parallel belong to the same super-step.**
**Nodes that run sequentially belong to different super-steps.**

```
Super-step 1:  orchestrator runs (sequential)
Super-step 2:  worker_A, worker_B, worker_C all run (parallel — same step)
Super-step 3:  synthesizer runs (sequential — waits for all workers)
```

This is why reducers are so important...when multiple parallel nodes in the same super-step all update the same state field, the reducer determines how to combine their contributions without conflict.

### Message Passing

The underlying mechanism is **message passing** — each node sends its state updates as messages along edges to the next node(s). This is the same model Pregel uses for distributed graph computation. It's what gives LangGraph its ability to handle both sequential and parallel execution within the same unified framework.

### The Full Execution Flow Visualised

```
                    ┌───────────┐
                    │   START   │
                    └─────┬─────┘
                          │ (initial state as message)
                    ┌─────▼─────┐
 Super-step 1:      │  Node A   │ → executes → sends messages to B and C
                    └─────┬─────┘
                     ┌────┴────┐
 Super-step 2:  ┌────▼───┐ ┌───▼────┐  ← B and C run IN PARALLEL
                │ Node B │ │ Node C │
                └────┬───┘ └───┬────┘
                     └────┬────┘
                    ┌─────▼─────┐
 Super-step 3:      │  Node D   │ → conditional edge →
                    └─────┬─────┘
                          │
              ┌───────────┴────────────┐
              ↓                        ↓
           (END)                  (loop to A)
```

---

## Part 9: LangGraph Core Concept 7 — Checkpointing

### What Is Checkpointing?

> **Checkpointing** saves a complete snapshot of the graph's state after every super-step — like a save point in a video game.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### What Does This Enable?

**1. Fault Tolerance**
If your app crashes mid-execution, the next run picks up from the last checkpoint — not from zero. For a 20-step workflow, a crash on step 19 doesn't mean starting over.

**2. Human-in-the-Loop**
Execution can be paused at a specific node, waiting for a human to review the state, make changes, and give the green light to continue:

```python
# Define an interrupt point before a sensitive action
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email_node"]  # pause here for human review
)

# First run: executes up to the interrupt
result = app.invoke(inputs, config={"configurable": {"thread_id": "123"}})

# Human reviews state, approves...

# Second run: resumes from checkpoint
result = app.invoke(None, config={"configurable": {"thread_id": "123"}})
```

**3. Long-Running Workflows**
Workflows that span hours, days, or even weeks — because state is persisted between sessions, not held in memory.

**4. Time Travel Debugging**
Rewind to any previous checkpoint and re-run from that point with different inputs or a modified state. Invaluable for debugging complex agent behaviour.

### Checkpointer Options

| Checkpointer | Storage | Use Case |
|---|---|---|
| `MemorySaver` | RAM | Development and testing |
| `SqliteSaver` | Local SQLite file | Single-machine production |
| `PostgresSaver` | PostgreSQL database | Distributed / production scale |
| Custom | Any store | Enterprise requirements |

---

## Part 10: Human-in-the-Loop — The Critical Safety Feature

For any high-stakes application like healthcare, finance, legal you need a human in the loop. LangGraph makes this a first-class feature.

### The Pattern

```
Agent proposes an action
    ↓
INTERRUPT → state saved to checkpoint
    ↓
Human reviews the state (reads the proposed action)
    ↓
Human approves, edits, or rejects
    ↓
Execution resumes from checkpoint with human's input incorporated
```

### Why This Matters

Without human-in-the-loop, a fully autonomous agent might:
- Send an email you didn't intend
- Execute a financial transaction based on a misunderstood instruction
- Take an irreversible action after misinterpreting ambiguous input

With interrupt points, the agent does all the heavy lifting (research, drafting, planning) and humans provide the final approval on consequential actions. You get the efficiency of automation with the safety of human oversight.

---

## Part 11: Putting It All Together — A Complete Example

Let's see how all the concepts combine into a real workflow: an **essay writer with quality evaluation**.

### The Workflow Design

```
User provides topic
    ↓
[research_node]  — searches for relevant information
    ↓
[draft_node]     — writes the essay using research
    ↓
[evaluate_node]  — scores quality (1-10) and provides feedback
    ↓
    ├─ score ≥ 8 → [polish_node] → END
    └─ score < 8  → back to [draft_node]  (loop: evaluator-optimizer pattern)
```

### The State

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class EssayState(TypedDict):
    topic: str                                  # user's input topic
    research: str                               # gathered research
    draft: str                                  # current draft
    feedback: str                               # evaluator's feedback
    quality_score: int                          # evaluator's score
    iteration: int                              # how many drafts we've tried
    final_essay: str                            # polished final output
    messages: Annotated[list, add_messages]     # full conversation log
```

### The Nodes

```python
def research_node(state: EssayState) -> dict:
    results = search_web(state["topic"])
    return {"research": results}

def draft_node(state: EssayState) -> dict:
    feedback = state.get("feedback", "")
    prompt = f"Write about {state['topic']}. Research: {state['research']}"
    if feedback:
        prompt += f"\n\nIncorporate this feedback: {feedback}"
    draft = llm.invoke(prompt)
    return {
        "draft": draft.content,
        "iteration": state.get("iteration", 0) + 1
    }

def evaluate_node(state: EssayState) -> dict:
    evaluation = evaluator_llm.invoke(
        f"Rate this essay 1-10 and provide feedback:\n{state['draft']}"
    )
    return {
        "quality_score": evaluation.score,
        "feedback": evaluation.feedback
    }

def polish_node(state: EssayState) -> dict:
    polished = llm.invoke(f"Polish this essay:\n{state['draft']}")
    return {"final_essay": polished.content}
```

### The Routing Logic (Conditional Edge)

```python
def should_continue(state: EssayState) -> str:
    if state["quality_score"] >= 8:
        return "polish"       # good enough, proceed
    elif state["iteration"] >= 3:
        return "polish"       # max iterations reached, proceed anyway
    else:
        return "redraft"      # loop back
```

### Building the Graph

```python
from langgraph.graph import StateGraph, END, START

builder = StateGraph(EssayState)

# Add nodes
builder.add_node("research",  research_node)
builder.add_node("draft",     draft_node)
builder.add_node("evaluate",  evaluate_node)
builder.add_node("polish",    polish_node)

# Add edges
builder.add_edge(START,        "research")
builder.add_edge("research",   "draft")
builder.add_edge("draft",      "evaluate")
builder.add_conditional_edges(
    "evaluate",
    should_continue,
    {"polish": "polish", "redraft": "draft"}
)
builder.add_edge("polish", END)

# Compile with checkpointing
from langgraph.checkpoint.memory import MemorySaver
app = builder.compile(checkpointer=MemorySaver())
```

### Running It

```python
result = app.invoke(
    {"topic": "The impact of AI on healthcare"},
    config={"configurable": {"thread_id": "essay-001"}}
)
print(result["final_essay"])
```

The graph will automatically loop between `draft` → `evaluate` → `draft` until the quality score reaches 8, then polish and finish. The checkpointer saves state after every step.

---

## Summary:

```
┌─────────────────────────────────────────────────────────────────┐
│                    A LangGraph Workflow                         │
│                                                                 │
│  PATTERNS used:                                                 │
│  • Prompt Chaining  — sequential steps via regular edges        │
│  • Routing          — conditional edges + routing functions     │
│  • Parallelization  — Send() API + reducers for fan-out         │
│  • Orchestrator-Worker — dynamic Send() from orchestrator node  │
│  • Evaluator-Optimizer — loop-back conditional edges            │
│                                                                 │
│  BUILT FROM:                                                    │
│  • Nodes    — Python functions that do the work                 │
│  • Edges    — connections that control flow                     │
│  • State    — shared TypedDict whiteboard all nodes access      │
│  • Reducers — rules for how state fields are updated            │
│                                                                 │
│  POWERED BY:                                                    │
│  • Pregel execution — message-passing, super-steps              │
│  • Checkpointing    — persistence, fault tolerance, HITL        │
│                                                                 │
│  RESULT:                                                        │
│  A stateful, cyclical, production-grade agentic workflow        │
└─────────────────────────────────────────────────────────────────┘
```

---

*Sources: LangGraph official documentation (docs.langchain.com/oss/python/langgraph), Anthropic "Building Effective Agents" research, Google Pregel paper, DataCamp, Towards Data Science (Sachin Soni, Shuvrajyoti Debroy), DEV Community (Sreeni5018), Medium (Armen Barsegyan), LangGraph Cheatsheet (sumanmichael.github.io), HuggingFace Blog (dcarpintero), Workflows.guru, Arize AX Docs, Neural Creators Substack, and Krish Naik (krishnaik.in).*