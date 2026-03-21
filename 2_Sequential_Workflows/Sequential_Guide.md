# Sequential Workflows in LangGraph

## The Assembly Line Pattern

Think of sequential workflows as an **assembly line for AI tasks**. Just as a car factory moves a chassis through stations where each worker adds something specific, LangGraph moves data through nodes where each node performs one focused operation.

This is **Prompt Chaining** — the first and most fundamental pattern in agentic AI design.

---

## What Makes Sequential Workflows Tick

### The Core Idea

A sequential workflow is a directed graph where execution flows in one direction:

```
START → [Node A] → [Node B] → [Node C] → END
```

Three principles govern this flow:

1. **Shared State** — All nodes read from and write to a common data object
2. **Partial Updates** — Nodes return only what they changed, not the entire state
3. **Fixed Order** — Execution follows a predetermined path with no branching

### When This Pattern Shines

Use sequential workflows when:

- Your task naturally breaks into ordered subtasks
- Each step depends on the previous step's output
- You want predictable, debuggable execution
- You're building your first LangGraph application

Avoid when you need conditional logic, parallel execution, or loops — those require advanced patterns.

---

## Anatomy of a Sequential Workflow

### 1. State: The Shared Whiteboard

State is a TypedDict that defines what data flows through your graph:

```python
from typing import TypedDict

class ResearchState(TypedDict):
    topic: str           # User's research question
    search_results: list # Web search results
    summary: str         # Generated summary
    citations: list      # Source references
```

Every node sees this entire structure. Every node can add to it.

### 2. Nodes: Specialized Workers

A node is a function that transforms state:

```python
def search_node(state: ResearchState) -> dict:
    """Search the web for relevant information"""
    results = web_search(state['topic'])
    return {'search_results': results}  # Partial update only

def summarize_node(state: ResearchState) -> dict:
    """Summarize search results"""
    summary = llm.summarize(state['search_results'])
    return {'summary': summary}
```

Notice the pattern: **read from state, do work, return changes**.

### 3. Edges: The Conveyor Belt

Edges connect nodes and define execution order:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ResearchState)

graph.add_node('search', search_node)
graph.add_node('summarize', summarize_node)

graph.add_edge(START, 'search')
graph.add_edge('search', 'summarize')
graph.add_edge('summarize', END)

workflow = graph.compile()
```

---

## Real-World Example: Content Generation Pipeline

Here's a production-ready pattern for generating structured content:

```python
class ContentState(TypedDict):
    topic: str
    outline: str
    draft: str
    word_count: int
    quality_score: float

def generate_outline(state: ContentState) -> dict:
    prompt = f"Create a detailed outline for: {state['topic']}"
    outline = llm.invoke(prompt).content
    return {'outline': outline}

def write_draft(state: ContentState) -> dict:
    prompt = f"""Write a comprehensive article using this outline:
    
    Topic: {state['topic']}
    Outline: {state['outline']}
    """
    draft = llm.invoke(prompt).content
    return {'draft': draft}

def count_words(state: ContentState) -> dict:
    count = len(state['draft'].split())
    return {'word_count': count}

def evaluate_quality(state: ContentState) -> dict:
    prompt = f"Score this article 0-10: {state['draft'][:500]}..."
    score = float(llm.invoke(prompt).content)
    return {'quality_score': score}

# Build the pipeline
graph = StateGraph(ContentState)
graph.add_node('outline', generate_outline)
graph.add_node('draft', write_draft)
graph.add_node('metrics', count_words)
graph.add_node('evaluate', evaluate_quality)

graph.add_edge(START, 'outline')
graph.add_edge('outline', 'draft')
graph.add_edge('draft', 'metrics')
graph.add_edge('metrics', 'evaluate')
graph.add_edge('evaluate', END)

pipeline = graph.compile()

# Execute
result = pipeline.invoke({'topic': 'Quantum Computing Basics'})
print(f"Quality: {result['quality_score']}/10, Words: {result['word_count']}")
```

---

## The Power of Decomposition

Why break tasks into chains instead of one big prompt?

### Single Prompt (Fragile)
```
"Write a research report on climate change"
```
Problems: Unstructured output, no validation points, hard to debug, inconsistent quality.

### Chained Prompts (Robust)
```
1. "Generate report outline for climate change"
2. [Validate outline has required sections]
3. "Research each section from the outline"
4. "Write full report using research and outline"
5. "Review and improve the report"
```

Benefits: Each step is focused, you can validate between steps, errors are isolated, quality is consistent.

---

## Adding Validation Gates

Gates are checkpoints that validate output before proceeding:

```python
def outline_validator(outline: str) -> bool:
    """Ensure outline meets quality standards"""
    required_sections = ['Introduction', 'Methods', 'Results', 'Conclusion']
    return all(section in outline for section in required_sections)

def generate_outline(state: ContentState) -> dict:
    prompt = f"Create outline for: {state['topic']}"
    outline = llm.invoke(prompt).content
    
    # Gate: validate before passing to next node
    if not outline_validator(outline):
        raise ValueError("Outline missing required sections")
    
    return {'outline': outline}
```

For more sophisticated handling, use conditional edges to retry or route to fallback nodes.

---

## Design Principles

### One Node, One Responsibility

Each node should have a single, clear purpose:

```python
# Good: Focused nodes
def extract_keywords(state): ...
def classify_sentiment(state): ...
def generate_response(state): ...

# Bad: Monolithic node
def process_everything(state):
    # 100 lines of extraction, classification, and generation
    ...
```

### Name Nodes Intentionally

Node names appear in logs, traces, and visualizations:

```python
# Clear
graph.add_node('validate_input', validate)
graph.add_node('fetch_context', fetch)
graph.add_node('generate_answer', generate)

# Unclear
graph.add_node('step1', func1)
graph.add_node('processor', func2)
```

### Return Minimal Updates

Only return fields that changed:

```python
# Correct
def process(state):
    result = compute(state['input'])
    return {'result': result}

# Unnecessary verbosity
def process(state):
    state['result'] = compute(state['input'])
    return state
```

---

## Common Sequential Patterns

### Extract → Transform → Load
```
Input → Parse JSON → Validate schema → Store in database → Done
```

### Plan → Execute → Verify
```
Problem → Generate plan → Execute plan → Check output → Result
```

### Translate Pipeline
```
Source → Detect language → Translate → Proofread → Format → Output
```

### Data Enrichment
```
Record → Fetch external data → Merge → Validate → Transform → Export
```

---

## State Evolution Example

Watch how state accumulates through a customer support workflow:

```
START
  ↓
{ticket: "Can't login", priority: null, response: null}
  ↓
[classify_node runs]
  ↓
{ticket: "Can't login", priority: "HIGH", response: null}
  ↓
[lookup_node runs]
  ↓
{ticket: "Can't login", priority: "HIGH", response: null, 
 context: {user_locked: True, attempts: 5}}
  ↓
[generate_response_node runs]
  ↓
{ticket: "Can't login", priority: "HIGH", 
 response: "Your account is locked. Reset link sent...", 
 context: {user_locked: True, attempts: 5}}
  ↓
END
```

Each node adds information. The final node sees everything.

---

## When Sequential Isn't Enough

Sequential workflows have limits. Recognize when you need more:

| Need | Pattern |
|------|---------|
| Route based on input type | Conditional edges |
| Process multiple items concurrently | Send API / Parallel |
| Retry until quality threshold met | Evaluator-optimizer loop |
| Dynamic task decomposition | Orchestrator-worker |

---

## Key Takeaways

1. **Sequential = Predictable** — Fixed order, easy to trace and debug
2. **State flows forward** — Each node builds on previous work
3. **Decompose complex tasks** — Multiple focused prompts beat one complex prompt
4. **Validate between steps** — Gates catch errors early
5. **Master this first** — Sequential workflows are the foundation for advanced patterns

---

## Practice Exercises

1. Build a three-node pipeline: Input → Uppercase → Count words → Output
2. Create a math tutor: Question → Solve step-by-step → Explain reasoning
3. Design a code reviewer: Code → Find bugs → Suggest fixes → Format report

Modify the notebooks in this folder to experiment with each pattern.

---

*Next: Explore parallel workflows in `3_Parallel_Workflows/` to learn fan-out/fan-in patterns.*
