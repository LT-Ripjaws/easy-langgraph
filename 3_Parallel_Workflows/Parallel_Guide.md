# Parallel Workflows in LangGraph

## The Power of Concurrent Execution

Parallel workflows unlock LangGraph's ability to execute multiple nodes simultaneously. Instead of processing tasks one after another, you can fan-out work to multiple workers and fan-in the results — dramatically reducing execution time for independent operations.

This pattern directly implements **Parallelization** — one of the five foundational LLM workflow patterns from Anthropic's research.

---

## Understanding Parallel Execution

### The Core Concept

In a parallel workflow, multiple nodes execute in the same **super-step** — LangGraph's term for a single iteration over active nodes. All parallel nodes receive the same input state, perform their work independently, and their outputs are merged before passing to the next node.

```
                START
                  ↓
        ┌─────────┼─────────┐
        ↓         ↓         ↓
    [Node A]  [Node B]  [Node C]   ← All run concurrently
        ↓         ↓         ↓
        └─────────┼─────────┘
                  ↓
              [Node D]         ← Waits for all three
                  ↓
                END
```

### When to Use Parallel Workflows

Use parallel execution when:

- Tasks are **independent** — no dependencies between sub-tasks
- You need **speed** — concurrent execution reduces total runtime
- You want **multiple perspectives** — different analyses on the same input
- You're doing **comparative evaluation** — run the same task multiple times for confidence

Avoid when:
- Tasks have dependencies on each other
- Order of execution matters
- Resources are limited (rate limits, API quotas)

---

## Two Types of Parallelism in LangGraph

### 1. Static Parallelism (Multiple START Edges)

The simplest form — connect multiple nodes directly to START. All execute immediately when the workflow begins.

```python
graph = StateGraph(MyState)

graph.add_node('analyze_a', func_a)
graph.add_node('analyze_b', func_b)
graph.add_node('analyze_c', func_c)
graph.add_node('combine', combine_func)

# All three start in parallel
graph.add_edge(START, 'analyze_a')
graph.add_edge(START, 'analyze_b')
graph.add_edge(START, 'analyze_c')

# All must complete before combine runs
graph.add_edge('analyze_a', 'combine')
graph.add_edge('analyze_b', 'combine')
graph.add_edge('analyze_c', 'combine')
graph.add_edge('combine', END)
```

**Use case:** Fixed number of parallel operations known at design time.

### 2. Dynamic Parallelism (Send API)

The `Send` API spawns worker nodes at **runtime** — the number of parallel branches is determined by the data, not the code.

```python
from langgraph.types import Send

def orchestrator(state: MyState):
    # Dynamically create workers based on input
    items = state['items_to_process']
    return [Send('worker', {'item': item}) for item in items]

graph.add_node('orchestrator', orchestrator)
graph.add_node('worker', worker_func)
graph.add_node('synthesizer', synthesize_func)

graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges('orchestrator', orchestrator)
graph.add_edge('worker', 'synthesizer')
graph.add_edge('synthesizer', END)
```

**Use case:** Variable number of parallel tasks determined at runtime.

---

## The Critical Role of Reducers

Parallel execution requires careful state management. When multiple nodes update the same field simultaneously, LangGraph needs rules for merging — this is where **reducers** come in.

### Default Behavior: Last Write Wins

Without a reducer, if two nodes update the same field, the last one to complete overwrites the first:

```python
class State(TypedDict):
    result: str  # No reducer → last update wins

# Node A returns {'result': 'A'}
# Node B returns {'result': 'B'}
# Final state: {'result': 'B'}  ← Node A's update lost!
```

### Using operator.add for Accumulation

For lists, use `operator.add` to concatenate results from all parallel nodes:

```python
from typing import Annotated
import operator

class State(TypedDict):
    scores: Annotated[list[int], operator.add]

# Node A returns {'scores': [7]}
# Node B returns {'scores': [8]}
# Node C returns {'scores': [6]}
# Final state: {'scores': [7, 8, 6]}  ← All preserved!
```

### Custom Reducers

For complex merging logic, define your own reducer function:

```python
def merge_scores(existing: list[int], update: list[int]) -> list[int]:
    return sorted(existing + update, reverse=True)

class State(TypedDict):
    top_scores: Annotated[list[int], merge_scores]
```

---

## Example 1: Cricket Statistics Analysis

A straightforward parallel workflow computing multiple metrics simultaneously.

**File:** `3_Parallel_Workflows/1_batsman_workflow.ipynb`

### State Definition

```python
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    
    # Computed metrics (parallel)
    sr: float
    bpb: float
    boundary_percent: float
    
    # Final summary
    summary: str
```

### Parallel Metric Computation

```python
def calculate_sr(state: BatsmanState):
    sr = (state['runs'] / state['balls']) * 100
    return {'sr': sr}

def calculate_bpb(state: BatsmanState):
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb': bpb}

def calculate_boundary_percent(state: BatsmanState):
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100
    return {'boundary_percent': boundary_percent}

def summary(state: BatsmanState):
    summary_text = f"""
    Strike Rate: {state['sr']}
    Balls per boundary: {state['bpb']}
    Boundary percent: {state['boundary_percent']}
    """
    return {'summary': summary_text}
```

### Graph Construction

```python
graph = StateGraph(BatsmanState)

# Add all nodes
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)

# Parallel execution: all three metrics computed simultaneously
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

# Fan-in: summary waits for all three
graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')
graph.add_edge('summary', END)

workflow = graph.compile()
```

### Why This Works

Each metric calculation is **independent** — none depends on another's output. Running them in parallel reduces execution time and demonstrates the fan-out/fan-in pattern clearly.

---

## Example 2: Multi-Dimensional Essay Evaluation

A more sophisticated parallel workflow using LLMs to evaluate an essay from multiple angles.

**File:** `3_Parallel_Workflows/2_Essay_LLM_workflow.ipynb`

### Structured Output Schemas

```python
from pydantic import BaseModel, Field

class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback')
    score: int = Field(description='Score 0-10', ge=0, le=10)

# Create structured model
structured_model = model.with_structured_output(EvaluationSchema)
```

### State with Reducer

```python
from typing import Annotated
import operator

class LLMEssayState(TypedDict):
    essay: str
    
    # Parallel evaluation results
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    
    # Accumulated scores from all evaluators
    individual_scores: Annotated[list[int], operator.add]
    
    # Final aggregated results
    overall_feedback: str
    avg_score: float
```

### Parallel Evaluator Nodes

```python
def evaluate_language(state: LLMEssayState):
    prompt = f'Evaluate the language quality and provide feedback with a score out of 10:\n{state["essay"]}'
    output = structured_model.invoke(prompt)
    return {
        'language_feedback': output.feedback,
        'individual_scores': [output.score]
    }

def evaluate_analysis(state: LLMEssayState):
    prompt = f'Evaluate the depth of analysis and provide feedback with a score out of 10:\n{state["essay"]}'
    output = structured_model.invoke(prompt)
    return {
        'analysis_feedback': output.feedback,
        'individual_scores': [output.score]
    }

def evaluate_thought(state: LLMEssayState):
    prompt = f'Evaluate the clarity of thought and provide feedback with a score out of 10:\n{state["essay"]}'
    output = structured_model.invoke(prompt)
    return {
        'clarity_feedback': output.feedback,
        'individual_scores': [output.score]
    }
```

### Aggregation Node

```python
def final_evaluation(state: LLMEssayState):
    # Synthesize all feedback
    prompt = f'''Based on the following feedbacks, create a summarized feedback:
    Language feedback: {state['language_feedback']}
    Analysis feedback: {state['analysis_feedback']}
    Clarity feedback: {state['clarity_feedback']}'''
    
    overall_feedback = model.invoke(prompt).content
    
    # Calculate average from accumulated scores
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])
    
    return {
        'overall_feedback': overall_feedback,
        'avg_score': avg_score
    }
```

### Graph with Parallel Execution

```python
graph = StateGraph(LLMEssayState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# Fan-out: all three evaluators run in parallel
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

# Fan-in: final evaluation waits for all
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')
graph.add_edge('final_evaluation', END)

workflow = graph.compile()
```

### The Power of This Pattern

Three independent LLM calls execute **concurrently**, each evaluating a different dimension. The reducer automatically collects all scores into a list, and the final node synthesizes everything. Total execution time is roughly that of a **single** LLM call instead of three sequential ones.

---

## The Send API: Dynamic Parallelism

For workflows where the number of parallel branches isn't known until runtime, use the `Send` API.

### Basic Send Pattern

```python
from langgraph.types import Send

def split_work(state: MyState):
    tasks = state['tasks']
    # Create one worker per task
    return [Send('worker_node', {'task': task}) for task in tasks]

graph.add_node('splitter', split_work)
graph.add_node('worker_node', worker_func)
graph.add_node('aggregator', aggregate_func)

graph.add_edge(START, 'splitter')
graph.add_conditional_edges('splitter', split_work)
graph.add_edge('worker_node', 'aggregator')
graph.add_edge('aggregator', END)
```

### Real-World Example: Document Processing

```python
def process_chapters(state: DocumentState):
    chapters = state['chapters']
    return [Send('analyze_chapter', {'chapter': chapter}) for chapter in chapters]

def analyze_chapter(state: DocumentState):
    chapter = state['chapter']
    summary = llm.summarize(chapter)
    return {'chapter_summaries': [summary]}

def compile_summary(state: DocumentState):
    all_summaries = state['chapter_summaries']
    final = llm.synthesize(all_summaries)
    return {'document_summary': final}
```

---

## Common Parallel Patterns

### Map-Reduce

```
Input → Split → [Process A, Process B, Process C] → Aggregate → Output
```

Use case: Processing large documents, batch data transformation.

### Multi-Perspective Analysis

```
Input → [Analyzer 1, Analyzer 2, Analyzer 3] → Combine Perspectives → Output
```

Use case: Essay evaluation, code review, sentiment analysis from multiple angles.

### Ensemble Voting

```
Input → [Model A, Model B, Model C] → Vote → Final Decision
```

Use case: Classification tasks requiring high confidence.

### Parallel Tool Execution

```
Query → [Search Web, Search Database, Search Cache] → Merge Results → Response
```

Use case: Agentic systems with multiple data sources.

---

## Performance Considerations

### When Parallelism Helps

- **Independent tasks** — no interdependencies
- **Expensive operations** — LLM calls, API requests, database queries
- **Sufficient resources** — API rate limits, compute capacity
- **Meaningful aggregation** — results can be combined effectively

### When It Doesn't

- **Sequential dependencies** — Task B needs Task A's output
- **Resource constraints** — Rate limits force serialization
- **Overhead exceeds benefit** — Small, fast operations
- **Complex merging** — Aggregation logic is expensive

### Rate Limiting

With LLM calls, parallel execution can hit rate limits quickly:

```python
# May hit rate limits with many parallel calls
graph.add_edge(START, 'evaluator_1')
graph.add_edge(START, 'evaluator_2')
graph.add_edge(START, 'evaluator_3')
# ... 10 more evaluators

# Solution: Batch or use sequential chunks
```

---

## Debugging Parallel Workflows

### Challenge: Non-Deterministic Ordering

Parallel nodes complete in unpredictable order. Don't rely on ordering:

```python
# Bad: Assumes order
def combine(state):
    first = state['results'][0]  # Unpredictable!
    
# Good: Order-independent
def combine(state):
    best = max(state['results'])  # Deterministic
```

### Visualization

Use LangGraph's built-in visualization to verify parallel structure:

```python
from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())
```

Parallel nodes appear at the same horizontal level in the diagram.

---

## Best Practices

### 1. Design for Concurrency

Ensure nodes are truly independent — no shared mutable state beyond what reducers handle.

### 2. Use Descriptive Node Names

Parallel nodes show up in traces and logs:

```python
graph.add_node('evaluate_grammar', eval_grammar)
graph.add_node('evaluate_style', eval_style)
graph.add_node('evaluate_coherence', eval_coherence)
```

### 3. Handle Partial Failures

If one parallel branch fails, decide: fail entire workflow, or continue with partial results.

### 4. Monitor Resource Usage

Parallel LLM calls multiply token usage and costs.

### 5. Test with Various Input Sizes

Parallel behavior may differ with small vs. large inputs.

---

## Summary

Parallel workflows enable **concurrent execution** in LangGraph:

- **Static parallelism** — Multiple START edges for fixed parallel branches
- **Dynamic parallelism** — Send API for runtime-determined workers
- **Reducers** — Critical for merging parallel outputs safely
- **Fan-out/fan-in** — Core pattern for splitting and aggregating work

**Key takeaways:**
1. Use parallel execution for independent, expensive operations
2. Always use reducers when multiple nodes update the same field
3. The Send API enables dynamic worker spawning
4. Parallel LLM calls reduce latency but increase costs
5. Visualize graphs to verify parallel structure

---

*Next: Explore conditional workflows in `4_Conditional_Workflows/` to learn dynamic routing and branching patterns.*
