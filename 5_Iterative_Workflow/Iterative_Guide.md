# Iterative Workflows in LangGraph

## The Generate-Evaluate-Refine Pattern

Iterative workflows introduce **feedback loops** into your LangGraph applications. Instead of flowing in one direction, these workflows can cycle back to earlier nodes, allowing the system to refine outputs based on evaluation criteria — mimicking how humans revise their work.

This pattern is a pure implementation of **Evaluator-Optimizer** — Pattern 5 from Anthropic's foundational LLM workflow research.

> **Key insight:** This pattern is about **depth** (iterating on one task until quality is maximized), not breadth (parallel subtasks like orchestrator-worker).

---

## Understanding Iterative Execution

### The Core Concept

An iterative workflow contains at least one **cycle** — a path that loops back to an earlier node:

```
                START
                  ↓
            [Generate]
                  ↓
           [Evaluate] ←──┐
              ↓          │
         [Approved?] ────┤ (if needs improvement)
              ↓          │
            [END]    [Optimize]
                          ↑
                          └──┘
```

The workflow continues looping until a **termination condition** is met.

### When to Use Iterative Workflows

| Use Case | Why |
|----------|-----|
| Code generation | Catch logic errors, enforce complexity requirements |
| Legal/technical documents | "Almost right" is effectively wrong |
| Complex math problems | Self-correction for hallucinations |
| Content with strict constraints | Word count, JSON schema, format rules |
| Quality-critical outputs | First drafts are usually flabby |

**Use when:** Quality > Speed, objective criteria exist, first drafts need refinement.

**Avoid when:**
- Speed is critical and quality variance is acceptable
- The task is simple enough for single-pass completion
- You cannot define clear termination criteria

---

## Key Components of Iterative Workflows

### 1. Generator Node (Creative)

Creates the initial output with higher temperature for creativity:

```python
def generate_tweet(state: XPostState):
    prompt = f"Write a funny tweet about: {state['topic']}"
    tweet = generator_llm.invoke(prompt).content
    return {'tweet': tweet, 'tweet_history': [tweet]}
```

### 2. Evaluator Node (Critical)

Assesses quality and provides feedback with lower temperature for consistency:

```python
def evaluate_tweet(state: XPostState):
    prompt = f"Rate this tweet 1-10: {state['tweet']}"
    score = int(evaluator_llm.invoke(prompt).content)
    return {'score': score, 'feedback': '...'}
```

### 3. Routing Function

Decides whether to continue or stop:

```python
def should_continue(state: XPostState) -> Literal['optimize', 'end']:
    if state['score'] >= 8:
        return 'end'
    elif state['iteration'] >= state['max_iter']:
        return 'end'  # Safety cutoff
    else:
        return 'optimize'
```

### 4. Optimizer Node (Refinement)

Refines the output based on feedback:

```python
def optimize_tweet(state: XPostState):
    prompt = f"""Improve this tweet based on feedback:
    Original: {state['tweet']}
    Feedback: {state['feedback']}"""
    new_tweet = optimizer_llm.invoke(prompt).content
    return {
        'tweet': new_tweet,
        'iteration': state['iteration'] + 1,
        'tweet_history': [new_tweet]
    }
```

---

## The Critical Safety Valve: Max Iterations

**Always include a maximum iteration count** to prevent infinite loops:

```python
class XPostState(TypedDict):
    topic: str
    tweet: str
    score: int
    feedback: str
    iteration: int          # Current iteration
    max_iter: int           # Maximum allowed iterations
    tweet_history: Annotated[list[str], operator.add]
```

The routing function must check this:

```python
def should_continue(state: XPostState) -> str:
    # Primary condition: quality threshold
    if state['score'] >= 8:
        return 'end'
    
    # Safety valve: prevent infinite loops
    if state['iteration'] >= state['max_iter']:
        return 'end'
    
    # Continue refinement
    return 'optimize'
```

Without exit conditions, poor-quality outputs could cause the workflow to loop forever.

---

## Example: Tweet Generation with Iterative Refinement

A complete example of generate-evaluate-refine in action.

**File:** `5_Iterative_Workflow/1_post_generate_evaluate.ipynb`

### Specialized LLM Roles

Using different temperatures for different tasks is a **critical best practice**:

```python
# Generator: creative, exploratory (temperature = 0.5-0.8)
generator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)

# Evaluator: strict, deterministic (temperature = 0.0-0.3)
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# Optimizer: creative refinement (temperature = 0.7-0.8)
optimizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.8)
```

> **Why different temperatures?** The generator needs creativity to try different approaches. The evaluator needs consistency for reliable judgments. The optimizer needs creativity to explore alternative formulations.

### State with History Tracking

```python
class XPostState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iter: int
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
```

### Structured Evaluation Schema

Using Pydantic for deterministic parsing — **this is essential for clean routing logic**:

```python
class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., 
        description="Final Evaluation Result"
    )
    feedback: str = Field(
        ..., 
        description="Constructive feedback for the tweet"
    )

structured_evaluator = evaluator_llm.with_structured_output(TweetEvaluationSchema)
```

### Node Functions

```python
def generate_tweet(state: XPostState):
    """Create initial tweet"""
    messages = [
        SystemMessage(content="You are a funny Twitter influencer."),
        HumanMessage(content=f"Write a hilarious tweet about: {state['topic']}")
    ]
    response = generator_llm.invoke(messages).content
    return {'tweet': response, 'tweet_history': [response]}

def evaluate_tweet(state: XPostState):
    """Critique the tweet"""
    messages = [
        SystemMessage(content="You are a ruthless Twitter critic."),
        HumanMessage(content=f"Evaluate this tweet: {state['tweet']}")
    ]
    response = structured_evaluator.invoke(messages)
    return {
        'evaluation': response.evaluation,
        'feedback': response.feedback,
        'feedback_history': [response.feedback]
    }

def optimize_tweet(state: XPostState):
    """Refine tweet based on feedback"""
    messages = [
        SystemMessage(content="You improve tweets based on feedback."),
        HumanMessage(content=f"""
            Improve this tweet:
            Original: {state['tweet']}
            Feedback: {state['feedback']}
        """)
    ]
    response = optimizer_llm.invoke(messages).content
    return {
        'tweet': response,
        'iteration': state['iteration'] + 1,
        'tweet_history': [response]
    }
```

### Routing Logic

```python
def route_evaluation(state: XPostState) -> Literal['approved', 'needs_improvement']:
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iter']:
        return 'approved'
    else:
        return 'needs_improvement'
```

### Graph Construction with Loop

```python
graph = StateGraph(XPostState)

graph.add_node('generate', generate_tweet)
graph.add_node('evaluate', evaluate_tweet)
graph.add_node('optimize', optimize_tweet)

# Initial flow
graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

# Conditional loop
graph.add_conditional_edges(
    'evaluate',
    route_evaluation,
    {
        'approved': END,
        'needs_improvement': 'optimize'
    }
)

# Loop back
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()
```

### Execution Example

```python
result = workflow.invoke({
    'topic': 'AI taking over the world',
    'max_iter': 5,
    'iteration': 0
})

print(f"Final tweet: {result['tweet']}")
print(f"Iterations: {result['iteration']}")
print(f"All versions: {result['tweet_history']}")
```

### Example Output Flow

```
🚀 Task: Write a funny tweet about "AI taking over the world"

--- Attempt 1/5 ---
Decision: NEEDS_IMPROVEMENT
Feedback: Too generic. Add specific cultural reference. Avoid question format.

--- Attempt 2/5 ---
Decision: NEEDS_IMPROVEMENT  
Feedback: Better! But punchline is weak. Make it sharper.

--- Attempt 3/5 ---
Decision: APPROVED
Feedback: Original, uses irony, under 280 chars. Good to post.

✅ Returning final tweet
```

---

## Iterative vs Conditional: What's the Difference?

Both patterns use conditional edges, but serve different purposes:

| Aspect | Conditional (Routing) | Iterative (Loop) |
|--------|----------------------|------------------|
| **Flow** | Branches to different paths | Cycles back to same node |
| **Purpose** | Handle different cases | Refine single output |
| **Termination** | All paths lead to END | Loop has exit condition |
| **Focus** | Breadth (different handlers) | Depth (quality through iteration) |
| **Example** | Route positive/negative reviews | Improve draft until quality met |

In practice, iterative workflows are a **special case** of conditional workflows where one branch loops back.

---

## Design Patterns for Iterative Workflows

### 1. Generate-Evaluate-Refine

The classic loop:

```
Generate → Evaluate → (loop if needed) → Finalize
```

Use case: Content creation, code generation, translation.

### 2. Progressive Enhancement

Start simple, add complexity iteratively:

```
Draft → Add Details → Add Examples → Polish → END
```

Use case: Documentation, tutorials, explanations.

### 3. Constraint Satisfaction

Keep refining until all constraints are met:

```
Generate → Check Constraints → (loop if violated) → Output
```

Use case: Format-compliant outputs, regulated content, JSON schema validation.

### 4. Multi-Criteria Optimization

Evaluate against multiple criteria, optimize for weakest:

```
Generate → [Check A, Check B, Check C] → Optimize weakest → Loop
```

Use case: Balanced outputs requiring multiple quality dimensions.

---

## State Management Best Practices

### Use Explicit State Schemas with Reducers

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages

class XPostState(TypedDict):
    # Overwrite - latest value wins
    topic: str
    tweet: str
    evaluation: str
    
    # Accumulate - preserves all history
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
    
    # Counters
    iteration: int
    max_iter: int
```

### Why Reducers Matter

| Reducer | Use Case | Behavior |
|---------|----------|----------|
| `operator.add` | Lists, counters | Accumulates values |
| `add_messages` | Chat history | Appends, handles deletions |
| None (default) | Drafts, current values | Latest value overwrites |

**Critical:** Without `Annotated[list, operator.add]`, parallel or sequential updates to list fields will **overwrite** instead of accumulate.

---

## Setting Quality Thresholds

### Too Low

```python
if state['score'] >= 3:  # Accepts poor quality
    return 'end'
```

Risk: Workflow exits prematurely with subpar output.

### Too High

```python
if state['score'] >= 10:  # Perfection required
    return 'end'
```

Risk: Workflow loops until `max_iter` every time, wasting resources.

### Balanced

```python
if state['score'] >= 7:  # Good enough for most purposes
    return 'end'
```

Calibrate thresholds based on:
- Task complexity
- LLM capability
- Acceptable quality variance
- Cost/time constraints

---

## Common Pitfalls

### 1. No Exit Condition

```python
# BAD: Infinite loop risk
def router(state):
    if state['score'] >= 8:
        return 'end'
    return 'optimize'  # No max_iter check!
```

### 2. Not Incrementing Counter

```python
# BAD: Counter never increases
def optimize(state):
    new_output = improve(state['output'])
    return {'output': new_output}  # Forgot iteration!
```

### 3. Losing History

```python
# BAD: Overwrites instead of accumulating
def optimize(state):
    return {'output': new_output}  # Should be {'output_history': [new_output]}
```

### 4. Vague Evaluation Criteria

```python
# BAD: Subjective, inconsistent
prompt = "Is this good?"

# GOOD: Specific, measurable
prompt = "Rate 1-10 based on: clarity, accuracy, completeness"
```

### 5. Unbalanced Evaluator Strictness

```python
# BAD: Too vague
feedback = "This is inefficient"

# GOOD: Actionable
feedback = "Uses O(n log n) sorting. Use Counter for O(n) complexity."
```

---

## Performance Considerations

### Cost of Iteration

Each loop multiplies LLM calls:

```
Single pass: 1 LLM call
2 iterations: 3 LLM calls (generate + 2× evaluate + 2× optimize)
5 iterations: 11 LLM calls
```

Factor this into:
- API budget planning
- Latency expectations
- Rate limit management

### When to Stop Early

Consider early termination when:
- Diminishing returns (score plateau)
- Time budget exhausted
- Token limits approaching

```python
def should_continue(state):
    # Quality met
    if state['score'] >= 8:
        return 'end'
    
    # Diminishing returns - good enough after 3 attempts
    if state['iteration'] >= 3 and state['score'] >= 6:
        return 'end'
    
    # Max iterations
    if state['iteration'] >= state['max_iter']:
        return 'end'
    
    return 'optimize'
```

### Managing Context Window Growth

Passing previous code + feedback grows tokens. For long tasks:

```python
# Summarize feedback instead of appending all history
def optimize(state):
    # Only use latest feedback, not full history
    prompt = f"Improve based on: {state['feedback']}"
```

---

## Debugging Iterative Workflows

### Visualize the Loop

```python
from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())
```

Look for the back-edge from optimizer to evaluator.

### Trace Execution

```python
# Enable thread-level tracing
config = {'configurable': {'thread_id': 'tweet_123'}}
result = workflow.invoke(input_state, config)

# View execution trace to see iteration count
```

### Log Each Iteration

```python
def evaluate_tweet(state: XPostState):
    # ... evaluation logic ...
    
    print(f"Iteration {state['iteration']}: Score = {score}")
    
    return {'evaluation': ..., 'feedback': ...}
```

---

## Best Practices

### 1. Use Structured Output for Evaluation

Ensures consistent, parseable results:

```python
class EvaluationSchema(BaseModel):
    decision: Literal["PASS", "NEEDS_IMPROVEMENT"] = Field(
        description="'PASS' or 'NEEDS_IMPROVEMENT'"
    )
    feedback: str = Field(description="Specific issues to fix")
    score: int = Field(ge=0, le=10, description="Quality score")

evaluator = evaluator_llm.with_structured_output(EvaluationSchema)
```

### 2. Test Evaluator Prompts Independently

Before integrating into the loop, test your evaluator:

```python
# Test evaluator with sample outputs
test_tweet = "Why did the AI cross the road? To optimize the chicken's path!"
test_result = evaluator.invoke({"tweet": test_tweet})
print(test_result)  # Should return valid JSON with decision + feedback
```

### 3. Separate Generator and Optimizer

Different prompts (or even models) for creation vs. refinement:

```python
# Generator: create from scratch
def generate(state):
    prompt = f"Create a {state['topic']} piece"

# Optimizer: improve existing
def optimize(state):
    prompt = f"""
    Improve this based on feedback:
    Original: {state['output']}
    Feedback: {state['feedback']}
    """
```

### 4. Make Feedback Actionable

Evaluator should say *what* is wrong and *how* to fix:

```python
# BAD: Vague
feedback = "Make it better"

# GOOD: Specific
feedback = "Add a punchline. Remove the question format. Keep under 280 chars."
```

### 5. Monitor Convergence

Track if iterations are actually improving:

```python
def optimize(state):
    # ... optimization ...
    
    if state['iteration'] > 1:
        prev_score = state['scores'][-1] if 'scores' in state else 0
        # Log if not improving
        if new_score <= prev_score:
            logger.warning(f"Score not improving: {prev_score} → {new_score}")
```

### 6. Generator Must Check for Feedback

First iteration = fresh generation. Subsequent = refinement:

```python
def generate_clause(state: State):
    if state.get("feedback"):
        # Refine based on feedback
        prompt = f"""
        Draft about: {state['topic']}.
        Fix these issues: {state['feedback']}
        """
    else:
        # Initial generation
        prompt = "Draft a short piece."
    
    clause = llm.invoke(prompt).content
    return {"clause": clause, "history": [clause]}
```

### 7. Evaluator Needs Binary Criteria

Clear pass/fail conditions prevent ambiguous routing:

```python
def evaluate_clause(state: State):
    # Explicit checklist - ALL must pass
    criteria = """
    A clause is ACCEPTABLE only if:
    1. Liability cap present
    2. Exclusion of indirect damages
    3. Exception for gross negligence
    """
    # If ANY item missing → "needs revision"
```

---

## Quick Reference: Do's and Don'ts

| ✅ Do | ❌ Don't |
|-------|----------|
| Use `Annotated[list, operator.add]` for history | Return full state from nodes |
| Set `max_iterations` safety valve | Forget the loop-back edge |
| Use structured output for evaluator | Skip testing evaluator independently |
| Make evaluation criteria binary | Use vague feedback like "improve this" |
| Track iteration history for debugging | Ignore convergence monitoring |

---

## Summary

Iterative workflows enable **self-improving AI systems** that mimic human "draft → review → revise" workflows:

- **Loops** allow refinement based on feedback
- **Exit conditions** prevent infinite execution
- **History tracking** enables debugging and audit
- **Quality thresholds** ensure output standards
- **Different temperatures** optimize each role (creative generator, strict evaluator, creative optimizer)

**Key takeaways:**
1. Always include `max_iter` as a safety valve
2. Use structured output for reliable evaluation parsing
3. Use different temperatures: generator (0.5-0.8), evaluator (0.0-0.3), optimizer (0.7-0.8)
4. Track iteration history with `Annotated[list, operator.add]`
5. Calibrate quality thresholds to balance quality vs. cost
6. Make feedback actionable — tell the optimizer *what* and *how* to fix
7. Test evaluator prompts independently before integrating
8. Consider diminishing returns when designing exit conditions
9. Generator must check for feedback to refine properly
10. Evaluator needs explicit binary criteria for consistent routing

> **Bottom line:** Ask yourself: **Does this need to be fast, or does it need to be right?** If "right," use the evaluator-optimizer pattern. The quality of your workflow depends more on evaluator criteria than generator capability.

---

*Next: Explore advanced patterns combining multiple workflow types, or revisit earlier chapters to see how sequential, parallel, and conditional patterns build the foundation for iterative systems.*
