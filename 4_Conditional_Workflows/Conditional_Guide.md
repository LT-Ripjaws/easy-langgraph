# Conditional Workflows in LangGraph

## Building Dynamic, Decision-Driven AI Agents

Conditional workflows transform LangGraph from a static pipeline into a **dynamic, decision-making system**. Instead of following a fixed path, your workflow can branch, loop, and adapt based on runtime conditions — enabling intelligent agents that respond to the data they process.

This pattern implements **Routing** and **Evaluator-Optimizer** — two of the five foundational LLM workflow patterns from Anthropic's research.

---

## Beyond Linear Execution

### The Limitation of Sequential Workflows

Sequential workflows follow a predetermined path:

```
START → A → B → C → END
```

Every input follows the same route. This works for simple tasks but fails when:

- Different inputs require different handling
- Quality checks should trigger retries
- Some cases need special processing
- The workflow should adapt based on intermediate results

### The Power of Conditional Execution

Conditional workflows introduce **dynamic routing**:

```
                    START
                      ↓
                 [Node A]
                      ↓
            ┌─────────┴─────────┐
            ↓         ↓         ↓
        [Path X]  [Path Y]  [Path Z]
            ↓         ↓         ↓
            └─────────┬─────────┘
                      ↓
                    END
```

The path taken depends on **runtime conditions** evaluated during execution.

---

## Conditional Edges: The Core Mechanism

### What Are Conditional Edges?

A conditional edge is a dynamic connection that routes execution to different nodes based on a **routing function**. Unlike regular edges that always go to the same destination, conditional edges decide the next step at runtime.

```python
# Regular edge - always goes to the same place
graph.add_edge('node_a', 'node_b')

# Conditional edge - destination determined by function
graph.add_conditional_edges('node_a', routing_function)
```

### The Routing Function

A routing function is a plain Python function that:

1. Receives the current state
2. Analyzes the state
3. Returns a string indicating which node to visit next

```python
from typing import Literal

def route_based_on_state(state: MyState) -> Literal['path_a', 'path_b', 'path_c']:
    if state['some_condition']:
        return 'path_a'
    elif state['another_condition']:
        return 'path_b'
    else:
        return 'path_c'

# Connect to conditional edge
graph.add_conditional_edges('decision_node', route_based_on_state)
```

### Mapping Return Values to Nodes

The routing function returns strings that map to actual node names:

```python
def router(state: MyState) -> str:
    if state['score'] >= 8:
        return 'approve'
    elif state['score'] >= 5:
        return 'review'
    else:
        return 'reject'

graph.add_conditional_edges('evaluate', router, {
    'approve': 'approval_node',
    'review': 'review_node',
    'reject': 'rejection_node'
})
```

If the return values match node names exactly, you can omit the mapping dictionary.

---

## Example 1: Quadratic Equation Solver

A clean example of mathematical branching based on computed conditions.

**File:** `4_Conditional_Workflows/1_Quadratic_Eq_workflow.ipynb`

### State Definition

```python
class QuadState(TypedDict):
    a: int
    b: int
    c: int
    
    # Computed values
    equation: str
    discriminant: float
    
    # Final result
    result: str
```

### Node Functions

```python
def show_equation(state: QuadState):
    equation = f'{state["a"]}x^2 + ({state["b"]})x + ({state["c"]})'
    return {'equation': equation}

def calculate_discriminant(state: QuadState):
    discriminant = state['b']**2 - (4 * state['a'] * state['c'])
    return {'discriminant': discriminant}

def real_roots(state: QuadState):
    root1 = (-state['b'] + state['discriminant']**0.5) / (2 * state['a'])
    root2 = (-state['b'] - state['discriminant']**0.5) / (2 * state['a'])
    result = f'The roots are {root1} and {root2}'
    return {'result': result}

def repeated_roots(state: QuadState):
    root = (-state['b']) / (2 * state['a'])
    result = f'The only repeating root is: {root}'
    return {'result': result}

def no_real_roots(state: QuadState):
    return {'result': 'No real roots'}
```

### Routing Function

```python
from typing import Literal

def check_condition(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'
```

### Graph Construction

```python
graph = StateGraph(QuadState)

# Add all nodes
graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

# Sequential start
graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')

# Conditional branching based on discriminant
graph.add_conditional_edges('calculate_discriminant', check_condition)

# All branches terminate
graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

workflow = graph.compile()
```

### Execution Examples

```python
# Case 1: Two real roots (discriminant > 0)
workflow.invoke({'a': 1, 'b': -5, 'c': 6})
# Path: show_equation → calculate_discriminant → real_roots → END
# Result: "The roots are 3.0 and 2.0"

# Case 2: One repeated root (discriminant = 0)
workflow.invoke({'a': 1, 'b': -4, 'c': 4})
# Path: show_equation → calculate_discriminant → repeated_roots → END
# Result: "The only repeating root is: 2.0"

# Case 3: No real roots (discriminant < 0)
workflow.invoke({'a': 1, 'b': 2, 'c': 5})
# Path: show_equation → calculate_discriminant → no_real_roots → END
# Result: "No real roots"
```

---

## Example 2: LLM Review Response Workflow

A sophisticated conditional workflow using LLMs to classify sentiment and route to appropriate handlers.

**File:** `4_Conditional_Workflows/2_LLM_review_reply_workflow.ipynb`

### Structured Output Schemas

```python
from pydantic import BaseModel, Field
from typing import Literal

class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Sentiment of the review'
    )

class DiagnosisSchema(BaseModel):
    issue_type: Literal['UX', 'Performance', 'Bug', 'Support', 'Other'] = Field(
        description='Category of the issue'
    )
    tone: Literal['angry', 'frustrated', 'disappointed', 'calm'] = Field(
        description='Emotional tone of the user'
    )
    urgency: Literal['low', 'medium', 'high'] = Field(
        description='How urgent the issue is'
    )

# Create structured models
sentiment_model = model.with_structured_output(SentimentSchema)
diagnosis_model = model.with_structured_output(DiagnosisSchema)
```

### State Definition

```python
class ReviewState(TypedDict):
    review: str
    
    # Classification results
    sentiment: str
    issue_type: str
    tone: str
    urgency: str
    
    # Response
    response: str
```

### Classification Nodes

```python
def classify_sentiment(state: ReviewState):
    prompt = f'Classify the sentiment of this review: {state["review"]}'
    output = sentiment_model.invoke(prompt)
    return {'sentiment': output.sentiment}

def diagnose_issue(state: ReviewState):
    prompt = f'Analyze this review and extract issue details: {state["review"]}'
    output = diagnosis_model.invoke(prompt)
    return {
        'issue_type': output.issue_type,
        'tone': output.tone,
        'urgency': output.urgency
    }
```

### Response Generation Nodes

```python
def generate_positive_response(state: ReviewState):
    prompt = f'''Generate a warm thank-you response for this positive review:
    {state['review']}'''
    response = model.invoke(prompt).content
    return {'response': response}

def generate_urgent_response(state: ReviewState):
    prompt = f'''Generate an urgent, empathetic response addressing this critical issue:
    Review: {state['review']}
    Issue Type: {state['issue_type']}
    Tone: {state['tone']}'''
    response = model.invoke(prompt).content
    return {'response': response}

def generate_standard_response(state: ReviewState):
    prompt = f'''Generate a professional response addressing this feedback:
    Review: {state['review']}
    Issue Type: {state['issue_type']}'''
    response = model.invoke(prompt).content
    return {'response': response}
```

### Routing Functions

```python
def route_by_sentiment(state: ReviewState) -> Literal['positive_handler', 'negative_handler']:
    if state['sentiment'] == 'positive':
        return 'positive_handler'
    else:
        return 'negative_handler'

def route_negative_reviews(state: ReviewState) -> Literal['urgent_response', 'standard_response']:
    if state['urgency'] == 'high':
        return 'urgent_response'
    else:
        return 'standard_response'
```

### Graph with Nested Conditionals

```python
graph = StateGraph(ReviewState)

# Add nodes
graph.add_node('classify_sentiment', classify_sentiment)
graph.add_node('diagnose_issue', diagnose_issue)
graph.add_node('positive_handler', generate_positive_response)
graph.add_node('urgent_response', generate_urgent_response)
graph.add_node('standard_response', generate_standard_response)

# Initial classification
graph.add_edge(START, 'classify_sentiment')
graph.add_edge('classify_sentiment', 'diagnose_issue')

# First branch: sentiment-based routing
graph.add_conditional_edges('diagnose_issue', route_by_sentiment)

# Second branch: urgency-based routing for negative reviews
graph.add_conditional_edges('negative_handler', route_negative_reviews)

# Termination
graph.add_edge('positive_handler', END)
graph.add_edge('urgent_response', END)
graph.add_edge('standard_response', END)

workflow = graph.compile()
```

### Execution Flow

```
                    START
                      ↓
            [classify_sentiment]
                      ↓
             [diagnose_issue]
                      ↓
            ┌─────────┴─────────┐
            ↓                   ↓
    [positive_handler]    [negative_handler]
            ↓                   ↓
          END           ┌───────┴───────┐
                        ↓               ↓
                [urgent_response]  [standard_response]
                        ↓               ↓
                      END             END
```

---

## Advanced Pattern: Loops and Retries

Conditional edges enable **cycles** — routing back to earlier nodes for refinement.

### Evaluator-Optimizer Loop

```python
def generate_draft(state: DraftState):
    draft = llm.generate(state['topic'])
    return {'draft': draft, 'iterations': state['iterations'] + 1}

def evaluate_quality(state: DraftState):
    score = llm.evaluate(state['draft'])
    return {'score': score}

def should_continue(state: DraftState) -> Literal['generate_draft', 'finalize']:
    if state['score'] >= 8:
        return 'finalize'
    elif state['iterations'] >= 3:
        return 'finalize'  # Max iterations reached
    else:
        return 'generate_draft'  # Loop back

graph = StateGraph(DraftState)
graph.add_node('generate_draft', generate_draft)
graph.add_node('evaluate_quality', evaluate_quality)
graph.add_node('finalize', finalize)

graph.add_edge(START, 'generate_draft')
graph.add_edge('generate_draft', 'evaluate_quality')
graph.add_conditional_edges('evaluate_quality', should_continue)
graph.add_edge('finalize', END)
```

### Key Considerations for Loops

**Always include exit conditions:**

```python
def should_continue(state: DraftState) -> str:
    # Primary condition: quality threshold
    if state['score'] >= 8:
        return 'finalize'
    
    # Safety valve: max iterations
    if state['iterations'] >= 5:
        return 'finalize'
    
    # Continue loop
    return 'generate_draft'
```

Without exit conditions, poor-quality outputs could cause infinite loops.

---

## Common Conditional Patterns

### 1. Router Pattern

Classify input and route to specialized handlers:

```
Input → Classifier → [Specialist A, Specialist B, Specialist C] → Output
```

Use case: Customer support triage, intent classification.

### 2. Guardrail Pattern

Validate before proceeding:

```
Input → Validator → [Proceed, Reject/Retry] → Output
```

Use case: Content moderation, input validation.

### 3. Evaluator-Optimizer Pattern

Generate, evaluate, refine loop:

```
Generate → Evaluate → (loop if needed) → Finalize
```

Use case: Draft writing, code generation, translation.

### 4. Escalation Pattern

Handle by complexity:

```
Input → Triage → [Simple Handler, Complex Handler, Human Escalation]
```

Use case: Support tickets, medical diagnosis assistance.

---

## Dynamic Routing with Command

For more advanced scenarios, LangGraph provides the `Command` primitive for runtime routing decisions:

```python
from langgraph.types import Command

def dynamic_router(state: MyState):
    # Complex logic here
    if should_process_parallel:
        return Command(
            goto=['worker_a', 'worker_b'],
            update={'status': 'processing'}
        )
    else:
        return Command(
            goto='single_worker',
            update={'status': 'queued'}
        )
```

This allows routing decisions that also update state atomically.

---

## Best Practices

### 1. Keep Routing Functions Simple

Routing functions should be fast, deterministic, and side-effect free:

```python
# Good: Simple, fast
def router(state):
    return 'path_a' if state['score'] > 5 else 'path_b'

# Bad: Complex, slow, unpredictable
def router(state):
    result = call_external_api(state)
    if result复杂:
        # 50 lines of logic
        ...
```

### 2. Use Type Hints for Clarity

```python
from typing import Literal

def router(state: MyState) -> Literal['accept', 'reject', 'review']:
    ...
```

This makes valid return values explicit and enables IDE autocomplete.

### 3. Handle All Branches

Ensure every possible return value from your router has a corresponding node:

```python
# Router can return 'yes', 'no', 'maybe'
def router(state): ...

# All three must have edges
graph.add_conditional_edges('decision', router, {
    'yes': 'yes_node',
    'no': 'no_node',
    'maybe': 'maybe_node'  # Don't forget this!
})
```

### 4. Log Routing Decisions

For debugging, log why certain paths were taken:

```python
def router(state):
    score = state['score']
    logging.info(f'Routing based on score: {score}')
    
    if score >= 8:
        return 'approve'
    ...
```

### 5. Test All Paths

Ensure every branch works correctly:

```python
# Test case triggering each path
test_positive = {'review': 'Great product!', ...}
test_negative_urgent = {'review': 'Broken! Need help NOW', ...}
test_negative_calm = {'review': 'Could be better', ...}
```

---

## Debugging Conditional Workflows

### Visualize the Graph

```python
from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())
```

Conditional edges appear as diamond-shaped decision nodes in the visualization.

### Trace Execution

LangGraph provides execution traces showing which path was taken:

```python
# Enable tracing
config = {'configurable': {'thread_id': '123'}}
result = workflow.invoke(input_state, config)

# View trace to see routing decisions
```

---

## Summary

Conditional workflows enable **dynamic, adaptive AI agents**:

- **Conditional edges** route execution based on runtime state
- **Routing functions** decide which path to take
- **Loops** enable iterative refinement with proper exit conditions
- **Nested conditionals** create complex decision trees

**Key takeaways:**
1. Use `add_conditional_edges` for dynamic routing
2. Routing functions return strings that map to node names
3. Always include exit conditions in loops
4. Type hints with `Literal` make routers self-documenting
5. Test all branches to ensure correctness
6. Visualize graphs to verify routing structure

---

*wait for next workflows*
