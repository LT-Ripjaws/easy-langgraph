# LangGraph Command Routing Guide
### Returning the Next Node From a Node

---

## Part 1: What Problem Does `Command` Solve?

So far, routing in this repo has worked like this: a node updates state, and then a
separate routing function reads that state and decides which node runs next.

```python
def find_sentiment(state: ReviewState):
    ...
    return {'sentiment': sentiment}

def check_sentiment(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
    if state["sentiment"] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

graph.add_conditional_edges('find_sentiment', check_sentiment)
```

That is two separate pieces: the node that produces data, and the conditional edge
that reads it back out of state to decide where to go. `Command` collapses both
into one return value, from inside the node itself:

```python
def find_sentiment(state: ReviewState) -> Command[Literal['positive_response', 'run_diagnosis']]:
    ...
    if sentiment == 'positive':
        return Command(update={'sentiment': sentiment}, goto='positive_response')
    return Command(update={'sentiment': sentiment}, goto='run_diagnosis')
```

No `add_conditional_edges` call is needed. The node decides the update and the
destination in the same place, in the same step.

---

## Part 2: What This Notebook Builds

```text
06_Command_Routing/
|-- 1_command_routing.ipynb
|-- Command_Routing_Guide.md
```

The notebook builds two graphs:

1. A tiny non-LLM graph: a `router` node looks at a number and sends the graph to
   `double` or `halve` using `Command`.
2. An LLM support-ticket triage graph: a `triage_node` classifies an incoming
   ticket with structured output, then routes to `billing`, `technical`, or
   `general` using `Command`.

Both graphs have exactly one `add_edge` call each (`START` to the first node).
Neither uses `add_conditional_edges`.

---

## Part 3: The `Command` Object

```python
from langgraph.types import Command

Command(update={...}, goto="some_node")
```

`Command` takes two arguments that matter here:

| Argument | Meaning |
|---|---|
| `update` | A dict merged into the graph state, exactly like a normal node's return dict |
| `goto` | The name of the node to run next |

A node that returns `Command(update=X, goto=Y)` is doing the work of a node
return *and* a conditional edge, in one object.

`goto` can also be `END`, which ends the run immediately, the same as routing to
`END` from a conditional edge.

---

## Part 4: The Return-Type Hint, `Command[Literal[...]]`

The triage node in the notebook is written as:

```python
def triage_node(state: TicketState) -> Command[Literal["billing", "technical", "general"]]:
```

This is not a cosmetic type hint. LangGraph inspects it when the graph is built
to learn which nodes `triage_node` might send execution to. That is how the
compiled graph can validate the node names and draw the graph diagram correctly,
even though there is no `add_conditional_edges("triage_node", ...)` call telling
it so directly.

```text
Without add_conditional_edges, LangGraph still needs to know:
"triage_node might go to billing, technical, or general."

It gets that from the Command[Literal[...]] return annotation.
```

If you leave the annotation as a plain `Command` (no `Literal`), the graph still
runs, but LangGraph has less information about the possible destinations when it
builds the graph structure. Always write the full `Command[Literal["a", "b"]]`
form when a node can go to more than one place.

---

## Part 5: Do the Destination Nodes Need Edges to `END`?

No. In LangGraph, a node with no outgoing edge simply ends the run when it
finishes, whether or not `Command` is involved. The notebook's `double`, `halve`,
`billing`, `technical`, and `general` nodes have no `add_edge(..., END)` call, and
the graphs still compile and run correctly. You only need edges for the paths you
want to be explicit about, such as `START -> triage_node`.

---

## Part 6: Command vs. Conditional Edges — When to Use Which

Both tools solve the same problem: "given the current state, where does the
graph go next?" The difference is where the decision and the state update live.

```text
Conditional edges:
    node -> (state update)
    separate routing function -> (reads state, returns a node name)
    add_conditional_edges(node, routing_function)

Command:
    node -> (state update AND next node, together, in one return value)
```

| Use conditional edges when... | Use `Command` when... |
|---|---|
| The routing decision does not need any new data — it can be made from state that already exists | The routing decision depends on something the node just computed (e.g. an LLM classification) |
| You want the routing logic reusable or testable separately from the node | The update and the destination naturally belong together |
| Multiple different nodes might reuse the same routing function | Only this node needs this particular routing logic |
| You are following a Send/map-style fan-out pattern | You want to avoid a second function just to re-read a value the node already has |

In practice: if you find yourself writing a node that stores a value in state
purely so a routing function can immediately read it back out, `Command` usually
removes a step. The ticket triage example is exactly that case — without
`Command`, you would need a `classify_node` that stores `category`, plus a
`route_by_category` function that reads `state["category"]` right back out.

Conditional edges are still the better fit when the same routing function is
shared across multiple nodes, or when you want the branching logic visible and
testable independently of any one node.

---

## Part 7: Walking Through the Notebook — Part 1 (Non-LLM Example)

```python
class NumberState(TypedDict):
    value: int
    path: str


def router(state: NumberState) -> Command[Literal["double", "halve"]]:
    if state["value"] % 2 == 0:
        return Command(update={"path": "double"}, goto="double")
    return Command(update={"path": "halve"}, goto="halve")


def double(state: NumberState) -> dict:
    return {"value": state["value"] * 2}


def halve(state: NumberState) -> dict:
    return {"value": state["value"] // 2}
```

`router` checks `state["value"]` and, in one step, records which path it chose
(`path`) and sends the graph to that node. Building the graph needs only one edge:

```python
number_graph = StateGraph(NumberState)
number_graph.add_node("router", router)
number_graph.add_node("double", double)
number_graph.add_node("halve", halve)
number_graph.add_edge(START, "router")

number_app = number_graph.compile()
```

Running it on `4` (even) takes the `double` path and returns `{'value': 8, 'path': 'double'}`.
Running it on `7` (odd) takes the `halve` path and returns `{'value': 3, 'path': 'halve'}`
(integer division of `7 // 2`). Both runs land on a different final node with no
conditional edge anywhere in the graph.

---

## Part 8: Walking Through the Notebook — Part 2 (Ticket Triage)

### Structured Output for the Routing Decision

```python
class TicketCategory(BaseModel):
    category: Literal["billing", "technical", "general"] = Field(
        description="The best-matching category for the support ticket"
    )


classifier = llm.with_structured_output(TicketCategory)
```

The classification is constrained to exactly three literal values. This matters
for routing specifically: `Command(goto=...)` needs a real node name. If the LLM
were allowed to answer in free text, it could return something like `"Billing
issue"` instead of `"billing"`, and `goto` would fail to find a matching node.
Structured output with a `Literal` field guarantees the value is always one of
the node names the graph actually has.

### State

```python
class TicketState(TypedDict):
    ticket: str
    category: str
    response: str
```

### The Triage Node

```python
def triage_node(state: TicketState) -> Command[Literal["billing", "technical", "general"]]:
    result = classifier.invoke(
        f"Classify this support ticket as billing, technical, or general:\n{state['ticket']}"
    )
    return Command(update={"category": result.category}, goto=result.category)
```

One call to the LLM produces `result.category`, which is used both as the state
update and as the literal node name passed to `goto`. There is no intermediate
"store it, then re-read it" step.

### The Handler Nodes

```python
def billing_node(state: TicketState) -> dict:
    return {"response": f"[Billing team] We received your ticket: '{state['ticket']}'"}

def technical_node(state: TicketState) -> dict:
    return {"response": f"[Technical team] We received your ticket: '{state['ticket']}'"}

def general_node(state: TicketState) -> dict:
    return {"response": f"[General support] We received your ticket: '{state['ticket']}'"}
```

Each handler is a plain node. It does not need to know how it was reached — it
just processes the ticket and returns a response.

### Building and Running the Graph

```python
ticket_graph = StateGraph(TicketState)
ticket_graph.add_node("triage_node", triage_node)
ticket_graph.add_node("billing", billing_node)
ticket_graph.add_node("technical", technical_node)
ticket_graph.add_node("general", general_node)
ticket_graph.add_edge(START, "triage_node")

ticket_app = ticket_graph.compile()
```

```text
                 START
                   |
                   v
            +--------------+
            | triage_node  |
            |  classifies  |
            +------+-------+
                   |
     goto = result.category
                   |
      +------------+------------+
      |            |            |
      v            v            v
  billing     technical      general
```

Running the graph on three tickets produced:

```text
billing -> [Billing team] We received your ticket: 'I was charged twice for my subscription this month.'
technical -> [Technical team] We received your ticket: 'The app crashes every time I try to upload a file.'
general -> [General support] We received your ticket: 'What are your support hours?'
```

Each ticket landed on the handler matching its actual content, confirmed by the
`category` in the printed output — not just an assumption about what the LLM
"should" have done.

---

## Part 9: Common Beginner Confusions

### Confusion 1: Do I still need `add_conditional_edges` anywhere?

Not for the routing that `Command` handles. `add_conditional_edges` and `Command`
are two different ways to express the same kind of decision. A graph can mix
both if some nodes route with `Command` and others use conditional edges, but
you never need both for the same routing decision.

### Confusion 2: Does the `Command[Literal[...]]` type hint change behavior at runtime?

It does not change what the node does when it runs — `goto=result.category` would
route correctly even without the annotation. What the annotation changes is what
LangGraph knows about the graph's shape when it is built (validation and the
graph diagram). Always add it for any node that can go to more than one place.

### Confusion 3: What happens if `goto` names a node that does not exist?

The graph raises an error when that node is invoked, because there is nothing
registered under that name. This is exactly why the notebook constrains the LLM's
output with `Literal["billing", "technical", "general"]` — it guarantees `goto`
only ever receives one of the three node names that actually exist in the graph.

### Confusion 4: Can `update` be left out?

Yes. `Command(goto="some_node")` with no `update` just routes, without changing
state. `update` is only needed when the node also has data to store.

### Confusion 5: Is `Command` only for LLM nodes?

No. Part 1 of the notebook routes on a plain integer check with no LLM involved.
`Command` is a general routing mechanism; the LLM classification in Part 2 is one
common use of it, not a requirement.

---

## Summary

`Command(update=..., goto=...)` lets a node update state and choose the next
node in a single return value, instead of splitting that decision across a node
and a separate conditional-edge function.

```text
Conditional edges:  node  ->  routing function reads state  ->  next node
Command:            node  ->  update state AND choose next node, together
```

Key points from this notebook:

| Piece | Role |
|---|---|
| `Command(update, goto)` | Updates state and names the next node in one step |
| `Command[Literal["a", "b"]]` return hint | Tells LangGraph which nodes this node might go to |
| No outgoing edge on a node | The run ends there, same with or without `Command` |
| Structured output (`Literal` field) | Keeps an LLM's routing decision to exact node names |

Use `Command` when a node already computes the value the routing decision needs.
Use `add_conditional_edges` when the routing logic is reused across nodes or
needs to stand on its own, independent of any single node's return value.
