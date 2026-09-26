# LangGraph Multi-Agent Systems Guide
### Splitting Work Across a Supervisor and Specialist Agents

---

## Part 1: Why Split Work Across Agents?

Every agent built so far in this repo (`15_Tools`, `16_Prebuilt_Agents`) is one
model with one system prompt and one tool list, handling a whole request by itself.
That works well while the task is small. As a task grows, one agent handling
everything runs into real problems:

- The system prompt has to describe every role at once ("do research, then write,
  then check facts, then format"), which makes it longer and easier to get wrong.
- Every tool is available at every step, even the ones that only make sense for one
  part of the task.
- There is no natural place to test or reuse just the "research" part on its own.

Splitting the work into specialist agents — each with a short, focused prompt and
only the tools it needs — keeps each piece easy to write, test, and reason about on
its own.

---

## Part 2: Two Ways to Connect Specialist Agents

### Pattern 1: Supervisor

A coordinator node looks at the task and the work done so far, decides which
specialist should act next, and always gets control back once that specialist
finishes.

```text
                +--------------+
      +-------->|  supervisor  |<--------+
      |         +------+-------+         |
      |                |                 |
      |        picks the next agent      |
      |                |                 |
      |   +------------+------------+    |
      |   |                         |    |
      |   v                         v    |
+-----+------+              +-------+----+
| researcher |              |   writer   |
+------------+              +------------+
```

The supervisor is always in charge of what happens next. Specialists never decide
where control goes; they just do their job and report back. This is the pattern
this notebook builds.

### Pattern 2: Handoffs

Agents transfer control directly to each other, without a central coordinator. A
handoff is a normal tool, except that instead of returning data, it returns a
`Command` that jumps execution to a different agent:

```python
from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

@tool
def transfer_to_billing_agent(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the conversation to the billing agent."""
    # Every tool call needs a matching ToolMessage, or the next model call is rejected.
    tool_message = ToolMessage(content="Transferred to billing agent", tool_call_id=tool_call_id)
    return Command(
        goto="billing_agent",
        graph=Command.PARENT,
        update={"messages": state["messages"] + [tool_message]},
    )
```

`InjectedState` and `InjectedToolCallId` are filled in by LangGraph when the tool
runs; the model never sees them as arguments.

`graph=Command.PARENT` matters here: it tells LangGraph that `goto` refers to a node
in the *parent* graph, not a node inside the current agent's own internal graph.
Without it, `goto="billing_agent"` would look for a node named `billing_agent`
inside the calling agent's own graph and fail to find one.

Handoffs suit conversational, back-and-forth situations where any agent might decide
a different specialist should take over next (a support agent recognizing a billing
question partway through a conversation, for example). A supervisor suits situations
where one place should own the routing logic and specialists should not need to know
about each other at all. This notebook uses the supervisor pattern because it is
easier to keep deterministic and to add a hard stopping point to, which matters for
a beginner example.

There are also two small helper packages on PyPI for these patterns —
`langgraph-supervisor` and `langgraph-swarm` — that provide ready-made supervisor and
handoff-based ("swarm") graph builders so you do not have to write the routing node
or handoff tools by hand. This notebook builds both by hand instead, since seeing the
`Command`-based routing directly is the point of the exercise.

---

## Part 3: What This Notebook Builds

```text
20_Multi_Agent/
|-- 1_supervisor.ipynb
|-- Multi_Agent_Guide.md
```

The notebook builds one supervisor graph with three nodes:

- `supervisor`: decides `researcher`, `writer`, or finish, using structured output.
- `researcher`: a `create_agent` agent with one small lookup tool.
- `writer`: a plain LLM call that turns research notes into a final answer.

Every node returns a `Command`, so the graph has exactly one `add_edge` call
(`START` to `"supervisor"`) and no `add_conditional_edges` calls at all — the same
`Command(goto=...)` idea introduced in `06_Command_Routing`, just with more than two
possible destinations.

---

## Part 4: The Researcher's Tool and Agent

The researcher gets one small tool: a lookup over a few hardcoded facts about a
fictional product, so the example needs no network access beyond the LLM call
itself.

```python
PRODUCT_FACTS = {
    "battery": "The Aurora Smart Thermostat runs for about 8 months per battery charge.",
    "price": "The Aurora Smart Thermostat costs $89.",
    "compatibility": "The Aurora Smart Thermostat works with most 24V HVAC systems and needs a C-wire.",
}


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about the Aurora Smart Thermostat. Topic is one of: battery, price, compatibility."""
    return PRODUCT_FACTS.get(topic.lower(), "No fact found for that topic.")
```

The researcher itself is a normal prebuilt agent from `16_Prebuilt_Agents`, scoped
down to one job:

```python
researcher_agent = create_agent(
    model=llm,
    tools=[lookup_fact],
    system_prompt="You are a researcher. Use lookup_fact to answer the request, then summarize what you found in one or two sentences.",
)
```

This is the same idea as `19_Subgraphs`: a fully compiled graph (here, an agent
built by `create_agent`) used as one step inside a bigger graph, instead of being run
on its own.

---

## Part 5: Shared State

```python
class SupervisorState(TypedDict, total=False):
    task: str
    research_notes: str
    final_answer: str
    steps: int
    history: Annotated[list[str], operator.add]
```

All three nodes read and write this same state. `history` uses `operator.add` as a
reducer, so every node's contribution is appended to the running list instead of
overwriting it — the same reducer idea as `add_messages` in earlier notebooks, just
applied to plain strings instead of chat messages. It exists purely so the notebook
can print the route the graph actually took.

---

## Part 6: The Supervisor Node

```python
class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]


MAX_STEPS = 6
```

```python
def supervisor(state: SupervisorState) -> Command[Literal["researcher", "writer", "__end__"]]:
    if state.get("steps", 0) >= MAX_STEPS:
        return Command(
            goto=END,
            update={"history": ["supervisor: step limit reached, stopping"]},
        )

    if state.get("final_answer"):
        return Command(goto=END, update={"history": ["supervisor: final answer ready, finishing"]})

    prompt = (
        f"Task: {state['task']}\n"
        f"Research notes so far: {state.get('research_notes') or 'none yet'}\n"
        f"Final answer so far: {state.get('final_answer') or 'none yet'}\n"
        "Choose 'researcher' if there are no research notes yet, 'writer' if there are "
        "research notes but no final answer yet, or 'FINISH' if a final answer already exists."
    )
    decision = llm.with_structured_output(RouteDecision).invoke(prompt)

    if decision.next == "FINISH":
        return Command(goto=END, update={"history": ["supervisor: chose FINISH"]})

    return Command(
        goto=decision.next,
        update={"steps": state.get("steps", 0) + 1, "history": [f"supervisor: routed to {decision.next}"]},
    )
```

Three things happen here, in order:

1. **The step-limit check runs first**, before asking the model anything. This is
   what guarantees the graph ends, discussed in Part 8.
2. **A short-circuit for the already-done case**: if `final_answer` is already set,
   there is no reason to call the model again just to be told to finish.
3. **Only then** does it call `llm.with_structured_output(RouteDecision)` to get an
   actual routing decision, and turns that decision straight into a `Command`.

`with_structured_output` here plays the same role as `response_format` did for a
whole agent in `16_Prebuilt_Agents`: it forces the model's output into a specific
pydantic shape (`RouteDecision`, with one field `next` restricted to three literal
values) instead of a free-text reply.

---

## Part 7: The Worker Nodes

```python
def researcher_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
    result = researcher_agent.invoke(
        {"messages": [{"role": "user", "content": f"Research this for the task: {state['task']}"}]}
    )
    notes = result["messages"][-1].content
    return Command(
        goto="supervisor",
        update={"research_notes": notes, "history": ["researcher: gathered notes"]},
    )


def writer_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
    prompt = (
        f"Task: {state['task']}\n"
        f"Research notes: {state.get('research_notes')}\n"
        "Write a short final answer using only these notes."
    )
    response = llm.invoke(prompt)
    return Command(
        goto="supervisor",
        update={"final_answer": response.content, "history": ["writer: wrote final answer"]},
    )
```

Neither worker decides where to go next beyond `"supervisor"`. `researcher_node`
calls the researcher agent as a normal function and pulls out its last message as
plain text. `writer_node` does not need `create_agent` at all — it has no tools, so
it is just one direct `llm.invoke(...)` call with a prompt built from the state.

---

## Part 8: Building the Graph, and Why the Step Limit Matters

```python
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_edge(START, "supervisor")

app = graph.compile()
```

Because every node already returns a `Command(goto=...)`, the graph only needs its
three nodes and one starting edge. There is no `add_conditional_edges` call anywhere
in this notebook — every routing decision is made and returned from inside the node
that made it.

Structured output is a model prediction, not a guarantee. Nothing stops a model from
choosing `"researcher"` over and over instead of ever moving on to `"writer"` or
`"FINISH"`, especially with a low-effort or very deterministic model. Without
`MAX_STEPS`, that would be a real infinite loop between `supervisor` and
`researcher`, with no error and no way out. The step-limit check in Part 6 is what
actually guarantees termination — it does not depend on the model ever cooperating.
`history` exists so a run that hits the limit is still visible and debuggable
instead of silently cutting off.

---

## Part 9: Running It

```python
result = app.invoke({
    "task": "How long does the Aurora Smart Thermostat's battery last?",
    "steps": 0,
})
print("Route taken:")
for line in result["history"]:
    print(" -", line)
print()
print("Final answer:", result.get("final_answer"))
```

A full, well-behaved run visits `researcher` (to get the battery fact), then
`writer` (to phrase a final answer), then the supervisor finishes. `history` prints
that sequence line by line, so it is visible which specialist ran and in what order,
not just the final text.

---

## Part 10: Common Beginner Confusions

### Confusion 1: Does the supervisor call the workers as functions, or route to them?

Both, in a sense, but they are different mechanisms. `supervisor` never calls
`researcher_node` or `writer_node` directly. It returns `Command(goto="researcher")`,
and LangGraph runs the `"researcher"` node next as a normal step in the graph. The
worker nodes, in turn, do call `researcher_agent.invoke(...)` directly, because
`researcher_agent` is a separate compiled graph, not a node registered in *this*
graph.

### Confusion 2: Why doesn't the researcher or writer decide when to stop?

Centralizing that decision in one place (the supervisor) is the whole point of this
pattern. If every specialist could also decide to end the graph, there would be no
single place to look at to understand the overall flow, and two specialists could
disagree about whether the task is done.

### Confusion 3: What happens if `MAX_STEPS` is removed?

If the model ever fails to choose `"FINISH"` (or state never naturally satisfies the
`final_answer` short-circuit), the graph loops between `supervisor` and whichever
worker it keeps picking, forever. This is not a hypothetical edge case — it is
`with_structured_output` returning a valid-but-unhelpful decision. Some kind of
step or iteration limit is standard practice for any agent loop whose exit
condition depends on a model's output, not just this one.

### Confusion 4: Is a handoff tool just a regular tool that returns a `Command`?

Mostly. Three things make a handoff tool different from `lookup_fact` above: it
returns a `Command` instead of a value; it needs `graph=Command.PARENT` so the `goto`
target is looked up in the right graph; and because it returns a `Command`, it must
add its own `ToolMessage` for its tool call to the update (a normal tool gets one
automatically from `ToolNode`). The
`@tool` decorator, the docstring, and the way it gets bound to a model are otherwise
identical to any other tool in this repo.

### Confusion 5: Should I write my own supervisor, or use `langgraph-supervisor`?

Either is reasonable. Writing it by hand, as this notebook does, makes the routing
logic and the `Command` mechanics fully visible, which is useful while learning.
`langgraph-supervisor` (and `langgraph-swarm` for the handoff pattern) exist on PyPI
and provide the same ideas as ready-made building blocks once the underlying pattern
is familiar.

---

## Summary

A supervisor graph keeps one node in charge of routing, while every specialist node
does one job and hands control back with `Command(goto="supervisor", ...)`. This
notebook uses `Command` for every transition, so there is no `add_conditional_edges`
call at all — the same idea as `06_Command_Routing`, extended to more than two
destinations. A specialist built with `create_agent` (the researcher here) plugs
into a node exactly like a plain Python function, the same way a compiled subgraph
plugs into a parent graph in `19_Subgraphs`. Because the routing decision itself
comes from a model, a hard step limit — not the model's good behavior — is what
actually guarantees the graph terminates. The alternative pattern, handoffs, moves
that routing decision into tools that any agent can call, useful when no single
place should own it; `langgraph-supervisor` and `langgraph-swarm` package both
patterns as reusable helpers once you understand what they are doing underneath.
