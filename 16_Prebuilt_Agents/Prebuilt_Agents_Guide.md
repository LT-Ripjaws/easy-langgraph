# LangGraph Prebuilt Agents Guide
### From a Hand-Built Tool Loop to `create_agent`

---

## Part 1: What Problem Does `create_agent` Solve?

`15_Tools/Tooling.ipynb` builds a tool-using chatbot by hand: a `StateGraph`, a
`chat_node` that calls the model, a `ToolNode` that runs whatever tool the model
asked for, and `tools_condition` to route between them. That graph is a completely
standard shape:

```text
START -> chat_node -> tools_condition -> tools -> chat_node -> ... -> END
```

Almost every tool-using agent needs that same shape. As of LangChain 1.0 (October
2025), `langchain.agents.create_agent` builds it for you. You still get a compiled
LangGraph graph back; you just do not have to write the `StateGraph`, the
`ToolNode`, or the conditional edge yourself.

```text
from langgraph.prebuilt import create_react_agent   # deprecated
from langchain.agents import create_agent            # current standard
```

`create_react_agent` (from `langgraph.prebuilt`) is the older name for this same
idea and is now deprecated. New code should use `langchain.agents.create_agent`.

---

## Part 2: What This Notebook Builds

```text
16_Prebuilt_Agents/
|-- 1_create_agent.ipynb
|-- Prebuilt_Agents_Guide.md
```

The notebook builds five small agents, each adding one capability on top of the
basic tool-using loop:

1. A basic agent with one tool, compared to the hand-built loop in `15_Tools`.
2. The same idea with `response_format` added, for structured output.
3. The same idea with a `checkpointer` added, for multi-turn memory.
4. An agent whose tool reads request-scoped data through `context_schema`.
5. An agent with `HumanInTheLoopMiddleware`, which pauses before a sensitive tool
   call and waits for approval.

None of these are separate graphs built from scratch. Each one is a different set of
keyword arguments to `create_agent`.

---

## Part 3: The Model

```python
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
```

This is the same model setup used across this repo. One cell later in the notebook
switches to `gemini-2.5-flash` for a specific reason explained in Part 5 below.

---

## Part 4: A Basic Agent With a Tool

The tool is a plain function decorated with `@tool`, exactly like in `15_Tools`:

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake_weather = {
        "paris": "18C and cloudy",
        "tokyo": "24C and sunny",
        "cairo": "33C and clear",
    }
    return fake_weather.get(city.lower(), "20C and mild")
```

The agent itself is one function call:

```python
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use tools when you need current data.",
)
```

Compared to `15_Tools/Tooling.ipynb`, this one call replaces:

| `15_Tools` (hand-built) | `create_agent` equivalent |
|---|---|
| `class ChatState(TypedDict): messages: Annotated[...]` | Built in automatically |
| `def chat_node(state): ...; return {"messages": [response]}` | Built in automatically |
| `tool_node = ToolNode(tools)` | Built in automatically |
| `graph.add_edge(START, "chat_node")` | Built in automatically |
| `graph.add_conditional_edges("chat_node", tools_condition)` | Built in automatically |
| `graph.add_edge("tools", "chat_node")` | Built in automatically |
| `graph.compile()` | Returned directly by `create_agent(...)` |

It is invoked the same way as any LangGraph app, with a `messages` list:

```python
result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
for m in result["messages"]:
    print(type(m).__name__, "-", getattr(m, "content", None) or m.tool_calls)
```

A full run produces four messages in `result["messages"]`: the human question, an
`AIMessage` containing a tool call for `get_weather`, a `ToolMessage` with the tool's
return value, and a final `AIMessage` with the model's answer. That is the exact
same reason/act/observe sequence documented in `15_Tools/Tools_Guide.md`; only the
code that builds the graph changed.

---

## Part 5: Structured Output

By default, an agent's last message is a plain-text `AIMessage`. Passing
`response_format=SomePydanticModel` makes the agent also return a validated object
matching that model, at `result["structured_response"]`:

```python
class WeatherReport(BaseModel):
    city: str
    condition: str
```

```python
flash_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

structured_agent = create_agent(
    model=flash_llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use tools when you need current data.",
    response_format=WeatherReport,
)

result = structured_agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Cairo?"}]})
print(result["structured_response"])
```

### Why This Cell Uses `gemini-2.5-flash` Instead of `gemini-2.5-flash-lite`

Internally, `response_format` works by adding one more forced tool call at the end
of the turn: a "tool" whose schema is the pydantic model, so the model's very last
action is producing structured data instead of free text.

While building this notebook, side-by-side testing showed a real difference between
the two models when a regular tool and `response_format` are used together:

```text
gemini-2.5-flash-lite:
  1. Calls get_weather.
  2. Reads the tool result.
  3. Answers in plain text.
  4. result["structured_response"] is None.

gemini-2.5-flash:
  1. Calls get_weather.
  2. Reads the tool result.
  3. Calls the WeatherReport tool.
  4. result["structured_response"] is a filled-in WeatherReport.
```

`gemini-2.5-flash-lite` simply stops after answering in text; it never makes the
forced structured-output call, even though `response_format` was set. This is not a
code bug — the exact same `create_agent(...)` call behaves differently depending on
the model. `gemini-2.5-flash` is used for this one cell because it produces the
structured output reliably; the rest of the notebook keeps using
`gemini-2.5-flash-lite`, since this particular tool-plus-structured-output
combination is the only place the difference showed up.

If you see `result["structured_response"]` come back `None` with a different model
or tool combination, this is the first thing to check.

---

## Part 6: Memory Across Turns

`checkpointer=InMemorySaver()` gives the agent memory. LangGraph saves the graph's
state after every step, keyed by a `thread_id` inside `config`. Invoking the agent
again with the same `thread_id` continues that saved conversation instead of
starting a new one — the same checkpointer idea as `09_Persistence`, wired into a
prebuilt agent instead of a hand-built graph.

```python
checkpointer = InMemorySaver()
memory_agent = create_agent(model=llm, tools=[get_weather], checkpointer=checkpointer)

config = {"configurable": {"thread_id": "weather-chat-1"}}

result = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]},
    config=config,
)
```

A second call with the same `config` can refer back to the first turn, because the
full message history was saved and is loaded back in before this turn runs:

```python
result = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "Which city did I just ask about?"}]},
    config=config,
)
```

Without the checkpointer, each `invoke` would start from an empty `messages` list
and the second question would have nothing to refer back to.

---

## Part 7: Runtime Context

Sometimes a tool needs information that should not live in the conversation itself —
a user ID, a permission level, a tenant name. `context_schema` declares the shape of
that data, and `agent.invoke(..., context=...)` passes a value in for one specific
call, without it ever becoming part of `messages`.

```python
@dataclass
class UserContext:
    user_name: str


@tool
def greet_user(runtime: ToolRuntime[UserContext]) -> str:
    """Greet the current user by name."""
    return f"Hello, {runtime.context.user_name}!"
```

A tool reads the context by adding a parameter named `runtime`, typed as
`ToolRuntime` (imported from `langchain.tools`, not from `langchain.agents`).
LangGraph inspects the tool's signature, sees a `runtime: ToolRuntime` parameter, and
injects a `ToolRuntime` object automatically — it is not something the caller passes
as a tool argument.

```python
context_agent = create_agent(model=llm, tools=[greet_user], context_schema=UserContext)

result = context_agent.invoke(
    {"messages": [{"role": "user", "content": "Greet me please."}]},
    context=UserContext(user_name="Chinmoy"),
)
```

`runtime.context` inside the tool is the exact `UserContext` instance passed to
`context=`. `ToolRuntime` also exposes `runtime.state` (the current graph state),
`runtime.tool_call_id`, `runtime.config`, and `runtime.store`, for tools that need
more than just the context object.

---

## Part 8: Middleware

Middleware is code that runs before or after the model or a tool call, without
changing how the model or the tool itself works. It is how `create_agent` adds
optional behavior — approval steps, retries, trimming old messages, guardrails —
without every project rebuilding the agent loop to add each one.

```python
middleware=[SomeMiddleware(...), AnotherMiddleware(...)]
```

is a plain list passed to `create_agent`. This notebook uses one built-in
middleware, `HumanInTheLoopMiddleware`, which pauses the agent right before a chosen
tool call runs and waits for a human decision.

```python
@tool
def send_email(to: str, body: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {to}: {body}"
```

```python
hitl_checkpointer = InMemorySaver()
email_agent = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt="You help the user send emails.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
    checkpointer=hitl_checkpointer,
)
```

`interrupt_on={"send_email": True}` marks `send_email` as a tool that always
requires approval before it runs. A checkpointer is required here: pausing and later
resuming a graph only works if its state was saved somewhere in between.

### The First Call Pauses

```python
config = {"configurable": {"thread_id": "email-1"}}
result = email_agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to bob@example.com saying hi"}]},
    config=config,
)
print("paused:", "__interrupt__" in result)
print(result["__interrupt__"][0].value)
```

Instead of a final answer, `result` contains an `"__interrupt__"` key. Its value
describes exactly what is waiting for approval:

```python
{
    "action_requests": [
        {
            "name": "send_email",
            "args": {"to": "bob@example.com", "body": "hi"},
            "description": "Tool execution requires approval\n...",
        }
    ],
    "review_configs": [
        {"action_name": "send_email", "allowed_decisions": ["approve", "edit", "reject", "respond"]}
    ],
}
```

### Resuming

To continue, invoke the agent again with `Command(resume=...)` instead of a new
message:

```python
result = email_agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
```

The exact shape of `resume` matters and is easy to get wrong by guessing:

- It must be a dictionary with a `"decisions"` key, containing one decision per
  pending tool call — not a bare list.
- Each decision's `"type"` must be one of the values listed in
  `allowed_decisions` from the interrupt payload above: `"approve"`, `"edit"`,
  `"reject"`, or `"respond"`. Some documentation and examples elsewhere use the word
  "accept"; the middleware code itself only recognizes `"approve"`.

After resuming with `"approve"`, `send_email` actually runs, and the agent produces
its normal final answer, appended to the same `result["messages"]` list as before.

---

## Part 9: What This Notebook's Cells Do, In Order

| Cell | Purpose |
|---|---|
| Imports | `create_agent`, `ToolRuntime`, middleware, model, checkpointer, `Command` |
| `load_dotenv()` | Loads `GOOGLE_API_KEY` from `.env` |
| `llm = ChatGoogleGenerativeAI(...)` | Default model for this notebook |
| `get_weather` tool | Shared tool used in Parts 1 and 2 |
| `agent = create_agent(...)` | Basic tool-using agent |
| `WeatherReport` + `flash_llm` + `structured_agent` | Structured output demo |
| `checkpointer` + `memory_agent` | Multi-turn memory demo |
| `UserContext` + `greet_user` + `context_agent` | Runtime context demo |
| `send_email` + `email_agent` | Middleware (human-in-the-loop) demo |

---

## Part 10: Common Beginner Confusions

### Confusion 1: Is `create_agent` a different kind of graph?

No. `agent = create_agent(...)` returns the same kind of object as
`graph.compile()` in every other notebook in this repo: a compiled LangGraph app you
call `.invoke(...)` on. `create_agent` is a function that builds a `StateGraph` for
you; it does not introduce a new execution model.

### Confusion 2: Does `response_format` replace the normal answer?

No. The agent still produces its normal `AIMessage` messages in
`result["messages"]`. `result["structured_response"]` is an additional field, filled
in by one extra forced tool call at the end.

### Confusion 3: Why does a tool need `ToolRuntime` instead of just a normal argument?

Because context values (like the current user) should not be something the model
has to type out as a tool argument — the model does not know or need to know they
exist. `ToolRuntime` is injected by the framework based on the parameter's type
annotation, not filled in by the model.

### Confusion 4: Does middleware change what a tool does?

No. `send_email` above is a completely normal tool. `HumanInTheLoopMiddleware` adds
a pause before it runs; it does not touch the tool's own code. The same tool works
identically with no middleware at all, just without the approval step.

### Confusion 5: Can I resume an interrupted agent with a plain string or a new message?

No. Resuming requires `Command(resume=...)`, not `agent.invoke({"messages": [...]})`.
Passing a new message starts a fresh turn in the same thread; it does not answer the
pending interrupt.

---

## Summary

`create_agent` builds the same LLM-decides / tool-executes / LLM-answers loop as the
hand-built graph in `15_Tools`, in one function call. On top of that basic loop, four
plain keyword arguments add real capability:

| Argument | Adds |
|---|---|
| `response_format` | A structured, validated object at `result["structured_response"]` |
| `checkpointer` | Multi-turn memory, keyed by `thread_id` |
| `context_schema` + `context=` | Request-scoped data a tool reads via `ToolRuntime` |
| `middleware` | Cross-cutting behavior around model and tool calls, such as human approval |

None of these require rebuilding the graph. They are the difference between the
basic agent in Part 4 and every other agent in this notebook. A model can also
matter more than the code around it: `gemini-2.5-flash-lite` and `gemini-2.5-flash`
ran the exact same `create_agent(...)` call with different results in Part 5, which
is worth remembering whenever an agent's behavior looks wrong — check the model
before assuming the code is broken.
