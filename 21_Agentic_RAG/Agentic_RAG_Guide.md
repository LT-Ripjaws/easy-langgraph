# Agentic RAG Guide
### Retrieving, Grading, and Deciding Whether to Try Again

---

## Part 1: What Is RAG?

Retrieval-augmented generation (RAG) answers a question in two steps instead of one:

```text
Plain LLM:
  question -> model -> answer (from training data only)

RAG:
  question -> retrieve relevant text -> model reads question + text -> answer
```

This matters because a model's training data is fixed and general. It does not know
about a specific product's documentation, a company's internal policies, or
anything written after training. Retrieval gives the model the actual source text to
read before it answers, instead of asking it to recall or guess.

## Part 2: What Makes RAG "Agentic"?

Plain RAG always retrieves once and always answers with whatever came back, even if
the retrieved text does not actually answer the question. Agentic RAG adds a
decision in the middle:

```text
retrieve -> is this text actually relevant? -> yes -> answer
                                             -> no  -> rewrite the question, try again
```

The "agentic" part is that the graph itself checks its own retrieval before
committing to an answer, and can act on that check (rewrite and retry) instead of
answering regardless. This notebook builds exactly that loop, with a hard limit on
how many times it will retry.

---

## Part 3: What This Notebook Builds

```text
21_Agentic_RAG/
|-- 1_agentic_rag.ipynb
|-- Agentic_RAG_Guide.md
```

```text
START -> agent -> tools_condition -> retrieve -> grade_documents
                                                        |
                                          +-------------+-------------+
                                          |                           |
                                       generate                    rewrite
                                          |                           |
                                         END                       agent (loop)
```

The notebook builds:

1. A small set of made-up product documents (no downloads needed).
2. An `InMemoryVectorStore`, embedded once with `GoogleGenerativeAIEmbeddings`.
3. A retriever exposed as a tool.
4. A graph: an agent node that can call the retriever, a grading step that checks
   whether what came back is useful, and either a final answer or a rewritten
   question that loops back to the start.
5. Two runs: one question the documents answer, one they do not.

---

## Part 4: The Document Set and Embeddings

```python
DOCS = [
    "The Nimbus Home Hub is a smart home hub that connects lights, locks, cameras, "
    "and thermostats over WiFi and Zigbee, and controls them from one app.",
    ...
]
```

Eight short paragraphs describe a fictional product, the "Nimbus Home Hub": what it
is, how to set it up, what it integrates with, its subscription tiers, its power
requirements, its warranty, troubleshooting steps, and how it handles data privacy.
Keeping the whole corpus inline means the notebook has no external dependency beyond
the embedding and chat model calls.

```python
documents = [Document(page_content=text) for text in DOCS]
vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

`GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")` turns each
paragraph into a vector once, when `from_documents` is called. Every question asked
later reuses this same `vectorstore`; no document is re-embedded per question, only
the incoming query is.

`models/gemini-embedding-001` produces 3072-dimensional vectors. The older
`text-embedding-004` model no longer exists — `gemini-embedding-001` is the current
embedding model for this API.

---

## Part 5: The Retriever as a Tool

```python
retriever_tool = create_retriever_tool(
    retriever,
    name="search_nimbus_docs",
    description="Search the Nimbus Home Hub product documentation for relevant passages.",
)
tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)
```

`create_retriever_tool` wraps any retriever as a normal LangChain tool: it gives it
a name and a description the model uses to decide when to call it, and it returns
the matched documents' text as the tool's result. This is the same tool-calling
mechanism used in `15_Tools` and `16_Prebuilt_Agents` — a retriever tool is not a
special case, just a tool whose job is "go fetch some text."

---

## Part 6: Graph State

```python
class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rewrite_count: int
    grade: str


MAX_REWRITES = 2
```

`messages` works exactly like every other message-based graph in this repo:
`add_messages` appends instead of overwriting. `rewrite_count` and `grade` are new —
`rewrite_count` tracks how many times the question has already been rewritten, and
`grade` holds the most recent yes/no relevance judgment, so the routing function in
Part 8 can make its decision from a plain state field instead of re-deriving it.

---

## Part 7: The Agent Node

```python
RAG_SYSTEM_PROMPT = (
    "You answer questions about the Nimbus Home Hub. Always use search_nimbus_docs "
    "before answering. If the search results do not contain the answer, say so plainly "
    "instead of guessing."
)


def agent_node(state: RAGState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=RAG_SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}
```

This is the same shape as `chat_node` in `15_Tools/Tooling.ipynb`: read the current
messages, call a tool-bound model, return whatever it produces. The system prompt is
prepended at call time rather than stored in `messages`, so it is not duplicated into
the saved conversation on every turn.

---

## Part 8: Grading, Then Generating or Rewriting

```python
class Grade(BaseModel):
    binary_score: Literal["no", "yes"]


def grade_documents(state: RAGState) -> dict:
    question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    retrieved = state["messages"][-1].content

    prompt = (
        f"Question: {question}\n"
        f"Retrieved text: {retrieved}\n"
        "Does the retrieved text answer the question? Answer yes or no."
    )
    result = llm.with_structured_output(Grade).invoke(prompt)
    return {"grade": result.binary_score}
```

`grade_documents` runs right after the retriever tool, while `state["messages"][-1]`
is still the `ToolMessage` holding the retrieved text. It asks the model a narrow
yes/no question — not "answer this," but "does this retrieved text answer this" —
using `with_structured_output(Grade)` so the result is a validated `"yes"` or
`"no"`, not a sentence to parse.

```python
def route_after_grade(state: RAGState) -> Literal["generate", "rewrite"]:
    if state["grade"] == "yes":
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "generate"
    return "rewrite"
```

Notice `grade_documents` does not route by itself — it only writes `"grade"` into
state. `route_after_grade` is a separate, plain conditional-edge function (the same
style as `tools_condition` and every conditional edge in `04_Conditional_Workflows`)
that reads `"grade"` back out and decides. Splitting these two responsibilities keeps
each function doing one simple thing: one produces a judgment, the other acts on it.

```python
def rewrite_node(state: RAGState) -> dict:
    question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    prompt = (
        f"Rewrite this question to make it easier to find in product documentation, "
        f"keeping the same meaning: {question}"
    )
    rewritten = llm.invoke(prompt)
    return {
        "messages": [HumanMessage(content=rewritten.content)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }
```

`rewrite_node` appends a *new* `HumanMessage` with the reworded question rather than
editing the old one in place, so the original question stays visible in the full history. It
increments `rewrite_count` every time it runs, which is what lets
`route_after_grade` eventually give up and generate anyway.

```python
def generate_node(state: RAGState) -> dict:
    question = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    retrieved = state["messages"][-1].content
    prompt = (
        f"Question: {question}\n"
        f"Retrieved text: {retrieved}\n"
        "Answer the question using only the retrieved text. If it does not contain "
        "the answer, say the documentation does not cover this."
    )
    answer = llm.invoke(prompt)
    return {"messages": [AIMessage(content=answer.content)]}
```

`generate_node` is explicitly told to answer only from the retrieved text and to
admit when that text does not cover the question, rather than filling the gap from
its own general knowledge. This matters most for the second run in Part 10.

---

## Part 9: Building the Graph

```python
graph = StateGraph(RAGState)
graph.add_node("agent", agent_node)
graph.add_node("retrieve", ToolNode(tools))
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate", generate_node)
graph.add_node("rewrite", rewrite_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
graph.add_edge("rewrite", "agent")
graph.add_edge("generate", END)

app = graph.compile()
```

Two conditional edges do all the branching:

- `tools_condition` after `"agent"`: this is the exact prebuilt function from
  `15_Tools`, routing to `"retrieve"` if the model asked for a tool call, or ending
  the graph if it answered directly.
- `route_after_grade` after `"grade_documents"`: the custom function from Part 8,
  choosing between `"generate"` and `"rewrite"`.

`"rewrite"` connects back to `"agent"` with a plain unconditional edge, which is what
makes this a loop: a rewritten question goes through `agent -> tools_condition ->
retrieve -> grade_documents` again, exactly like the first attempt.

---

## Part 10: Running Both Questions

### A Question the Docs Answer

```python
result = app.invoke({
    "messages": [HumanMessage(content="What is the warranty period for the Nimbus Home Hub?")],
    "rewrite_count": 0,
})
print(result["messages"][-1].content)
```

The warranty paragraph is directly in `DOCS`, so a good run retrieves it, grades it
as relevant, and answers from it on the first pass.

### A Question the Docs Do Not Answer

```python
result = app.invoke({
    "messages": [HumanMessage(content="What programming language does the Nimbus Home Hub's firmware use?")],
    "rewrite_count": 0,
})
print(result["messages"][-1].content)
```

Nothing in `DOCS` mentions firmware or programming languages. A good run either
grades the retrieved text as not relevant and rewrites the question (possibly more
than once, up to `MAX_REWRITES`), or reaches the rewrite limit — either way, the
final answer should say the documentation does not cover this, not invent a
plausible-sounding language.

---

## Part 11: Why `MAX_REWRITES` Matters

Grading is also a model prediction, not a guarantee. If the documents genuinely do
not contain an answer, no amount of rewriting will make `grade_documents` come back
`"yes"` — the model could keep grading the retrieved text as unhelpful forever.
`MAX_REWRITES` (set to `2`) caps how many times `route_after_grade` will send the
question back through the loop before forcing `"generate"` regardless of the grade,
the same role `MAX_STEPS` plays for the supervisor's routing loop in
`20_Multi_Agent`. Without it, a genuinely unanswerable question would loop between
`agent` and `rewrite` with no way out.

---

## Part 12: Common Beginner Confusions

### Confusion 1: Does `grade_documents` decide where the graph goes next?

No. `grade_documents` only writes a `"grade"` value into state. `route_after_grade`,
registered separately with `add_conditional_edges`, is what reads that value and
picks the next node. Keeping them separate mirrors `tools_condition`, which also
only reads state — it never runs a node's actual logic.

### Confusion 2: Why does `rewrite_node` add a new message instead of editing the question?

It could edit it: `add_messages` replaces a message when you send one with the same
`id`, and deletes one when you send `RemoveMessage(id=...)` (see
`13_Short_term_memory`). This lesson appends a new `HumanMessage` on purpose, so the
history shows both the original question and the rewrite, and the model treats the
rewrite as the latest turn.

### Confusion 3: Does the retriever tool guarantee relevant results?

No. A retriever always returns its `k` closest matches by embedding similarity, even
if none of them are actually relevant — "closest available" is not the same as
"good enough." That gap is exactly why `grade_documents` exists: retrieval and
relevance are two different questions, and only the second one decides whether to
trust the retrieved text.

### Confusion 4: Is embedding the documents repeated on every question?

No. `InMemoryVectorStore.from_documents(...)` embeds `DOCS` exactly once, when the
notebook builds the vector store. Every question after that only embeds the query
text (inside `retriever.invoke(...)`, called through the tool), not the documents
again.

### Confusion 5: What happens after `MAX_REWRITES` is reached?

`route_after_grade` sends the graph to `"generate"` regardless of the last grade.
`generate_node`'s prompt explicitly tells the model to say the documentation does
not cover the question if the retrieved text does not contain the answer, so hitting
the limit should still produce an honest answer, not a wrong one dressed up as a
confident one.

---

## Summary

Plain RAG retrieves text once and answers from it unconditionally. Agentic RAG adds
one checkpoint: `grade_documents` decides whether the retrieved text is actually
useful before the graph commits to an answer. When it is not, `rewrite_node` gives
the retriever a better-worded question and the graph tries again through the same
`agent -> tools_condition -> retrieve` path as the first attempt — the loop is just
the normal tool-calling path, run more than once. `MAX_REWRITES` stops that loop from
running forever when no rewrite will help, the same guard role `MAX_STEPS` plays for
the supervisor loop in `20_Multi_Agent`. The result is a RAG pipeline that can tell
the difference between "I found the answer" and "I looked, and it is not there" —
which is worth more than a system that always answers confidently either way.
