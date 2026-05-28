# Long-Term Memory in LangGraph

Long-term memory in LangGraph means information that survives across different conversations, sessions, and thread IDs. It is used when an agent should remember facts, preferences, lessons, examples, or instructions beyond the current chat.

In one sentence:

```text
Long-term memory = cross-thread data saved in a store under namespaces and keys.
```

---

## 1. Why Long-Term Memory Exists

Short-term memory remembers a single thread. That is useful for one ongoing conversation, but it does not solve personalization across conversations.

Example:

```text
Thread 1:
User says "Remember that I prefer short answers."

Thread 2:
User starts a new chat and asks a question.
```

If you only use short-term memory, thread 2 does not know what happened in thread 1.

Long-term memory solves this by saving selected information outside the thread. Any future thread for the same user can retrieve it.

---

## 2. Core Idea

Long-term memory has four main parts:

| Part | Meaning |
|------|---------|
| Store | Persistent key-value memory interface |
| Namespace | A tuple-like path for grouping memories |
| Key | Unique ID for one memory item |
| Value | JSON-like document stored as the memory |

Conceptually:

```text
namespace = ("users", "user_123", "preferences")
key       = "response_style"
value     = {"preference": "short, direct answers"}
```

The agent can later search or retrieve that memory from any thread.

---

## 3. Short-Term vs Long-Term Memory

| Question | Short-Term Memory | Long-Term Memory |
|----------|-------------------|------------------|
| Scope | One thread | Many threads |
| LangGraph primitive | Checkpointer | Store |
| Main ID | `thread_id` | Namespace and key |
| Typical data | Chat history, temporary state | User facts, preferences, instructions |
| Updated automatically | Graph state is checkpointed automatically | You decide what to save |
| Retrieval | Load latest thread state | `get`, `search`, filters, semantic search |
| Example | "What did we discuss earlier in this chat?" | "What does this user prefer in all chats?" |

Short-term memory is the working memory of a conversation.

Long-term memory is the durable memory of a user, organization, app, or agent.

---

## 4. What Long-Term Memory Stores

Long-term memory should store durable, useful information.

Good candidates:

- User name
- User preferences
- Writing style preferences
- Product settings
- Organization rules
- Learned project facts
- Reusable examples
- Instructions that improve future behavior
- Important outcomes from past tasks
- Stable domain knowledge specific to the user or app

Poor candidates:

- Every message from every chat
- Temporary scratchpad data
- One-off tool outputs
- Large raw documents without indexing
- Sensitive secrets
- Information that is likely to become stale quickly

Good long-term memory is selective. The agent should not remember everything.

---

## 5. How LangGraph Implements It

LangGraph uses stores for long-term memory.

A store is separate from the checkpointer:

```text
checkpointer -> saves thread state
store        -> saves cross-thread memories
```

You can compile a graph with a store:

```python
graph = builder.compile(store=store)
```

You can also compile with both:

```python
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)
```

That combination is common:

- The checkpointer remembers the current conversation.
- The store remembers durable facts across conversations.

---

## 6. Namespaces and Keys

Long-term memory is organized with namespaces and keys.

Think of a namespace like a folder:

```text
("users", "user_123", "memories")
```

Think of a key like a filename:

```text
"food_preference"
```

Think of the value like the file contents:

```text
{"likes": ["pizza"], "dislikes": ["shellfish"]}
```

Namespaces can represent:

- A user
- An organization
- An application
- An agent
- A project
- A memory type
- A permission boundary

Examples:

| Namespace | Meaning |
|-----------|---------|
| `("users", user_id, "profile")` | User profile document |
| `("users", user_id, "preferences")` | User preferences |
| `("orgs", org_id, "policies")` | Organization policies |
| `("projects", project_id, "facts")` | Project-specific facts |
| `("agents", agent_id, "instructions")` | Agent-specific learned instructions |

Good namespaces make memory easier to search, secure, and delete.

---

## 7. Store Operations

Common store operations:

| Operation | Purpose |
|-----------|---------|
| `put(namespace, key, value)` | Save or update a memory |
| `get(namespace, key)` | Retrieve one memory by exact key |
| `search(namespace, query=...)` | Search memories in a namespace |
| `search(namespace, filter=...)` | Search by metadata or content filters |
| `list_namespaces()` | Discover stored namespace paths |

The returned item usually includes:

- `value`: The JSON-like memory document
- `key`: The memory key
- `namespace`: The namespace path
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

This metadata is useful for debugging, sorting, freshness checks, and cleanup.

---

## 8. Store Backends

Common long-term memory stores:

| Store | Use Case |
|-------|----------|
| `InMemoryStore` | Local demos and tests |
| `PostgresStore` | Durable production memory |
| `RedisStore` | Fast memory and semantic lookup |
| `MongoDBStore` | Document-oriented production storage |

`InMemoryStore` is not durable. It disappears when the process restarts.

For production, use a database-backed store and run any required setup or migrations before serving traffic.

---

## 9. Reading Memory in a Graph

A graph node can read from the store before calling the model.

High-level flow:

```text
1. Runtime context provides user_id.
2. Node builds a namespace from user_id.
3. Node searches the store for relevant memories.
4. Node formats memories into system context.
5. Model responds using both current messages and recalled memory.
```

Example shape:

```text
User asks: "What should I cook tonight?"

Store search:
("users", "user_123", "preferences")
-> "User likes vegetarian meals"
-> "User dislikes mushrooms"

Model prompt includes:
"User info: likes vegetarian meals, dislikes mushrooms"
```

The model can now personalize its answer even if this is a brand-new thread.

---

## 10. Writing Memory

Long-term memory is not usually automatic. You need a strategy for what gets stored.

There are two major approaches.

## Hot Path Memory

The agent writes memory during the user interaction.

Flow:

```text
User message arrives
Agent decides something should be remembered
Agent writes to store before or during response
New memory is immediately available
```

Benefits:

- Memory is available right away.
- User can be told when something was remembered.
- Useful for explicit commands like "remember this."

Costs:

- Adds latency.
- The model must decide what to save while also answering.
- Can over-save noisy memories.
- Requires careful permissions.

Use hot path memory when memory creation is part of the user experience.

## Background Memory

The system writes memory after the main interaction.

Flow:

```text
Conversation happens
Background job reviews messages
Important facts are extracted
Store is updated asynchronously
Future conversations use the new memory
```

Benefits:

- Main response stays fast.
- Memory extraction can use a specialized prompt/model.
- Easier to batch, audit, and clean.
- Good for summarizing long histories.

Costs:

- New memories may not be available immediately.
- Requires background infrastructure.
- More moving pieces.

Use background memory when memory quality matters more than immediate availability.

---

## 11. Types of Long-Term Memory

The LangGraph conceptual docs describe three useful memory categories: semantic, episodic, and procedural.

## Semantic Memory

Semantic memory stores facts and concepts.

Examples:

- "User prefers Python examples."
- "User is building a LangGraph tutorial repo."
- "The company uses PostgreSQL."
- "Project X has a Streamlit frontend."

Semantic memory is useful for personalization and factual grounding.

Two common patterns:

| Pattern | Meaning |
|---------|---------|
| Profile | One document that gets updated over time |
| Collection | Many smaller memory documents |

Profile example:

```text
("users", user_id, "profile") -> "main"
{
  "name": "Noah",
  "language": "Python",
  "style": "concise"
}
```

Collection example:

```text
("users", user_id, "memories") -> uuid_1
{"fact": "User prefers concise explanations"}

("users", user_id, "memories") -> uuid_2
{"fact": "User is learning LangGraph"}
```

Profiles are compact but can become hard to update safely as they grow.

Collections are easier to append to and search, but can accumulate duplicates.

## Episodic Memory

Episodic memory stores experiences or past events.

Examples:

- "Last time the agent tried approach A, it failed because the API key was missing."
- "In session 2026-05-25, the user approved the database guide structure."
- "This debugging strategy solved a Windows path problem."

Episodic memory is useful when the agent should learn from past attempts.

It can be represented as:

- Past task traces
- Input-output examples
- Few-shot examples
- Summaries of previous sessions
- Success/failure records

## Procedural Memory

Procedural memory stores rules or instructions for how to act.

Examples:

- "When creating docs for this repo, use a tutorial-style guide."
- "Prefer small notebooks for demos."
- "When adding a Streamlit app, document how to run it."
- "Always ask for approval before destructive commands."

For AI agents, procedural memory often appears as:

- System prompt updates
- Agent instructions
- Playbooks
- Tool-use policies
- Learned workflows

Procedural memory is powerful but risky. A bad instruction can affect many future outputs, so it should be versioned, reviewed, and scoped carefully.

---

## 12. Semantic Search

Stores can support semantic search when configured with embeddings.

Exact lookup:

```text
get namespace + key
```

Semantic search:

```text
find memories similar in meaning to this query
```

Example:

```text
Query: "How should I explain this to the user?"

Relevant memories:
- "User likes short explanations"
- "User prefers beginner-friendly examples"
- "User is learning with notebooks"
```

Semantic search is useful when:

- You have many memory documents
- The exact key is unknown
- You want relevant facts by meaning
- Memory is stored as a collection

Be careful:

- Embeddings can retrieve irrelevant memories.
- Similarity is not truth.
- Retrieved memories should be formatted clearly and often filtered by namespace.

---

## 13. Using Context

Long-term memory usually needs a stable identity.

For user memory, the graph must know:

```text
which user is this?
```

For organization memory, it must know:

```text
which organization is this?
```

In LangGraph, runtime context can carry values such as:

- `user_id`
- `org_id`
- `project_id`
- `agent_id`
- permissions
- memory access policy

The node then uses that context to build namespaces.

Good pattern:

```text
context.user_id -> namespace -> search store -> inject relevant memories
```

Avoid deriving identity from untrusted user text.

---

## 14. Combining Short-Term and Long-Term Memory

Most real agents need both.

Example chatbot:

```text
Short-term memory:
- Current conversation messages
- The user's latest question
- Tool results from this chat

Long-term memory:
- User prefers short explanations
- User is learning LangGraph
- User uses Windows and PowerShell
```

Response flow:

```text
1. Load current thread state from checkpointer.
2. Search long-term store for relevant user memories.
3. Build prompt from system instructions, memories, and thread messages.
4. Generate response.
5. Save new thread state.
6. Optionally update long-term memory.
```

This separation keeps the system clean:

- Checkpointer handles "what is happening now."
- Store handles "what should persist across conversations."

---

## 15. Memory Quality

Long-term memory can make an agent much better, but bad memory can make it worse.

High-quality memories are:

- Specific
- Stable
- Useful for future behavior
- Properly scoped
- Not overly broad
- Not sensitive unless necessary and permitted
- Easy to update or delete

Weak memories:

- "User likes things"
- "User asked about code"
- "User may like Python maybe"
- "User was unhappy"

Better memories:

- "User prefers Python examples over JavaScript examples."
- "User is building an educational LangGraph repo with numbered folders."
- "For this repo, guides should be markdown files inside each topic folder."

---

## 16. Memory Update Strategies

## Explicit Memory

Only save when the user says:

```text
remember this
save this
from now on
my preference is
```

This is safest and easiest to explain.

## Inferred Memory

The agent infers memories from behavior.

Example:

```text
User repeatedly asks for concise docs.
Agent saves: "User prefers concise documentation."
```

This can be helpful but should be used carefully. Inferred memory can be wrong.

## Reviewed Memory

The agent proposes a memory and asks the user to approve it.

Example:

```text
Should I remember that you prefer short guides?
```

This gives the user control and improves trust.

## Background Consolidation

A background job reads recent thread history and updates long-term memory.

Good for:

- Session summaries
- Duplicate removal
- Profile updates
- Extracting stable facts from many messages

---

## 17. Deleting and Updating Memories

Long-term memory needs lifecycle management.

You should support:

- Updating stale memories
- Deleting incorrect memories
- User-requested deletion
- Namespace deletion for account removal
- Memory expiration policies
- Audit trails for sensitive systems

Memory should not be treated as permanent just because it is long-term.

---

## 18. Security and Privacy

Long-term memory is sensitive because it persists beyond one conversation.

Good practices:

- Do not store API keys or passwords.
- Scope memory by user or organization.
- Enforce authorization before reading a namespace.
- Avoid cross-user memory leakage.
- Let users inspect and delete stored memories.
- Encrypt sensitive stores where required.
- Keep memory schemas explicit.
- Separate personal memory from application memory.
- Avoid storing unverified claims as facts.

For example, store:

```text
"User said they prefer Python examples."
```

instead of:

```text
"User is a Python expert."
```

The first is grounded in what happened. The second may be an unsupported inference.

---

## 19. Production Design

## Choose the Right Backend

Use `InMemoryStore` for demos only.

For production, use a persistent backend such as Postgres, Redis, or MongoDB.

## Run Migrations

Database-backed stores often require setup before use.

Run setup or migrations as part of deployment, not as a surprise during the first user request.

## Design Namespaces Carefully

Namespace design affects:

- Search relevance
- Access control
- Deletion
- Multi-tenant isolation
- Debugging

Good namespace:

```text
("orgs", org_id, "users", user_id, "preferences")
```

Risky namespace:

```text
("memories",)
```

The risky version can mix unrelated users and memory types.

## Keep Values Structured

Prefer structured JSON-like memory documents.

Example:

```text
{
  "type": "preference",
  "subject": "answer_style",
  "value": "concise",
  "source": "explicit_user_request"
}
```

Structured memory is easier to filter, update, audit, and delete.

---

## 20. Common Pitfalls

## Pitfall 1: Saving Everything

More memory is not always better. Too much memory creates noise and retrieval problems.

## Pitfall 2: Mixing Users in One Namespace

Always include user or tenant identity in the namespace when memory is user-specific.

## Pitfall 3: Treating Long-Term Memory as Chat History

Long-term memory should usually store distilled facts or useful summaries, not every raw message.

## Pitfall 4: No Update Path

Users change preferences. Projects change. Memory needs updates and deletion.

## Pitfall 5: No Permission Model

If memory crosses threads, it needs access control.

## Pitfall 6: Blindly Trusting Retrieved Memory

Retrieved memory is context, not guaranteed truth. The model should treat it as prior information and resolve conflicts carefully.

## Pitfall 7: Confusing Semantic Memory with Semantic Search

Semantic memory means facts and concepts stored by the agent.

Semantic search means retrieval by meaning using embeddings.

They are related but not the same.

---

## 21. Practical Mental Model

Think of long-term memory like a notebook the agent keeps outside any single conversation.

```text
thread_id = current conversation
checkpointer = saves conversation state
store = cross-conversation notebook
namespace = notebook section
key = page title
value = page content
```

The agent should read only the relevant notebook pages and write only information worth keeping.

---

## 22. Summary

Long-term memory in LangGraph is:

- Cross-thread
- Stored in a store
- Organized by namespace and key
- Used for facts, preferences, examples, and learned instructions
- Retrieved intentionally with exact lookup, filters, or semantic search
- Updated through hot path or background workflows
- Best combined with short-term memory

Core pattern:

```text
Store + namespace + key + value = long-term memory
```

Use it when the graph needs to remember durable information across conversations.

---

## References

- LangGraph memory guide: https://docs.langchain.com/oss/python/langgraph/add-memory
- LangGraph persistence guide: https://docs.langchain.com/oss/python/langgraph/persistence
- LangChain long-term memory guide: https://docs.langchain.com/oss/python/langchain/long-term-memory
- Memory conceptual overview: https://docs.langchain.com/oss/python/concepts/memory
