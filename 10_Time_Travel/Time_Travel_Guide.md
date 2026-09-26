# LangGraph Time Travel Guide
### Inspecting, Replaying, and Forking Past Checkpoints

---

## Part 1: Why This Needs a Checkpointer

`09_Persistence/Persistence_Guide.md` already introduces the idea of time
travel: a checkpointer saves the state after every step, so you can look back
at any earlier point in a run. This notebook is the hands-on version of that
idea — it does not repeat the conceptual case for persistence at length, only
what is new here: the actual functions you call to look backward and to
change history.

Without a checkpointer, a graph forgets everything the moment `invoke` returns.
With one attached:

```python
story_app = story_graph.compile(checkpointer=InMemorySaver())
```

every step of every run is saved under a `thread_id`, and stays inspectable
after the run finishes.

---

## Part 2: What This Notebook Builds

```text
10_Time_Travel/
|-- 1_time_travel.ipynb
|-- Time_Travel_Guide.md
```

A two-step story pipeline, `topic -> draft -> final`, using `InMemorySaver` as
the checkpointer. The notebook runs it once, inspects its checkpoint history,
replays it unchanged from an old checkpoint, and forks it by rewriting the
topic at an old checkpoint and continuing forward from there.

```text
START -> draft_step -> polish_step -> END
```

---

## Part 3: Running the Pipeline

```python
config = {"configurable": {"thread_id": "story-1"}}

original = story_app.invoke({"topic": "a lost umbrella", "draft": "", "final": ""}, config)
```

`thread_id` is the same idea used in `09_Persistence` and `08_Basic_Chatbot`:
it names which conversation's checkpoints these are. Time travel always
operates within one `thread_id` — you inspect and rewind a specific thread's
history, not the whole checkpointer.

---

## Part 4: `get_state` — Where a Thread Currently Stands

```python
current = story_app.get_state(config)
print(current.values)
print(current.next)
```

`get_state(config)` returns a single snapshot: the most recent checkpoint for
that thread. Two fields matter most:

| Field | Meaning |
|---|---|
| `.values` | The state dict as of this checkpoint |
| `.next` | The node(s) that would run next from here — empty once the graph has finished |

After a completed run, `current.next` is `()` — there is nothing left to run.

---

## Part 5: `get_state_history` — Every Checkpoint, Newest First

```python
history = list(story_app.get_state_history(config))

for snapshot in history:
    print(snapshot.next, "->", snapshot.values)
```

`get_state_history(config)` yields one snapshot per checkpoint that was ever
saved for this thread, **starting with the most recent**. For the two-node
story graph, a completed run produces four snapshots:

```text
next=()                 -> {topic, draft, final}          (finished)
next=('polish_step',)   -> {topic, draft, final: ''}      (right after draft_step)
next=('draft_step',)    -> {topic, draft: '', final: ''}  (right after input, before anything ran)
next=('__start__',)     -> {}                             (the graph's internal starting point)
```

Each snapshot also has a `.config` — this is the important part for time
travel. `.config` pins down that *exact* checkpoint (it carries a
`checkpoint_id` alongside the `thread_id`), and it is what you pass back into
`invoke` or `update_state` to act on that specific point in the past, instead
of the thread's current, most recent state.

---

## Part 6: Replaying an Old Checkpoint

```python
before_polish = next(s for s in history if s.next == ("polish_step",))

replayed = story_app.invoke(None, before_polish.config)
```

`before_polish` is the checkpoint saved right after `draft_step` finished,
before `polish_step` ran. Two things matter about this call:

1. The input is `None`, not a new state dict. `None` means "don't add new
   input, just continue."
2. The config is `before_polish.config`, not the thread's current config. This
   tells LangGraph to resume from *that* checkpoint's position, not from
   wherever the thread currently stands.

The result: `polish_step` runs again, using the exact `draft` that was saved
at that checkpoint. `draft_step` does not run again — it already ran, and this
checkpoint already reflects that.

---

## Part 7: Forking — Rewrite State, Then Continue Differently

Replaying reruns history unchanged. Forking rewrites a value at an old
checkpoint first, then continues forward from the changed state:

```python
before_draft = next(s for s in history if s.next == ("draft_step",))

forked_config = story_app.update_state(before_draft.config, {"topic": "a haunted toaster"})
forked = story_app.invoke(None, forked_config)
```

`before_draft` is the checkpoint saved right after the input was received, but
before `draft_step` ran — the very start of the pipeline for this thread.

`update_state(config, updates)` merges `updates` into that checkpoint's values
and saves the result as a **new** checkpoint, returning a new config that
points at it. It does not touch the original checkpoint or overwrite history —
the original run's checkpoints are still there in `history`, unchanged.

`invoke(None, forked_config)` then continues forward from that new checkpoint.
Because `before_draft.next` was `('draft_step',)`, execution resumes there:
`draft_step` runs again, this time reading `topic = "a haunted toaster"`,
followed by `polish_step`. The result is a story about a completely different
topic than `original`, produced without re-running `invoke` on the original
input from scratch — the fork branches off an existing checkpoint instead.

```text
original run:  topic="a lost umbrella"     -> draft_step -> polish_step -> final A
                                                  |
                                    (checkpoint saved here, before draft_step)
                                                  |
forked run:    topic="a haunted toaster"   -> draft_step -> polish_step -> final B
```

`final A` and `final B` are different stories, even though both runs share the
same graph and the same thread — they diverge at the point where the fork
rewrote `topic`.

---

## Part 8: `as_node` — What It Actually Changes

`update_state` takes an optional `as_node` argument:

```python
story_app.update_state(config, updates, as_node="draft_step")
```

`as_node` tells LangGraph "treat this update as if it were produced by this
node," which changes what `.next` becomes on the resulting checkpoint —
normally, whatever would follow that node's outgoing edges. This matters
because it can skip a node entirely: if you call
`update_state(before_draft.config, {"topic": "..."}, as_node="draft_step")`
without actually supplying the `draft` value that `draft_step` would have
produced, the graph's `.next` advances straight to `polish_step`, and
`polish_step` runs on whatever `draft` already holds. In this notebook that is the
empty string from the input, so it quietly polishes an empty draft; no error
tells you the step was skipped.

The safe default is to leave `as_node` out, as this notebook does. Reach for
it only when you are directly supplying the output a specific node would have
produced (for example, injecting a human-approved value in place of what a
node normally computes), not when you only want to change an upstream input
like `topic` and let the graph regenerate everything after it.

---

## Part 9: Common Beginner Confusions

### Confusion 1: Does replaying or forking affect the original run's history?

No. `history` (captured before either call) still reflects the original run.
Both replaying and forking add *new* checkpoints; neither call rewrites or
deletes the ones already saved.

### Confusion 2: Why does `get_state_history` list a `('__start__',)` entry
with an empty `.values`?

That is the checkpoint LangGraph saves before your input is even applied — the
graph's true starting point. The next entry after it (`.next ==
('draft_step',)`) is the one with your actual input already merged in, and is
the more useful "start of my run" checkpoint to fork from in practice.

### Confusion 3: What is the difference between `get_state` and
`get_state_history`?

`get_state(config)` returns exactly one snapshot: the current one for that
config. `get_state_history(config)` returns every snapshot ever saved for that
thread, newest first. Time travel always starts by getting the history, then
picking one snapshot's `.config` out of it.

### Confusion 4: Do I need to pass `None` as input when resuming?

Yes, when you want to continue rather than start over. Passing a real state
dict as input starts a fresh run using the given config as a starting
point; passing `None` tells LangGraph to just continue the run that
checkpoint was already partway through.

### Confusion 5: Can I fork from any checkpoint, or only from the very first
one?

Any checkpoint with a `.config` works. This notebook forks from
`before_draft` because changing `topic` before `draft_step` runs is the
clearest demonstration, but the same `update_state` + `invoke(None, ...)`
pattern works from `before_polish`, or from the final checkpoint, to change
different values at different points in the pipeline.

### Confusion 6: Does forking use a new `thread_id`?

No. Both the replay and the fork in this notebook use `story-1`, the same
`thread_id` as the original run. `update_state` and `invoke(None, config)`
identify the point in history through `checkpoint_id` inside `.config`, not
through the thread. Every checkpointer, including `InMemorySaver`, keeps every branch under that one
thread, so `get_state_history` on `story-1` after forking shows both the original
checkpoints and the forked ones. The only difference is that `InMemorySaver` loses
all of them when the Python process exits, while a database-backed checkpointer
keeps them.

---

## Part 10: Why This Matters Beyond Debugging

Time travel is not only for fixing mistakes after the fact. The same
mechanism supports a few patterns that come up once a graph does real work:

- **Human-in-the-loop corrections.** A node pauses (see `18_Human_in_the_loop`
  for the interrupt mechanism), a person reviews the state, and `update_state`
  applies their correction before the graph continues — this is exactly the
  fork pattern in Part 7, just triggered by a person instead of by a topic
  change written in a notebook cell.
- **Comparing alternatives from the same starting point.** Forking the same
  checkpoint more than once, with a different `update_state` call each time,
  lets you compare several continuations of the same run without re-running
  the steps that came before the fork point.
- **Debugging a failed run.** If a node several steps in raised an exception,
  `get_state_history` shows exactly what state each earlier step produced,
  so you can find where things went wrong without re-running the whole
  pipeline from the beginning.

None of this needs new API surface beyond what Parts 3–7 already cover — it is
the same `get_state_history`, `update_state`, and `invoke(None, config)` used
for different reasons.

---

## Summary

A checkpointer turns a run's history into something you can read back and
branch from, not just something that happened once.

```text
get_state(config)          -> the current checkpoint
get_state_history(config)  -> every checkpoint, newest first
invoke(None, snap.config)  -> replay forward from snap, unchanged
update_state(snap.config, updates) -> new checkpoint with updates merged in
invoke(None, new_config)   -> continue forward from the updated checkpoint (a fork)
```

| Piece | Role |
|---|---|
| `.values` | State at that checkpoint |
| `.next` | What would run next from that checkpoint |
| `.config` | Pins down that exact checkpoint; pass it back into `invoke`/`update_state` |
| `invoke(None, config)` | Continue a run from `config`, without new input |
| `update_state(config, updates)` | Write new values at `config`, return a new config for the result |
| `as_node` | Pretends the update came from a given node — changes `.next`; use only when supplying that node's actual output |

Replaying answers "what happens if I just let this checkpoint continue?"
Forking answers "what happens if I change something at this checkpoint and
then continue?" Both start from the same tool: a `.config` taken from
`get_state_history`.
