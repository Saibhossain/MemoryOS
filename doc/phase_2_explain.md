# Phase 2 — Context Window Management (Coming Next)

**Status:** not yet implemented. This file is a placeholder so the link from
[phase_1_explain.md](./phase_1_explain.md) resolves — it will be filled in
once we build Phase 2 together.

## What Phase 2 will cover

Phase 1 gave every conversation *perfect, unlimited* memory — the
checkpointer stores every message forever. That's great for durability, but
it creates a real problem: your Qwen model's context window is finite, and
feeding it the entire history of a long conversation every turn will
eventually blow past that limit, slow inference down, and waste tokens.

Phase 2 is about the engineering trade-off between:
- **Checkpointer memory** (full fidelity, replayable, already built)
- **What you actually feed the model** on each turn (must be pruned)

Planned pieces:

1. **A summarization node** — once message count crosses a threshold,
   summarize older turns into a compact `summary` field in state, and only
   keep the last N raw messages verbatim.
2. **Trimming strategy** — using LangChain's message trimming utilities
   (`trim_messages`) as an alternative/complement to full summarization.
3. **Storing the summary in the same Postgres checkpoint** — so it persists
   and survives restarts just like everything else in Phase 1.
4. **Visualizing token savings** — extending `DB/monitor_app.py` to show
   before/after message counts per thread once summarization kicks in.

No embeddings or RAG here either — summarization is a text-in/text-out call
to the same local Qwen model, not a retrieval mechanism.

---

⬅️ Back to [phase_1_explain.md](./doc/phase_1_explain.md)