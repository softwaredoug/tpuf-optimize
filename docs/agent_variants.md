# Agent-backed retrieval benchmark

`agent-main` compares three retrievers behind the same OpenAI Agents policy:

| Variant | Retrieval behavior |
|---|---|
| `naive-bm25` | Locale-filtered weighted title/description BM25, title:description 2:1, top 10 |
| `voyage4-ann` | Turbopuffer native Voyage-4 ANN on `passage__embed__voyage_4`, top 10 |
| `best-typed-computed` | Top-100 Voyage-4 ANN and BM25 candidates, existing computed BM25 features, then the frozen `6508e87` typed+computed ranker |

The agent uses `gpt-6-luna`, the same prompt, locale restriction, three-tool-call
budget, and `search_products` interface for all variants. Each tool response
contains up to 20 products with document ID, title, description (up to 1,200
characters), and available brand/color/category values. The agent is asked to
rank the results and can use descriptions to decide whether to reformulate. Its
final answer is constrained to a structured ordered list of up to ten result
objects, each with a `doc_id`. Array order determines rank. The evaluator
accepts only IDs actually returned by a tool call for that query, avoiding
free-text ID-label parsing.

A single agent makes tool calls serially, but the runner allows up to eight
independent query agents in flight (`--concurrency` can be set from 1 to 32).
The concurrency value is fixed in each checkpoint so resumed latency samples
are not silently mixed across loads.

## Split and execution

This evaluation reuses the existing ESCI Referee session with 1,000 training
queries and a separate 1,000-query holdout. It uses only the training token at
`esci-search-agent-training-eval-1000/referee.token`. `prepare` creates a
separate task for each variant and verifies that all three have the exact query
IDs, text, and locales of the assigned training task. It never reads or
accesses the holdout token or task.

```sh
export ALLOW_LIVE_SEARCH_AGENT_EVAL=1
uv run search-agent-variants prepare

# Run variants sequentially; within each run, independent agents are concurrent.
uv run search-agent-variants run --variant naive-bm25 --concurrency 8 --allow-live
uv run search-agent-variants run --variant voyage4-ann --concurrency 8 --allow-live
uv run search-agent-variants run --variant best-typed-computed --concurrency 8 --allow-live
uv run search-agent-variants compare --concurrency 8
```

Each completed query is checkpointed privately with only IDs, scores, timing,
tool-call counts, and failure/format counts. A failed query is not turned into a
no-hit result; Referee submission waits until the checkpoint covers the entire
task. The three completed rankings are submitted to Referee separately, each
against its matched training task. The comparison reports NDCG@10, paired
agent-variant deltas, end-to-end and backend p50/p95 latency, tool calls,
invalid IDs, missing IDs, and formatting failures. It is a training result,
not a holdout result.

`OPENAI_API_KEY` is used by the Agents SDK and `TURBOPUFFER_API_KEY` by the
retrieval tool. Neither key is accepted as an argument or written to outputs.
The query and retrieved product data are sent to OpenAI; traces are disabled.

Previous free-text parser results are not included in this protocol. Structured
`doc_id` result runs use fresh Referee tasks on the same training split; tasks
completed under the prior output format are not reused.
