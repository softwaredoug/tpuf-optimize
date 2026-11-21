# Paired search-agent evaluation (training split only)

This protocol compares the existing `ProductSearch` direct/original-query path
(using the frozen computed-feature ranker from `e64100d`) with the exact
OpenAI Agents SDK policy at `77fc0d3`, using `gpt-5-mini`. It does not tune
prompts, models, retrieval, or ranking. The agent returns its existing
free-text response; a deterministic parser extracts only explicitly labeled
`Document ID:`, `doc_id:`, `document_id:`, or `Product ID:` values. An ID is
ranked only if it appeared in a tool response for that query. No LLM extraction
or fill-in from unseen backend candidates is permitted. Unsupported/missing ID
formats yield no valid recommendation (the Referee sentinel), and their counts
are reported. Thus the output format is the existing free-text policy plus a
strict evaluation adapter; the agent output/prompt is unchanged.

## Assigned task data and privacy

The orchestrator pre-created two matched Referee tasks from this subagent's
single assigned training split. Their private responses are at
`/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/opencode/esci-search-agent-training-eval/{direct,agent}.task.json`.
Both have 500 identical queries. The only permitted token file is the
`referee.token` in that same directory. The evaluator never creates tasks,
reads other tokens, or selects another split. Task files, checkpoints, CSVs,
and results remain outside the repository with restrictive permissions.

No live retrieval or OpenAI call has been made during implementation. Mocked
tests are safe to run at any time. Live execution requires both an explicit
flag and `ALLOW_LIVE_SEARCH_AGENT_EVAL=1`; do not use them until the
orchestrator explicitly says the serial benchmark has finished and grants
launch approval. Run the arms serially, never concurrently:

```sh
export ALLOW_LIVE_SEARCH_AGENT_EVAL=1
uv run search-agent-eval run --arm direct --allow-live
uv run search-agent-eval run --arm agent --allow-live
uv run search-agent-eval compare
```

Set `OPENAI_API_KEY` and `TURBOPUFFER_API_KEY` only in the execution
environment; neither is accepted as a CLI argument or written to output. The
live runner requires the two exact pre-created task files and the assigned
token path. It checkpoints each successful query privately, skips those
queries when resumed, and does not turn API/tool failures into no-hits. Any
failed query prevents Referee completion until it is successfully resumed.
Because Referee completion is single-use and has no recovery endpoint, the
runner writes a private completion-pending marker before the request and will
not replay an ambiguous completion request.

The CSVs use Referee's `query_id,doc_id,rank,score` schema, contiguous ranks,
unique document IDs, finite rank-derived scores, and exactly one `-1` sentinel
row for no valid recommendation. Referee scores use rank (not score).

## Paired analysis and call budget

After both arms complete, `compare` checks identical 500-query ID coverage,
calculates per-query agent-minus-direct NDCG@10 and a deterministic paired
bootstrap 95% CI, and reports end-to-end and backend-only p50/p95 separately.
The agent's residual (end-to-end minus measured backend) summarizes SDK,
model, and orchestration overhead. Counts include tool calls, invalid IDs,
missing-ID queries, and output-format failures. The aggregate summary is
private; only aggregate metrics should be committed after live measurement.

For 500 queries per arm, a direct run normally makes approximately 1,000
Turbopuffer HTTP operations: one multi-query candidate request and one
ID-filtered feature request per query. The agent makes two such operations per
tool call, up to three tool calls per query: at most about 3,000 Turbopuffer
operations. Across both arms, the expected total is about 4,000 such
operations at the agent's maximum tool budget; serial fallback and 429 retries
may add calls. Agent orchestration can make up to four model turns per query
(one initial turn plus up to three follow-up/tool turns), or at most about
2,000 OpenAI requests for the 500-query agent arm. The exact number of
successful tool calls and measured latencies are captured by the run.
