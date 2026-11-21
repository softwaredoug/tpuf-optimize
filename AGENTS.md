# Turbopuffer experiments

This is a python repo that interacts with Turbopuffer to index
and then experiment with query parameters of Turbopuffer.

## Python practices

This is a python project using uv, flake8, pyright. We should install pre-commit hook and
fix any issues that arise from these tools during a commit.

Use python available via asdf in 3.15.2

Use the official turbopuffer python client library.
Use tpuf-helpers here: http://github.com/softwaredoug/tpuf-helpers (install via git, pin to a revision)
Use cheat-at-search to access datasets: https://github.com/softwaredoug/cheat-at-search/ (install via git, pin to a revision)

## Indexing

Create a CLI to index ESCI dataset into Turbopuffer.

- Use `from cheat_at_search.esci_data import corpus` to get a pandas dataframe of the ESCI corpus
- Index text fields for title, description, etc
- Create different variations of common text indexing strategies
- Pick one embedding from TPuf's built-in embeddings for indexing (ie voyager-4, etc)

## Querying

Create a CLI that will issue multiple queries, and package up results in a CSV file for analysis
by an external tool.


With multple queries in a file, using the keywords + locale that ESCI supports

queries.csv
```
query,locale
red shoe,en-US
blue shoe,en-US
...
```


```
uv run query --queries queries.csv --output results.csv
```

Where results.csv is a CSV file with the following format at a minimum

```
query_id,doc_id,rank,score
q1,doc123,1,0.9
q1,doc456,2,0.8
q2,doc789,1,0.95
...
```

Every query in the task must be represented by at least one row, and every
submitted query ID must belong to the task. For a query with results, ranks are
positive integers starting at 1, with no gaps or ties; duplicate documents for
the same query are invalid. A query with no results is represented by exactly
one sentinel row with `doc_id=-1`, `rank=-1`, and `score=-1`. It cannot have
other result rows. The sentinel is the only case where rank may be negative.

```
query_id,doc_id,rank,score
q1,doc123,1,0.9
q1,doc456,2,0.8
q2,doc789,1,0.95
q3,-1,-1,-1
...

```

### Modular structure

We want to have an outer set of code that handles batching indexing and querying to 
the tpuf namespace.

But the meat of what gets optimized is an inner TPufStrategy object it defines

- namespace schema
- how its queried

With a clear search strategy (both index + query) within TPufStrategy.

Generally, we want to create a single index that encompasess a large set of possible experiments
and then delegate sub-agents different ways to use those ingredients. That way sub-agents don't
have to re-index, and can instead quickly experiment with an existing set of ingredients.

An outer Indexer will batch documents for indexing. An an outer Searcher will batch documents
for search. These won't be the focus of relevance optimization.


## Optimizing relevance

We will split off `relevance-optimizer` subagents to work on optimizing relevance.
With a strategy of split management to avoid overfitting using the referee service.

We will focus only on the query module, using what's already been indexed.

For parallel optimizing subagents, create one unique OpenCode Git worktree per coding subagent with   
worktree_create. Pass the returned directory to that subagent. Its first action must be           
session_move to that directory; it must not inspect or modify repository files before the move    
completes. Have it report its branch and commit when finished.                                    

## Use Referee to generate splits and delegate them

A service, called referee, exists to securely generate splits between holdout and training sets.
This lets us delegate optimizations to `relevance-optimizer` subagents, giving them a 
a training split of queries, then keeping your own holdout to accept / reject the subagent's results.

Review the service here:
http://localhost:8080/docs

When you start work, and before delegation create a referee session for the esci dataset for
~1000 qureies with two splits - one for holdout, the other for training.

Keep one bearer token (identifying a split) for yourself. NEVER LEAK IT TO A SUBAGENT

Assign the other bearer token (identifying a different set of queries) to each subagent.
The subagents can share the same training split, but they must not have access to your holdout split.

Tell the subagent how to use the referee service to get its training split, and how to submit results for evaluation
to referee.

When a subagent appears to have a successful solution. You visit its worktree, and run your holdout data. 
Doing your best never to leak holdout queries. Once done, you signal to the subagent acceptance / rejection

Respond with ACCEPTED if the holdout was improved
Respond with REJECTED if the holdout was not improved

Store accepted solutions in a git branch for later review and analysis.

## Keep looping

Loop working with the subagent, for up to 5-10 iterations.

After the subagent submits a change, run on he holdout. Returning to the subagent ACCEPTED / REJECTED

When responding, you can tell the overall metric delta, but NEVER SPECIFIC QUERIES about the holdout. We need to keep good seperation between the holdout and training sets.

## Use the batch runner to generate CSVs to submit solutions

The batch query runner should generate CSVs conforming to referee's exected format of search results.
