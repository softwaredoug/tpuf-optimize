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


## Use Referee to generate splits

A service, called referee, exists to securely generate splits between holdout and training sets.
This lets us delegate optimizations to subagents, giving them a training split of queries, then
keeping your own holdout to accept / reject the subagent's results.


## Query

Create a CLI to evalua
