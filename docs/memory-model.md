# History, memory, Git and the knowledge graph

The lab keeps six kinds of record. They overlap in subject (they all talk about experiments) but answer
different questions, and they are **not interchangeable**. The database is the single source of truth.
Everything else either derives from it or is linked to it by id.

| Record | Question it answers | Where it lives | Mutable? |
|---|---|---|---|
| Experiment history | *What exactly ran, and what was measured?* | `experiments`, `experiment_metrics`, `run_events` | append-only, except lifecycle fields |
| Episodic memory | *What happened?* (one experience per experiment) | `memory_records` (category `episodic`) | no |
| Semantic memory | *What do we know?* (generalisable findings and hypotheses) | `findings` + `memory_records` (`semantic`) | only `verification_status` |
| Procedural memory | *How must we work?* (rules, protocols, limits) | `memory_records` (`procedural`, namespace `global`) | updated when settings change |
| Git DAG | *Which configuration came from which, and what changed?* | `workspace/experiments` (`refs/experiments/*`) | append-only |
| Knowledge graph | *How are these things connected, and what is the evidence?* | `graph_relationships` → NetworkX | append-only, rebuilt on demand |

## Experiment history

The full, typed record of each experiment includes:

- objective, hypothesis and expected effects;
- configuration, its hash and its diff from the parent;
- parent and comparison-baseline ids;
- status, attempts and error;
- the decision with its reason, primary and secondary deltas, and threshold;
- the hypothesis outcome;
- Git commit, seed, dataset and protocol versions, and timestamps.

Metrics are rows in `experiment_metrics`, separated by split (`validation` holds the 5-fold cross-validation
means used for decisions, `test` the final evaluation only). Every experiment records its `protocol_version`;
results from different protocols are never compared. The accepted baseline is the explicit pointer `baseline_pointers[namespace]`. It changes
only after a completed, valid, accepted evaluation.

History is **complete but not curated**. It is what an auditor reads.

## Episodic memory

One `experiment_outcome` record per experiment, failures included. It is a compact narrative ("Experiment X
(RF, n_estimators=100…) completed. Validation F1 0.9577 (−0.0073 vs. B). Decision: rejected — …") with
evidence pointing back to the experiment and its metrics. Agents use it to answer "have we tried this
configuration?" and "what happened when we made this move?". It is an *experience*, not a conclusion.

## Semantic memory

**Findings** are statements derived by the Evaluator from stored measurements:

- `primary_improvement`, `regression`, `no_primary_effect`;
- `secondary_improvement` (a rejected experiment that used fewer resources);
- `failure`, `hypothesis_outcome`, `baseline_established`.

Each finding gets a semantic memory record and is `verified`, because it is computed from measurements.

**Research suggestions** come from the Researcher. They are hypotheses, so they start as `unverified`. When
the follow-up experiment is evaluated, the suggestion becomes `verified` if every expected effect was met, or
`contradicted` otherwise. Agent claims are never treated as evidence until measured.

Findings report measured associations on one fixed split. They do not claim causation.

## Procedural memory

Rules every agent works under: the acceptance rule and threshold, the evaluation protocol, execution
constraints (no generated code, model whitelist, timeouts), budgets and retry limits, and the allowed search
space. Records are seeded from the current settings at startup and shared by all namespaces. Procedural
memory is *instructions*, not learned weights; nothing in the lab updates model weights.

## Git DAG

A versioning record of **configuration artifacts**. Each experiment's `config.json` is committed with its
derivation parent's commit as the Git parent, so the DAG mirrors "derived from". It also includes research
follow-ups that derive from a *rejected* experiment. `git diff parent child` shows the exact change. Git
knows nothing about metrics or decisions, and the latest commit is **not** the baseline.

## Knowledge graph

Typed, evidence-backed relationships across all of the above: who proposed and executed what, which
configuration and model an experiment used, which findings it produced and what supports them, which memories
influenced a proposal, which commit recorded it, and which baseline it was compared against. Relationships
are validated on insert (endpoints and evidence must exist) and the graph is rebuilt from the database.

The graph adds *navigation and multi-hop questions*; it never adds facts. For example:
"show the evidence path for this finding", or "which memories influenced this experiment?".

## How they work together: one iteration

1. The **Researcher** queries semantic memory for verified `secondary_improvement` findings from rejected
   experiments. It writes an **unverified** `research_suggestion` and sends it to the Planner as a message.
2. The **Planner** checks episodic memory (tested configurations, the measured effect of each move) and the
   suggestion. It proposes a configuration and cites the memory ids it used.
3. The **Executor** validates the proposal against **procedural** rules, commits `config.json` to the
   **Git DAG**, trains, and returns measured metrics.
4. The **Evaluator** applies the procedural acceptance rule and updates **experiment history**. It moves the
   baseline pointer only on acceptance.
5. The Manager stores an **episodic** outcome and **semantic** findings, updates the suggestion's verification
   status, and writes the **graph** relationships (`INFLUENCED`, `DERIVED_FROM`, `SUPPORTED_BY`, …), each with
   an evidence reference.

## Planner policies (deterministic)

- **Memory disabled:** choose uniformly, with a seeded RNG, among the bounded one-step neighbours of the
  accepted baseline. No history is read, so configurations can repeat.
- **Memory enabled:**
  1. Use an untested Researcher suggestion if there is one.
  2. Otherwise drop neighbours with a recorded outcome.
  3. Score each remaining move by the mean measured F1 delta of past experiments that made the same move
     (0 if unexplored).
  4. Take the best, with a seeded tie-break.
  5. If every neighbour is tested, extend one step from the best tested neighbour.
  6. Stop only when nothing new remains within two steps.

## Retrieval ranking

```
score = 0.25·text_match + 0.15·model_match + 0.25·config_similarity + 0.25·metric_relevance + 0.10·recency
```

Hard filters run first in SQL. Every result returns its component scores, the reasons, and evidence copied
from stored measurements. See `backend/app/memory/retrieval.py` for each component's exact definition.

## Autoresearch namespace

Autoresearch sessions write into the `autoresearch` namespace with protocol `autoresearch-cv5`. There, the
"configuration" is code: the hash of `train.py`. The Git DAG has one branch per session, holding kept commits;
discarded attempts are kept under `refs/autoresearch/attempts/*`. The baseline pointer always names the best
kept `train.py`, which the next session branches from. Episodic memories of every attempt (kept, discarded or
crashed) stop the proposers from re-running code that was already tried, and they give the LLM proposer the
history of what worked.

