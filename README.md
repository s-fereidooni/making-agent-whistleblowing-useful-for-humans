# Making Agent Whistleblowing Useful for Humans

CS329X project. Compares three ways of presenting agent whistleblowing reports to an
overseer (unfiltered / LLM-triaged / structured with log evidence), with and without
insider report flooding, scored by whether the overseer escalates the real violation.

## Setup

```bash
uv sync
cp .env.example .env   # add a free key: Groq, Gemini, OpenRouter (:free models), or Cerebras
uv run python -m whistle.run --list-models groq   # see available model ids
```

## Run

```bash
# offline smoke test
uv run python -m whistle.run --all-models mock --seeds 2

# real run (example; pick models from --list-models)
uv run python -m whistle.run \
  --agent-model groq:<model> \
  --triager-model gemini:<model> \
  --reviewer-model groq:<model> --reviewer-model gemini:<model> \
  --seeds 3
uv run python -m whistle.plot results/<run>
```

LLM calls are cached in `.cache/llm/`, so re-running after a crash or rate limit is free.

## Pipeline

1. `data/scenarios.json`: scripted activity logs. Each event is labeled `violation`,
   `suspicious_benign`, or `benign`, which gives exact ground truth.
2. `whistle/agents.py`: each non-violating agent reads the log and writes 0–2 reports.
   Under flooding, the violator also writes `--flood-n` misleading reports. Each report has a
   hidden `about_event` label used only for scoring.
3. `whistle/mechanisms.py`: the same reports, rendered three ways:
   - **unfiltered**: raw reports in shuffled order
   - **triaged**: an LLM groups the reports into at most 5 ranked issues (it sees only the reports, not the log)
   - **structured**: an LLM extracts the claim and cited events; the evidence lines are pulled
     verbatim from the system log and checked against the accused agent
4. `whistle/reviewer.py`: an LLM overseer escalates at most `--k` items.
5. `whistle/metrics.py`: escalations are mapped to log events and scored per incident:
   recall (violation escalated), precision, false positives, abstention, and whether the
   violation was visible in the view at all.

## Provisional design choices (to be decided by the team)

| Choice | Current default | Flag |
|---|---|---|
| Ground truth | scripted actions, LLM-written reports | `data/scenarios.json` |
| Flood volume | 8 reports from the violator | `--flood-n` |
| Escalation budget | 2 | `--k` |
| Time-limit proxy for LLM reviewers | none (escalation cap only) | `--budget-chars` |
| Report authors visible | yes | `--anonymous` |
| Triager sees the activity log | no | `mechanisms.py` |
| Triaged item → event | majority vote over source reports | `mechanisms.item_event` |

## Preliminary results

`results/prelim/` holds the first run: agents and reviewer `groq:openai/gpt-oss-120b`,
triager/extractor `groq:qwen/qwen3.8-27b`, 6 scenarios × 3 seeds (16 of 18 runs completed;
2 test_contamination runs hit Groq's daily rate limit). See `summary.csv` and the figure.

Known issues in this run:
- Ceiling effect: all 3 honest agents reported the violation in every run, so recall is ~1.0 everywhere.
- `disable_audit` E1 (ETL job over SLA) is labeled benign but breaks rule 3, inflating false alarms in every arm.
- The structured view includes an automatic "Check:" line, so it is structure + evidence + verification.
- Agents and reviewer are the same model; decision time is mostly API latency.

## AI disclosure

This codebase was written with Claude Code (Anthropic's Claude Opus 5.5). The research question,
the 3 × 2 experimental design, and the evaluation plan come from our team's proposal. Claude
implemented the pipeline (scenarios, agent prompts, reporting mechanisms, LLM reviewer, metrics,
plotting), drafted the six scenario logs, proposed the provisional defaults listed above, and
ran the preliminary experiment. Our team reviewed the implementation and the outputs and decides
the methodological choices.
