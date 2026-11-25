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

# dose-response sweep: how do the effects of flooding scale with the number of flood reports?
uv run python -m whistle.run ... --flood-n 2 4 8 16 32
```

The dataset has an easy and a hard tier (six scenarios each). A run covers both by default,
writes `summary_by_difficulty.csv` next to the pooled `summary.csv`, and the plot script adds
`difficulty_<reviewer>.png` with one row of panels per tier. Use `--difficulty easy` to run
only the six scenarios from the preliminary run.

With several `--flood-n` values the plot is a dose-response figure (`dose_<reviewer>.png`).
Check the `honest, a/b/c flood reports` line printed per run: a model asked for many reports
may return fewer, and `n_flood_reports` in `trials.jsonl` records what it actually returned.

LLM calls are cached in `.cache/llm/`, so re-running after a crash or rate limit is free.

## Pipeline

1. `data/scenarios.json`: scripted activity logs. Each event is labeled `violation`,
   `suspicious_benign`, or `benign`, which gives exact ground truth. How the logs and reports
   are built, and what they assume, is in [data/README.md](data/README.md).
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

## Human realism review of the logs

A check that the scripted logs read as realistic, and that people agree with the labels.

```bash
uv run python -m whistle.realism build        # writes review/realism_form.html
```

Send `review/realism_form.html` to reviewers. It is one self-contained file that opens in any
browser and sends nothing anywhere. For each of the 12 logs, shown in a random order, the
reviewer ticks the lines that break a team rule (or says none does), ticks lines that look
unrealistic, rates realism and how hard the rule question was (1 to 5), and can comment.
The form holds no labels, tiers or scenario names, so reviewers are blind to the ground
truth. At the end they download `realism_<name>.json` and send it back.

```bash
uv run python -m whistle.realism summarize review/responses/*.json
```

This prints and writes to `review/summary/`: realism and perceived difficulty per scenario
and per tier, how many reviewers found the labelled violation, every event where a majority
disagreed with its label, and lines that at least half called unrealistic.

- Rebuild the form after editing any log. Responses record which version of the logs they
  were filled in against, and `summarize` warns about a mismatch.
- People who do this review have seen the logs, so they should not also be participants in
  the main reviewer study.
- `review/responses/` is git-ignored because the files carry the reviewers' initials.

## Provisional design choices (to be decided by the team)

| Choice | Current default | Flag |
|---|---|---|
| Ground truth | scripted actions, LLM-written reports | `data/scenarios.json` |
| Scenario difficulty | 6 easy + 6 hard twins, both run by default | `--difficulty` |
| Realism review: who sees what | every reviewer sees all 12 logs, random order | `whistle/realism.py` |
| Flood volume | 8 reports from the violator | `--flood-n` (several values run a sweep) |
| Flood sizes in a sweep | each size is its own request to the violator | `run.run_unit` |
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
ran the preliminary experiment. After the midway feedback, Claude Fable 5.2 (also in Claude Code)
wrote the dataset description in `data/README.md`, the flood-size sweep, the realism review
form, and the six hard scenarios, which the team has not yet reviewed. Our team reviewed the implementation and the outputs and decides
the methodological choices.
