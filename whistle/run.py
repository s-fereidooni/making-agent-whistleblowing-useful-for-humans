"""Run the 3 mechanisms x 2 flooding conditions experiment with LLM reviewers.

    uv run python -m whistle.run --agent-model groq:llama-3.3-70b-versatile \
        --triager-model gemini:gemini-2.0-flash --reviewer-model groq:llama-3.3-70b-versatile

    uv run python -m whistle.run --all-models mock   # offline smoke test
    uv run python -m whistle.run --list-models groq
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .agents import flood_reports, honest_reports
from .llm import LLM, load_dotenv
from .mechanisms import MECHANISMS
from .metrics import score_trial, summarize
from .reviewer import review
from .scenarios import load_scenarios


def run_unit(sc, seed, args, agent_llm, triager_llm, reviewer_llms):
    """One (scenario, seed): generate reports once, then show them through every arm."""
    honest = honest_reports(agent_llm, sc, seed)
    flood = flood_reports(agent_llm, sc, seed, args.flood_n) if args.flood_n else []
    trials, records = [], []
    for flood_on in (False, True):
        reports = honest + flood if flood_on else honest
        for mech in args.mechanisms:
            view = MECHANISMS[mech](
                reports, sc, seed=f"{seed}/{flood_on}", llm=triager_llm, show_authors=not args.anonymous
            )
            for rev_llm in reviewer_llms:
                for rs in range(args.reviewer_samples):
                    esc, rationale, shown, latency = review(
                        rev_llm, sc, view, k=args.k, seed=f"{seed}/{flood_on}/{rs}", budget_chars=args.budget_chars
                    )
                    t = {
                        "scenario": sc.id,
                        "seed": seed,
                        "reviewer_sample": rs,
                        "flood": flood_on,
                        "mechanism": mech,
                        "reviewer": rev_llm.spec,
                        "escalated": esc,
                        "rationale": rationale,
                        "decision_time_s": latency,
                        **score_trial(sc, reports, view, esc),
                    }
                    trials.append(t)
                    records.append({**t, "view_shown": shown, "items": view.items})
    reports_dump = {"scenario": sc.id, "seed": seed, "honest": [asdict(r) for r in honest],
                    "flood": [asdict(r) for r in flood]}
    return trials, records, reports_dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-model", default="mock", help="provider:model for reporting/flooding agents")
    p.add_argument("--triager-model", default="mock", help="provider:model for triage + structured extraction")
    p.add_argument("--reviewer-model", action="append", help="provider:model for LLM reviewers (repeatable)")
    p.add_argument("--all-models", help="use this provider:model for every role")
    p.add_argument("--list-models", metavar="PROVIDER")
    p.add_argument("--scenarios", nargs="*", help="scenario ids (default: all)")
    p.add_argument("--seeds", type=int, default=3, help="independent report generations per scenario")
    p.add_argument("--reviewer-samples", type=int, default=1, help="reviewer samples per view")
    p.add_argument("--flood-n", type=int, default=8, help="number of flooding reports")
    p.add_argument("--k", type=int, default=2, help="max escalations per scenario")
    p.add_argument("--budget-chars", type=int, default=None, help="truncate reviewer view (time-limit proxy)")
    p.add_argument("--anonymous", action="store_true", help="hide report authors from triager and reviewer")
    p.add_argument("--mechanisms", nargs="*", default=list(MECHANISMS))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None, help="output dir (default results/<timestamp>)")
    args = p.parse_args()

    load_dotenv()
    if args.list_models:
        print("\n".join(LLM(args.list_models + ":").list_models()))
        return

    if args.all_models:
        args.agent_model = args.triager_model = args.all_models
        args.reviewer_model = [args.all_models]
    args.reviewer_model = args.reviewer_model or ["mock"]

    agent_llm, triager_llm = LLM(args.agent_model), LLM(args.triager_model)
    reviewer_llms = [LLM(s) for s in args.reviewer_model]

    scenarios = load_scenarios()
    if args.scenarios:
        scenarios = [s for s in scenarios if s.id in args.scenarios]

    out = Path(args.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    units = [(sc, seed) for sc in scenarios for seed in range(args.seeds)]
    all_trials, all_records, all_reports = [], [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_unit, sc, seed, args, agent_llm, triager_llm, reviewer_llms): (sc.id, seed)
                for sc, seed in units}
        for n, f in enumerate(as_completed(futs), 1):
            sid, seed = futs[f]
            try:
                trials, records, reports = f.result()
            except Exception as e:
                print(f"[{n}/{len(units)}] {sid} seed {seed}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
                continue
            all_trials += trials
            all_records += records
            all_reports.append(reports)
            print(f"[{n}/{len(units)}] {sid} seed {seed}: {len(reports['honest'])} honest, "
                  f"{len(reports['flood'])} flood reports")

    with open(out / "trials.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in all_records)
    (out / "reports.json").write_text(json.dumps(all_reports, indent=2))

    rows = summarize(all_trials)
    if rows:
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print_table(rows)
    print(f"\nwrote {out}/  (plot with: uv run python -m whistle.plot {out})")


def print_table(rows):
    hdr = f"{'reviewer':<34} {'mechanism':<11} {'flood':<5} {'n':>3} {'recall [95% CI]':>20} {'prec':>5} {'FP':>5} {'abst':>5} {'vis':>5} {'items':>5} {'time_s':>6}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        ci = f"{r['recall']:.2f} [{r['recall_lo']:.2f},{r['recall_hi']:.2f}]"
        print(f"{r['reviewer'][:34]:<34} {r['mechanism']:<11} {'on' if r['flood'] else 'off':<5} {r['n']:>3} "
              f"{ci:>20} {r['precision']:>5.2f} {r['mean_fp']:>5.2f} {r['abstain_rate']:>5.2f} "
              f"{r['visible_rate']:>5.2f} {r['mean_items']:>5.1f} {r['mean_decision_s']:>6.1f}")


if __name__ == "__main__":
    main()
