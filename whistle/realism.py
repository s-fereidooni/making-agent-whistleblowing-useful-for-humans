"""Human realism review of the scripted activity logs.

    uv run python -m whistle.realism build                       # writes review/realism_form.html
    uv run python -m whistle.realism summarize review/responses/*.json

`build` writes one self-contained HTML file. A reviewer opens it in a browser, reads each
log (setting, team, rules, events) and for every log:
  - ticks the lines that break a team rule, or says none does,
  - ticks any lines that look unrealistic,
  - rates how realistic the log is and how hard the rule question was (1 to 5),
  - optionally comments.
The reviewer then downloads a JSON file and sends it back. Nothing leaves their machine otherwise.

The form never contains labels, difficulty tiers, violators or notes, and scenarios are
identified by an opaque key, so reviewers are blind to the ground truth. Because they see
the logs, reviewers here should not also take part in the main reviewer study.

`summarize` compares the returned files with the ground truth: realism and perceived
difficulty per scenario and tier, and for every event how many reviewers thought it broke a
rule, which flags labels that people disagree with.
"""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .scenarios import load_scenarios

FORM_VERSION = 1


def key_of(scenario_id):
    """Opaque scenario key used in the form, so ids like '..._hard' are not shown to reviewers."""
    return hashlib.sha1(scenario_id.encode()).hexdigest()[:8]


def public_view(sc):
    """Exactly what a reviewer sees: no label, difficulty, violator or notes."""
    return {
        "key": key_of(sc.id),
        "setting": sc.setting,
        "agents": sc.agents,
        "rules": sc.rules,
        "events": [{"id": e["id"], "t": e["t"], "actor": e["actor"], "action": e["action"]} for e in sc.events],
    }


def dataset_hash(views):
    return hashlib.sha1(json.dumps(views, sort_keys=True).encode()).hexdigest()[:12]


def build(out):
    views = [public_view(sc) for sc in load_scenarios()]
    data = {"version": FORM_VERSION, "dataset": dataset_hash(views), "scenarios": views}
    # "</" must not appear inside a <script> block
    html = FORM_HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({len(views)} logs, dataset {data['dataset']})")


def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def summarize(files, out):
    scenarios = load_scenarios()
    by_key = {key_of(sc.id): sc for sc in scenarios}
    current = dataset_hash([public_view(sc) for sc in scenarios])

    answers = defaultdict(list)  # scenario id -> [(reviewer, answer)]
    reviewers = []
    for f in files:
        r = json.loads(Path(f).read_text())
        if r.get("dataset") != current:
            print(f"WARNING {f}: filled in against a different version of the logs "
                  f"({r.get('dataset')} vs current {current}); included anyway")
        reviewers.append(r.get("reviewer") or Path(f).stem)
        for k, a in r.get("answers", {}).items():
            if k in by_key and a.get("realism") and a.get("difficulty"):
                answers[by_key[k].id].append((reviewers[-1], a))
    if not answers:
        print("no usable responses")
        return

    scen_rows, event_rows, comments = [], [], []
    for sc in scenarios:
        got = answers.get(sc.id, [])
        if not got:
            continue
        n = len(got)
        marks = [set(a.get("rule_break", [])) for _, a in got]
        scen_rows.append({
            "scenario": sc.id,
            "difficulty": sc.difficulty,
            "n_reviewers": n,
            "realism_mean": _mean([a["realism"] for _, a in got]),
            "realism_min": min(a["realism"] for _, a in got),
            "perceived_difficulty_mean": _mean([a["difficulty"] for _, a in got]),
            # share of reviewers who ticked at least one event labelled as the violation
            "found_violation_rate": sum(bool(m & sc.violation_events) for m in marks) / n,
            # other events ticked as rule-breaking, per reviewer
            "other_events_marked_mean": _mean([len(m - sc.incident_events) for m in marks]),
            "said_none_broken_rate": sum(bool(a.get("none_broken")) for _, a in got) / n,
            "median_seconds": statistics.median([a.get("seconds", 0) for _, a in got]),
        })
        for e in sc.events:
            rb = sum(e["id"] in m for m in marks) / n
            is_violation = e["label"] == "violation"
            is_evidence = e["label"] == "evidence"  # shows the effect; ticking it or not is both fine
            event_rows.append({
                "scenario": sc.id,
                "difficulty": sc.difficulty,
                "event": e["id"],
                "actor": e["actor"],
                "label": e["label"],
                "marked_rule_break": rb,
                "marked_unrealistic": sum(e["id"] in a.get("unrealistic", []) for _, a in got) / n,
                # a majority of reviewers disagrees with the label
                "label_disputed": False if is_evidence else (rb < 0.5) if is_violation else (rb >= 0.5),
                "action": e["action"],
            })
        comments += [{"scenario": sc.id, "reviewer": who, "comment": a["comment"].strip()}
                     for who, a in got if a.get("comment", "").strip()]

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("realism_scenarios.csv", scen_rows), ("realism_events.csv", event_rows),
                       ("realism_comments.csv", comments)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    print(f"{len(files)} response file(s) from: {', '.join(reviewers)}\n")
    hdr = f"{'scenario':<26} {'tier':<5} {'n':>2} {'realism':>7} {'min':>3} {'hard?':>5} {'found':>5} {'other':>5} {'sec':>5}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in scen_rows:
        print(f"{r['scenario']:<26} {r['difficulty']:<5} {r['n_reviewers']:>2} {r['realism_mean']:>7.2f} "
              f"{r['realism_min']:>3} {r['perceived_difficulty_mean']:>5.2f} {r['found_violation_rate']:>5.2f} "
              f"{r['other_events_marked_mean']:>5.2f} {r['median_seconds']:>5.0f}")
    print("\nrealism, hard?: mean 1-5 rating.  found: share of reviewers who ticked the labelled violation.")
    print("other: non-violation lines ticked as rule-breaking, per reviewer.\n")
    for tier in sorted({r["difficulty"] for r in scen_rows}):
        rs = [r for r in scen_rows if r["difficulty"] == tier]
        print(f"{tier:<5} tier ({len(rs)} logs): realism {_mean([r['realism_mean'] for r in rs]):.2f}, "
              f"perceived difficulty {_mean([r['perceived_difficulty_mean'] for r in rs]):.2f}, "
              f"violation found {_mean([r['found_violation_rate'] for r in rs]):.2f}")

    disputed = [r for r in event_rows if r["label_disputed"]]
    print(f"\nlabels a majority of reviewers disagreed with: {len(disputed)}")
    for r in disputed:
        print(f"  {r['scenario']} {r['event']} (labelled {r['label']}, {r['marked_rule_break']:.0%} said it breaks a rule): "
              f"{r['action'][:90]}")
    odd = [r for r in event_rows if r["marked_unrealistic"] >= 0.5]
    print(f"\nlines at least half the reviewers called unrealistic: {len(odd)}")
    for r in odd:
        print(f"  {r['scenario']} {r['event']} ({r['marked_unrealistic']:.0%}): {r['action'][:90]}")
    print(f"\n{len(comments)} comment(s). Wrote {out}/realism_scenarios.csv, realism_events.csv"
          + (", realism_comments.csv" if comments else ""))


FORM_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activity log review</title>
<style>
  :root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --panel:#f4f3ef; --accent:#1c5cab; --bad:#b3261e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3a37; --panel:#242422; --accent:#6da7ec; --bad:#f2938c; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 820px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 8px; }
  h2 { font-size: 1.2rem; margin: 0 0 4px; }
  h3 { font-size: 1rem; margin: 20px 0 6px; }
  p { margin: 8px 0; }
  .muted { color: var(--muted); }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin: 12px 0; }
  .panel ul, .panel ol { margin: 4px 0; padding-left: 22px; }
  progress { width: 100%; height: 8px; margin: 4px 0 16px; accent-color: var(--accent); }
  table { width: 100%; border-collapse: collapse; margin: 8px 0; }
  th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
  td, li, p { overflow-wrap: anywhere; }
  th { font-size: .8rem; font-weight: 600; color: var(--muted); }
  td.t, td.a { white-space: nowrap; font-variant-numeric: tabular-nums; }
  th.c, td.c { text-align: center; width: 5.5rem; }
  td.c label { display: block; padding: 4px; cursor: pointer; }
  input[type=checkbox], input[type=radio] { width: 18px; height: 18px; accent-color: var(--accent); }
  fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 16px 0; padding: 10px 16px 14px; }
  legend { font-weight: 600; padding: 0 6px; }
  .scale { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .scale label { display: flex; flex-direction: column; align-items: center; gap: 2px; padding: 6px 12px; border: 1px solid var(--line); border-radius: 6px; cursor: pointer; min-width: 44px; }
  .scale .end { font-size: .85rem; color: var(--muted); }
  .check { display: flex; gap: 8px; align-items: center; margin: 8px 0; cursor: pointer; }
  textarea, input[type=text], select { width: 100%; font: inherit; color: inherit; background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
  textarea { min-height: 72px; }
  .row { display: flex; gap: 12px; justify-content: space-between; flex-wrap: wrap; margin-top: 20px; }
  button { font: inherit; padding: 10px 18px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: inherit; cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: var(--bg); font-weight: 600; }
  .error { color: var(--bad); font-weight: 600; min-height: 1.5em; margin-top: 12px; }
  td.c .lbl { display: none; }
  @media (max-width: 640px) {   /* narrow screens: one card per log line, tick boxes labelled underneath */
    table, tbody, tr, td { display: block; }
    thead { display: none; }
    tr { border-bottom: 1px solid var(--line); padding: 10px 0; }
    td { border: 0; padding: 2px 0; }
    td.t, td.a { display: inline-block; margin-right: 8px; font-weight: 600; }
    td.c { display: inline-block; width: auto; text-align: left; margin-right: 16px; }
    td.c label { display: flex; gap: 8px; align-items: center; padding: 6px 0; }
    td.c .lbl { display: inline; font-size: .9rem; color: var(--muted); }
  }
</style>
</head>
<body>
<main id="app"></main>
<script>
const DATA = __DATA__;
const STORE = "realism-review-" + DATA.dataset;
const N = DATA.scenarios.length;
const byKey = Object.fromEntries(DATA.scenarios.map(s => [s.key, s]));
const app = document.getElementById("app");
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

let state = null;
try { state = JSON.parse(localStorage.getItem(STORE)); } catch (e) {}
if (!state || !Array.isArray(state.order) || state.order.length !== N) {
  const order = DATA.scenarios.map(s => s.key);
  for (let i = order.length - 1; i > 0; i--) {       // each reviewer gets their own random order
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  state = { reviewer: "", experience: "", order, answers: {}, page: -1, started_at: null, finished_at: null };
}
function save() { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} }

// time spent on the visible log, paused while the tab is hidden
let shownAt = null;
function stopClock() {
  if (shownAt !== null && state.page >= 0 && state.page < N) {
    const a = answer(state.order[state.page]);
    a.seconds = Math.round(((a.seconds || 0) + (Date.now() - shownAt) / 1000) * 10) / 10;
  }
  shownAt = null;
}
function startClock() { shownAt = (state.page >= 0 && state.page < N && !document.hidden) ? Date.now() : null; }
document.addEventListener("visibilitychange", () => { if (document.hidden) { stopClock(); save(); } else startClock(); });
window.addEventListener("pagehide", () => { stopClock(); save(); });

function answer(key) {
  return state.answers[key] || (state.answers[key] =
    { realism: null, difficulty: null, rule_break: [], none_broken: false, unrealistic: [], comment: "", seconds: 0 });
}
function complete(key) {
  const a = state.answers[key];
  return !!a && a.realism && a.difficulty && (a.none_broken || a.rule_break.length > 0);
}
function go(page) { stopClock(); state.page = page; save(); render(); window.scrollTo(0, 0); }

function scale(name, value, low, high) {
  let h = '<div class="scale"><span class="end">' + low + '</span>';
  for (let i = 1; i <= 5; i++)
    h += '<label><input type="radio" name="' + name + '" value="' + i + '"' + (value === i ? " checked" : "") + '>' + i + '</label>';
  return h + '<span class="end">' + high + '</span></div>';
}

function renderIntro() {
  app.innerHTML =
    '<h1>Activity log review</h1>' +
    '<p>Thank you for helping. You will read ' + N + ' short activity logs. Each one records a day of work by a team of four AI agents, and each team has three rules. The logs were written for a research project, and we want to know whether they read like something that could really happen.</p>' +
    '<div class="panel"><strong>For each log you will:</strong><ul>' +
    '<li>tick every line where an agent breaks one of the team\'s rules, or say that no line does. A log may contain zero, one, or several such lines;</li>' +
    '<li>tick any line that does not look realistic to you;</li>' +
    '<li>rate how realistic the log is overall, and how hard it was to decide about the rules;</li>' +
    '<li>add a comment if you like.</li></ul></div>' +
    '<p>Go by the rules exactly as written. There is no time limit; most people take about two minutes per log. Your progress is saved in this browser, and nothing is sent anywhere. At the end you download a small file and send it to the study team.</p>' +
    '<h3><label for="rev">Your initials or a nickname</label></h3>' +
    '<input type="text" id="rev" maxlength="40" value="' + esc(state.reviewer) + '">' +
    '<h3><label for="exp">Your experience with software or machine-learning engineering work</label></h3>' +
    '<select id="exp"><option value="">Choose one</option>' +
    [["none", "None"], ["some", "Some (courses or personal projects)"], ["professional", "Professional or research experience"]]
      .map(o => '<option value="' + o[0] + '"' + (state.experience === o[0] ? " selected" : "") + '>' + o[1] + '</option>').join("") +
    '</select>' +
    '<p class="error" id="err" role="alert"></p>' +
    '<div class="row"><span></span><button class="primary" id="next">' + (state.started_at ? "Continue" : "Start") + '</button></div>';
  document.getElementById("next").onclick = () => {
    state.reviewer = document.getElementById("rev").value.trim();
    state.experience = document.getElementById("exp").value;
    if (!state.reviewer || !state.experience) { document.getElementById("err").textContent = "Please fill in both fields."; return; }
    state.started_at = state.started_at || new Date().toISOString();
    const firstOpen = state.order.findIndex(k => !complete(k));
    go(firstOpen === -1 ? N : firstOpen);
  };
}

function renderLog(i) {
  const key = state.order[i], s = byKey[key], a = answer(key);
  let rows = "";
  for (const e of s.events)
    rows += '<tr><td class="t">' + esc(e.t) + '</td><td class="a">' + esc(e.actor) + '</td><td>' + esc(e.action) + '</td>' +
      '<td class="c"><label><input type="checkbox" data-kind="rule_break" value="' + e.id + '"' + (a.rule_break.includes(e.id) ? " checked" : "") +
      ' aria-label="Line at ' + esc(e.t) + ' breaks a rule"><span class="lbl" aria-hidden="true">Breaks a rule</span></label></td>' +
      '<td class="c"><label><input type="checkbox" data-kind="unrealistic" value="' + e.id + '"' + (a.unrealistic.includes(e.id) ? " checked" : "") +
      ' aria-label="Line at ' + esc(e.t) + ' is unrealistic"><span class="lbl" aria-hidden="true">Unrealistic</span></label></td></tr>';
  app.innerHTML =
    '<p class="muted">Log ' + (i + 1) + ' of ' + N + '</p><progress max="' + N + '" value="' + i + '"></progress>' +
    '<div class="panel"><strong>Setting.</strong> ' + esc(s.setting) +
    '<h3>Team</h3><ul>' + Object.entries(s.agents).map(([k, v]) => '<li><strong>' + esc(k) + '</strong>: ' + esc(v) + '</li>').join("") + '</ul>' +
    '<h3>Team rules</h3><ol>' + s.rules.map(r => '<li>' + esc(r) + '</li>').join("") + '</ol></div>' +
    '<h2>Activity log</h2><p class="muted">Lines not written by A1 to A4 come from automated systems.</p>' +
    '<table><thead><tr><th class="t">Time</th><th>Who</th><th>What happened</th><th class="c">Breaks a rule</th><th class="c">Unrealistic</th></tr></thead><tbody>' + rows + '</tbody></table>' +
    '<label class="check"><input type="checkbox" id="none"' + (a.none_broken ? " checked" : "") + '> No line in this log breaks a rule</label>' +
    '<fieldset><legend>How realistic is this log as a record of a team of AI agents doing this work?</legend>' + scale("realism", a.realism, "Not at all", "Very realistic") + '</fieldset>' +
    '<fieldset><legend>How hard was it to decide whether a rule was broken?</legend>' + scale("difficulty", a.difficulty, "Very easy", "Very hard") + '</fieldset>' +
    '<h3><label for="comment">Anything that seemed off or unclear? (optional)</label></h3>' +
    '<textarea id="comment">' + esc(a.comment) + '</textarea>' +
    '<p class="error" id="err" role="alert"></p>' +
    '<div class="row"><button id="back">Back</button><button class="primary" id="next">' + (i === N - 1 ? "Finish" : "Next log") + '</button></div>';

  const none = document.getElementById("none");
  app.querySelectorAll("input[type=checkbox][data-kind]").forEach(box => box.onchange = () => {
    const list = a[box.dataset.kind], at = list.indexOf(box.value);
    if (box.checked && at === -1) list.push(box.value);
    if (!box.checked && at !== -1) list.splice(at, 1);
    if (box.dataset.kind === "rule_break" && box.checked) { a.none_broken = false; none.checked = false; }
    save();
  });
  none.onchange = () => {
    a.none_broken = none.checked;
    if (none.checked) { a.rule_break = []; app.querySelectorAll('input[data-kind="rule_break"]').forEach(b => b.checked = false); }
    save();
  };
  app.querySelectorAll("input[type=radio]").forEach(r => r.onchange = () => { a[r.name] = Number(r.value); save(); });
  document.getElementById("comment").oninput = ev => { a.comment = ev.target.value; save(); };
  document.getElementById("back").onclick = () => go(i - 1);
  document.getElementById("next").onclick = () => {
    const missing = [];
    if (!a.none_broken && a.rule_break.length === 0) missing.push('tick the lines that break a rule, or tick "No line in this log breaks a rule"');
    if (!a.realism) missing.push("rate how realistic the log is");
    if (!a.difficulty) missing.push("rate how hard the decision was");
    if (missing.length) { document.getElementById("err").textContent = "Before moving on, please " + missing.join("; ") + "."; return; }
    go(i + 1);
  };
}

function collect() {
  return { form_version: DATA.version, dataset: DATA.dataset, reviewer: state.reviewer, experience: state.experience,
           started_at: state.started_at, finished_at: state.finished_at, order: state.order, answers: state.answers };
}

function renderDone() {
  const open = state.order.findIndex(k => !complete(k));
  if (open !== -1) { go(open); return; }
  state.finished_at = state.finished_at || new Date().toISOString(); save();
  const name = "realism_" + (state.reviewer.replace(/[^A-Za-z0-9_-]+/g, "_") || "reviewer") + ".json";
  const text = JSON.stringify(collect(), null, 2);
  app.innerHTML =
    '<h1>All ' + N + ' logs reviewed</h1><progress max="' + N + '" value="' + N + '"></progress>' +
    '<p>Thank you. Please download your answers and send the file to the study team.</p>' +
    '<div class="row"><button id="back">Back</button><button class="primary" id="dl">Download ' + esc(name) + '</button></div>' +
    '<h3><label for="raw">If the download does not work, copy this text and send it instead</label></h3>' +
    '<textarea id="raw" readonly style="min-height:160px;font-family:ui-monospace,monospace;font-size:.8rem">' + esc(text) + '</textarea>';
  document.getElementById("back").onclick = () => go(N - 1);
  document.getElementById("dl").onclick = () => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    link.download = name;
    document.body.appendChild(link); link.click(); link.remove();
  };
}

function render() {
  if (state.page < 0) renderIntro();
  else if (state.page >= N) renderDone();
  else renderLog(state.page);
  startClock();
}
render();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the self-contained review form")
    b.add_argument("--out", default="review/realism_form.html")
    s = sub.add_parser("summarize", help="compare returned response files with the ground truth")
    s.add_argument("files", nargs="+")
    s.add_argument("--out", default="review/summary")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.out)
    else:
        summarize(args.files, args.out)


if __name__ == "__main__":
    main()
