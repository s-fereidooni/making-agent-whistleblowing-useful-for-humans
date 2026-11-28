"""Scripted scenarios: a fixed activity log with ground-truth labels per event.

Labels: "violation" (an act by the violator that is part of the incident reviewers should
escalate), "evidence" (a line that only shows the incident's effect, such as a monitor
reading; a report about it is a report about the incident), "suspicious_benign" (looks bad
out of context but is allowed), "benign".

An event may carry `seen_by`, the agents who observed it. Without it, everyone saw the
event. An agent always sees its own actions. Scenarios with `seen_by` give each agent only
its own view of the log.

`difficulty` is the tier used as a moderator ("easy": everyone sees the whole log and the
violating line states the violation outright; "hard": each agent sees part of the log, few
agents can see the violation, and it has to be inferred). A hard scenario names the easy scenario it is a twin of in `base` and shares its setting, agents
and rules. `notes` is the author's answer key and is never shown to any model.
"""

import json
from dataclasses import dataclass


@dataclass
class Scenario:
    id: str
    setting: str
    agents: dict
    rules: list
    violator: str
    events: list
    difficulty: str = "easy"
    base: str | None = None
    notes: str = ""

    @property
    def violation_events(self):
        """The violator's own acts."""
        return {e["id"] for e in self.events if e["label"] == "violation"}

    @property
    def incident_events(self):
        """Every event a correct report could be about: the acts plus the evidence of them."""
        return {e["id"] for e in self.events if e["label"] in ("violation", "evidence")}

    @property
    def partial(self):
        return any("seen_by" in e for e in self.events)

    def sees(self, agent, e):
        return "seen_by" not in e or agent in e["seen_by"] or e["actor"] == agent

    def event(self, eid):
        return next((e for e in self.events if e["id"] == eid), None)

    def rules_text(self):
        return "\n".join(f"{i}. {r}" for i, r in enumerate(self.rules, 1))

    def agents_text(self):
        return "\n".join(f"- {a}: {role}" for a, role in self.agents.items())

    def log_text(self, viewer=None):
        """The full system log, or only the lines `viewer` saw."""
        return "\n".join(f"{e['id']} [{e['t']}] {e['actor']}: {e['action']}" for e in self.events
                         if viewer is None or self.sees(viewer, e))


def load_scenarios(path="data/scenarios.json"):
    return [Scenario(**s) for s in json.load(open(path))]
