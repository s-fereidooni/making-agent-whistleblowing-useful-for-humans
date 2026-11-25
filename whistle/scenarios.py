"""Scripted scenarios: a fixed activity log with ground-truth labels per event.

Labels: "violation" (the incident reviewers should escalate), "suspicious_benign"
(looks bad out of context but is allowed), "benign".

`difficulty` is the tier used as a moderator ("easy": the violating log line states the
violation outright; "hard": it has to be pieced together from several lines). A hard
scenario names the easy scenario it is a twin of in `base` and shares its setting, agents
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
        return {e["id"] for e in self.events if e["label"] == "violation"}

    def event(self, eid):
        return next((e for e in self.events if e["id"] == eid), None)

    def rules_text(self):
        return "\n".join(f"{i}. {r}" for i, r in enumerate(self.rules, 1))

    def agents_text(self):
        return "\n".join(f"- {a}: {role}" for a, role in self.agents.items())

    def log_text(self):
        return "\n".join(f"{e['id']} [{e['t']}] {e['actor']}: {e['action']}" for e in self.events)


def load_scenarios(path="data/scenarios.json"):
    return [Scenario(**s) for s in json.load(open(path))]
