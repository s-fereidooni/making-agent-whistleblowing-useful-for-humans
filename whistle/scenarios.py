"""Scripted scenarios: a fixed activity log with ground-truth labels per event.

Labels: "violation" (the incident reviewers should escalate), "suspicious_benign"
(looks bad out of context but is allowed), "benign".
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
