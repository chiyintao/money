"""Live feature drift monitoring.

The drift detector in app/drift.py compares two populations. This decides which two, how
often, and what happens when they differ.

The reference population is a sample of the training dataset -- the exact rows the loaded
models were fitted on. The current population is the feature snapshots the decision layer
has actually decided on. Comparing those two answers the question that matters: are the
inputs arriving now drawn from the same distribution as the inputs the model learned?
Comparing the last hour to the previous hour answers a different and much less useful
question, and would report drift every time the market changed session character.

Like the venue check, this reports rather than acts by default. Under "block" it records a
reason code the decision layer can refuse on, because a model being asked about inputs it
has never seen is not a model whose output means anything.
"""
import json
import random
import time
from pathlib import Path

from .drift import DEFAULT_BINS, drift_report
from ..features.feature_spec import FEATURES


def sample_dataset(path, features=FEATURES, limit=2000, seed=17):
    """A reproducible random sample of a jsonl dataset, as column lists.

    Random rather than the first N rows: a dataset is written in chronological order, so
    the first two thousand rows are one symbol on one day in one regime and would make the
    reference population narrower than the training data actually was.
    """
    path = Path(path)
    if not path.is_file():
        return None
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                seen += 1
                if len(reservoir) < limit:
                    reservoir.append(row)
                else:
                    index = rng.randrange(seen)
                    if index < limit:
                        reservoir[index] = row
    except OSError:
        return None
    if not reservoir:
        return None
    columns = {}
    for name in features:
        columns[name] = [row[name] for row in reservoir
                         if isinstance(row.get(name), (int, float))]
    return {"columns": columns, "rows": seen, "sampled": len(reservoir)}


class DriftMonitor:
    """Collects live feature snapshots and compares them against the training sample."""

    def __init__(self, reference=None, window=2000, threshold=0.2, bins=DEFAULT_BINS,
                 interval_seconds=3600.0, features=FEATURES, policy="warn"):
        self.reference = reference or {}
        self.window = int(window)
        self.threshold = float(threshold)
        self.bins = int(bins)
        self.interval_seconds = float(interval_seconds)
        self.features = tuple(features)
        self.policy = str(policy or "warn").lower()
        self.observations = []
        self.last_run_ms = 0
        self.last_report = None
        self.stats = {"observations": 0, "runs": 0, "drifted": 0, "skipped": 0}

    @classmethod
    def from_dataset(cls, path, **kwargs):
        sample = sample_dataset(path, features=kwargs.get("features") or FEATURES)
        if sample is None:
            return cls(reference={}, **kwargs)
        return cls(reference=sample["columns"], **kwargs)

    def observe(self, features):
        """Record one live feature snapshot."""
        if not features:
            return
        row = {}
        for name in self.features:
            value = features.get(name)
            if isinstance(value, (int, float)):
                row[name] = float(value)
        if not row:
            return
        self.observations.append(row)
        self.stats["observations"] += 1
        # Bounded: the comparison only ever uses the tail, so keeping more would grow
        # without changing an answer.
        if len(self.observations) > self.window:
            del self.observations[:len(self.observations) - self.window]

    def due(self, now_ms=None):
        now = float(now_ms if now_ms is not None else time.time() * 1000)
        return (now - self.last_run_ms) >= self.interval_seconds * 1000

    def check(self, now_ms=None):
        """Compare the collected window against the reference and record the result."""
        self.last_run_ms = float(now_ms if now_ms is not None else time.time() * 1000)
        if not self.reference:
            self.stats["skipped"] += 1
            self.last_report = {"status": "unavailable", "reason": "no_reference_sample"}
            return self.last_report
        if len(self.observations) < max(self.bins, 50):
            self.stats["skipped"] += 1
            self.last_report = {"status": "insufficient", "reason": "not_enough_live_rows",
                                "observations": len(self.observations)}
            return self.last_report
        current = {}
        for name in self.features:
            current[name] = [row[name] for row in self.observations if name in row]
        report = drift_report(self.reference, current, threshold=self.threshold,
                              bins=self.bins, names=self.features)
        report["observations"] = len(self.observations)
        report["policy"] = self.policy
        report["checked_at"] = int(self.last_run_ms)
        self.stats["runs"] += 1
        if report["drifted"]:
            self.stats["drifted"] += 1
        self.last_report = report
        return report

    def blocks_entries(self):
        """Whether the current drift verdict should stop new entries."""
        if self.policy != "block":
            return False
        report = self.last_report or {}
        return bool(report.get("drifted"))

    def health(self):
        report = self.last_report or {}
        return {**self.stats, "window": self.window, "threshold": self.threshold,
                "policy": self.policy, "status": report.get("status"),
                "drifted": report.get("drifted") or [],
                "checked_at": report.get("checked_at"),
                "due": self.due(),
                "reference_features": len(self.reference)}
