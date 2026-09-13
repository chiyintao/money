"""Rebuild a dataset profile, and say whether its bounds can refuse anything.

The serving gate compares live features against the training bounds. Those bounds live in a
sidecar profile, and the profile is only as good as the code that last wrote it. The stored
one was written by a version that used the observed extremes of features which had already
been clipped, so for four of the ten features the bounds equalled the clip range and the
gate could not fire at all -- the exact case it exists to catch. The code that detects this
reports it rather than repairing it, and nothing rebuilt the profile, so the condition was
permanent.

Run this after retraining or whenever the model panel reports degenerate bounds. It is
idempotent: the sidecar is keyed on the file size and mtime, so a rebuild that produces the
same file is a no-op for the service.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.dataset_io import cached_profile, profile_cache_path, stream_profile, bounds_are_informative
from app.features.feature_spec import FEATURES  # noqa: E402


def rebuild(path, features=FEATURES, dry_run=False):
    path = Path(path)
    if not path.is_file():
        raise SystemExit("dataset not found: %s" % path)
    sidecar = profile_cache_path(path)
    before = {}
    if sidecar.is_file():
        try:
            before = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            before = {}
    before_quality = bounds_are_informative(before.get("bounds") or {}, features)
    print("before: method=%s rows=%s degenerate=%s"
          % (before.get("bounds_method"), before.get("rows"),
             before_quality["degenerate"] or "none"))
    if dry_run:
        return before_quality
    # stream_profile, not cached_profile: the cache would be served straight back, since a
    # stale profile is keyed on the same size and mtime as the file it describes.
    profile = stream_profile(path, features)
    stat = path.stat()
    payload = dict(profile, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                   features=list(features))
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    quality = bounds_are_informative(profile["bounds"], features)
    print("after:  method=%s rows=%s degenerate=%s"
          % (profile.get("bounds_method"), profile.get("rows"),
             quality["degenerate"] or "none"))
    print("sidecar written: %s" % sidecar)
    for name in features:
        print("  %-18s %s" % (name, profile["bounds"].get(name)))
    return quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?",
                        default="data/research_v3/training_dataset.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the stored bounds without re-reading the dataset")
    args = parser.parse_args(argv)
    quality = rebuild(args.path, dry_run=args.dry_run)
    return 0 if quality["informative"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
