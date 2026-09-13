"""Load a promoted model together with the calibrator the promotion gate checked."""
from .calibration import calibration_path, load_calibrator

# Why a production model could not be loaded, split by what it means for the caller.
#
# "There is no production model" and "there is one and it is not usable" arrive here as
# the same exception type and used to leave as the same status, so the loader treated an
# expired or corrupt production model exactly like an empty registry -- and fell through
# to the candidate weights. Those are the weights the promotion gate refused. Degrading
# from an approved-but-stale model to a rejected one is a loss of provenance wearing the
# costume of availability, and the operator sees only that the service is trading.
ABSENT_REASONS = ("no_production_model",)
UNUSABLE_REASONS = ("production_model_expired", "production_model_missing",
                    "calibrator_missing", "calibrator_artifact_missing",
                    "calibrator_invalid")


def classify_reason(reason):
    """Whether a load failure means "nothing promoted" or "something promoted, unusable"."""
    text = str(reason or "")
    for name in UNUSABLE_REASONS:
        if name in text:
            return "unusable"
    for name in ABSENT_REASONS:
        if name in text:
            return "absent"
    # An unrecognised failure is treated as unusable rather than absent: the safe reading
    # of "the production load did not work for a reason I do not recognise" is not to
    # silently substitute weights that were never approved.
    return "unusable"



def load_calibrated_model(registry):
    try:
        model, manifest = registry.load_production()
    except (ValueError, FileNotFoundError) as exc:
        reason = str(exc)
        return {"status": "fallback", "reason": reason, "kind": classify_reason(reason),
                "model": None, "calibrator": None, "manifest": None}
    version = manifest["version"]
    if not calibration_path(registry.root, version).exists():
        return {"status": "fallback", "reason": "calibrator_missing", "kind": "unusable",
                "model": None, "calibrator": None, "manifest": manifest}
    # The same validator the promotion gate used, from the same module. Two implementations
    # of "is this calibrator usable" is how a model gets promoted against one and refused
    # by the other, and the failure is a silent fallback at load time.
    calibrator = load_calibrator(registry.root, version)
    if calibrator is None:
        return {"status": "fallback", "reason": "calibrator_invalid", "kind": "unusable",
                "model": None, "calibrator": None, "manifest": manifest}
    return {"status": "active", "reason": None, "kind": "active", "model": model,
            "calibrator": calibrator, "manifest": manifest}
