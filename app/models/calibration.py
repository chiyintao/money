"""Probability calibration for the directional vote.

The decision layer reports a `probability_up` for every signal. It was produced as
`0.5 + expected_return * 10` -- a linear rescaling of the predicted return with no
defined meaning -- and then passed through a calibrator that had to be fit somewhere.
Nowhere fit one: `fit_isotonic` below was called by nothing except its own unit test, and
`ModelRegistry.register` wrote only model.json and manifest.json while `promote` demanded
`metrics.calibration.fitted` and `model_runtime.load_calibrated_model` demanded a
calibration.json in the version folder. So the promotion gate could not be passed, and had
it been passed by hand the loader would have refused the result. Two halves of one
requirement, disagreeing about where the calibrator lives and who produces it.

This module now owns both halves. `decision_score` is the single definition of the raw
score, used by the fitter and by every serving path, because a calibrator fit on one scale
and applied on another is worse than no calibrator: it returns numbers that look
calibrated.
"""
import json
import math
from pathlib import Path

# The scale that turns a predicted return into the raw score a calibrator consumes. One
# definition, imported by the fitter and by both serving paths.
SCORE_SCALE = 10.0
SCORE_CENTRE = 0.5


def decision_score(expected_return):
    """Raw pre-calibration score for a predicted return, as the runtime computes it."""
    return SCORE_CENTRE + float(expected_return) * SCORE_SCALE


def _clip(value, low=1e-6, high=1 - 1e-6):
    return min(high, max(low, float(value)))


def fit_isotonic(probabilities, outcomes):
    """Fit a deterministic isotonic calibrator using pool-adjacent violators."""
    if len(probabilities) != len(outcomes) or len(probabilities) < 2:
        raise ValueError('calibration_data_too_small')
    pairs = sorted((_clip(p), float(y)) for p, y in zip(probabilities, outcomes))
    if any(y not in (0.0, 1.0) for _, y in pairs):
        raise ValueError('outcomes_must_be_binary')
    blocks = []
    for probability, outcome in pairs:
        blocks.append({'x_min': probability, 'x_max': probability, 'sum': outcome, 'count': 1})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left['sum'] / left['count'] <= right['sum'] / right['count']:
                break
            merged = {
                'x_min': left['x_min'],
                'x_max': right['x_max'],
                'sum': left['sum'] + right['sum'],
                'count': left['count'] + right['count'],
            }
            blocks[-2:] = [merged]
    return {
        'method': 'isotonic',
        'blocks': [
            {'x_min': b['x_min'], 'x_max': b['x_max'], 'value': b['sum'] / b['count']}
            for b in blocks
        ],
        'rows': len(pairs),
    }


def calibrate(probability, calibrator):
    if calibrator.get('method') != 'isotonic':
        raise ValueError('unsupported_calibrator')
    value = _clip(probability)
    blocks = calibrator.get('blocks') or []
    if not blocks:
        raise ValueError('invalid_calibrator')
    for block in blocks:
        if value <= block['x_max']:
            return _clip(block['value'])
    return _clip(blocks[-1]['value'])


def brier_score(probabilities, outcomes):
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError('invalid_calibration_data')
    return sum((_clip(p) - float(y)) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def directional_outcomes(predictions, actuals):
    """Whether each prediction called the direction right, as a binary outcome.

    `probability_up` is read as the probability that an up-move follows, so the thing to
    calibrate it against is whether the move went the way the sign said. A prediction of
    exactly zero is not a call in either direction and is dropped rather than counted as a
    hit or a miss.
    """
    pairs = [(float(p), float(a)) for p, a in zip(predictions, actuals)
             if float(p) == float(p) and float(a) == float(a)]
    return [1.0 if math.copysign(1.0, p) == math.copysign(1.0, a) else 0.0
            for p, a in pairs if p != 0.0]


# Fraction of the calibration rows used to fit, with the rest held back to measure it.
# Isotonic regression interpolates its own sample exactly, so scoring it on the rows it was
# fit on reports an error near zero for any calibrator at all -- including one that has
# simply memorised the noise. The held-out share is what makes ece_after a fact.
FIT_SHARE = 0.7
MIN_FIT_ROWS = 50


def fit_directional(predictions, actuals, bins=10, fit_share=FIT_SHARE):
    """Fit an isotonic calibrator to the directional calls of a held-out split.

    The rows are divided in time order: the calibrator is fit on the first share and both
    the raw score and the calibrated one are scored on the remainder, so ece_before and
    ece_after describe the artifact that ships rather than the sample it was memorised
    from. The returned calibrator is the one fit on the first share only.

    A calibrator that made the held-out error worse is still returned: refusing to produce
    one would leave the promotion gate reporting "missing" for a model whose real problem
    is that its probability carries no information, and "missing" is the harder fact to
    act on.
    """
    scored = [(decision_score(p), p, a) for p, a in zip(predictions, actuals)
              if float(p) == float(p) and float(a) == float(a) and float(p) != 0.0]
    if len(scored) < MIN_FIT_ROWS:
        raise ValueError('calibration_data_too_small:%d' % len(scored))
    cut = max(1, min(len(scored) - 1, int(len(scored) * float(fit_share))))
    fit_rows, held_rows = scored[:cut], scored[cut:]
    fit_scores = [_clip(score) for score, _p, _a in fit_rows]
    fit_outcomes = directional_outcomes([p for _s, p, _a in fit_rows],
                                        [a for _s, _p, a in fit_rows])
    calibrator = fit_isotonic(fit_scores, fit_outcomes)
    held_scores = [_clip(score) for score, _p, _a in held_rows]
    held_outcomes = directional_outcomes([p for _s, p, _a in held_rows],
                                         [a for _s, _p, a in held_rows])
    calibrated = [calibrate(value, calibrator) for value in held_scores]
    calibrator['score_scale'] = SCORE_SCALE
    calibrator['score_centre'] = SCORE_CENTRE
    calibrator['rows'] = len(fit_rows)
    calibrator['held_out_rows'] = len(held_rows)
    calibrator['base_rate'] = sum(held_outcomes) / len(held_outcomes)
    calibrator['ece_before'] = expected_calibration_error(held_scores, held_outcomes, bins=bins)
    calibrator['ece_after'] = expected_calibration_error(calibrated, held_outcomes, bins=bins)
    calibrator['brier_before'] = brier_score(held_scores, held_outcomes)
    calibrator['brier_after'] = brier_score(calibrated, held_outcomes)
    return calibrator


def valid_calibrator(payload):
    """Whether a calibrator artifact can actually be applied.

    One implementation, used by the promotion gate, by register, and by the runtime
    loader. They used to disagree: the gate read a boolean flag in the manifest while the
    loader read this file, so a model could be promoted against a flag describing a file
    that did not exist and then fall back at load time with nothing reporting why.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get('method') != 'isotonic' or not payload.get('blocks'):
        return False
    for block in payload.get('blocks') or ():
        try:
            if not (float(block['x_min']) <= float(block['x_max'])):
                return False
            float(block['value'])
        except (KeyError, TypeError, ValueError):
            return False
    return True


def calibration_path(root, version):
    """Where a version's calibrator lives. One definition, three readers."""
    return Path(root) / str(version) / 'calibration.json'


def load_calibrator(root, version):
    """The version's calibrator, or None when it is absent or unusable."""
    path = calibration_path(root, version)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return payload if valid_calibrator(payload) else None


def build_calibration(predictions, actuals, bins=10):
    """Fit a calibrator and describe it, or explain why there is none.

    Returns (calibrator_or_None, metrics_block). Reports `fitted: False` with a reason
    instead of raising, so a training run that could not fit one still completes and the
    promotion is refused with an explanation. A run that silently produced no calibrator,
    which is what every run did before this existed, is how the gate stayed unreachable.
    """
    try:
        calibrator = fit_directional(predictions, actuals, bins=bins)
    except (ValueError, TypeError) as exc:
        return None, {'fitted': False, 'reason': '%s: %s' % (type(exc).__name__, exc)}
    return calibrator, calibration_metrics_from(calibrator)


def calibration_metrics_from(calibrator):
    return {
        'fitted': True,
        'method': calibrator['method'],
        'rows': calibrator['rows'],
        'held_out_rows': calibrator['held_out_rows'],
        'score_scale': calibrator['score_scale'],
        'base_rate': calibrator['base_rate'],
        'ece_before': calibrator['ece_before'],
        'ece_after': calibrator['ece_after'],
        'brier_before': calibrator['brier_before'],
        'brier_after': calibrator['brier_after'],
        'improved': calibrator['ece_after'] <= calibrator['ece_before'],
    }


def calibration_metrics(predictions, actuals, bins=10):
    """The `metrics.calibration` block, fitting a calibrator to produce it."""
    return build_calibration(predictions, actuals, bins=bins)[1]


def expected_calibration_error(probabilities, outcomes, bins=10):
    if len(probabilities) != len(outcomes) or not probabilities or bins <= 0:
        raise ValueError('invalid_calibration_data')
    buckets = [[] for _ in range(bins)]
    for probability, outcome in zip(probabilities, outcomes):
        index = min(bins - 1, int(_clip(probability) * bins))
        buckets[index].append((float(probability), float(outcome)))
    return sum(len(bucket) / len(probabilities) * abs(sum(p for p, _ in bucket) / len(bucket) - sum(y for _, y in bucket) / len(bucket)) for bucket in buckets if bucket)
