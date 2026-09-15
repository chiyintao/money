"""The deflated Sharpe needs the size of the search that produced it.

A candidate reported `deflated_sharpe` 0.999959 and `survives=True` while its own verdict,
in the same manifest, read "out-of-sample edge is negative (-10.33 bp): this would lose
money". Both statements came from the same run, and the reason they could coexist is that
`trials` was 1.

`expected_max_sharpe` returns 0.0 for a single trial -- correctly, because nothing was
selected -- so the benchmark being subtracted was zero and the "deflated" Sharpe was the raw
one. The parameter defaulted to 1 and no caller overrode it, so every candidate trained
through the ordinary path was undeflated.

The size of the effect is not marginal. Measured on the stored figures, adding the search
that actually happened moves that candidate from survives=True to survives=False:

    trials=1     DSR 0.999959   survives True
    trials=10    DSR 0.992772   survives True
    trials=100   DSR 0.938255   survives False

These tests pin the requirement, the arithmetic, and the fact that production states a
number instead of inheriting one.
"""
import pytest

from app.models import training_job
from app.models.tabular_model import train_tabular
from app.models.validation import deflated_sharpe, expected_max_sharpe
from helpers import row


def dataset():
    return [row(i, label_end_time=i + 1, future_return=(.005 if i % 2 else -.005))
            for i in range(150)]


def test_one_trial_means_no_correction_at_all():
    """The mechanism behind the bug, stated as arithmetic."""
    assert expected_max_sharpe(1, 100) == 0.0
    assert expected_max_sharpe(0, 100) == 0.0


def test_the_benchmark_grows_with_the_search():
    """A longer search looks better by luck, so the bar has to rise."""
    values = [expected_max_sharpe(n, 200) for n in (2, 10, 100, 1000)]
    assert values == sorted(values), values
    assert values[0] > 0.0


def test_a_real_search_size_can_flip_the_verdict():
    """The exact numbers from the stored candidate."""
    sharpe, observations = 0.45087617, 85
    assert deflated_sharpe(sharpe, 1, observations) >= 0.95
    assert deflated_sharpe(sharpe, 100, observations) < 0.95


def test_training_refuses_to_invent_a_search_size(tmp_path):
    with pytest.raises(ValueError, match="trials_required"):
        train_tabular(dataset(), "lightgbm", str(tmp_path), rounds=10)


def test_the_recorded_search_reaches_the_manifest(tmp_path):
    result = train_tabular(dataset(), "lightgbm", str(tmp_path), rounds=10, trials=100,
                           search_note="horizon x edge_multiple")
    assert result["search"]["trials"] == 100
    assert result["validation"]["trials"] == 100
    assert "horizon" in result["search"]["note"]


def test_production_states_a_search_size_rather_than_inheriting_one():
    """The job fits both backends and promotes one, so the selection is at least that."""
    assert training_job.BACKEND_TRIALS >= 2
    source = (__import__("pathlib").Path(training_job.__file__).read_text(encoding="utf-8"))
    assert "trials, search_note))" in source, (
        "the training job stopped passing an explicit trial count, which silently "
        "re-enables the undeflated Sharpe")