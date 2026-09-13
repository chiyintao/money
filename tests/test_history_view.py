"""The replay must not copy a hundred thousand pointers to answer one lookup.

The entry loop asks for "the history up to this bar" once per (timestamp, symbol) pair.
Answering that with a list slice allocated a fresh list every time -- 1.26 million
allocations and roughly 650 GB of copying for one year across eleven symbols, which is
where the twelve minutes of the portfolio replay went. HistoryView answers the same
question in constant time, and these tests pin that it still behaves like the sequence
the strategies were written against.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backtest.backtest import HistoryView  # noqa: E402


def rows(count):
    return [{"i": i} for i in range(count)]


def test_it_reports_only_the_prefix_length():
    view = HistoryView(rows(100), 7)
    assert len(view) == 7


def test_the_last_element_is_the_current_bar():
    # This is the only access the out-of-sample strategy makes, and the whole point of the
    # change is that it costs nothing.
    view = HistoryView(rows(100), 7)
    assert view[-1] == {"i": 6}


def test_positive_and_negative_indices_agree():
    view = HistoryView(rows(100), 10)
    assert view[0] == view[-10]
    assert view[9] == view[-1]


def test_an_index_past_the_prefix_raises():
    view = HistoryView(rows(100), 5)
    for bad in (5, 99, -6, -100):
        try:
            view[bad]
        except IndexError:
            continue
        raise AssertionError("expected IndexError for %r" % (bad,))


def test_iteration_yields_exactly_the_prefix():
    view = HistoryView(rows(100), 4)
    assert [item["i"] for item in view] == [0, 1, 2, 3]


def test_slicing_the_view_matches_slicing_the_list():
    backing = rows(100)
    view = HistoryView(backing, 20)
    assert list(view[:3]) == backing[:3]
    assert list(view[5:8]) == backing[5:8]
    assert list(view[-2:]) == backing[18:20]


def test_it_does_not_copy_the_backing_list():
    backing = rows(1000)
    view = HistoryView(backing, 999)
    # The underlying list is shared, which is what makes construction O(1).
    assert view._rows is backing
