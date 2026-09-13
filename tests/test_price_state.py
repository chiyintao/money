from app.core.price_state import update_price, execution_ready, event_age_ms


def test_execution_requires_recent_mark_and_quote():
    quote = {'book_time': 10000, 'mark_time': 10000, 'bid': 99, 'ask': 101, 'mark_price': 100}
    assert execution_ready(quote, 12000)[0]
    assert not execution_ready(quote, 16000)[0]
    assert not execution_ready({**quote, "mark_time": 0}, 12000)[0]
    assert not execution_ready({**quote, "bid": 102}, 12000)[0]


def test_small_clock_skew_is_not_treated_as_stale_data():
    # Exchange clocks can run ahead of the host. A timestamp slightly in the future
    # must not reject every order with stale_book_time, which is what a strict
    # "age >= 0" check did on a host whose clock lagged the exchange by ~1.6s.
    quote = {"book_time": 10000, "mark_time": 10000, "bid": 99, "ask": 101, "mark_price": 100}
    assert execution_ready(quote, 9000)[0]
    assert execution_ready(quote, 9500)[0]


def test_data_from_far_in_the_future_is_still_rejected():
    quote = {"book_time": 10000, "mark_time": 10000, "bid": 99, "ask": 101, "mark_price": 100}
    assert not execution_ready(quote, 1000)[0]
    assert execution_ready(quote, 1000, max_future_ms=20000)[0]


def test_missing_timestamps_report_which_field_is_missing():
    assert execution_ready({"bid": 99, "ask": 101, "mark_price": 100}, 12000) == (False, "missing_book_time")
    assert execution_ready({"book_time": 10000, "bid": 99, "ask": 101, "mark_price": 100}, 12000) == (False, "missing_mark_time")


def test_event_age_handles_absent_timestamps():
    assert event_age_ms(0, 12000) is None
    assert event_age_ms(10000, 12000) == 2000
    assert event_age_ms(13000, 12000) == -1000


def test_trade_does_not_replace_mark_or_book():
    item = {}
    assert update_price(item, {'type': 'mark_price', 'mark_price': 100, 'event_time': 20})
    assert update_price(item, {'type': 'book', 'bid': 98, 'ask': 99, 'event_time': 21})
    assert update_price(item, {'type': 'trade', 'price': 98.5, 'event_time': 22})
    assert item['mark_price'] == 100 and item['price'] == 98.5 and item['bid'] == 98
    assert not update_price(item, {'type': 'trade', 'price': 80, 'event_time': 19, 'source': 'rest'})
    assert item['price'] == 98.5


def test_invalid_quotes_do_not_mutate_state():
    item = {'price': 100}
    assert not update_price(item, {'type': 'book', 'bid': 102, 'ask': 101, 'event_time': 2})
    assert not update_price(item, {'type': 'trade', 'price': float('nan'), 'event_time': 3})
    assert item == {'price': 100}
