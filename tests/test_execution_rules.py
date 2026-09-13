from types import SimpleNamespace

from app.trading.execution_rules import evaluate_bar_exit


def test_long_stop_wins_ambiguous_ohlc_bar():
    position = SimpleNamespace(side='LONG', stop=95, target=105)
    result = evaluate_bar_exit(position, {'low': 94, 'high': 106})
    assert result == {'reason': 'stop_loss', 'price': 95.0, 'ambiguous': True}


def test_short_target_and_no_exit():
    position = SimpleNamespace(side='SHORT', stop=105, target=95)
    assert evaluate_bar_exit(position, {'low': 96, 'high': 104}) is None
    assert evaluate_bar_exit(position, {'low': 94, 'high': 104})['reason'] == 'take_profit'
