from app.strategy.live_models import RealModelRuntime
from app.backtest.simulation_session import SimulationSession
from app.storage.storage import Store


def test_trained_selection_requires_verified_dataset(tmp_path):
    runtime = RealModelRuntime(tmp_path, chronos_enabled=False)
    market = {'all': [{'symbol': 'BTCUSDT', 'price': 100}, {'symbol': 'ALTUSDT', 'price': 1}]}
    runtime.feature_space = {'verified': False, 'symbols': ['BTCUSDT']}
    assert runtime.trained_market(market) == []
    runtime.feature_space['verified'] = True
    market['trained'] = runtime.trained_market(market)
    store = Store(str(tmp_path))
    session = SimulationSession(store)
    session.start(1000, 2, 'trained', 3)
    assert session.source == 'trained'
    assert session.choose(market) == ['BTCUSDT']
    assert SimulationSession.restore(store, session.snapshot()).source == 'trained'
    store.close()
    runtime.fast.shutdown()
    runtime.slow.shutdown()
