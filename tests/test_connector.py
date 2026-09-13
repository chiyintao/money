from app.market.connector import ConnectorHealth


def test_connector_health_rejects_time_regression():
    health = ConnectorHealth('binance')
    assert health.observe(100)
    assert not health.observe(99)
    assert health.snapshot()['last_error'] == 'event_time_regression'


def test_connector_health_tracks_disconnects():
    health = ConnectorHealth('binance')
    health.observe(100)
    health.disconnected(ConnectionError('reset'))
    state = health.snapshot()
    assert state['connected'] is False and state['reconnects'] == 1
    assert 'reset' in state['last_error']
