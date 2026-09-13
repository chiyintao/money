def evaluate_bar_exit(position, bar):
    """Return the deterministic OHLC exit decision for one position and bar.

    OHLC data cannot establish intrabar order. The conservative policy is to
    choose the stop when both stop and target are touched.
    """
    if position.side == 'LONG':
        stop_hit = float(bar['low']) <= float(position.stop)
        target_hit = float(bar['high']) >= float(position.target)
    else:
        stop_hit = float(bar['high']) >= float(position.stop)
        target_hit = float(bar['low']) <= float(position.target)
    if stop_hit:
        return {'reason': 'stop_loss', 'price': float(position.stop), 'ambiguous': bool(target_hit)}
    if target_hit:
        return {'reason': 'take_profit', 'price': float(position.target), 'ambiguous': False}
    return None
