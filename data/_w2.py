import io
p='app/main.py'
s=io.open(p,encoding='utf-8').read()
old = '''    decisions = ModelDecision(models, fee_rate=broker.fee_rate, slippage_bps=broker.slippage_bps,
                              min_edge_bps=settings.model_min_edge_bps,''')
new = '''    decisions = ModelDecision(models, fee_rate=broker.fee_rate, slippage_bps=broker.slippage_bps,
                              maker_fee_rate=settings.maker_fee_rate,
                              entry_order_type=settings.entry_order_type,
                              min_edge_bps=settings.model_min_edge_bps,''')
assert old in s, 'ModelDecision call not found'
s = s.replace(old, new, 1)
io.open(p,'w',encoding='utf-8').write(s)
print('main.py wired')