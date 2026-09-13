import io
p = 'app/trading/simulation.py'
s = io.open(p, encoding='utf-8').read()

# 1) 抽出 Lot 的字段名，供重建时过滤
anchor = '    @classmethod' + chr(10) + '    def restore(cls,data,default_cash=10000):'
assert anchor in s, 'restore anchor'

helper = chr(10).join([
    '@classmethod',
    '    def restore_position(cls, payload):',
    '        """Rebuild one position from a snapshot, lots included.',
    '',
    '        `asdict()` is recursive. It turned each Lot into a plain dict as well as the',
    '        Position around them, and `restore` only rebuilt the outer object -- so after',
    '        any restart `position.lots` was a list of dicts while every method on Position',
    '        reads `lot.qty`. `take_lots` raised AttributeError on the first read, which',
    '        means every exit a restored position could take was broken at once: the stop,',
    '        the target, the exit policy and the manual close button all end in close(),',
    '        and close() calls take_lots. The effect an operator saw was a close button',
    '        that answered HTTP 500 and a position that survived every attempt to flatten',
    '        it, with the real reason only in the server log.',
    '        """',
    '        data = dict(payload or {})',
    '        raw_lots = data.pop("lots", None) or []',
    '        position = Position(**{key: value for key, value in data.items()',
    '                              if key in _POSITION_FIELDS})',
    '        fields = {name for name in Lot.__dataclass_fields__}',
    '        position.lots = [',
    '            lot if isinstance(lot, Lot) else Lot(**{k: v for k, v in lot.items()',
    '                                                    if k in fields})',
    '            for lot in raw_lots]',
    '        return position',
    '',
    '    @classmethod',
    '    def restore(cls,data,default_cash=10000):',
])
s = s.replace(anchor, helper, 1)

# 2) 用它替换原来的一行重建
old = "account.positions={p['symbol']:Position(**p) for p in data.get('positions',[])}"
new = "account.positions={p['symbol']:cls.restore_position(p) for p in data.get('positions',[])}"
assert old in s, 'positions line'
s = s.replace(old, new, 1)

# 3) 在 Lot/Position 都定义之后加字段集合
marker = '@dataclass' + chr(10) + 'class PaperAccount:'
assert marker in s, 'PaperAccount anchor'
s = s.replace(marker,
              '#: The field names restore_position accepts, resolved once at import.',
              '_POSITION_FIELDS = frozenset(Position.__dataclass_fields__)',
              '',
              marker, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('restore rebuilds lots')