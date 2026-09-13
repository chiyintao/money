import io
p = 'web/js/render.js'
s = io.open(p, encoding='utf-8').read()
old = chr(10).join([
    '      ${pair("止盈价格", num(x.take_profit_price, 8))}',
    '      ${pair("止损价格", num(x.stop_price, 8))}',
])
assert old in s, 'stop/target rows'
new = chr(10).join([
    '      ${pair("止损价格", stopCell(x, "stop"))}',
      ${pair("止盈价格", stopCell(x, "target"))}',
])
s = s.replace(old, new, 1)
anchor = 'function pair(label, value) {'
assert anchor in s
helper = "// The exit policy moves the stop: breakeven_at_rr raises it to the entry once the\n// trade is 0.6R in profit, and the trailing rule follows the best price after that.\n// Showing only the final level made almost every trade read as \"stop equals entry\",\n// which looks like a bug and is not one. Both levels are shown, and the moved one is\n// labelled, so a raise is visible as a decision rather than as a wrong number.\nfunction stopCell(x, kind) {\n  const final = kind === \"stop\" ? x.stop_price : x.take_profit_price;\n  const initial = kind === \"stop\" ? x.initial_stop : x.initial_target;\n  const shown = num(final, 8);\n  if (initial === undefined || initial === null || !isFinite(initial) || initial <= 0) {\n    return shown;\n  }\n  if (Math.abs(initial - final) < 1e-12) {\n    return shown;\n  }\n  const moved = kind === \"stop\" ? \"已移保本/跟踪\" : \"已调整\";\n  return shown + \" <span class=\\\"level-note\\\">(原 \" + num(initial, 8) + \" · \" + moved + \")</span>\";\n}\n\nfunction pair(label, value) {\n"
s = s.replace(anchor, helper, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('frontend updated')