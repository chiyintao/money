import io
p='app/models/train.py'
s=io.open(p,encoding='utf-8').read()
old='''            entry["degraded_features"] = dict(sorted(missing_seen.items()))
            report.append(entry)'''
new='''            entry["degraded_features"] = dict(sorted(missing_seen.items()))'''
assert old in s, 'not found'
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('removed stray report.append')