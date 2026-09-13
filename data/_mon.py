import ctypes, json, subprocess, sys, time, urllib.request

class MEM(ctypes.Structure):
    _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),
              ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
              ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
              ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),
              ('ullAvailExtendedVirtual',ctypes.c_ulonglong)]

def sample():
    m=MEM(); m.dwLength=ctypes.sizeof(MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.dwMemoryLoad, m.ullAvailPhys/1e9

def procs():
    out = subprocess.run(['tasklist','/FI','IMAGENAME eq python.exe','/FO','CSV'],
                         capture_output=True, text=True).stdout
    return max(0, len(out.strip().splitlines()) - 1)

def stage():
    try:
        d=json.load(urllib.request.urlopen('http://127.0.0.1:8101/api/training/state', timeout=30))
        c=d.get('current') or {}
        return c.get('status'), c.get('stage')
    except Exception:
        return '?', '?'

if __name__ == '__main__':
    body=json.dumps({'tier':'mainstream','interval':'5m','days':365,'horizon':12,
                     'folds':5,'rounds':120,'cost_bps':12}).encode()
    req=urllib.request.Request('http://127.0.0.1:8101/api/training/start', data=body,
                               headers={'Content-Type':'application/json'}, method='POST')
    print('run =', json.load(urllib.request.urlopen(req, timeout=100)).get('run_id'))
    print()
    print('  %-7s %-14s %-6s %-8s %s' % ('t(s)','stage','procs','mem%','avail GB'))
    peak=0
    for i in range(18):
        st, sg = stage()
        load, avail = sample()
        peak=max(peak, load)
        print('  %-7d %-14s %-6d %-8d %.1f' % (i*30, sg, procs(), load, avail))
        if st == 'done':
            break
        time.sleep(30)
    print()
    print('峰值内存占用 = %d%%' % peak)