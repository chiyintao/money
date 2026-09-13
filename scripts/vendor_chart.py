from pathlib import Path
import requests
root=Path('web/vendor'); root.mkdir(exist_ok=True)
for name,url in [('lightweight-charts.js','https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js'),('LICENSE','https://unpkg.com/lightweight-charts@4.2.3/LICENSE')]:
    response=requests.get(url,timeout=30); response.raise_for_status(); (root/name).write_bytes(response.content)
