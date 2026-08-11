import requests, os
from pathlib import Path
DATA_DIR = Path(os.environ.get("CORNEAL_FM_DATA", str(Path(__file__).resolve().parent.parent / "data")))
url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193101/suppl/GSE193101_RAW.tar'
r = requests.head(url, timeout=15)
sz = int(r.headers.get('content-length', 0))
print(f'HTTP {r.status_code}, Size: {sz/(1024**2):.0f}MB')
if r.status_code == 200:
    dest = str(DATA_DIR / 'epidermis' / 'wiedemann2023' / 'GSE193101_RAW.tar')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {}
    tmp = dest + '.part'
    if os.path.exists(tmp):
        headers['Range'] = f'bytes={os.path.getsize(tmp)}-'
    r = requests.get(url, headers=headers, stream=True, timeout=(30, 7200))
    mode = 'ab' if 'Range' in headers else 'wb'
    dl = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    with open(tmp, mode) as f:
        for chunk in r.iter_content(1024*1024):
            if chunk:
                f.write(chunk)
                dl += len(chunk)
                pct = dl/sz*100 if sz else 0
                print(f'\r{sz/(1024**2):.0f}MB: {dl/(1024**2):.0f}MB ({pct:.0f}%)', end='')
    os.rename(tmp, dest)
    print(f'\nDone')
else:
    print('Not downloadable')
