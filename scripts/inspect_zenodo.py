import urllib.request
import re
import json

url = 'https://zenodo.org/records/20053147'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        print('Zenodo HTML length:', len(html))
        title = re.findall(r'<title>(.*?)</title>', html)
        print('Title:', title)
        # Find all files links
        files = re.findall(r'/records/20053147/files/([a-zA-Z0-9_\-\.\%]+)', html)
        print('Files found in html:', set(files))
        # Find any other links
        links = re.findall(r'href="([^"]+)"', html)
        rel_links = [l for l in links if 'files' in l or '.zip' in l or '.csv' in l or '.tar' in l]
        print('Relevant links:', rel_links[:10])
except Exception as e:
    print('Zenodo error:', e)

# Also try API with same user agent
api_url = 'https://zenodo.org/api/records/20053147'
try:
    req_api = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req_api) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print('API metadata title:', data.get('metadata', {}).get('title'))
        print('API files count:', len(data.get('files', [])))
        for f in data.get('files', []):
            print('  File:', f.get('key'), f.get('size'), f.get('links', {}).get('self'))
except Exception as e:
    print('API error:', e)
