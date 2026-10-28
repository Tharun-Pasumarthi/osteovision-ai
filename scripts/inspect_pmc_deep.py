"""
Inspect PMC11375658 details: Jintan cohort, Soochow cohort, OSF repository contents.
"""
import urllib.request
import re
import json

def inspect_pmc():
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11375658/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
    
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', content)
    text = ' '.join(text.split())

    keywords = ['Jintan', 'Soochow', 'Ethics', 'Institutional', 'DXA', 'dual-energy', 'dataset', 'availability']
    for kw in keywords:
        matches = [m.start() for m in re.finditer(kw, text, re.IGNORECASE)]
        print(f"Keyword '{kw}': {len(matches)} occurrences")
        for idx in matches[:3]:
            snippet = text[max(0, idx-100):min(len(text), idx+200)]
            print(f"   ...{snippet}...\n")

def inspect_osf():
    # The paper mentioned: https://osf.io/7aujc and https://osf.io/zjahc
    for osf_id in ['7aujc', 'zjahc']:
        url = f"https://api.osf.io/v2/nodes/{osf_id}/files/"
        print(f"\n--- Checking OSF node {osf_id} ---")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                providers = data.get('data', [])
                for prov in providers:
                    prov_name = prov.get('attributes', {}).get('name')
                    files_link = prov.get('relationships', {}).get('files', {}).get('links', {}).get('related', {}).get('href')
                    print(f"  Storage Provider: {prov_name}, Files link: {files_link}")
                    if files_link:
                        try:
                            f_req = urllib.request.Request(files_link, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(f_req) as f_resp:
                                f_data = json.loads(f_resp.read().decode('utf-8'))
                                for item in f_data.get('data', []):
                                    item_name = item.get('attributes', {}).get('name')
                                    item_kind = item.get('attributes', {}).get('kind')
                                    print(f"    Item: {item_name} ({item_kind})")
                        except Exception as fe:
                            print(f"    Failed to query files link: {fe}")
        except Exception as e:
            print(f"  OSF error for {osf_id}: {e}")

if __name__ == '__main__':
    inspect_pmc()
    inspect_osf()
