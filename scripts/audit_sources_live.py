"""
Script to audit the 3 target data sources:
1. Mendeley dataset: https://data.mendeley.com/datasets/fxjm8fb6mw/2
2. Zenodo combined collection: https://zenodo.org/records/20053147
3. Published few-shot learning study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11375658/
"""
import urllib.request
import json
import re

def check_mendeley():
    print("=== Checking Mendeley dataset (fxjm8fb6mw/2) ===")
    url = "https://data.mendeley.com/datasets/fxjm8fb6mw/2"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            print(f"Mendeley page fetched successfully ({len(content)} bytes)")
            # Look for direct file download links
            links = re.findall(r'href="([^"]+)"', content)
            file_links = [l for l in links if any(ext in l.lower() for ext in ['.zip', '.rar', '.csv', '.xlsx', '/files/'])]
            print(f"Found {len(file_links)} relevant file links:")
            for l in file_links[:10]:
                print("  ", l)
            # Check for API endpoint
            api_url = "https://data.mendeley.com/public-api/datasets/fxjm8fb6mw/2"
            try:
                req_api = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                with urllib.request.urlopen(req_api, timeout=10) as a_resp:
                    data = json.loads(a_resp.read().decode('utf-8'))
                    print("API Title:", data.get("name"))
                    print("API Files count:", len(data.get("files", [])))
                    for f in data.get("files", []):
                        print("  File:", f.get("name"), "Size:", f.get("size"), "URL:", f.get("content_details", {}).get("download_url"))
            except Exception as e:
                print("Mendeley public-api note:", e)
    except Exception as e:
        print("Mendeley error:", e)

def check_zenodo():
    print("\n=== Checking Zenodo record (20053147) ===")
    url = "https://zenodo.org/api/records/20053147"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("Zenodo Title:", data.get("metadata", {}).get("title"))
            print("Zenodo DOI:", data.get("doi"))
            print("Files count:", len(data.get("files", [])))
            for f in data.get("files", []):
                print("  File:", f.get("key"), "Size:", f.get("size"), "Download:", f.get("links", {}).get("self"))
    except Exception as e:
        print("Zenodo API error:", e)
        # Try web page directly
        page_url = "https://zenodo.org/records/20053147"
        try:
            req_page = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_page, timeout=15) as p_resp:
                print("Zenodo web page status:", p_resp.status)
        except Exception as pe:
            print("Zenodo web page error:", pe)

def check_pmc():
    print("\n=== Checking PMC11375658 ===")
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11375658/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            print(f"PMC page fetched successfully ({len(content)} bytes)")
            # Search for Data Availability Statement
            data_avail = re.findall(r'(?:data availability|availability of data)[\s\S]{0,1000}?(?:</div>|</section>|<p>)', content, re.IGNORECASE)
            if data_avail:
                clean_text = re.sub(r'<[^>]+>', ' ', data_avail[0])
                print("Data Availability excerpt:\n", " ".join(clean_text.split()[:150]))
            else:
                print("Searching for mentions of Jintan, Soochow, cohorts...")
                mentions = re.findall(r'([^.]{0,150}(?:Jintan|Soochow|cohort|hospital)[^.]{0,150}\.)', content, re.IGNORECASE)
                for m in mentions[:4]:
                    print("  Snippet:", " ".join(re.sub(r'<[^>]+>', '', m).split()))
    except Exception as e:
        print("PMC error:", e)

if __name__ == '__main__':
    check_mendeley()
    check_zenodo()
    check_pmc()
