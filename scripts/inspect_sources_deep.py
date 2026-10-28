"""
Detailed inspection of Mendeley and PMC HTML.
"""
import urllib.request
import re
import json

def inspect_mendeley_details():
    url = "https://data.mendeley.com/datasets/fxjm8fb6mw/2"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            # Look for JSON in HTML or data-react-helmet or state
            print("Searching Mendeley for files...")
            # Look for file names or downloads
            files = re.findall(r'(\b[\w\-\.]+\.(?:zip|rar|csv|xlsx|png|jpg|jpeg)\b)', content, re.IGNORECASE)
            print("Extracted file mentions in Mendeley page:", set(files))
            
            # Look for urls or download endpoints
            download_urls = re.findall(r'https?://[^\s"\'<>]+download[^\s"\'<>]*', content, re.IGNORECASE)
            print("Download URLs:", download_urls[:10])

            # Check if there is an api call pattern in scripts
            api_calls = re.findall(r'https?://data\.mendeley\.com/api/[^\s"\'<>]+', content)
            print("API calls mentioned:", api_calls[:5])

            # Search for zip or direct download
            direct_links = re.findall(r'https?://prod-dcd-datasets-cache-zipfiles[^\s"\'<>]+', content)
            print("Prod zip links:", direct_links)
    except Exception as e:
        print("Mendeley error:", e)

def inspect_pmc_details():
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11375658/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            # Find data availability section
            idx = content.find("Data availability")
            if idx == -1:
                idx = content.find("Data Availability")
            if idx != -1:
                snippet = content[idx:idx+1500]
                clean = re.sub(r'<[^>]+>', ' ', snippet)
                print("PMC Data Availability Section:\n", " ".join(clean.split()))
            else:
                print("Could not find 'Data availability' heading directly.")
                
            # Check title and abstract
            title = re.findall(r'<title>(.*?)</title>', content)
            print("PMC Title:", title)
    except Exception as e:
        print("PMC error:", e)

def inspect_zenodo_details():
    url = "https://zenodo.org/records/20053147"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print("Zenodo status:", resp.status)
    except urllib.error.HTTPError as e:
        print(f"Zenodo HTTPError {e.code}: {e.reason}")
    except Exception as e:
        print("Zenodo error:", e)

if __name__ == '__main__':
    inspect_mendeley_details()
    inspect_pmc_details()
    inspect_zenodo_details()
