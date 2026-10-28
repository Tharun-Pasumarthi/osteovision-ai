"""
Download Zenodo dataset zip and extract to data/raw/zenodo/
"""
import os
import urllib.request
import zipfile
import time

DEST_DIR = "d:/anti/data/raw/zenodo"
ZIP_PATH = os.path.join(DEST_DIR, "Knee_Osteoporosis_Dataset.zip")
URL = "https://zenodo.org/records/20053147/files/Knee_Osteoporosis_Dataset.zip?download=1"

os.makedirs(DEST_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 30000000:
    print(f"Downloading Zenodo archive (~33 MB) to {ZIP_PATH}...")
    start_t = time.time()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, 'wb') as out_f:
        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 512 # 512KB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (1024 * 1024 * 5) < chunk_size:
                print(f"  {downloaded*100/total_size:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", flush=True)
    elapsed = time.time() - start_t
    print(f"Download complete in {elapsed:.1f}s. Size: {os.path.getsize(ZIP_PATH)} bytes")
else:
    print(f"Archive already downloaded ({os.path.getsize(ZIP_PATH)} bytes)")

print(f"Extracting {ZIP_PATH} to {DEST_DIR}...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(DEST_DIR)
print("Extraction complete. Extracted contents:")
for root, dirs, files in os.walk(DEST_DIR):
    rel = os.path.relpath(root, DEST_DIR)
    if rel == '.':
        print(f"  Root: {len(dirs)} dirs, {len(files)} files")
    else:
        print(f"  Subdir '{rel}': {len(dirs)} dirs, {len(files)} files")
