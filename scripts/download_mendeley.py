"""
Download Mendeley dataset zip and extract to data/raw/mendeley/
"""
import os
import urllib.request
import zipfile
import time

DEST_DIR = "d:/anti/data/raw/mendeley"
ZIP_PATH = os.path.join(DEST_DIR, "Osteoporosis_Knee_X-ray.zip")
URL = "https://data.mendeley.com/public-files/datasets/fxjm8fb6mw/files/b7a95139-dfef-4db3-b3b9-e0a36315f334/file_downloaded"

os.makedirs(DEST_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 180000000:
    print(f"Downloading Mendeley archive (~181 MB) to {ZIP_PATH}...")
    start_t = time.time()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, 'wb') as out_f:
        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
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
