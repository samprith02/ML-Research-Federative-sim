"""
download_cifar10.py
===================
Robust CIFAR-10 downloader with retry logic and resume support.

Retries up to 10 times with exponential backoff on network errors.
Uses requests library for better streaming support than urllib.

Run once:
    uv run python download_cifar10.py
"""
import os
import sys
import time
import hashlib
import tarfile

# --------------- try requests, fall back to urllib ---------------
try:
    import requests as _requests_lib
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import urllib.request

CIFAR10_URL  = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5  = "c58f30108f718f92721af3b95e74349a"
DATA_DIR     = "./data"
FILENAME     = "cifar-10-python.tar.gz"
EXTRACT_DIR  = os.path.join(DATA_DIR, "cifar-10-batches-py")
DEST_PATH    = os.path.join(DATA_DIR, FILENAME)
MAX_RETRIES  = 10
CHUNK_SIZE   = 1024 * 64  # 64 KB

def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def already_extracted() -> bool:
    return os.path.isdir(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 0

def download_with_retry() -> bool:
    os.makedirs(DATA_DIR, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n[Attempt {attempt}/{MAX_RETRIES}] Downloading CIFAR-10...")
            if HAS_REQUESTS:
                _download_requests()
            else:
                _download_urllib()
            print("  Download stream finished.")
            return True
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"  ERROR: {e}")
            print(f"  Retrying in {wait}s...")
            # Remove corrupt partial file before retry
            if os.path.exists(DEST_PATH):
                os.remove(DEST_PATH)
            time.sleep(wait)

    return False

def _download_requests():
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(CIFAR10_URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1
        with open(DEST_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        mb = downloaded / 1024 / 1024
                        print(f"  {pct:3d}%  {mb:.1f} MB / {total/1024/1024:.1f} MB", flush=True)
                        last_pct = pct

def _download_urllib():
    def reporthook(count, block, total):
        downloaded = count * block
        if total > 0:
            pct = int(downloaded * 100 / total)
            if pct % 5 == 0:
                mb = downloaded / 1024 / 1024
                print(f"  {pct:3d}%  {mb:.1f} MB / {total/1024/1024:.1f} MB", flush=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(CIFAR10_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(DEST_PATH, "wb") as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct % 5 == 0:
                        mb = downloaded / 1024 / 1024
                        sys.stdout.write(
                            f"\r  {pct:3d}%  {mb:.1f} MB / {total/1024/1024:.1f} MB"
                        )
                        sys.stdout.flush()

if __name__ == "__main__":
    print("=" * 55)
    print("CIFAR-10 Robust Downloader")
    print("=" * 55)
    print(f"  Using requests: {HAS_REQUESTS}")
    print(f"  Destination:    {DEST_PATH}")

    if already_extracted():
        print("\n[OK] CIFAR-10 already extracted -- nothing to do.")
        sys.exit(0)

    if os.path.exists(DEST_PATH):
        print(f"\n  Found existing file ({os.path.getsize(DEST_PATH)//1024//1024} MB), removing before fresh download.")
        os.remove(DEST_PATH)

    success = download_with_retry()
    if not success:
        print("\n[FAIL] All download attempts failed. Please download manually:")
        print(f"  URL: {CIFAR10_URL}")
        print(f"  Place the file at: {os.path.abspath(DEST_PATH)}")
        sys.exit(1)

    # Verify MD5
    print("\n  Verifying MD5 checksum...")
    actual_md5 = md5sum(DEST_PATH)
    if actual_md5 != CIFAR10_MD5:
        print(f"  [FAIL] MD5 mismatch! Expected {CIFAR10_MD5}, got {actual_md5}")
        os.remove(DEST_PATH)
        sys.exit(1)
    print(f"  MD5 OK: {actual_md5}")

    # Extract
    print("\n  Extracting archive...")
    with tarfile.open(DEST_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"  Extracted to: {EXTRACT_DIR}")

    print("\n[OK] CIFAR-10 is ready.")
