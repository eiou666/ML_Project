"""Download the configured official CUDA wheel in verified, resumable ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
NAME = "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"
URL = "https://download.pytorch.org/whl/cu121/" + NAME.replace("+", "%2B")
CHUNK = 8 * 1024 * 1024


def main():
    index = urlopen("https://download.pytorch.org/whl/cu121/torch/", timeout=60).read().decode()
    expected = re.search(r"torch-2\.5\.1%2Bcu121-cp312-cp312-win_amd64\.whl#sha256=([a-f0-9]{64})", index).group(1)
    with urlopen(Request(URL, headers={"Range": "bytes=0-0"}), timeout=60) as response:
        total = int(response.headers["Content-Range"].split("/")[-1])
    directory = ROOT / "tmp" / "wheels"
    parts = directory / "torch-parts"
    parts.mkdir(parents=True, exist_ok=True)
    count = (total + CHUNK - 1) // CHUNK
    started = time.monotonic()

    def get_part(i):
        start, end = i * CHUNK, min(total, (i + 1) * CHUNK) - 1
        part = parts / f"{i:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(4):
            try:
                request = Request(URL + f"?part={i}", headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=90) as response:
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Server did not return requested range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError("Incomplete range")
                part.write_bytes(data)
                return part
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    print(f"Official wheel: {total / 1e6:.1f} MB; {count} ranges; sha256={expected}", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(get_part, i) for i in range(count)]
        for finished, future in enumerate(as_completed(futures), 1):
            future.result()
            if finished % 8 == 0 or finished == count:
                print(f"Downloaded {finished}/{count} ranges in {time.monotonic()-started:.1f}s", flush=True)
    target = directory / NAME
    temporary = target.with_suffix(".download")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for i in range(count):
            data = (parts / f"{i:04d}.part").read_bytes()
            digest.update(data)
            output.write(data)
    if digest.hexdigest() != expected:
        raise RuntimeError("Official SHA256 verification failed")
    temporary.replace(target)
    print(f"SHA256 verified: {target}", flush=True)


if __name__ == "__main__":
    main()
