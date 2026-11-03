"""Download the AI4I 2020 dataset from UCI into data/raw/ai4i2020.csv."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        OUT.write_bytes(z.read(name))
    print(f"saved {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
