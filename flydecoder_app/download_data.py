"""Download only the official MaleCNS files required by the FlyBrain simulator."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "flybrain" / "data" / "manifest.json"
REQUIRED_FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, destination: Path, expected: str) -> None:
    if destination.exists() and _sha256(destination) == expected:
        print(f"Вже перевірено: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "FlyDecoder/0.1 (MaleCNS CC BY 4.0)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:
            raise
        response = None
    if response is not None:
        with response, part.open("ab" if response.status == 206 else "wb") as output:
            received = existing if response.status == 206 else 0
            total = received + int(response.headers.get("Content-Length") or 0)
            reported = -1
            while block := response.read(1 << 20):
                output.write(block)
                received += len(block)
                percent = int(received * 100 / total) if total else -1
                if percent >= 0 and percent // 10 != reported:
                    reported = percent // 10
                    print(f"  {destination.name}: {percent}% ({received / 1e6:,.0f} MB)", flush=True)
                elif total == 0 and received // (100 << 20) > reported:
                    reported = received // (100 << 20)
                    print(f"  {destination.name}: {received / 1e6:,.0f} MB", flush=True)
    if not part.exists() or _sha256(part) != expected:
        raise RuntimeError(f"SHA-256 не збігається: {destination.name}; файл залишено як .part")
    os.replace(part, destination)
    print(f"Завантажено й перевірено: {destination.name}")


def fetch_required(data_dir: str | Path | None = None) -> list[Path]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Не знайдено upstream маніфест: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest["malecns"]["files"]
    selected = [item for item in files if item["path"] in REQUIRED_FILES]
    found = {item["path"] for item in selected}
    if found != REQUIRED_FILES:
        raise RuntimeError(f"upstream маніфест не містить потрібних файлів: {sorted(REQUIRED_FILES - found)}")
    target = Path(data_dir or os.environ.get("FLYBRAIN_DATA", ROOT / "data" / "malecns"))
    target.mkdir(parents=True, exist_ok=True)
    print(f"Офіційні файли MaleCNS v1.0 · CC BY 4.0 → {target}", flush=True)
    for item in selected:
        _fetch(item["url"], target / item["path"], item["sha256"])
    return [target / name for name in sorted(REQUIRED_FILES)]


def main() -> int:
    try:
        fetch_required()
    except Exception as exc:
        print(f"Помилка отримання MaleCNS: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
