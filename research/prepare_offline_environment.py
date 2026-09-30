"""Recover cached wheels for an isolated MVP environment, with no network.

Links immutable wheel archives, never installed package files. Installation is
a separate pip operation. Source cache entries are neither edited nor removed.
"""
import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile


def collect(cache, output):
    if shutil.disk_usage("/mnt/c").free < 20 * 2**30:
        raise RuntimeError("physical_disk_floor")
    output.mkdir(parents=True, exist_ok=False)
    wheels = []
    for source in sorted(cache.rglob("*")):
        if not source.is_file() or not (source.name.endswith(".body") or source.suffix == ".whl"):
            continue
        try:
            with zipfile.ZipFile(source) as archive:
                metadata_path = next((n for n in archive.namelist() if n.endswith(".dist-info/METADATA")), None)
                if metadata_path is None:
                    continue
                folder = metadata_path.rsplit("/", 1)[0]
                metadata = email.message_from_bytes(archive.read(metadata_path))
                wheel = email.message_from_bytes(archive.read(folder + "/WHEEL"))
                tags = wheel.get_all("Tag")
                tag = next((t for t in tags if t.startswith("py3-")), tags[0])
                name = re.sub(r"[-_.]+", "_", metadata["Name"])
                filename = f"{name}-{metadata['Version']}-{tag}.whl"
                if Path(filename).name != filename:
                    raise ValueError("unsafe wheel filename")
                extracted = sum(i.file_size for i in archive.infolist())
            target = output / filename
            if target.exists():
                continue
            # Same filesystem required: do not silently duplicate gigabytes.
            os.link(source, target)
            wheels.append({"file": filename, "archive_bytes": source.stat().st_size,
                           "extracted_bytes": extracted})
        except (zipfile.BadZipFile, KeyError):
            continue
    (output / "inventory.json").write_text(json.dumps(wheels, indent=2) + "\n")
    print(json.dumps({"wheel_count": len(wheels), "output": str(output)}))


def lock(report, output, wheelhouse):
    data = json.loads(report.read_text())
    rows = []
    extracted = 0
    for item in data["install"]:
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
        from urllib.parse import unquote, urlparse
        url = urlparse(item["download_info"]["url"])
        if url.scheme != "file":
            raise ValueError("network source forbidden")
        path = Path(unquote(url.path))
        if path.parent.resolve() != wheelhouse.resolve():
            raise ValueError("wheel outside local wheelhouse")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("wheel identity mismatch")
        with zipfile.ZipFile(path) as archive:
            extracted += sum(i.file_size for i in archive.infolist())
        rows.append(f"{name}==={version} --hash=sha256:{digest}")
    # Reserve installation plus a conservative 256 MiB overhead.
    reserve = extracted + 256 * 2**20
    if reserve > 5 * 2**30 or shutil.disk_usage("/mnt/c").free < 20 * 2**30 + reserve:
        raise RuntimeError("installation exceeds bounded budget or disk floor")
    with output.open("x") as stream:
        stream.write("# Linux x86_64 / CPython 3.12; local wheel SHA-256 lock\n")
        stream.write("\n".join(sorted(rows)) + "\n")
    print(json.dumps({"packages": len(rows), "extracted_bytes": extracted, "reserved_bytes": reserve}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--cache", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    lock_parser = commands.add_parser("lock")
    lock_parser.add_argument("--report", type=Path, required=True)
    lock_parser.add_argument("--output", type=Path, required=True)
    lock_parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.cache, args.output)
    else:
        lock(args.report, args.output, args.wheelhouse)
