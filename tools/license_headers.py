#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Adds or checks the copyright and SPDX header on every source file this
project wrote. Vendored code keeps its own license and is skipped.

  tools/license_headers.py          add the header where it is missing
  tools/license_headers.py --check  list files without it and exit 1
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = "Copyright 2026 Peter Marelas"
SPDX = "SPDX-License-Identifier: Apache-2.0"
VENDORED = ("eval/oracle/", "Sources/DramaEngine/CodecConversion/dac_reference/")
SLASH = {".swift", ".metal"}
HASH = {".py", ".sh"}
HASH_NAMES = {"tools/with-gpu-lock"}


def style(path):
    if path.suffix in SLASH:
        return "//"
    if path.suffix in HASH or str(path) in HASH_NAMES:
        return "#"
    return None


def files():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    for name in tracked:
        path = Path(name)
        if name.startswith(VENDORED) or style(path) is None:
            continue
        yield path


def with_header(text, mark):
    header = f"{mark} {OWNER}\n{mark} {SPDX}\n"
    lines = text.splitlines(keepends=True)
    keep = 0
    if lines and (lines[0].startswith("#!") or lines[0].startswith("// swift-tools-version")):
        keep = 1
    return "".join(lines[:keep]) + header + "".join(lines[keep:])


def main():
    check = "--check" in sys.argv
    missing = []
    for path in files():
        text = (ROOT / path).read_text()
        if SPDX in text.split("\n", 4)[-1] or SPDX in "\n".join(text.split("\n")[:4]):
            continue
        missing.append(path)
        if not check:
            (ROOT / path).write_text(with_header(text, style(path)))
    if check:
        for path in missing:
            print(path)
        sys.exit(1 if missing else 0)
    print(f"added the header to {len(missing)} files")


if __name__ == "__main__":
    main()
