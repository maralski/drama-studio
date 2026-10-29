#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""List DramaEngine's public API and check its documentation rules.

    swift package dump-symbol-graph --minimum-access-level public
    python3 -I tools/api_audit.py [--list]

Fails when a public symbol declared in Sources/DramaEngine has no doc comment,
or a public struct or enum does not conform to Sendable. --list prints every
public symbol with its kind and source line.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / ".build/arm64-apple-macosx/symbolgraph/DramaEngine.symbols.json"
SENDABLE = "s:s8SendableP"


def main():
    if not GRAPH.exists():
        sys.exit(f"{GRAPH} is missing; run swift package dump-symbol-graph --minimum-access-level public")
    graph = json.loads(GRAPH.read_text())
    sendable = {r["source"] for r in graph["relationships"] if r["kind"] == "conformsTo" and r["target"] == SENDABLE}
    rows, problems = [], []
    for symbol in graph["symbols"]:
        location = symbol.get("location")
        if not location or "/Sources/DramaEngine/" not in location["uri"]:
            continue
        path = ".".join(symbol["pathComponents"])
        kind = symbol["kind"]["identifier"].removeprefix("swift.")
        where = location["uri"].split("/Sources/DramaEngine/")[1] + ":" + str(location["position"]["line"] + 1)
        rows.append((path, kind, where))
        if "docComment" not in symbol:
            problems.append(f"{where}: {path} has no doc comment")
        if kind in ("struct", "enum") and symbol["identifier"]["precise"] not in sendable:
            problems.append(f"{where}: {path} is not Sendable")
    if "--list" in sys.argv:
        for path, kind, where in sorted(rows):
            print(f"{path}\t{kind}\t{where}")
    for problem in sorted(problems):
        print(problem, file=sys.stderr)
    print(f"{len(rows)} public symbols, {len(problems)} problems", file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
