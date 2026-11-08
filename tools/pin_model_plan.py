#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Writes a pinned download plan the app verifies every byte against.

    python3 -I tools/pin_model_plan.py [drama3|ming]            write Sources/DramaEngine/Install/Drama3Plan.swift or MingDownloadPlan.swift
    python3 -I tools/pin_model_plan.py [drama3|ming] --check    HEAD every file URL and compare with the committed plan

The engine defaults to drama3.

The Hugging Face tree API gives the sha256 of each LFS file (lfs.oid). Small
files live in git, where the API gives only the git blob sha1, so this
downloads each one, checks its blob sha1 against the tree at the pinned commit,
and records its sha256. Run it at development time only: the app never fetches
hashes, so a changed upstream file fails verification instead of being trusted.
"""
import hashlib, json, re, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANS = {
    "drama3": dict(out="Drama3Plan.swift", repo="fishaudio/drama-3", revision="e0a9218aa48888270f06ef1ed9ea67622c0e17d2",
                   skip=lambda path: path == ".gitattributes", big=lambda path: path.startswith("model-") or path == "codec.pth",
                   doc="the engine, its tuning and its packed-weight\n    /// cache were verified on"),
    "ming": dict(out="MingDownloadPlan.swift", repo="inclusionAI/Ming-3.0-tts-0.5B", revision="7247f4c94a5b2e987404ed28a5cfec41b1a4ca61",
                 skip=lambda path: path == ".gitattributes" or path.endswith(".pdf") or path.startswith("figures/"),
                 big=lambda path: path in ("model.safetensors", "campplus.onnx"),
                 doc="the Ming engine was verified on, without\n    /// the report and figures"),
}
ENGINE = next((a for a in sys.argv[1:] if not a.startswith("--")), "drama3")
if ENGINE not in PLANS:
    sys.exit(f"unknown engine {ENGINE}; pick one of {', '.join(PLANS)}")
PLAN = PLANS[ENGINE]
OUT = ROOT / "Sources/DramaEngine/Install" / PLAN["out"]
REPO = PLAN["repo"]
REVISION = PLAN["revision"]
AGENT = {"User-Agent": "drama-studio-pin"}


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120)


def resolve_url(path):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"


def tree():
    with fetch(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true") as response:
        entries = json.load(response)
    return [e for e in entries if e["type"] == "file" and not PLAN["skip"](e["path"])]


def pinned(entry):
    if entry.get("lfs"):
        return entry["path"], entry["size"], entry["lfs"]["oid"]
    with fetch(resolve_url(entry["path"])) as response:
        data = response.read()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if len(data) != entry["size"] or blob != entry["oid"]:
        sys.exit(f"{entry['path']}: downloaded {len(data)} bytes with blob {blob}; the tree says {entry['size']} and {entry['oid']}")
    return entry["path"], entry["size"], hashlib.sha256(data).hexdigest()


def write():
    rows = sorted((pinned(e) for e in tree()), key=lambda row: (PLAN["big"](row[0]), row[0]))
    lines = "\n".join(f'        .init(path: "{p}", size: {s}, sha256: "{h}"),' for p, s, h in rows)
    OUT.write_text(f"""// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Written by tools/pin_model_plan.py; do not edit.

extension ModelDownloadPlan {{
    /// `{REPO}` at the commit {PLAN["doc"]}. Small files come first.
    package static let {ENGINE} = ModelDownloadPlan(repo: "{REPO}", revision: "{REVISION}", files: [
{lines}
    ])
}}
""")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} files, {sum(s for _, s, _ in rows):,} bytes")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def check():
    """Hugging Face answers a HEAD of a resolve URL with X-Repo-Commit and a
    redirect carrying X-Linked-Size and X-Linked-Etag. For LFS files the etag
    is the sha256; for small files it is the git blob sha1, so those are
    fetched and hashed."""
    opener = urllib.request.build_opener(NoRedirect)
    rows = re.findall(r'\.init\(path: "([^"]+)", size: (\d+), sha256: "([0-9a-f]{64})"\)', OUT.read_text())
    problems = 0
    for path, size, sha in rows:
        try:
            response = opener.open(urllib.request.Request(resolve_url(path), method="HEAD", headers=AGENT), timeout=60)
        except urllib.error.HTTPError as redirect:
            response = redirect
        headers = response.headers
        commit = headers.get("X-Repo-Commit") or ""
        etag = (headers.get("X-Linked-Etag") or "").strip().removeprefix("W/").strip('"')
        linked = re.fullmatch(r"[0-9a-f]{64}", etag) is not None
        got_size = int(headers.get("X-Linked-Size") or headers.get("Content-Length") or -1)
        if linked:
            got_sha = etag
        else:
            with fetch(resolve_url(path)) as body:
                data = body.read()
            got_sha, got_size = hashlib.sha256(data).hexdigest(), len(data)
        ok = commit == REVISION and got_size == int(size) and got_sha == sha
        problems += not ok
        print(f"{'ok ' if ok else 'BAD'} {response.status} {path:36} {got_size:>11} bytes  sha256 {got_sha[:12]}  commit {commit[:7]}  {'etag' if linked else 'hashed'}")
    print(f"{len(rows)} files, {problems} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    check() if "--check" in sys.argv else write()
