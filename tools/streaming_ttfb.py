#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Time to first audio byte of `stream: true` through `curl -N`, against two servers, interleaved.

    tools/streaming_ttfb.py --side before=http://127.0.0.1:8781:PID --side after=http://127.0.0.1:8782:PID --runs 5

Each run sends the same two-sentence line with a fixed seed to every side, in
alternating order, and prints one TSV row per request: the time from starting
curl to the first body byte on its stdout, the time to the last byte, the PCM
bytes, curl's exit code and HTTP status, the server's CPU seconds over the
request (from `ps -o time=`), and the peak CPU cores over any 0.5 s window.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

TEXT = ("The storm had passed by morning, leaving the harbor strangely quiet. "
        "Old Tom walked the pier, counting the boats that had survived the night.")


def cpu_seconds(pid):
    out = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    if not out:
        return float("nan")
    parts = out.split(":")
    seconds = float(parts[-1])
    if len(parts) >= 2:
        seconds += 60 * float(parts[-2])
    if len(parts) == 3:
        seconds += 3600 * float(parts[0])
    return seconds


def run(base, pid, token, seed):
    body = json.dumps({"input": TEXT, "stream": True, "response_format": "pcm", "seed": seed})
    headers = tempfile.NamedTemporaryFile(delete=False)
    command = ["curl", "-sSN", "-D", headers.name, "-H", "Content-Type: application/json", "-d", body, base + "/v1/audio/speech"]
    if token:
        command[1:1] = ["-H", f"Authorization: Bearer {token}"]
    peak = [0.0]
    done = threading.Event()

    def sample():
        last_t, last_cpu = time.time(), cpu_seconds(pid)
        while not done.wait(0.5):
            t, cpu = time.time(), cpu_seconds(pid)
            peak[0] = max(peak[0], (cpu - last_cpu) / (t - last_t))
            last_t, last_cpu = t, cpu

    cpu0 = cpu_seconds(pid)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    first = proc.stdout.read(1)
    first_at = time.time() - start
    rest = proc.stdout.read()
    total_at = time.time() - start
    code = proc.wait()
    done.set()
    sampler.join()
    cpu = cpu_seconds(pid) - cpu0
    head = open(headers.name, newline="").read()
    os.unlink(headers.name)
    status = head.split("\r\n", 1)[0].split(" ")[1] if head else "none"
    fields = dict(line.split(": ", 1) for line in head.split("\r\n")[1:] if ": " in line)
    return {
        "first_byte_s": first_at if first else float("nan"),
        "total_s": total_at,
        "pcm_bytes": len(first) + len(rest),
        "curl_exit": code,
        "status": status,
        "estimated_s": fields.get("X-Drama-Estimated-Seconds", ""),
        "cpu_s": cpu,
        "avg_cores": cpu / total_at,
        "peak_cores": peak[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", action="append", required=True, help="name=url:pid")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--token")
    args = parser.parse_args()
    sides = []
    for spec in args.side:
        name, rest = spec.split("=", 1)
        url, pid = rest.rsplit(":", 1)
        sides.append((name, url, int(pid)))
    columns = ["ts_utc", "run", "side", "first_byte_s", "total_s", "pcm_bytes", "audio_s", "curl_exit", "status",
               "estimated_s", "cpu_s", "avg_cores", "peak_cores"]
    print("\t".join(columns))
    for index in range(1, args.runs + 1):
        order = sides if index % 2 == 1 else list(reversed(sides))
        for name, url, pid in order:
            r = run(url, pid, args.token, args.seed)
            r.update(ts_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), run=index, side=name,
                     audio_s=r["pcm_bytes"] / 2 / 44100)
            print("\t".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in columns))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
