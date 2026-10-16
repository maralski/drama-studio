#!/usr/bin/env python3
"""Speaks text from stdin through POST /v1/audio/speech/live as it is typed or piped.

Each stdin line goes out as one HTTP chunk the moment it is read; an empty line
flushes the text so far; EOF ends the request. The PCM that comes back
(16-bit little-endian mono at 44.1 kHz) is written to stdout, so pipe it into
a player:

    some-agent | tools/live_speech.py --voice Narrator | ffplay -f s16le -ar 44100 -ch_layout mono -nodisp -

With --events it prints the Server-Sent Events instead, each stamped with the
seconds since the request started.
"""
import argparse
import os
import socket
import sys
import threading
import time
import urllib.parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("DRAMA_API_URL", "http://127.0.0.1:8780"))
    parser.add_argument("--voice")
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--token", default=os.environ.get("DRAMA_API_TOKEN"))
    args = parser.parse_args()

    token = args.token
    token_file = os.path.expanduser("~/Library/Application Support/DramaStudio/api-token")
    if token is None and os.path.exists(token_file):
        token = open(token_file).read().strip()

    url = urllib.parse.urlparse(args.url)
    query = urllib.parse.urlencode({"voice": args.voice} if args.voice else {})
    head = [f"POST /v1/audio/speech/live{'?' + query if query else ''} HTTP/1.1",
            f"Host: {url.hostname}", "Transfer-Encoding: chunked", "Content-Type: text/plain; charset=utf-8"]
    if token:
        head.append(f"Authorization: Bearer {token}")
    if args.events:
        head.append("Accept: text/event-stream")
    sock = socket.create_connection((url.hostname, url.port or 80))
    sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode())
    start = time.time()

    def send_input():
        for line in sys.stdin:
            data = line.encode()
            sock.sendall(b"%x\r\n%s\r\n" % (len(data), data))
        sock.sendall(b"0\r\n\r\n")

    threading.Thread(target=send_input, daemon=True).start()
    reader = sock.makefile("rb")
    status = reader.readline().decode().strip()
    while reader.readline() not in (b"\r\n", b""):
        pass
    if " 200 " not in status:
        sys.stderr.write(status + "\n" + reader.read().decode(errors="replace"))
        sys.exit(1)
    out = sys.stdout.buffer
    while True:
        size = int(reader.readline().split(b";")[0], 16)
        if size == 0:
            break
        payload = reader.read(size)
        reader.readline()
        if args.events:
            for line in payload.decode().splitlines():
                if line.startswith("event:"):
                    out.write(b"%6.2f %s\n" % (time.time() - start, line.encode()))
                elif line.startswith("data: {"):
                    out.write(b"       %s\n" % line.encode())
        else:
            out.write(payload)
        out.flush()


if __name__ == "__main__":
    main()
