#!/usr/bin/env python3
"""Exercise the shipped recognizer and private protocol on a local WAV."""
import argparse
import json
from pathlib import Path
import socket
import struct
import subprocess
import uuid
import wave
import os

ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / "Sources/WordRecognitionProtocol/Assets.json").read_text())

def receive(connection):
    def exact(count):
        data = bytearray()
        while len(data) < count:
            part = connection.recv(count - len(data))
            if not part:
                raise RuntimeError("recognizer closed the private channel")
            data.extend(part)
        return data
    length = struct.unpack(">I", exact(4))[0]
    if not 0 < length <= 256 * 1024:
        raise RuntimeError("invalid recognizer reply length")
    return json.loads(exact(length))

def main():
    import base64
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bin", type=Path)
    group.add_argument("--app", type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--expect-transcript")
    args = parser.parse_args()
    directory = args.app / "Contents/MacOS" if args.app else args.bin
    resources = args.app / "Contents/Resources" if args.app else args.bin
    asset = next(m for m in ASSETS["models"] if m["englishOnly"] == (args.language == "en"))
    with wave.open(str(args.audio), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (44100, 1, 2):
            raise RuntimeError("expected delivered mono 44.1kHz PCM16")
        pcm = audio.readframes(audio.getnframes())
    parent, child = socket.socketpair()
    parent.settimeout(30)
    process = subprocess.Popen([str((directory / ASSETS["helperName"]).resolve()), args.language,
                                str((resources / ASSETS["modelDirectory"] / asset["fileName"]).resolve()), str(os.getpid())],
                               stdin=child, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child.close()
    try:
        ready = receive(parent)["ready"]["_0"]
        assert ready["modelID"] == asset["id"] and ready["modelSHA256"] == asset["sha256"], ready
        assert ready["configurationID"] == ASSETS["configurationID"]
        assert ready["protocolVersion"] == ASSETS["protocolVersion"] and args.language in ready["languages"]
        request_id = str(uuid.uuid4()).upper()
        request = json.dumps(dict(id=request_id, language=args.language, pcm16=base64.b64encode(pcm).decode())).encode()
        parent.sendall(struct.pack(">I", len(request)) + request)
        reply = receive(parent)["transcript"]
        assert reply["id"] == request_id
        if args.expect_transcript is not None:
            assert reply["text"] == args.expect_transcript, reply
        print(json.dumps(dict(model=ready["modelID"], model_sha256=ready["modelSHA256"], transcript=reply["text"])))
    finally:
        parent.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=3)
        process.stderr.close()

if __name__ == "__main__":
    main()
