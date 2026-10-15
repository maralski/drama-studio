"""Decode engine frames ([N, 11]) or codes ([10, T]) to a 44.1 kHz wav with the reference DAC codec on the CPU."""
import argparse, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from make_spec import load_codec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes"); ap.add_argument("out")
    ap.add_argument("--checkpoint", type=Path, default=next(Path.home().glob(".cache/huggingface/hub/models--fishaudio--drama-3/snapshots/*/")))
    a = ap.parse_args()
    arr = np.load(a.codes)
    codes = arr[:, 1:].T if arr.ndim == 2 and arr.shape[1] == 11 else arr
    codec = load_codec(a.checkpoint / "codec.pth")
    t0 = time.time()
    with torch.inference_mode():
        audio = codec.from_indices(torch.from_numpy(np.ascontiguousarray(codes)).long()[None])[0, 0].float().numpy()
    sf.write(a.out, audio, 44100)
    rms = np.sqrt(np.mean(audio ** 2))
    frame_rms = np.sqrt((audio[: len(audio) // 2048 * 2048].reshape(-1, 2048) ** 2).mean(1))
    print(f"{codes.shape[1]} frames -> {len(audio) / 44100:.2f}s in {time.time() - t0:.1f}s; rms {rms:.3f}; quiet frames (<0.01): {(frame_rms < 0.01).sum()} of {len(frame_rms)}")

if __name__ == "__main__":
    main()
