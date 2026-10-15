"""Check the Core ML codec against the PyTorch reference.

    tools/.venv/bin/python -I tools/verify_codec.py [--models DIR] [--work DIR] [--fixtures DIR]

Reports, per clip: encoder code agreement per codebook, decoder error against the fp32
reference for the same codes, round-trip SNR, the same numbers for the rewritten PyTorch
model (proving the rewrites), compute-unit placement from MLComputePlan, and timings
per compute-unit setting on a 10 s clip. With --fixtures it also writes the Swift test
fixtures.
"""

import argparse
import collections
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_codec import DEFAULT_OUT, FRAME_SIZES  # noqa: E402
from dac_reference import load_dac  # noqa: E402
from dac_reference.exportable import SAMPLES_PER_FRAME, CodecDecoder, CodecEncoder, make_exportable  # noqa: E402

SR = 44100
SPEECH = {
    "daniel_en_gb": ("Daniel", "The quick brown fox jumps over the lazy dog. Meanwhile, the orchestra tuned "
                     "their instruments, and the audience slowly settled into their seats for the evening."),
    "samantha_en_us": ("Samantha", "Could you send me the report by Friday? I would really appreciate it, "
                       "because the meeting starts early on Monday morning."),
    "anna_de_de": ("Anna", "Guten Morgen! Heute scheint die Sonne, und wir gehen spazieren."),
}
COMPUTE_UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]
TIMING_SECONDS = 10.0


def make_clips(work: Path) -> dict[str, np.ndarray]:
    clips = {}
    t = np.arange(int(3.0 * SR)) / SR
    f0, f1 = 50.0, 16000.0
    k = np.log(f1 / f0) / 3.0
    clips["chirp_3s"] = (0.5 * np.sin(2 * np.pi * f0 * (np.exp(k * t) - 1) / k)).astype(np.float32)
    for name, (voice, text) in SPEECH.items():
        aiff, wav = work / f"{name}.aiff", work / f"{name}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), "--data-format=BEF32@44100", text], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_f32le", str(wav)], check=True)
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == SR and audio.ndim == 1, (sr, audio.shape)
        clips[name] = audio
    return clips


def padded_frames(frames: int) -> int:
    return next(s for s in FRAME_SIZES if s >= frames)


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    err = np.sum((ref.astype(np.float64) - test) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / err))


class CoreMLCodec:
    def __init__(self, models: Path, units: str):
        cu = getattr(ct.ComputeUnit, units)
        self.encoder = ct.models.MLModel(str(models / "DramaCodecEncoder.mlpackage"), compute_units=cu)
        self.decoder = ct.models.MLModel(str(models / "DramaCodecDecoder.mlpackage"), compute_units=cu)

    def encode(self, audio: np.ndarray) -> np.ndarray:
        frames = -(-len(audio) // SAMPLES_PER_FRAME)
        x = np.zeros((1, 1, padded_frames(frames) * SAMPLES_PER_FRAME), np.float32)
        x[0, 0, : len(audio)] = audio
        return self.encoder.predict({"audio": x})["codes"][0, :, :frames]

    def decode(self, codes: np.ndarray) -> np.ndarray:
        frames = codes.shape[1]
        x = np.zeros((1, 10, padded_frames(frames)), np.int32)
        x[0, :, :frames] = codes
        return self.decoder.predict({"codes": x})["audio"][0, : frames * SAMPLES_PER_FRAME]


class TorchCodec:
    def __init__(self, model, rewritten: bool):
        self.model, self.rewritten = model, rewritten
        if rewritten:
            self.enc, self.dec = CodecEncoder(model).eval(), CodecDecoder(model).eval()

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(audio)[None, None]
        if self.rewritten:
            frames = -(-len(audio) // SAMPLES_PER_FRAME)
            x = torch.nn.functional.pad(x, (0, frames * SAMPLES_PER_FRAME - len(audio)))
            return self.enc(x)[0].numpy()
        return self.model.encode(x)[0][0].numpy().astype(np.int32)

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.rewritten:
            return self.dec(torch.from_numpy(codes.astype(np.int32))[None])[0].numpy()
        return self.model.from_indices(torch.from_numpy(codes.astype(np.int64))[None])[0, 0].numpy()


def compare(name, audio, oracle: TorchCodec, test, label: str) -> dict:
    ref_codes = oracle.encode(audio)
    codes = test.encode(audio)
    ref_audio = oracle.decode(ref_codes)
    out = test.decode(ref_codes)
    roundtrip = test.decode(codes)[: len(audio)]
    row = {
        "clip": name,
        "impl": label,
        "frames": int(ref_codes.shape[1]),
        "code_match": [round(float(v), 4) for v in (codes == ref_codes).mean(axis=1)],
        "decode_max_abs": float(np.abs(out - ref_audio).max()),
        "decode_snr_db": round(snr_db(ref_audio, out), 2),
        "roundtrip_snr_db": round(snr_db(audio, roundtrip), 2),
        "reference_roundtrip_snr_db": round(snr_db(audio, ref_audio[: len(audio)]), 2),
    }
    print(json.dumps(row))
    return row


def check_shapes(oracle: TorchCodec, coreml: CoreMLCodec, codes: np.ndarray) -> None:
    """Decode every enumerated size (codes tiled to fill it) and compare with the reference."""
    for t in FRAME_SIZES:
        tiled = np.tile(codes, (1, -(-t // codes.shape[1])))[:, :t]
        ref = oracle.decode(tiled)
        out = coreml.decode(tiled)
        enc_ref = oracle.encode(ref)
        enc = coreml.encode(ref)
        print(json.dumps({"shape_frames": t, "decode_snr_db": round(snr_db(ref, out), 2),
                          "decode_max_abs": float(np.abs(out - ref).max()),
                          "encode_code_match": [round(float(v), 4) for v in (enc == enc_ref).mean(axis=1)]}))


def compute_plan(package: Path) -> None:
    compiled = Path(ct.models.utils.compile_model(str(package)))
    try:
        for units in ["CPU_AND_NE", "ALL"]:
            plan = ct.models.compute_plan.MLComputePlan.load_from_path(
                path=str(compiled), compute_units=getattr(ct.ComputeUnit, units))
            program = plan.model_structure.program
            by_device = collections.Counter()
            cost_by_device = collections.Counter()
            off_ane = collections.Counter()
            for fn_name, fn in program.functions.items():
                for op in fn.block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if usage is None:
                        continue
                    device = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
                    cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                    by_device[device] += 1
                    cost_by_device[device] += cost.weight if cost else 0.0
                    if device != "NeuralEngine":
                        supported = sorted(type(d).__name__.replace("ML", "").replace("ComputeDevice", "")
                                           for d in usage.supported_compute_devices)
                        off_ane[(op.operator_name, device, ",".join(supported))] += 1
            total = sum(cost_by_device.values()) or 1.0
            print(f"{package.name} [{units}] ops per device: {dict(by_device)}; "
                  f"estimated cost share: { {k: round(v / total, 4) for k, v in cost_by_device.items()} }")
            for (op_name, device, supported), n in sorted(off_ane.items(), key=lambda kv: -kv[1]):
                print(f"    off ANE: {n:4d} x {op_name} on {device} (supported: {supported})")
    finally:
        shutil.rmtree(compiled, ignore_errors=True)


def timings(models: Path, audio: np.ndarray) -> None:
    for units in COMPUTE_UNITS:
        start = time.perf_counter()
        codec = CoreMLCodec(models, units)
        load = time.perf_counter() - start
        codes = codec.encode(audio)
        codec.decode(codes)
        enc, dec = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            codes = codec.encode(audio)
            t1 = time.perf_counter()
            codec.decode(codes)
            enc.append(t1 - t0)
            dec.append(time.perf_counter() - t1)
        print(json.dumps({"compute_units": units, "audio_s": round(len(audio) / SR, 2),
                          "padded_frames": padded_frames(-(-len(audio) // SAMPLES_PER_FRAME)),
                          "load_s": round(load, 2), "encode_median_ms": round(1000 * float(np.median(enc)), 1),
                          "decode_median_ms": round(1000 * float(np.median(dec)), 1)}))
        del codec


def write_fixtures(out: Path, oracle: TorchCodec, audio: np.ndarray, frames: int = 64) -> None:
    clip = audio[: frames * SAMPLES_PER_FRAME]
    codes = oracle.encode(clip)
    decoded = oracle.decode(codes)
    reencoded = oracle.encode(decoded)
    out.mkdir(parents=True, exist_ok=True)
    (out / "codec_codes.json").write_text(json.dumps({
        "codes": codes.astype(int).tolist(),
        "reencoded_codebook0": reencoded[0].astype(int).tolist(),
    }) + "\n")
    (out / "codec_audio_ref.json").write_text(json.dumps({
        "samples": [float(f"{v:.8g}") for v in decoded[:SAMPLES_PER_FRAME]],
    }) + "\n")
    print(f"wrote fixtures to {out}: {codes.shape[1]} frames; reference re-encode keeps "
          f"{(reencoded[0] == codes[0]).mean():.3f} of codebook 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="drama_codec_"))
    work.mkdir(parents=True, exist_ok=True)

    clips = make_clips(work)
    oracle = TorchCodec(load_dac(), rewritten=False)
    if args.fixtures:
        write_fixtures(args.fixtures, oracle, clips["daniel_en_gb"])

    print(f"torch {torch.__version__}, coremltools {ct.__version__}, models in {args.models}")
    rewritten = TorchCodec(make_exportable(load_dac(), max_frames=max(FRAME_SIZES)), rewritten=True)
    coreml = CoreMLCodec(args.models, "ALL")
    for name, audio in clips.items():
        compare(name, audio, oracle, rewritten, "torch-rewritten-fp32")
        compare(name, audio, oracle, coreml, "coreml-ALL")
    check_shapes(oracle, coreml, oracle.encode(clips["daniel_en_gb"]))

    for package in ["DramaCodecEncoder.mlpackage", "DramaCodecDecoder.mlpackage"]:
        compute_plan(args.models / package)
    if not args.skip_timing:
        speech = np.concatenate([clips[k] for k in SPEECH])
        timings(args.models, np.resize(speech, int(TIMING_SECONDS * SR)).astype(np.float32))


if __name__ == "__main__":
    main()
