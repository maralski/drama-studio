"""Check the Core ML codec against the PyTorch reference.

    tools/.venv/bin/python -I tools/verify_codec.py [--models DIR] [--work DIR] [--fixtures DIR]
                                                   [--decoder-units CPU_ONLY] [--timing-units ...]

Reports, per clip: encoder code agreement per codebook, decoder error against the fp32
reference for the same codes, and round-trip SNR, for both the Core ML models and the
rewritten PyTorch model (which proves the rewrites). Then the windowed long-input error,
compute-unit placement from MLComputePlan, and per-model timings on a 10 s clip for each
compute-unit setting. With --fixtures it also writes the Swift test fixtures.
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
from convert_codec import CONTEXT_FRAMES, DEFAULT_OUT, WINDOW_FRAMES  # noqa: E402
from dac_reference import load_dac  # noqa: E402
from dac_reference.exportable import SAMPLES_PER_FRAME, CodecDecoder, CodecEncoder, make_exportable  # noqa: E402

SR = 44100
F = SAMPLES_PER_FRAME
SPEECH = {
    "daniel_en_gb": ("Daniel", "The quick brown fox jumps over the lazy dog. Meanwhile, the orchestra tuned "
                     "their instruments, and the audience slowly settled into their seats for the evening."),
    "samantha_en_us": ("Samantha", "Could you send me the report by Friday? I would really appreciate it, "
                       "because the meeting starts early on Monday morning."),
    "anna_de_de": ("Anna", "Guten Morgen! Heute scheint die Sonne, und wir gehen spazieren."),
}
ALL_UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]
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


def windows(frames: int) -> list[tuple[int, int, int]]:
    """(start, keep, end) per model call; mirrors DramaCodec.windows in Swift."""
    out, keep = [], 0
    while keep < frames:
        start = 0 if keep == 0 else keep - CONTEXT_FRAMES
        end = min(frames, start + WINDOW_FRAMES)
        out.append((start, keep, end))
        keep = end
    return out


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    err = np.sum((ref.astype(np.float64) - test) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / err))


# coremltools frees a predict() input when Core ML resets its execution stream, which can
# happen after Python has dropped the array; holding every input avoids a use-after-free.
_INPUTS: list[np.ndarray] = []


def predict(model: ct.models.MLModel, name: str, value: np.ndarray, key: str) -> np.ndarray:
    _INPUTS.append(value)
    return model.predict({name: value})[key]


def load_mlmodel(path: Path, units: str) -> tuple[ct.models.MLModel, float]:
    start = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
    return model, time.perf_counter() - start


class CoreMLCodec:
    """Pads and windows exactly as the Swift DramaCodec does."""

    def __init__(self, encoder: ct.models.MLModel, decoder: ct.models.MLModel):
        self.encoder, self.decoder = encoder, decoder

    def encode(self, audio: np.ndarray) -> np.ndarray:
        frames = -(-len(audio) // F)
        codes = np.zeros((10, frames), np.int32)
        for start, keep, end in windows(frames):
            x = np.zeros((1, 1, WINDOW_FRAMES * F), np.float32)
            chunk = audio[start * F : end * F]
            x[0, 0, : len(chunk)] = chunk
            out = predict(self.encoder, "audio", x, "codes")[0]
            codes[:, keep:end] = out[:, keep - start : end - start]
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        frames = codes.shape[1]
        audio = np.zeros(frames * F, np.float32)
        for start, keep, end in windows(frames):
            x = np.zeros((1, 10, WINDOW_FRAMES), np.int32)
            x[0, :, : end - start] = codes[:, start:end]
            out = predict(self.decoder, "codes", x, "audio")[0]
            audio[keep * F : end * F] = out[(keep - start) * F : (end - start) * F]
        return audio


class TorchCodec:
    def __init__(self, model, rewritten: bool):
        self.model, self.rewritten = model, rewritten
        if rewritten:
            self.enc, self.dec = CodecEncoder(model).eval(), CodecDecoder(model).eval()

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(audio)[None, None]
        if self.rewritten:
            x = torch.nn.functional.pad(x, (0, -(-len(audio) // F) * F - len(audio)))
            return self.enc(x)[0].numpy()
        return self.model.encode(x)[0][0].numpy().astype(np.int32)

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.rewritten:
            return self.dec(torch.from_numpy(codes.astype(np.int32))[None])[0].numpy()
        return self.model.from_indices(torch.from_numpy(codes.astype(np.int64))[None])[0, 0].numpy()


def compare(name, audio, oracle: TorchCodec, test, label: str) -> None:
    ref_codes = oracle.encode(audio)
    codes = test.encode(audio)
    ref_audio = oracle.decode(ref_codes)
    out = test.decode(ref_codes)
    roundtrip = test.decode(codes)[: len(audio)]
    print(json.dumps({
        "clip": name,
        "impl": label,
        "frames": int(ref_codes.shape[1]),
        "code_match": [round(float(v), 4) for v in (codes == ref_codes).mean(axis=1)],
        "decode_max_abs": round(float(np.abs(out - ref_audio).max()), 5),
        "decode_snr_db": round(snr_db(ref_audio, out), 2),
        "roundtrip_snr_db": round(snr_db(audio, roundtrip), 2),
        "reference_roundtrip_snr_db": round(snr_db(audio, ref_audio[: len(audio)]), 2),
    }), flush=True)


def check_long(oracle: TorchCodec, coreml: CoreMLCodec, audio: np.ndarray) -> None:
    """A clip longer than one window: windowed Core ML against a single full reference pass."""
    ref_codes = oracle.encode(audio)
    ref = oracle.decode(ref_codes)
    out = coreml.decode(ref_codes)
    codes = coreml.encode(audio)
    print(json.dumps({"long_frames": int(ref_codes.shape[1]), "windows": len(windows(ref_codes.shape[1])),
                      "decode_max_abs": round(float(np.abs(out - ref).max()), 5),
                      "decode_snr_db": round(snr_db(ref, out), 2),
                      "code_match": [round(float(v), 4) for v in (codes == ref_codes).mean(axis=1)]}), flush=True)


def compute_plan(package: Path, units_list: list[str]) -> None:
    compiled = Path(ct.models.utils.compile_model(str(package)))
    try:
        for units in units_list:
            plan = ct.models.compute_plan.MLComputePlan.load_from_path(
                path=str(compiled), compute_units=getattr(ct.ComputeUnit, units))
            by_device, cost_by_device, off_ane = collections.Counter(), collections.Counter(), collections.Counter()
            for fn in plan.model_structure.program.functions.values():
                for op in fn.block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if usage is None:
                        continue
                    name = lambda d: type(d).__name__.replace("ML", "").replace("ComputeDevice", "")  # noqa: E731
                    device = name(usage.preferred_compute_device)
                    cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                    by_device[device] += 1
                    cost_by_device[device] += cost.weight if cost else 0.0
                    if device != "NeuralEngine":
                        supported = ",".join(sorted(name(d) for d in usage.supported_compute_devices))
                        off_ane[(op.operator_name, device, supported)] += 1
            total = sum(cost_by_device.values()) or 1.0
            print(f"{package.name} [{units}] ops per device {dict(by_device)}; estimated cost share "
                  f"{ {k: round(v / total, 4) for k, v in cost_by_device.items()} }", flush=True)
            for (op_name, device, supported), n in sorted(off_ane.items(), key=lambda kv: -kv[1]):
                print(f"    off ANE: {n:4d} x {op_name} on {device} (supported: {supported})")
    finally:
        shutil.rmtree(compiled, ignore_errors=True)


def timings(models: Path, oracle: TorchCodec, audio: np.ndarray, units_list: list[str]) -> None:
    ref_codes = oracle.encode(audio)
    ref_audio = oracle.decode(ref_codes)
    x = np.zeros((1, 1, WINDOW_FRAMES * F), np.float32)
    x[0, 0, : len(audio)] = audio
    c = np.zeros((1, 10, WINDOW_FRAMES), np.int32)
    c[0, :, : ref_codes.shape[1]] = ref_codes
    for units in units_list:
        for name, inp, value, key in [("DramaCodecEncoder", "audio", x, "codes"), ("DramaCodecDecoder", "codes", c, "audio")]:
            model, load = load_mlmodel(models / f"{name}.mlpackage", units)
            out = predict(model, inp, value, key)
            runs = []
            for _ in range(5):
                t0 = time.perf_counter()
                predict(model, inp, value, key)
                runs.append(time.perf_counter() - t0)
            row = {"model": name, "compute_units": units, "audio_s": round(len(audio) / SR, 2),
                   "window_frames": WINDOW_FRAMES, "load_s": round(load, 2),
                   "predict_median_ms": round(1000 * float(np.median(runs)), 1)}
            if key == "codes":
                row["code_match"] = [round(float(v), 4) for v in (out[0, :, : ref_codes.shape[1]] == ref_codes).mean(axis=1)]
            else:
                row["decode_snr_db"] = round(snr_db(ref_audio, out[0, : len(ref_audio)]), 2)
            print(json.dumps(row), flush=True)
            del model


def test_tone(frames: int = 64) -> np.ndarray:
    """A vibrato tone the Swift test regenerates sample for sample (float64 math, then float32)."""
    t = np.arange(frames * F, dtype=np.float64) / SR
    x = 0.3 * np.sin(2 * np.pi * (220 * t + 3 * np.sin(2 * np.pi * 5 * t))) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    return (x + 0.15 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)


def write_fixtures(out: Path, oracle: TorchCodec, audio: np.ndarray, frames: int = 64) -> None:
    clip = audio[: frames * F]
    codes = oracle.encode(clip)
    decoded = oracle.decode(codes)
    reencoded = oracle.encode(decoded)
    out.mkdir(parents=True, exist_ok=True)
    (out / "codec_codes.json").write_text(json.dumps({
        "codes": codes.astype(int).tolist(),
        "reencoded_codebook0": reencoded[0].astype(int).tolist(),
        "tone_codes": oracle.encode(test_tone(frames)).astype(int).tolist(),
    }) + "\n")
    (out / "codec_audio_ref.json").write_text(json.dumps({
        "samples": [float(f"{v:.8g}") for v in decoded[:F]],
    }) + "\n")
    print(f"wrote fixtures to {out}: {codes.shape[1]} frames; reference re-encode keeps "
          f"{(reencoded[0] == codes[0]).mean():.3f} of codebook 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--fixtures", type=Path)
    ap.add_argument("--decoder-units", default="CPU_ONLY", choices=ALL_UNITS)
    ap.add_argument("--timing-units", nargs="*", default=ALL_UNITS, choices=ALL_UNITS)
    args = ap.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="drama_codec_"))
    work.mkdir(parents=True, exist_ok=True)

    clips = make_clips(work)
    oracle = TorchCodec(load_dac(), rewritten=False)
    if args.fixtures:
        write_fixtures(args.fixtures, oracle, clips["daniel_en_gb"])

    print(f"torch {torch.__version__}, coremltools {ct.__version__}, models in {args.models}", flush=True)
    rewritten = TorchCodec(make_exportable(load_dac(), max_frames=2048), rewritten=True)
    coreml = CoreMLCodec(load_mlmodel(args.models / "DramaCodecEncoder.mlpackage", "CPU_ONLY")[0],
                         load_mlmodel(args.models / "DramaCodecDecoder.mlpackage", args.decoder_units)[0])
    for name, audio in clips.items():
        compare(name, audio, oracle, rewritten, "torch-rewritten-fp32")
        compare(name, audio, oracle, coreml, f"coreml enc CPU_ONLY fp32, dec {args.decoder_units} fp16")
    speech = np.concatenate([clips[k] for k in SPEECH] * 2)
    check_long(oracle, coreml, speech)

    for package in ["DramaCodecEncoder.mlpackage", "DramaCodecDecoder.mlpackage"]:
        compute_plan(args.models / package, ["CPU_AND_NE", "ALL"])
    if args.timing_units:
        timings(args.models, oracle, np.resize(speech, int(TIMING_SECONDS * SR)).astype(np.float32), args.timing_units)


if __name__ == "__main__":
    main()
