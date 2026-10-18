"""Export the Drama 3 DAC codec to Core ML.

    tools/.venv/bin/python -I Sources/DramaEngine/CodecConversion/convert_codec.py [--codec PATH] [--out DIR] [--only encoder|decoder]

Writes DramaCodecEncoder.mlpackage (fp32) and DramaCodecDecoder.mlpackage (fp16), both
mlprogram for macOS 15 with one static shape of WINDOW_FRAMES frames. Callers pad
shorter inputs on the right and run longer ones in windows of CONTEXT_FRAMES of left
context plus WINDOW_FRAMES - CONTEXT_FRAMES new frames (see DramaCodec in Swift).
"""

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dac_reference import CODEC_PATH, load_dac  # noqa: E402
from dac_reference.exportable import SAMPLES_PER_FRAME, CodecDecoder, CodecEncoder, make_exportable  # noqa: E402

WINDOW_FRAMES = 512
CONTEXT_FRAMES = 256
DEFAULT_OUT = Path.home() / "Library/Application Support/DramaStudio/Codec"


def _convert(module, example, inp, out, precision, name):
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[inp],
        outputs=[out],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    mlmodel.short_description = f"Drama 3 DAC codec {name}, 44.1 kHz, {SAMPLES_PER_FRAME} samples per frame"
    mlmodel.user_defined_metadata["window_frames"] = str(WINDOW_FRAMES)
    mlmodel.user_defined_metadata["context_frames"] = str(CONTEXT_FRAMES)
    mlmodel.user_defined_metadata["samples_per_frame"] = str(SAMPLES_PER_FRAME)
    mlmodel.user_defined_metadata["source"] = CODEC_PATH.name
    return mlmodel


def convert_encoder(model) -> ct.models.MLModel:
    # fp32: in fp16 a third of the residual codes flip, and encoding a voice is a one-off cost
    shape = (1, 1, WINDOW_FRAMES * SAMPLES_PER_FRAME)
    return _convert(
        CodecEncoder(model).eval(),
        torch.zeros(shape),
        ct.TensorType(name="audio", shape=shape, dtype=np.float32),
        ct.TensorType(name="codes", dtype=np.int32),
        ct.precision.FLOAT32,
        "encoder",
    )


def convert_decoder(model) -> ct.models.MLModel:
    shape = (1, 10, WINDOW_FRAMES)
    return _convert(
        CodecDecoder(model).eval(),
        torch.zeros(shape, dtype=torch.int32),
        ct.TensorType(name="codes", shape=shape, dtype=np.int32),
        ct.TensorType(name="audio", dtype=np.float32),
        ct.precision.FLOAT16,
        "decoder",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", type=Path, default=CODEC_PATH, help="codec.pth from the Drama 3 snapshot")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", choices=["encoder", "decoder"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = make_exportable(load_dac(args.codec), max_frames=WINDOW_FRAMES)
    jobs = {"encoder": (convert_encoder, "DramaCodecEncoder"), "decoder": (convert_decoder, "DramaCodecDecoder")}
    for key, (fn, name) in jobs.items():
        if args.only and args.only != key:
            continue
        start = time.time()
        path = args.out / f"{name}.mlpackage"
        fn(model).save(str(path))
        print(f"wrote {path} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
