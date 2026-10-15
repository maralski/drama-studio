"""Export the Drama 3 DAC codec to Core ML.

    tools/.venv/bin/python -I tools/convert_codec.py [--out DIR] [--only encoder|decoder]

Writes DramaCodecEncoder.mlpackage and DramaCodecDecoder.mlpackage (fp16, mlprogram,
macOS 15) with enumerated frame counts FRAME_SIZES.
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

FRAME_SIZES = (16, 32, 64, 128, 256, 512, 768)
DEFAULT_OUT = Path.home() / "Library/Application Support/DramaStudio/Codec"
TRACE_FRAMES = 32


def _convert(module: torch.nn.Module, example: torch.Tensor, inp: ct.TensorType, out: ct.TensorType, name: str):
    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[inp],
        outputs=[out],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=True,
    )
    mlmodel.short_description = f"Drama 3 DAC codec {name}, 44.1 kHz, 2048 samples per frame"
    mlmodel.user_defined_metadata["frame_sizes"] = ",".join(map(str, FRAME_SIZES))
    mlmodel.user_defined_metadata["samples_per_frame"] = str(SAMPLES_PER_FRAME)
    mlmodel.user_defined_metadata["source"] = str(CODEC_PATH.name)
    return mlmodel


def convert_encoder(model) -> ct.models.MLModel:
    shapes = [(1, 1, t * SAMPLES_PER_FRAME) for t in FRAME_SIZES]
    return _convert(
        CodecEncoder(model).eval(),
        torch.zeros(1, 1, TRACE_FRAMES * SAMPLES_PER_FRAME),
        ct.TensorType(name="audio", shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[0]), dtype=np.float32),
        ct.TensorType(name="codes", dtype=np.int32),
        "encoder",
    )


def convert_decoder(model) -> ct.models.MLModel:
    shapes = [(1, 10, t) for t in FRAME_SIZES]
    return _convert(
        CodecDecoder(model).eval(),
        torch.zeros(1, 10, TRACE_FRAMES, dtype=torch.int32),
        ct.TensorType(name="codes", shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[0]), dtype=np.int32),
        ct.TensorType(name="audio", dtype=np.float32),
        "decoder",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", choices=["encoder", "decoder"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = make_exportable(load_dac(), max_frames=max(FRAME_SIZES))
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
