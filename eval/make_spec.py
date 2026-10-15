"""Write the example prompt specs under eval/specs; clone.json also encodes its clip."""

import argparse
import functools
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dump_reference import DEFAULT_CHECKPOINT
from oracle.dac.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
from oracle.dac.rvq import DownsampleResidualVectorQuantize

EVAL_DIR = Path(__file__).resolve().parent
SPECS, DATA = EVAL_DIR / "specs", EVAL_DIR / "data"
PLAIN_TEXT = "<|speaker:0|> The quick brown fox jumps over the lazy dog."
CLIP_WORDS = "Every morning I walk to the little bakery at the end of our street."
CLONE_TEXT = "<|speaker:0|> I hope you have a wonderful day."
SAMPLE_RATE = 44100


def window_transformer() -> WindowLimitedTransformer:
    config = ModelArgs(
        block_size=2048,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    return WindowLimitedTransformer(config, input_dim=1024, window_size=128, causal=True)


def load_codec(path: Path) -> DAC:
    """fish-speech configs/modded_dac_vq.yaml, built without hydra."""
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=[2, 2],
        pre_module=window_transformer(),
        post_module=window_transformer(),
        semantic_codebook_size=4096,
    )
    codec = DAC(
        sample_rate=SAMPLE_RATE,
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        quantizer=quantizer,
        transformer_general_config=functools.partial(
            ModelArgs,
            block_size=8192,
            n_local_heads=-1,
            head_dim=64,
            rope_base=10000,
            norm_eps=1e-5,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            channels_first=True,
        ),
    )
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    if any("generator." in key for key in state):
        state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
    result = codec.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys:
        raise ValueError(f"codec weights missing: {result.missing_keys[:8]}")
    return codec.eval()


def make_clip(wav: Path) -> None:
    caf = wav.with_suffix(".caf")
    subprocess.run(
        ["say", "-o", str(caf), f"--data-format=LEF32@{SAMPLE_RATE}", CLIP_WORDS],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(caf),
         "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)],
        check=True,
    )
    caf.unlink()


@torch.inference_mode()
def encode(codec: DAC, wav: Path) -> np.ndarray:
    audio, rate = sf.read(wav, dtype="float32")
    assert rate == SAMPLE_RATE and audio.ndim == 1, (rate, audio.shape)
    codes, _ = codec.encode(torch.from_numpy(audio)[None, None])
    return codes[0].long().numpy()


def write_spec(name: str, spec: dict) -> None:
    SPECS.mkdir(exist_ok=True)
    (SPECS / name).write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {SPECS / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    write_spec("plain.json", {"text": PLAIN_TEXT, "references": [], "history": []})

    DATA.mkdir(exist_ok=True)
    wav, codes_path = DATA / "clip.wav", DATA / "clip_codes.npy"
    make_clip(wav)
    codes = encode(load_codec(args.checkpoint / "codec.pth"), wav)
    np.save(codes_path, codes)
    print(f"wrote {codes_path}: {codes.dtype} {codes.shape}")
    write_spec(
        "clone.json",
        {
            "text": CLONE_TEXT,
            "references": [
                {
                    "transcript": f"<|speaker:0|> {CLIP_WORDS}",
                    "codes": f"../{DATA.name}/{codes_path.name}",
                }
            ],
            "history": [],
        },
    )


if __name__ == "__main__":
    main()
