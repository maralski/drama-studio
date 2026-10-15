from functools import partial
from pathlib import Path

import torch

from .modded_dac import DAC, ModelArgs, WindowLimitedTransformer
from .rvq import DownsampleResidualVectorQuantize

CODEC_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--fishaudio--drama-3/snapshots"
    / "e0a9218aa48888270f06ef1ed9ea67622c0e17d2/codec.pth"
)


def _quantizer_transformer() -> WindowLimitedTransformer:
    return WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=ModelArgs(
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
        ),
    )


def build_dac() -> DAC:
    """The model `modded_dac_vq.yaml` describes, instantiated without hydra."""
    general = partial(
        ModelArgs,
        block_size=8192,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=[2, 2],
        post_module=_quantizer_transformer(),
        pre_module=_quantizer_transformer(),
        semantic_codebook_size=4096,
    )
    return DAC(
        sample_rate=44100,
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=general,
        quantizer=quantizer,
    )


def load_dac(path: Path = CODEC_PATH) -> DAC:
    """Same steps as `load_model` in fish_speech/models/dac/inference.py, on the CPU."""
    model = build_dac()
    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected = [k for k in result.unexpected_keys if not k.endswith(("causal_mask", "freqs_cis"))]
    if result.missing_keys or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {result.missing_keys}, unexpected {unexpected}")
    return model.eval()
