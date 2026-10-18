# DAC reference (vendored)

The PyTorch oracle for the Core ML codec. `convert_codec.py` beside it and
`tools/verify_codec.py` import it; nothing in the app does. Both live in
`Sources/DramaEngine/CodecConversion/`, which ships in DramaEngine's resource
bundle, so `DramaSetup.prepare` can run the conversion with a Python you name.

## Origin

- `modded_dac.py`, `rvq.py`, `modded_dac_vq.yaml`: fish-speech tag `v3.0.0-beta`
  (commit `1ff3c3ed2f80d3106a83909dd25b83129c3d816f`,
  https://github.com/fishaudio/fish-speech), files
  `fish_speech/models/dac/{modded_dac,rvq}.py` and
  `fish_speech/configs/modded_dac_vq.yaml`. License: Fish Audio Community
  License, copied as `LICENSE-fish-speech`.
- `dac_nn.py`: descript-audio-codec 1.0.0 (https://github.com/descriptinc/descript-audio-codec),
  `dac/nn/layers.py` and `dac/nn/quantize.py`. License: MIT, copied as
  `LICENSE-descript-audio-codec`.

## Edits to the copies

- `modded_dac.py`: imports `Snake1d`, `WNConv1d`, `WNConvTranspose1d` from
  `dac_nn` instead of `dac.nn.layers`; drops the `audiotools` imports; `DAC`
  derives from `nn.Module` instead of `BaseModel, CodecMixin` (save/load and
  chunked-compress helpers), so the `self.delay = self.get_delay()` line goes too;
  the `__main__` demo is dropped.
- `rvq.py`: imports `ResidualVectorQuantize` from `dac_nn`; `__main__` demo dropped.
- `dac_nn.py`: the two descript files merged; docstrings, `from_latents`, and
  the `__main__` demo dropped.

The numerics are untouched. `__init__.py` builds the model the YAML describes
without hydra and loads `codec.pth` the way `load_model` in
`fish_speech/models/dac/inference.py` does.

## Core ML rewrites

`exportable.py` holds every change made for conversion, as replacement modules
swapped in by `make_exportable`. Each is algebraically equal to the reference
op; `verify_codec.py` checks the rewritten fp32 model against the reference.
