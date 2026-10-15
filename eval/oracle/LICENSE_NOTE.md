# Notice

These Fish Audio Materials are licensed under the Fish Audio Community License,
Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.

The full agreement is in `LICENSE` next to this file.

## Origin

Copied from [fish-speech](https://github.com/fishaudio/fish-speech) at tag
`v3.0.0-beta`:

| here | upstream |
| --- | --- |
| `fish_omni/{config,layers,moe,fp8_moe,backbone,tower,model,generate,chunking}.py` | `fish_speech/models/text2semantic/fish_omni/` |
| `errors.py`, `sampling.py` | `fish_speech/models/text2semantic/` |
| `dac/{modded_dac,rvq}.py` | `fish_speech/models/dac/` |

## Changes

- Absolute `fish_speech.models.text2semantic...` imports became relative
  imports within `eval/oracle`.
- The `if __name__ == "__main__":` demo blocks at the end of
  `dac/modded_dac.py` and `dac/rvq.py` were removed. They need hydra, omegaconf
  and CUDA.

No model code was changed. The Triton kernels in `fish_omni/fp8_moe.py` stay
as they are. They are guarded by an `ImportError` check and are not used on
macOS.
