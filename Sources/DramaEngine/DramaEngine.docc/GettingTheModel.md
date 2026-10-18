# Getting the model

Download the weights, build the Core ML codec models once, and tune the
kernels for the Mac.

## Overview

The engine reads three things from disk: the Drama 3 snapshot from Hugging
Face, two Core ML codec models built from that snapshot's `codec.pth`, and a
kernel schedule tuned for the Mac on the first load. ``DramaSetup`` reports
each one and builds the last two.

| What | Where | Size | Who makes it |
| --- | --- | --- | --- |
| Snapshot: 8 weight shards, `config.json`, `tokenizer.json`, `codec.pth` | `~/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots/<id>/` | 41.5 GB (shards 39.6 GB) | `hf download` |
| `DramaCodecEncoder.mlpackage` (fp32), `DramaCodecDecoder.mlpackage` (fp16) | ``DramaCodec/defaultDirectory``, `~/Library/Application Support/DramaStudio/Codec/` | 809 MB and 353 MB | `convert_codec.py`, once |
| Compiled `.mlmodelc` beside each package | the same folder | about the same | ``DramaCodec`` on its first load |
| Tuned kernel schedule, one JSON file | ``DramaSynthesizer/defaultTuningDirectory``, `~/Library/Application Support/DramaStudio/Tuning/` | about 4 KB | the engine's first load on a Mac, about 30 s |

### Memory

The weights are memory-mapped, not copied. A loaded engine keeps about 40 GB
resident: the 39.6 GB of weights plus about 2 GB of state and codec. Only a
128 GB M3 Max has been tested; how a Mac with less unified memory than the
weights behaves has not been measured.

### Download the snapshot

Run the Hugging Face CLI through `uv`:

```bash
uvx --from huggingface_hub hf download fishaudio/drama-3
```

``ModelLocation/huggingFaceCache(repo:)`` finds the newest snapshot it
downloaded. To keep the snapshot elsewhere, pass the folder that holds
`config.json` to ``ModelLocation/init(directory:)``.

### Build the codec models

The conversion runs PyTorch on the CPU once, for a few minutes, and needs a
Python with torch 2.7.1 and coremltools. DramaEngine ships the script,
`convert_codec.py`, and the reference codec code it imports, in its resource
bundle. ``DramaSetup/conversionScript`` is its location.

1. Create a Python environment with the packages:

   ```bash
   uv venv --python 3.12 ~/.cache/drama-codec-venv
   uv pip install --python ~/.cache/drama-codec-venv/bin/python "torch==2.7.1" coremltools numpy einops soundfile
   ```

2. Run the conversion through ``DramaSetup``:

   ```swift
   let python = FileManager.default.homeDirectoryForCurrentUser
       .appending(path: ".cache/drama-codec-venv/bin/python")
   try await DramaSetup.prepare(location: location, python: python) { line in print(line) }
   ```

``DramaSetup/prepare(location:codecDirectory:python:tuningDirectory:progress:)``
never downloads Python. Without a `python` it throws
``DramaSetupError/pythonRequired(command:)`` with the exact shell commands,
which ``DramaSetup/codecConversionCommand(python:location:codecDirectory:)``
also returns. In a checkout of the repository the same script runs as
`Sources/DramaEngine/CodecConversion/convert_codec.py --codec <snapshot>/codec.pth`.

### Tune the kernels

The first load on a Mac times each kernel family on the real weights, keeps
a faster candidate only when its output stays within a relative L2 of 1e-5 of
the default's, and caches the result. Later loads read the cache and measure
nothing. A new macOS version, a new chip, or new kernel sources invalidate
the cache, and the next load tunes again. On an M3 Max the tuned schedule
makes a one-row forward about 20 percent faster.

``DramaSetup/prepare(location:codecDirectory:python:tuningDirectory:progress:)``
also tunes ahead of time by loading the model once. It returns without
loading anything when ``SetupReport/isPrepared`` is already true.

### Check an install

``DramaSetup/check(location:codecDirectory:tuningDirectory:)`` reads files
only. It names a missing shard, a missing codec model, or a Mac without a
tuned schedule, each with a fix a person can act on:

```swift
let report = DramaSetup.check(location: .huggingFaceCache())
for item in report.items {
    print(item.summary, item.fix ?? "")
}
```

### App Sandbox

A sandboxed app cannot read `~/.cache/huggingface` or start Python. Ask the
person for the snapshot folder with an open panel, keep a security-scoped
bookmark, and pass the folder to ``ModelLocation/init(directory:)``. Build
the codec models outside the app, or ship them, and pass their folder as
`codecDirectory`. Inside the sandbox, ``DramaCodec/defaultDirectory`` and
``DramaSynthesizer/defaultTuningDirectory`` resolve inside the app's
container.
