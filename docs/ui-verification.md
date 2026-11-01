# UI verification

This pass changes the editor, settings, take layout and waveform display.
Audio generation, sampling, retry policy and codec output are unchanged.
The parser reuses prepared scripts, matches ASCII dialogue with a compiled
Foundation regex and retains Swift Regex grapheme matching for Unicode input.
Window titles stop after five words. Export availability checks no longer
concatenate document audio just to decide whether a button should be enabled.

## Validation results

- App and SDK documentation suite: 202 tests, zero failures, four optional
  checks skipped. The separate release run exercised the two timing checks.
- Layout audit: 26 rendered views, zero failures. The capture helper now enables
  intrinsic sizing before measuring windows that fit their content.
- Release parser and responsiveness checks: 16 tests, zero failures.
- Final editor comparison: ten successful test-process runs, five per side.
- Live Clear Script, Undo and Redo restored and removed the exact test text.
- Local Markdown link checks and `git diff --check` passed.

## Common workflows

The app tests exercise text and document import, screenplay conversion and
casting, generation through the mock synthesizer, retakes, playback and seeking,
export selection, audio and caption exports, project save/reopen, recovery,
voice recording states, setup and update installation cancellation.

The layout audit renders the studio at its 900 by 640 point minimum, including
one and eight takes, document narration, long messages, settings, help and dialogs.
The mock produces placeholder audio, so these checks do not measure speech quality.

The interactive preview check also covered Settings > Updates > General with
a real model directory, the app-menu update action, screenplay validation and
its help popover, Clear Script, mock generation, New Take, playback and changing
the exported take. The preview used a separate bundle and settings directory;
the installed application was not replaced.

## Requirements evidence

- `Package.swift` sets macOS 15 as the minimum platform.
- `MacRequirements` checks for Apple silicon and warns below 64 GiB of memory.
  The warning can be acknowledged; 64 GiB is a recommendation, not an enforced
  or measured minimum.
- `ModelDownloadPlan.drama3` contains 41,502,592,187 bytes of downloads.
  `ModelSetup.reservedBytes` adds 7,000,000,000 bytes for setup, totaling
  48.5 decimal GB before user audio and projects.
- `ContentView.minimumSize` requires 900 by 640 points for the studio.
- This UI verification uses an Apple M3 Max with 16 CPU cores, 128 GiB of
  memory and macOS 27.0.1. It cannot establish behavior on smaller machines.
  The separate 16 GiB hardware test below failed during GPU prompt processing.
  32 and 64 GiB hardware remain untested. `DRAMA_STUDIO_MEMORY_GB` simulates
  only the onboarding warning.

## 32 GiB memory trial

On 1 November 2026, a release CLI run generated and decoded the sentence
"Hello. This is a short test of Drama Studio with a thirty-two gigabyte memory
budget." on the same 128 GiB Mac. It used the installed model, packed weights,
tuned schedule, seed 42 and a 300-frame cap. It completed with exit status zero,
139 frames and 6.455 seconds of mono 44.1 kHz float WAV audio. The engine retried
once after a short initial result, then reached EOS. The WAV contained finite,
non-silent samples, with peak amplitude 0.493 and RMS 0.110. Its spoken content
was not independently transcribed.

A parent process sampled `proc_pid_rusage` every 10 ms and would terminate the
child if resident size or physical footprint exceeded 32 GiB. Across 3,891
samples, peak reported resident size was 3.281 GiB and peak physical footprint
was 2.137 GiB. The monitor did not terminate the run. Total elapsed time was
47 seconds, including model loading, codec setup and retry.

This is **not a verified 32 GiB hardware result**. macOS rejected the small `RLIMIT_RSS` probe with `EINVAL`. The SDK
defines `RLIMIT_RSS` as an alias for `RLIMIT_AS`, which limits virtual address
space rather than forcing excess resident memory to page out. The monitor observes process accounting and
cannot impose a system-wide physical-memory budget or establish the full cost
of file-backed Metal mappings and shared caches. The host still had 128 GiB,
and the engine maps roughly 41 GB of model weights. Consequently, neither the
successful generation nor these process counters establish that the app works
on a 32 GiB Mac. The 64 GiB recommendation remains unchanged.

## 16 GiB Mac Mini hardware test

On 1 November 2026, two generation attempts on Peter's Apple M1 Mac Mini
failed with Metal command-buffer error 8, `Insufficient Memory`, during prompt
processing. Neither attempt produced frames or a WAV. This is a failure of the
current model and engine on this tested configuration, not a measurement of
the minimum memory needed on every Mac.

The Mini runs macOS 27.0.1, build 26A434, with 17,179,869,184 bytes of unified
memory. Metal reports a recommended working set of 12,713,115,648 bytes.
The test used the same arm64 release CLI and resource bundle as the local
trial. The full Fish Audio snapshot, 41,502,593,757 bytes, and 1,587,864,272
bytes of Core ML codec packages were copied from the local Mac over SSH.
The Mini downloaded no model files. Copies completed successfully with rsync.

The prompt was "Hello. This is a short test of Drama Studio on a Mac with
sixteen gigabytes of memory." Both attempts used seed 42, a 300-frame cap,
normal generation defaults and the default kernel schedule. The Mini had no
device-specific tuning cache. No memory cap, quantization change or OS memory
setting was applied. A parent monitor allowed normal paging and imposed only
a ten-minute timeout; it did not terminate either run.

| Attempt | Packed cache | Result | Elapsed |
| --- | --- | --- | --- |
| First | Built during startup | GPU out of memory during prefill; exit 133 | 30 seconds |
| Repeat | Loaded existing cache | Same error; exit 133 | 16 seconds |

The engine propagated the GPU error to the CLI, whose top-level Swift error
handler terminated with signal 5. These durations include setup and failed
work; they are not generation throughput measurements. System swap use rose
from 182.56 MiB before the first run to 1,582.31 MiB afterward and remained at
that level after the repeat. These are whole-system readings, not exclusively
the model's memory. Paging occurred but did not make this workload succeed.

Raw evidence: [first-run log](verification/mini-16gb/first/generation.log),
[repeat log](verification/mini-16gb/repeat/generation.log),
[initial system state](verification/mini-16gb/first/before.txt), and
[final system state](verification/mini-16gb/repeat/after.txt).

The copied model remains at `/Users/m/drama-memory-test/model` on `mini`.
The executable, runner and remote logs remain under `/Users/m/drama-memory-test`.
Codec packages and the generated packed cache remain in that Mac's
`~/Library/Application Support/DramaStudio` directory for future tests.

## Documentation review

Reviewed the repository's 16 Markdown files, plus `SettingHelp` and the setup
requirements text, against the relevant app, engine, API, packaging and test
code. Corrected update navigation, model discovery and restart instructions,
partial-streaming examples, codec model count, requirements and obsolete
precision-mode descriptions. Marked older timing and reliability studies as
historical instead of presenting their configurations as current defaults.

Applied the unslop pass to project prose. Preserved technical identifiers,
quoted research evidence, licensing notices and numerical study results.
Local Markdown link targets were checked. External services and historical
engine benchmarks were not rerun during this UI pass.

## Reproduction

```sh
swift test --filter 'DramaStudioTests|SDKUsageTests'
DRAMA_LAYOUT_AUDIT=/tmp/drama-layout DRAMA_STUDIO_HOME=/tmp/drama-ui-home \
  swift test --filter AppLayoutTests
DRAMA_UI_BENCH=1 swift test -c release --filter AppResponsivenessTests
```

The optional responsiveness tests use a 400-line screenplay and a three-minute
waveform. They assert output correctness while measuring UI work; they never
load the speech model. Release optimization is required for timing comparisons.

## Responsiveness results

Measured on 1 November 2026 in release builds. The Mac was shared with a virtual
machine, the window server and other applications; its initial load average was
9.8 on 16 cores. Results are specific to these workloads, not a claim that every
interaction is 80% faster.

| Work | Before, median and range | After, median and range | Reduction |
| --- | --- | --- | --- |
| 400-line screenplay edit and UI layout | 44.78 ms (39.64 to 46.99) | 10.18 ms (9.31 to 15.83) | 77.3% |
| Three-minute waveform peak calculation | 4.49 ms (4.43 to 4.58) | 0.46 ms (0.44 to 0.50) | 89.7% |

The editor comparison alternated the original uncached script preparation and
original title derivation against the final implementation across five pairs
of test-process runs. Each run warmed up once, then performed five batches of
20 edits. The table reports the median and range of those five per-run medians,
per edit. Each edit replaces the 400-line input, reads the same status and
casting controls, and requests NSHostingView layout. Assertions verify the
prepared speaker names and generation availability. This measures synchronous
editor work, not keyboard-event delivery or time until pixels appear on screen.

| Pair | Before, ms/edit | After, ms/edit |
| --- | --- | --- |
| 1 | 46.99 | 12.72 |
| 2 | 39.64 | 9.31 |
| 3 | 42.35 | 15.83 |
| 4 | 44.78 | 10.18 |
| 5 | 45.33 | 9.44 |

The baseline profile showed repeated title derivation through regular-expression
compilation and string splitting. Script preparation also repeated across the
footer, casting and transport consumers. The final implementation prepares the
script once per edit and reuses the result. ASCII dialogue uses a compiled
Foundation matcher; Unicode input retains the original grapheme-aware matcher.
Full text replacement and AppKit layout remain part of the measured work.

The waveform comparison alternated the original scalar reduction and Accelerate
within one process, five measured rounds after warmup. Every result matched the
baseline, with 160 normalized peaks for 7,938,000 samples. The vector reduction
still scans the samples, so memory reads remain necessary. The UI now schedules
that work outside the main actor and discards stale results. This timing is the
calculation alone; it does not include drawing or scheduling delay.

The 80% target was exceeded for waveform preparation, but not for the measured
editor path. No audio-generation optimization was attempted.
