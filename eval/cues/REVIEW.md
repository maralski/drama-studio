# How other Drama 3 front ends handle performance cues

Read-only review, 2026-10-18, for the cue-dropout investigation (`REPORT.md`
in `eval/reliability/`, section "Performance cues: investigation"). Sources
were read in full at the pins named below; quotes are under fifteen words.

## Overview

Nobody else cuts a line as finely as we do, and nobody else checks that the
text was covered. The Space generates the whole script (up to 600 characters)
as one take, SGLang's HTTP route does the same with no cap on length, and the
hosted API cuts at 300 characters with a 50-character minimum. Our sentence
mode makes "[laughs] Well, maybe that's fair." a generation of its own, and the
model is free to laugh and stop. On sampling, every Drama 3 reference uses
temperature 1.0 and top-p 0.9; the Space also removes the top-k cap and says
why. On guards, the only common element is the 16-frame repetition-aware
redraw, which we already match; the Space adds a silence stall of 4 s and a
single retry of a silent opening, and nothing anywhere compares the audio with
the text.

## Fish Audio documentation and model cards

Sources: https://docs.fish.audio/features/text-to-speech/drama-3.md,
https://docs.fish.audio/developer-guide/core-features/emotions.md,
https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech.md,
https://huggingface.co/fishaudio/drama-3/raw/main/README.md,
https://huggingface.co/fishaudio/s2-pro/raw/main/README.md.

- Bracket cues are free text. The Drama 3 page says "Put a direction in square
  brackets before the words it applies to." There is no list of point tags to
  map to; the S2 Pro card says "15,000+ unique tags supported".
- The paired tags are a fixed list: `<whisper>`, `<emphasis>`, `<soft>`,
  `<fast>`, `<slow>`, `<stress>`, `<forceful>`. Language tags such as
  `<french>` are experimental.
- Cues at the start of a line are the documented norm. The page's first
  example opens with `[barely holding it together, forcing a smile]`, and the
  multi-speaker example puts a direction right after each `<|speaker:N|>`.
  Sounds go "at the point where they happen", as in `I'm fine. [shaky breath]
  Really.`
- The S2 emotions page (S2, not Drama 3) tells writers to add text after a
  sound effect, with the example `[excited][laughing] We won! Ha ha!`, and says
  `Ha,ha,ha` alone produces laughter. A model trained on such scripts saying
  "Ha ha" aloud for `[laughs]` is a plausible origin of our spoken "Ha ha";
  inferred, not shown.
- Sampling: the hosted `/v1/tts` schema defaults to temperature 0.7, top-p
  0.7, repetition penalty 1.2, `max_new_tokens` 1024 per chunk, `chunk_length`
  300 with `min_chunk_length` 50 and `condition_on_previous_chunks` true, and
  has no top-k field. The Drama 3 model card benchmarks with temperature 1.0,
  top-p 0.9, top-k 30 and `max_new_tokens` 500.

## SGLang-omni

Sources, at `main` 53e0bfd (2026-10-17):
https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html and,
under https://raw.githubusercontent.com/sgl-project/sglang-omni/main/,
`sglang_omni/models/fishaudio_drama3/{prompt,speech_request,sampling,decode_state,config,request_builders}.py`
and `sglang_omni/serve/speech_ws.py`.

- Prompt: the same `build_drama3_prompt` layout we port (system instruction,
  `[Reference n]\nText: … \nSpeech: ` plus codes, user turn with
  `<|speaker:0|> `, `<|im_start|>assistant\n<|voice|>`), every literal
  tokenized on its own.
- Cues: no handling at all. Brackets and `<whisper>` reach the tokenizer as
  text.
- Sampling: `DEFAULT_TEMPERATURE = 1.0`, `DEFAULT_TOP_P = 0.9`,
  `DEFAULT_TOP_K = 30`; `repetition_penalty` other than 1.0 is refused. Top-p
  drops the token that crosses the mass and keeps rank 0, as ours does.
- Loop guard: only the repetition-aware redraw, `RAS_WINDOW_FRAMES = 16`,
  temperature 1.0, top-k 30, top-p 0.95, main token only, eos never redrawn,
  both draws always made. Ours matches on every point.
- Max tokens: `min(2048, context − prompt rows)` when the caller omits it;
  nothing derives it from the text. Stop on eos only; no silence stop.
- Chunking: the HTTP route sends the whole input as one prompt. The WebSocket
  session cuts after `.!?。！？` with no minimum length, no folding and no
  protection of cue spans, so `[laughs] I can't believe it!` stays with its
  words, and each segment is an independent request with no history.
- Retries: none.
- The cookbook's own example is our cloned-voice line: `[sighs] Fine. If we
  miss it, it's on you. [laughs] Well, maybe that's fair.`

## The Hugging Face Space fishaudio/Drama-3

Sources, at c368729: https://huggingface.co/spaces/fishaudio/Drama-3/raw/main/
`server.py`, `drama3_engine.py`, `drama3_space_ext.py`, `gradio_ui.py`,
`studio_content.json`, `fish_speech/models/text2semantic/fish_omni/tower.py`,
`tests/test_generation_reliability.py`.

- Sampling: `temperature` 1.0, `top_p` 0.9, `top_k` default 0 meaning no cap;
  the request sends `top_k or ALL_CANDIDATES` with `ALL_CANDIDATES = 4097`.
  The comment says "top-k 30 (the SGLang Omni default) left 11 of 40 takes
  silent" and no cap left none; the UI hint says "Low values make takes go
  silent mid-line."
- Cues: the palette (`studio_content.json`) has point tags laughs, chuckles,
  sighs, gasps, pause, clears throat, hesitates; scoped tags whisper, slow,
  emphasis; and free-text styles such as "with bright eager excitement".
  Nothing normalises cues. Many examples open with a cue, including our
  `[with bright eager excitement] We won! …` line.
- Chunking: none. The script, at most `MAX_TEXT_CHARS = 600`, is one take.
- `spoken_text` strips `[…]` and `<tag>`/`</tag>` with
  `\[[^\[\]\n]*\]|</?[A-Za-z][^<>\n]{0,40}>`, keeps the words inside paired
  tags and collapses white space. It feeds the retake script, the transcripts
  of history clips and the duration budget. Our `DirectorsNote.spokenWords`
  does the same.
- Retake: the note (brackets, angle brackets and bars removed, 120 characters
  at most) is repeated before every sentence, `Direction: …\nLine: [note] s1
  [note] s2`, with up to two earlier takes as history turns. Ours matches.
- Duration budget (`token_budget`): word units `sum(max(1, len/5))`, seconds
  `max(18, 10 + 1.2 × units)`, times 1.6 when `slow|slowly|drawn-out|unhurried|long pause`
  appears in the text, the note or the character, capped at 1200 frames. The
  docstring calls it "a safety limit, not an estimate". Every take gets at
  least 18 s. Ours is 1.5 × (characters / 16) + 2 s per chunk, so
  `[laughs] Well, maybe that's fair.` gets 4.4 s where the Space allows 18 s.
- Silence guard: 50 ms frames, silent under RMS 0.003, internal pauses kept up
  to 0.75 s, a 0.15 s tail kept, and 4 s of continuous silence ends the take
  as a stall. Ours stops after 2 s of trailing silence.
- Retry: `for attempt in range(2)`, the second request identical. An attempt
  counts as complete once 2.5 s of kept audio exists or the take ends by eos
  with any audible audio. It retries once on a stall under 2.5 s, on eos with
  no audible audio, or on a length stop under 2.5 s. It never compares the
  audio with the text, so a laugh followed by eos counts as a complete
  opening.

## The fish-speech reference, v3.0.0-beta (`eval/oracle/`)

Sources: `eval/oracle/fish_omni/chunking.py`, `generate.py`, `tower.py`.

- `PROTECTED_SPAN` is `<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]`:
  a cut never lands inside a special token, a paired tag or a bracket cue, but
  a cut may land right before a cue, since the free text before it is broken
  at sentence ends. Our `Drama3Chunking.protectedSpan` is the same regex.
- `split_by_length` keeps text of 300 bytes or fewer whole; all our cue lines
  are shorter, so the reference generates each as one prompt.
- `generate_drama3` runs to eos or `max_new_tokens` (600 per chunk in the
  chunker, 2048 by default) with no guard against loops, silence or a missing
  tail beyond the 16-frame redraw in `tower.py`.

## The mlx-audio port

Sources: `mlx_audio/tts/models/fish_drama3/drama3.py` and `chunking.py` in the
scratch copy.

- Same `UNCUTTABLE` regex as the reference, same 300-byte packing, same
  pinned-plus-latest history; no cue normalisation. The docstring says
  "Delivery cues go inline, e.g. "[whispers]"" and refuses an `instruct`
  argument.
- Same sampling recipe (0.7, 30, 0.7) and the same repetition-aware redraw,
  `DEFAULT_MAX_FRAMES = 1024` per chunk, stop on eos only. No silence stop, no
  retry, no loop guard.

## What we do that nobody else does

| area | us | elsewhere |
| --- | --- | --- |
| chunking | one sentence per turn, fragments under 24 bytes folded forward, so a cue can open a chunk of three words | whole line (Space, SGLang HTTP, reference, mlx-audio for lines under 300 bytes); 300 characters with a 50-character minimum (hosted API); sentences with no minimum (SGLang WebSocket) |
| sampling | temperature 0.7, top-k 30, top-p 0.7 | 1.0, 30, 0.9 (SGLang, model card); 1.0, no top-k, 0.9 (Space, with silent takes as the stated reason) |
| repetition redraw | 16 frames, 1.0, 30, 0.95 | identical everywhere |
| frame cap | 1.5 × characters/16 + 2 s per chunk | at least 18 s per take, 1.6 × for slow directions (Space); 500 to 2048 fixed elsewhere |
| silence stop | 2 s trailing | 4 s stall with 0.75 s pauses kept (Space); none elsewhere |
| retry | twice, when spoken seconds under 0.6 of characters/16 | once, same request, silent or stalled opening only (Space); none elsewhere |
| text coverage check | duration only | none |
| cue normalisation | none | none; bracket cues are free text by design |

Variant (f) of the experiment, cue normalisation, therefore has no official
list to map onto; the only mapping that exists is the Space's own palette of
styles, which are themselves free text.
