# Speech reliability study: full against balanced

Question: does every sentence come out as speech, every time, in the full
model and in the int8 "balanced" mode, under different conditions?

Method. 33 conditions in `slices/{a,b,c}.json`: short lines, two-sentence
lines, long paragraphs, numbers and dates, abbreviations, all caps,
performance cues, multi-speaker tags, Spanish, French and German, a tongue
twister, a cloned voice, low and high expressiveness, plus two control
conditions that are not the app's defaults (retries off, and the reference's
300-byte chunking). Five seeds each, the same seeds in both modes, 330 rows.
Generation through the `drama` CLI from main (built 2026-10-18 08:36), audio
through the reference PyTorch codec, transcription with Whisper small.en, and
the three non-English conditions rescored with the multilingual `small`
model (`rescore_multilingual.py`), because small.en translates French and
German into English. A row is `ok` when the word error rate is under the
condition's threshold (0.15, up to 0.3 for numbers and non-English) and eos
was reached; otherwise `dropped`, `garbled` or `runaway`. The machine was
shared by three workers and one engineering agent, so wall times are not a
benchmark. Rows: `eval/out/reliability/rescored/*.jsonl`; summary with
`eval/.venv/bin/python -I eval/reliability.py --summary eval/out/reliability/rescored`.

## Results

| rows | full | balanced |
| --- | --- | --- |
| shipped defaults, every condition but the two controls | 146 of 150 (97.3%) | 140 of 150 (93.3%) |
| prose only (the above without performance-cue lines) | 130 of 130 (100%) | 126 of 130 (96.9%) |
| performance-cue lines (`[laughs]`, `<whisper>`, opening styles) | 16 of 20 | 14 of 20 |
| retries off (control) | 1 of 5 | 3 of 5 |
| reference 300-byte chunking (control) | 7 of 10 | 9 of 10 |

Paired by seed on the 130 prose rows: full is fine where balanced fails in
4 pairs, and balanced is never fine where full fails. Over all 165 pairs the
mean word error rate differs by 0.011 with a standard error of 0.013.

Balanced's four prose misses: "Yes." spoken as "Yeah" over 3.3 s; the
two-sentence weather line stopping after its first sentence despite two
retries; the invoice line stopping after "due within 30 days" and running to
the cap; the all-caps warning spoken as "Warning" after two retries.

## What fails in both modes

Performance cues. "[with bright eager excitement] We won! We actually won!
[laughs] I can't believe it!" lost its last clause after `[laughs]` in 2 of 5
full runs and 2 of 5 balanced runs, and on the cloned voice one `[laughs]`
became a laughter loop that ran to the frame cap (31 s in the CLI, whose cap
is a fixed 600 frames; the app's text-derived cap would stop it near 8 s).
The retry rule does not catch these because the spoken duration is close to
the expected one.

The two control conditions fail as the earlier hill-climb predicted: without
retries, or with the reference chunker, two-sentence lines often stop after
the first sentence. The shipped defaults (sentence per turn plus retries)
are what makes prose reliable.

## Conclusion

Full precision speaks plain prose reliably: 130 of 130 prose rows across
every condition, including three languages, numbers, abbreviations, all caps,
cloned voices and both expressiveness extremes. Balanced is close but not
equal: 4 prose misses in 130 against none, all in the direction of balanced
being worse. The int8 mode stays an Advanced option with that stated in its
help, and is not recommended for narration that is exported. Compact is held
to the same study before it can be offered. Performance cues are the one
area where both modes drop content a few times in twenty; that is a model
behaviour the app should warn about next to the cue palette until a better
guard exists.

## Performance cues: investigation

Question: why do cue lines lose content, and which chunking, retry, cap or
sampling policy keeps every sentence without hurting prose?

Method. `eval/cues/run.py` runs every condition in `eval/cues/conditions.json`
over the eight lines in `eval/cues/lines.json` (the three study lines that
failed, plus a cue at the start, in the middle, at the end, a paired
`<whisper>` span, and a cue plus three words) through the `drama` CLI under
the app's configuration (`--text-cap`: the text-derived frame cap, the Core
ML silence probe, the 2 s silence stop, the duration retry), scored as the
study scores (PyTorch codec, Whisper small.en, word error rate, eos) plus one
column the duration rule cannot see: whether every sentence of the line is in
the transcript (`sentences_present`: a sentence counts when some window of the
transcript's words is within a third of its words in edits). Five seeds per
condition (the three study lines have ten under `a`), one engine process at a
time after the first hour, on a machine shared with other work, so wall times
are not a benchmark. Rows: `eval/out/cues/*.jsonl`; summary with
`eval/.venv/bin/python -I eval/cues/run.py --summary`. What other
implementations do is in `eval/cues/REVIEW.md`.

Two measurements came first. The laughter loop is visible in the code stream:
the repetition redraw forbids a main token within 16 frames, so the loop
settles into a period of 17; over the last 48 main tokens the match at lag 17
or 34 was 1.00 on both study loops and never above 0.75 on 267 speech runs
(`MainTokenLoop`: window 48, lags 17 to 40, threshold 0.9). A pause count
against the sentence count does not separate failures from good runs at any
gap length (3 to 35 good rows flagged out of 154, both loops missed), so the
transcript-free retry signal is eos plus duration, not pauses. The silence
stop is also weaker than designed: one probe is a 512-frame Core ML decode,
1.4 s on an idle machine and 3 to 5 s under load, so it fires 50 to 180
frames late and a short cue chunk reaches its cap first; the audio tail goes
silent about 140 frames before the main tokens settle to a constant, so a
code-space silence stop would be no faster.

| variant | what changes | runs | all sentences | WER under 0.15 | eos | runaways | retries | mean s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | today's defaults | 55 | 44 (80%) | 42 (76%) | 38 | 17 | 8 | 6.14 |
| b | a cue and its sentence join the previous chunk | 40 | 26 (65%) | 26 (65%) | 22 | 18 | 10 | 6.37 |
| b2 | a cue ends the previous chunk, its sentence opens the next | 40 | 35 (88%) | 32 (80%) | 25 | 15 | 4 | 5.88 |
| c | the whole line as one chunk | 40 | 27 (68%) | 27 (68%) | 22 | 18 | 10 | 6.35 |
| d | the Space's sampling: temperature 1.0, top-p 0.9, top-k off | 40 | 39 (98%) | 34 (85%) | 35 | 5 | 3 | 4.99 |
| e | retry every chunk that misses eos; loop guard | 43 | 39 (91%) | 38 (88%) | 38 | 5 | 25 | 5.57 |
| e2 | retry a chunk that misses eos only under 1.0 of expected seconds, or on a loop | 40 | 37 (92%) | 36 (90%) | 32 | 8 | 9 | 5.73 |
| f | `[with bright eager excitement]` written as `[excited]` (that line only) | 5 | 5 (100%) | 5 (100%) | 5 | 0 | 1 | 5.57 |
| g | a cap floor like the Space's: 8 s per chunk, 1.6 x for slow directions | 40 | 34 (85%) | 33 (82%) | 31 | 9 | 5 | 6.10 |
| he2 | e2 plus 1.5 s of cap per bracket cue (shipped) | 40 | 38 (95%) | 36 (90%) | 38 | 2 | 11 | 5.76 |
| dhe2 | he2 with the Space's sampling | 40 | 39 (98%) | 35 (88%) | 39 | 1 | 3 | 5.08 |

Per line, all sentences present (a has ten seeds on the first three lines):

| line | a | b | b2 | c | d | e2 | g | he2 | dhe2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cue-style: `[with bright eager excitement] We won! We actually won! [laughs] I can't believe it!` | 4/10 | 3/5 | 3/5 | 4/5 | 5/5 | 4/5 | 3/5 | 4/5 | 5/5 |
| clone-cues (cloned voice): `[sighs] Fine. If we miss the train, it's on you. [laughs] Well, maybe that's fair.` | 8/10 | 0/5 | 5/5 | 0/5 | 5/5 | 4/5 | 4/5 | 5/5 | 5/5 |
| cue-whisper: `Promise me <whisper>you won't tell her</whisper>. [pause] Not a word. Not even a hint.` | 10/10 | 3/5 | 5/5 | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| cue-start: `[gasps] Someone has been in this room, and they left in a hurry.` | 4/5 | 4/5 | 4/5 | 4/5 | 4/5 | 5/5 | 4/5 | 5/5 | 4/5 |
| cue-middle: `I told you it would work. [chuckles] You never listen to me, do you?` | 5/5 | 3/5 | 5/5 | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| cue-end: `Well, that went about as badly as it possibly could have. [sighs]` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| cue-span: `<whisper>Don't move. It's right behind you.</whisper> Now run, as fast as you can!` | 3/5 | 3/5 | 3/5 | 3/5 | 5/5 | 4/5 | 3/5 | 4/5 | 5/5 |
| cue-three: `[laughs] Oh, stop it.` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

What the rows say. The model's failure is to stop at a cue that follows a
sentence end: "We actually won! [laughs] I can't believe it!" loses its last
clause whether the cue opens a chunk or sits mid-chunk, and longer chunks (b,
c) make it worse, because more text sits behind the cue. Sentence chunking is
protective, not the cause; the 24-byte fold that gives "[laughs] I can't
believe it!" its own chunk is not what loses it. Three things help. Retrying
a chunk that stopped without eos recovers most dropped tails, but only when
gated by duration (e2): a trailing `[sighs]` never reaches eos and is
complete at 1.2 to 1.5 of its expectation, whereas every no-eos attempt that
lost a sentence spoke 0.64 to 0.90 of it or was a loop. The loop guard ends
laughter loops in the code stream, about 3 s in, without a decode. A cap
allowance of 1.5 s per bracket cue stops slow cue chunks from being cut at a
cap that counted no time for the laugh or the pause (cue-whisper and
cue-three). The Space's sampling (d) is the single largest effect on content,
98% against 80%, and dhe2 reaches 39 of 40 with three retries; its word error
rate is a little higher (Whisper writes the laughs it now performs as "Ha
ha", and one "Oh, stop it" became "I'll stop it"), it speaks more slowly into
the cap, and its effect on the 130-row prose baseline was not measured here,
so it is documented as a per-request option and not made the default. Cue
normalisation (f) has nothing to normalise to, because bracket cues are free
text by design (REVIEW.md); the five-seed probe of `[excited]` is suggestive
only.

Shipped (he2): the loop guard on by default, the retry rule extended to a
chunk that misses eos while under its expected duration or stopped by the
loop guard, and 1.5 s of frame cap per cue, each behind a `GenerationConfig`
field with the measured default; the old behaviour stays reachable from the
CLI for controls (`--eos-retry-below 0 --no-loop-guard --cap-per-cue 0`).
Sampling stays as it was.

Prose regression (`eval/cues/prose-lines.json`, the six short and
two-sentence items of slice `a`, five seeds, rows in `eval/out/cues-prose`):
today's defaults 30 of 30 with every sentence present (29 eos), he2 29 of 30
(28 eos), eight retries each. The one miss is the two-sentence weather line
at seed 1, a single 81-byte chunk that stops after its first sentence in most
attempts under both variants (four of five seeds retry it); under today's
rule its third attempt happened to speak the whole line, under he2 the first
attempt ended earlier (the loop guard ends a constant-token tail before the
cap), so the next attempts drew a different random path and all three
stopped short. The same seed is fine where the paths coincide (seeds 2 to
5 are identical in both columns). A seed-path difference, not a rule the
prose lines meet; the per-chunk retry rule itself is unchanged for a chunk
that reaches eos.

The balanced mode was removed from the engine after this study, and compact
was never offered, so `reliability.py` now runs the one precision and its
rows carry no mode; the summary reads such rows as `full`.
