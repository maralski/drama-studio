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
so it is documented as a per-request option and not made the default (the
prose measurement follows in "The Space's sampling on prose"). Cue
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

## The Space's sampling on prose

Question: the cue investigation found the Hugging Face Space's sampling
(temperature 1.0, top-p 0.9, top-k off) the single largest effect on cue
lines, 39 of 40 against 38 of 40, but did not measure it on prose. Does it
lose a prose row, what does it do to the word error rate and to speed, and
should it be the default, the top of the Expressiveness slider, or a
per-request option only?

Method. The 33 conditions of slices `a`, `b` and `c`, five seeds, through
`reliability.py` with `--drama-flags "--temperature 1.0 --top-p 0.9 --top-k
4097"` (4097 is the candidate count, so top-k keeps everything) appended after
each condition's own flags, so the four expressiveness-slider conditions ran
under the Space's sampling too; everything else as the study ran it: sentence
chunking, the CLI's fixed 600-frame cap, the Core ML silence stop, the
duration retry. Binary built from this branch (main at f55dfb4 plus the
harness flag), so the three cue guards shipped after the study (loop guard,
eos-gated retry, cap per cue) were on; the rows they are compared with ran
before those guards, which matters for the cue and control rows and is
checked below for prose. One engine process at a time, slices one after
another (16:11 to 16:58, 47 minutes), on a machine with a load of 5 to 28
from other sessions' work and the app open but idle, so wall times are not a
benchmark. Non-English rows rescored with the multilingual model as before.
Rows: `eval/out/reliability-space/rescored/*.jsonl`, paired by condition and
seed against `eval/out/reliability/rescored/*__full.jsonl` with
`eval/.venv/bin/python -I eval/sampling_compare.py eval/out/reliability/rescored eval/out/reliability-space/rescored`.
"All sentences" is the cue study's check (every sentence within a third of
its words in edits of some window of the transcript).

| rows, paired by seed | shipped sampling | the Space's sampling |
| --- | --- | --- |
| prose, 130: ok | 130 | 130 (none lost, none won) |
| prose: all sentences | 130 | 128 |
| prose: mean WER | 0.018 | 0.025, paired shift +0.007 (standard error 0.004); 12 rows worse, 6 better |
| prose: retries | 19 | 5 |
| prose: spoken s / expected s | 1.26 | 1.16, shift -0.101 (standard error 0.029) |
| prose: frames per row | 158 | 150 (-5%) |
| prose, the 110 rows without the slider conditions: ok, WER | 110, 0.021 | 110, 0.029, shift +0.008 (standard error 0.004) |
| cue lines, 20: ok | 16 | 18 (1 lost, 3 won) |
| cue lines: all sentences | 17 | 20 |
| cue lines: eos, frames per row | 17, 210 | 20, 121 |
| controls, 15 (retries off, reference chunking): ok | 8 | 15 |
| every row, 165: ok, mean WER | 154, 0.051 | 163, 0.023 (shift -0.028, standard error 0.011) |

No prose row is lost. The two prose rows that fail the all-sentences check
are the abbreviation line at seeds 2 and 4, where Whisper wrote "10am" and
"US" for "10 a.m." and "U.S." (word error rates 0.29 and 0.19 under that
condition's 0.3 threshold); the words are there. The word error rate rises
by 0.007, about two standard errors, and the rows behind it are of two kinds.
Transcript spelling: "10am", "US", "rest rooms", "weekend", "40" for
"forty", "gray" for "grey", "Alright"; these go both ways (the lighthouse
line is worse at one seed and better at three). Spoken slips: "Warming" for
"WARNING" at two seeds, "pack" for "peck" at two seeds, "room 2 or 7" for
"room 207", and the invoice amount at one seed; the shipped sampling has two
such rows ("on 21 March" for "on the 21st of March", a merged "Not me, don't
look"). So the Space's sampling slurs a word in about 6 rows of 110 against
2, every one under the verdict threshold, which is the "more risk of slurred
words" the Expressiveness help already describes for the top of the slider.

What it gains. Retries fall from 19 to 5 on prose, all five of them the
duration rule on a chunk that spoke under 1.3 s (none the eos-gated rule, so
the guards do not explain the drop), which is 14 fewer chunk regenerations in
130 rows. The two-sentence weather line, which the shipped sampling stops
after its first sentence in most attempts (four of five seeds retry it), came
out whole on its first attempt at every seed, and so did the retries-off
control: 5 of 5 with eos on the first attempt against 1 of 5, and the
reference-chunking control 10 of 10 against 7. On the cue lines every
sentence is present in 20 of 20; two of the three rows won are the
cue-style line's two runaways, which the loop guard and the eos-gated retry
also address, so that gain is shared with the guards; the one row lost is the cloned cue line at seed 2,
where Whisper wrote the performed laugh as "Ha ha" and the word error rate
landed at 0.154 against a 0.15 threshold, with every word present.

Speed. The Space's sampling does not speak more slowly: spoken audio per
expected second falls from 1.26 to 1.16 on prose (standard error 0.029), and
frames per row fall 5% on prose and 43% on the cue lines, where the loops are
gone. Wall time per row went from 39.9 s to 13.3 s, but the earlier rows ran
on a machine shared by three workers and these at a load of 12, with the GPU
at 36.7 ms per frame against 25.6 to 27 ms idle, so the comparable figure is
the frame count. The text-derived cap is a separate matter: estimated at row
level (1.5 x expected plus 2 s per chunk), the number lines run close to it
under both samplings (two shipped rows over it, three Space rows, all
numbers or a long paragraph), because "$1,249.99" and "1987" take longer to
say than their characters predict.

Recommendation: the top of the Expressiveness slider, not the default.
The default's job is exported narration, where the shipped sampling has the
lowest word error rate and no reliability gap to close (130 of 130 both
ways), and a default change would move every API and MCP caller that never
sets `expressiveness`. The Space's sampling earns the top of the slider: it
keeps every sentence of cue lines, retries a quarter as often, and reads
more varied, at a cost of a slurred word in a few rows in a hundred. Done
on this branch: `GenerationConfig.make` runs from steady (0) through the
vendor's sampling (0.5, unchanged, still the default) to the Space's at 1,
temperature and top-p linear between, top-k 30 up to the middle and off
above it; the help, the API table, the MCP tool and the changelog say so;
unit tests pin the five points, the top end, and that temperature and top-p
are monotone and continuous across the range. The old top (0.9, 0.9, top-k
30) was measured fine in this study (15 of 15 at 0.008) and the Space's
point replaces it on the strength of 165 rows; the points between 0.5 and 1
are interpolated, not measured.

## Silence probe latency

Question: the investigation above measured that the 2 s silence stop fires 50
to 180 frames late, because each reading is one 512-frame Core ML decode, and
that short cue chunks reach their frame cap first. Can the reading be made
cheap enough for the stop to fire within about 16 frames of the audio going
silent, without losing content?

Instrument. Each chunk line the `drama` CLI logs now ends in `end=<why>` and
`silence_at=<frame>`, the frame at which the chunk's full decode first held
2 s of silence (`ChunkResult.silenceStopFrame`, a pure function with unit
tests). From it and `spoken=`, `eval/cues/probe_latency.py` reads how many
frames a chunk ran past the point at which its trailing silence reached 2 s:
for a chunk the stop ended, how late the stop fired; for one that eos or the
cap ended, how long the stop had already failed to act. A chunk whose audio
held 2 s of silence and then spoke again paused mid-line; the stop firing
there is the designed behaviour, and the retry rule covers it.

Candidates, measured with `drama codec-bench` (the codec alone, no Drama
model) on six generated clips of 70 to 600 frames, 1546 frames in all, under
a load average of 7 to 15 from other sessions' jobs:

| decoder shape | context | ms per call, median (range), 5 interleaved calls | silent-flag disagreements with the full decode |
| --- | --- | --- | --- |
| 512 (shipped decoder) | 256 | 1636 (1622 to 1653) | reference |
| 128 | 64 | 381 (374 to 404) | 0 of 1546 |
| 64 | 48 | 190 (187 to 205) | 0 of 1546 |
| 64 | 32 | 199 (187 to 211) | 2 of 1546, both at the 0.01 RMS edge (0.0096 against 0.0101) |
| 32 | 16 | 98 (96 to 127) | 3 of 1546, all at the edge |

The cost is linear in the frames decoded (about 3 ms a frame; the transformer
is 128 frames wide and the convolutions dominate), so the choice is how much
context a reading needs. 48 frames of context reproduce the full decode's
silent flags exactly on every clip, and the stop frame a reading every 16
frames would give is the first multiple of 16 at or after the full decode's in
every clip. Code-space proxies were not pursued: the investigation above
measured that the main tokens settle about 140 frames after the audio goes
silent. Core ML's enumerated shapes still crash the CPU backend (the 512
export's history), so the probe is a second export of the same fp16 decoder
at 64 frames, `DramaCodecProbe.mlpackage` (353 MB), which `convert_codec.py`
writes beside the other two and `DramaCodec` loads with them, checking the
frame count each file records. (The conversion venv recipe pins
`numpy<2.4`: numpy 2.5 turned a deprecation inside coremltools 9.0's `int`
cast into an error, so a fresh venv could not build any of the models.)

Shipped. `SilenceProbe` decodes only the frames not yet classified, 16 new
after 48 of context through the 64-frame shape every 16 frames, on its own
queue, and keeps the count of silent frames at the tail; the audio is dropped.
The chunk's audio is decoded once, in full windows, at the end of the attempt
(sentence mode) or as streamed partials (low latency), where before the probe
re-decoded a 512-frame window every 16 frames whenever the queue was free. A
reading costs 190 ms, about 7 frames at 38 frames/s, so a stop fires 7 to 23
frames after the audio holds 2 s of silence: the cadence rounds up to the
next multiple of 16, then the reading lands.

Measured on the cue harness, variant he2 (the shipped defaults), eight lines,
seeds 1 to 5, each binary the only engine process, load average 3 to 11 from
other sessions (`eval/out/cues-before`, `eval/out/cues-after`;
`eval/cues/probe_latency.py` prints the table):

| | before (`f55dfb4` plus the instrument) | after |
| --- | --- | --- |
| all sentences present | 38 of 40 | 39 of 40 |
| chunks ended by the silence stop | 0 of 60 | 4 of 60 |
| chunks that ended with 2 s of trailing silence | 2: eos 26 frames past the mark, cap 45 frames past | 4, all by the stop |
| frames late, stop-ended chunks | the stop never fired | 9, 17, 18, 21 (median 18) |
| chunks that paused 2 s and spoke again | 7 | 3 |
| eos reached | 38 | 36 |
| retries | 11 | 17 |
| mean audio | 5.71 s | 5.50 s |

The two before chunks that sat in 2 s of silence are the failure the
investigation described: cue-end seed 3 ran to its cap of 192 frames with
the last 89 silent, and cue-end seed 2 reached eos 70 silent frames in. After,
every chunk that ended in 2 s of silence was ended by the stop, 9 to 21
frames late, inside the 7 to 23 the reading's cost predicts. The one line
still missing a sentence, cue-span seed 3, missed the same sentence before
(runaway to the cap of 209 frames) and after (three attempts, the last ended
by the stop at 153 frames); the extra retries are stops at a mid-line pause
that the retry rule sent back, as designed. The gain of one line
(cue-style seed 3, garbled before) is a seed-path difference, not a claim.
Wall time is not a benchmark here (10.7 s a run before, 9.2 s after, under
different loads), but sentence mode now runs one full-window decode per
attempt instead of one per 16 frames, so the codec CPU per chunk falls.

Variant a (the pre-study defaults: no loop guard, no eos retry, no cap per
cue) at the same five seeds:

| | before | after |
| --- | --- | --- |
| all sentences present | 33 of 40 | 34 of 40 |
| chunks ended by the silence stop | 3 of 60 | 6 of 60 |
| frames late, stop-ended chunks | 86, 96; the third stop-ended chunk (cue-span seed 3) was cut by a stale reading, its audio having resumed speech with no trailing silence left when the stop landed | 10, 14, 17, 20, 22, 22 (median 18) |
| chunks that reached the cap already in 2 s of silence | 6, by 0 to 45 frames (median 28) | 2, by 0 and 12 frames |
| eos reached | 28 | 29 |

Without the cue guards the stop matters more, and the two numbers the
investigation measured by hand both move: the stop that fired 86 and 96
frames late now fires 10 to 22 frames late, and the chunks cut at their cap
while already silent fall from six to two, both with the cap inside the
reading's 7 to 23 frame window (one at the very frame the silence reached
2 s). The stale stop is a third failure the slow reading caused: a 512-frame
decode of a snapshot landed seconds later and ended a chunk that had paused
and spoken again in the meantime. A reading is now at most about 7 frames
old when it lands. Per line the two columns trade seeds (cue-start 4 to 3, cue-span 2 to
4), a seed-path difference again.

Not done: the machine was not idle for any of these runs (other sessions'
jobs held the load average between 3 and 15), so the 190 ms reading and the
9 to 22 frame lateness are upper bounds for an idle Mac, and the 512-frame
figure of 1.6 s is the under-load one. A reading every 8 frames would cut the
expected lateness to 7 to 15 frames at twice the probe CPU (a reading takes
190 of every 210 ms); the measured median of 18 met the brief, so the cadence
stays at 16. Chunks longer than 512 frames now decode in two full windows at
the end of the attempt instead of overlapping generation; no cue or prose
line reaches that length.

## Silence probe cadence

Question: the decode hillclimb (`.audit/hillclimb.tsv`, the W7 row) measured
that the silence probe's readings, Core ML decodes of the newest frames on
the CPU every 16 frames, slow the GPU's frames while they run, most likely
by sharing memory bandwidth. How often should the probe read? Fewer
readings cost stop latency, because the stop fires at the first reading
after the audio holds 2 s of silence.

Two levers were measured on the app's path (sentence chunking, the text
cap, the Core ML codec and its probe, `drama generate` without
`--no-codec`), one process at a time under `tools/with-gpu-lock` with the
app quit:

- the cadence: a reading every 16 frames (today), 32 or 64
  (`--probe-every N`);
- the start: from the chunk's first frames (today, `--probe-after-expected
  no`), or only from the chunk's expected end less the 2 s the stop needs.
  A chunk that speaks its text cannot hold 2 s of trailing silence before
  that frame, so an earlier reading can only end an attempt that has
  already failed.

Three changes go with the late start. A reading that has fallen more than
44 frames behind (the silent run the stop waits for) classifies only the
newest 44: either they are all silent and the stop fires whatever came
before, or one is not and older frames could not extend the tail; so the
gated first reading costs at most three probe calls, however long the
chunk. The loop guard keeps its own 16-frame cadence
(`FrameGenerator.loopGuardEveryFrames`), so the probe setting cannot move a
seed's frames except through where the silence stop fires. And a reading
that falls due while the previous one still runs now waits for the probe's
queue and covers only the frames up to its due point, where before it was
dropped (`SilenceProbeSchedule.due(at:)`). The measurements below found why
that matters.

### Passes

A first pass (23 October) is superseded. The loop under test checked the
probe's reading only on frames where a reading was offered, not on every
frame as the shipped code does, which added up to one cadence of lateness
to every stop; it also ran under another project's Metal benchmarks (80 to
500 ms GPU frames against 21 quiet). The second and third passes ran on
29 October after that fix, with the GPU quiet by the sampler label
(`eval/quiet_label.sh`: samplers 0.65 to 0.99 ms per frame across the
passes, under the 1 ms gate) and the app quit; a virtual machine used about
80 percent of a core throughout, and the backbone slowed from 12.4 to 16.8
ms per frame over each pass, the sustained-load slowdown the quiet-GPU pass
described. Pass 2 measured every cadence and the late start as first
written; pass 3 measured the late start with waiting readings, the shipped
policy.

### Cost

Five interleaved rounds (pass 2) and three (pass 3) of a four-sentence,
461-frame paragraph with a reference clip, seed 1 (`eval/probe_cadence.sh`,
`eval/probe_cadence_summary.py`). GPU time comes from `--trace-frames`,
probe seconds from the chunk log's new `probe=` field. Each delta pairs a
condition with the probe-off run of the same round, which cancels the
drift; each round ran the conditions in the order of the columns, so the
drift within a round penalises the later, gated ones:

| | off | 16 (today) | 32 | 64 | 16 from the expected end (pass 2) | 32 from the expected end | 16 from the expected end, waiting readings (shipped, pass 3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probe CPU, share of decode | 0 | 35% | 32% | 24% | 20% | 17% | 21% |
| GPU ms per frame over off, median [range] | 0 | +2.10 [1.78..2.85]; pass 3 +2.43 [1.20..2.43] | +1.43 [0.65..3.26] | +1.24 [0.06..3.83] | +0.54 [-0.15..3.81] | -0.10 [-0.53..4.22] | +0.26 [-0.02..0.79] |
| frames per second against off, median | | -7.4%; pass 3 -8.5% | -5.1% | -4.3% | -1.9% | +0.4% | -1.0% |
| frames byte-identical to today's | every round | | every round | every round | every round | every round | every round |

Today's cadence costs 2 to 2.4 ms of every frame in every round. Halving or
quartering the readings recovers less than half of it, since a reading of
more frames costs more calls; starting late recovers nearly all of it,
because the readings before the expected end are most of them.

### Lateness and coverage

The cue harness (`eval/cues/run.py`, eight cue lines) and the six prose
lines, seeds 1 to 5, through `eval/probe_cadence_harness.sh`;
`eval/cues/probe_latency.py` prints the lateness. Variant a has no loop
guard and no eos retry, so the stop matters most there:

| variant a | all sentences, cue | all sentences, prose | frames late, chunks the stop ended (cue; prose) | chunks at their cap already 2 s silent | byte-identical to today's (cue, prose) |
| --- | --- | --- | --- | --- | --- |
| 16 (today) | 34 of 40 | 27 of 30 | 8, 16, 17, 17, 21, 23; 11, 11, 24 | 2 | |
| 16, A/A rerun | 33 of 40 | | 10, 13, 18, 19, 19, 21, 22 | 2 | 30 |
| 32 | 33 of 40 | 30 of 30 | 32, 33, 35, 38, 42; none | 3 | 30, 24 |
| 64 | 34 of 40 | 29 of 30 | 22, 46; 83 | 6 | 30, 24 |
| 16 from the expected end, readings dropped (pass 2) | 32 of 40 | 29 of 30 | 10, 15, 16, 20, 23, 40, 48; 22 | 2 | 31, 24 |
| 16 from the expected end, waiting readings (shipped) | 34 of 40 | 29 of 30 | 11, 17, 21, 23, 26, 28; 22 | 2 | 32, 24 |

Under the shipped defaults (variant he2):

| he2 | all sentences, cue | all sentences, prose | frames late (cue; prose) | byte-identical to today's (cue, prose) |
| --- | --- | --- | --- | --- |
| 16 (today) | 38 of 40 | 29 of 30 | 10, 13, 19, 22; 23 | |
| 16, A/A rerun | 34 of 40 | 28 of 30 | 9, 17, 19, 19, 20, 27, 29; 10, 24 | 29, 27 |
| 16 from the expected end, readings dropped (pass 2) | 35 of 40 | 28 of 30 | 11, 13, 16, plus an eos and a cap 41 and 45 frames past the mark; 26, 45 | 30, 24 |
| 16 from the expected end, waiting readings (shipped) | 39 of 40 | 29 of 30 | 10, 15, 19, 22; none | 32, 26 |

The cadence sets the lateness: a reading every 32 frames puts the stop
about 35 frames late against 17, and every 64 frames triples the chunks
that run to their cap while already silent. The A/A rows give the
harness's own spread: today's cadence run twice scores 34 and 33 of 40 cue
lines under variant a and 38 and 34 under he2, with only 29 or 30 of 40
generations byte-identical between its own runs. Where a stop lands
depends on when its reading finishes, not only on the seed, and every
frame after that follows a different seed path, so a difference of three
or four lines between two settings is not a difference between them.

These lines barely exercise the late start: every sentence in them is
expected to last at most 3.6 s (77 frames), so the gate lands at or before
frame 48, where today's first reading lands anyway. A third set was built
for it, `eval/cues/long-lines.json`: eight lines from the reliability
slices and `eval/sentences.json` with 13 sentences of 4.2 to 16.5 s (the
lighthouse paragraph, long prose, numbers, a cloned voice), variant a,
seeds 1 to 5:

| long sentences | all sentences | probe CPU, share of decode | byte-identical to today's |
| --- | --- | --- | --- |
| 16 (today) | 38 of 40 | 39% | |
| 16, A/A rerun | 39 of 40 | 39% | 36 of 40 |
| 16 from the expected end, readings dropped (pass 2) | 39 of 40 | 26% | 36 of 40 |
| 16 from the expected end, waiting readings (shipped) | 39 of 40 | 28% | 35 of 40 |

### Dropped readings

The pass-2 gated rows carried stops 40 to 48 frames late that neither run
of today's cadence showed, and a cue-span chunk that went silent at the
same frame in all three runs (2 s of silence at frame 115) stopped at 136
and 137 under today's cadence and at 163 gated. The mechanism: a reading
offered while the previous one still ran was dropped, so the next one had
32 frames to classify, two probe calls of about 190 ms; at 45 frames per
second (a quiet GPU, and faster still once the probe stops slowing it) two
calls take longer than the 16 frames a cadence step generates, so the next
offer is dropped as well, and the probe stays at an effective cadence of 32
for the rest of the chunk. Any slow reading starts it: the gate's
catch-up, a load spike, or, under today's cadence, a busy CPU. Waiting
readings break the chain, since each reading after a slow one classifies
one cadence in one call and the probe is back on time within a reading
(`SilenceProbeTests.testASlowReadingDelaysTheNextOneInsteadOfDroppingIt`
simulates the queue and fails with readings of three then two calls when
the drop is put back). With them the late 40 to 48 frame stops are gone
and every set scores at or above today's first run.

One stop still moves on the long set: weather seed 1, an attempt that
stopped speaking after its first clause and held 2 s of silence at frame
78, one frame before its gate. Today's stop landed at frame 89; the gated
one lands at 95 (98 with dropped readings), because the gated first
reading decodes 44 frames instead of 16. Attempts that fall silent before
their expected end are failures the retry rule regenerates, so that cost
is about 0.15 s of GPU time on an attempt thrown away.

### Shipped

A reading every 16 frames from the chunk's expected end less 2 s, with
waiting readings (`SilenceProbePolicy.shipped`). On the paragraph it gives
back nearly all of the 2.1 to 2.4 ms per frame the probe cost the GPU (8.5
percent of frames per second down to 1.0), cuts the probe's CPU from about
a third of the decode to a fifth, keeps every frame of a generation the
same until a stop lands, and on all three harness sets scores within the
spread of today's cadence run against itself. The cadences of 32 and 64
frames were not taken: they recover less speed and make the stop two to
three times later.

Not done: the GPU cost is measured on one paragraph at one seed; a chunk
expected to last under about 4 s never reaches the gate, so short lines
keep today's few readings. The harness's run-to-run spread (four lines of
40 between two runs of the same setting) is measured and reported, not
removed; a stop decision that does not depend on timing would need the
generation loop to wait for each reading, which costs the GPU the time the
probe takes.
