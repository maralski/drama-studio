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
