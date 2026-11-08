# Word-check evaluation

The word check decides whether a generated clip speaks its script line. This
directory measures that decision on five sets of recorded Drama 3 clips. Each
layer needs more than the one before it, so CI can run the cheap layers on any
macOS runner and the expensive ones where the models are installed.

Every layer runs through one entry point:

```
tools/word_check_eval/run.sh replay|audio|mutations|regenerate
```

It exits 0 on pass and non-zero on fail.

## The sets

| Set | Clips | Purpose |
|---|---|---|
| tuning | 200 | The matcher was tuned on these clips. |
| heldout1 | 166 | First held-out set. |
| heldout2 | 125 | Second held-out set. |
| cues | 132 | Lines with every kind of cue Drama Studio offers. |
| final | 126 | Hidden set that no implementer saw before the fix. |

`sets/<set>/items.json` defines each set. An item has an `id`, the `text` the
clip speaks, and a generation `seed`. An item with `check` is checked against
that line instead of `text`. An item with `audio` reuses another item's clip.
An item with `voice` uses that named library voice.

The id's second part says what the clip should get:

- `pos` and `spell` clips speak the line. The check should accept them.
- `neg` and `negspell` clips speak a different line, and `drop` clips leave a
  word out. The check must reject them.

`Tests/DramaEngineTests/Fixtures/word-check-sets.json` lists the sets and their
floors. A floor is the fewest correct clips a set may accept. Each floor is
today's count, so a change that loses a correct clip fails.

`sets/<set>/baseline.json` lists the clips the check accepted when the audio
was first recognized. For the v1 audio that is the check shipped before
pull request 12. The audio layer reports clips the baseline accepted and the
current check rejects. That report does not fail the run.

## Replay

**What it measures.** `WordCheckCorpusTests` replays each set's recorded
transcripts through the matcher. It uses the Apple-profile gate on the Apple
Speech transcript and the exact gate on the Whisper transcript. It fails when a
set accepts any wrong clip, names that clip, and fails when a set falls below
its `replayFloor`.

**What it needs.** macOS and Swift. No audio and no models. The run downloads
the pinned whisper framework, because the test target links it.

**Command.** `tools/word_check_eval/run.sh replay`

**Time.** The test takes under a second. The whole command took 55 seconds
on an M3 Max with a warm build. The other times below come from the same machine.

**Output.** One line per set, for example
`word-check final: 69/82 correct, 0/44 wrong accepted`.

| Set | Correct accepted | Wrong accepted |
|---|---|---|
| tuning | 132/150 | 0/50 |
| heldout1 | 87/102 | 0/64 |
| heldout2 | 57/78 | 0/47 |
| cues | 76/96 | 0/36 |
| final | 69/82 | 0/44 |

The replay hears each clip as recorded, without the silence the full check
appends after a clip that ends quietly, so it accepts fewer clips than the
full check. In `final`, two `rent` clips end with "month." at confidence
0.33. The replay rejects them, and the full check accepts them once the
recognizer hears an endpoint after the last word.

To add a set, record its fixture with `record.py` and add a row to
`word-check-sets.json`. The test needs no change.

`DRAMA_WORD_CHECK_BENCH=1` also times the matcher on the tuning set.

## Audio

**What it measures.** The production `LocalWordVerifier.check` on each set's
recorded clips. This includes the silence the check appends after a clip
that ends quietly, which the replay cannot do. The run fails when a set
accepts any wrong clip, falls below its `audioFloor`, or has a clip with no
decision.

**What it needs.** macOS 26 with the Apple Speech en_AU assets installed. The
pinned Whisper recognizer, which `tools/bootstrap_word_recognition.py`
downloads. The clip archive, which `fetch_audio.py` downloads.

**Command.**

```
tools/word_check_eval/run.sh audio
tools/word_check_eval/run.sh audio --archive /path/to/word-check-eval-audio-v1.zip
```

The run builds `word-check-lab` and the recognizer helper, stages the Whisper
models beside them, fetches the clips, and runs `word-check-lab decide` on all
five sets at once. `--archive` uses a local copy of the zip instead of the
download.

**Time.** The decisions took about 2 minutes for all five sets at once, and
the whole command about 3 minutes with a warm build. A first build adds the
full release build of the engine.

**The archive.** `audio.json` pins the zip's URL, size and SHA-256. The zip
holds one folder per set and one lossless FLAC per clip, 704 clips in
104 MB. `fetch_audio.py` rejects a zip whose size or hash differs, and unpacks
it into `cache/`, which git ignores. `pack_audio.py` builds the zip from WAV
folders. It checks that each FLAC decodes to the WAV's samples, and it writes
the same bytes each time it runs on the same WAVs.

**Output.** One line per set with the counts, the floor, and the clips lost
against the baseline, then `audio: PASS` or `audio: FAIL` with one `FAIL` line
per reason. Decisions land in `cache/decisions/<set>.json`.

| Set | Correct | Spelled | Wrong accepted | Floor |
|---|---|---|---|---|
| tuning | 82/90 | 52/60 | 0/50 | 134 |
| heldout1 | 53/60 | 34/42 | 0/64 | 87 |
| heldout2 | 34/45 | 23/33 | 0/47 | 57 |
| cues | 77/96 | | 0/36 | 77 |
| final | 52/60 | 19/22 | 0/44 | 71 |

## Mutations

**What it measures.** Whether the word-check tests notice a wrong change to the
matcher. `mutations.toml` lists 107 source edits. For each one the tool applies
the edit, runs `WordCoverageTests`, `BoundaryVerificationTests`,
`WordCheckCorpusTests` and `LocalWordVerifierTests`, and restores the file.

**What it needs.** macOS, Swift, Python 3.11 or later, and an APFS volume.
Each worker clones the checkout with copy-on-write, so the checkout itself is
never edited.

**Command.**

```
tools/word_check_eval/run.sh mutations --workers 6
tools/word_check_eval/run.sh mutations --workers 6 'no cardinal' 'no spellings'
tools/word_check_eval/run.sh mutations --check
```

Names limit the run to those mutations. `--check` only confirms that every
mutation's `old` text occurs exactly once, without running tests.

**Time.** Each worker first builds its clone and runs the tests unmutated.
Each mutation then costs one incremental build and one test run. Twelve
mutations on 6 workers took 229 seconds. A full run of all 107 took 1222
seconds with the scratch scripts this tool replaced.

**Output.** One line per mutation: `killed`, `SURVIVED`, `BUILD ERROR`,
`equivalent` or `EQUIVALENT KILLED`, with the failing tests. It ends with
`mutations: PASS` or `mutations: FAIL`. The run fails when a mutation survives
or fails to build. Three mutations change nothing the tests can observe, and
`equivalent = true` marks them. They must survive. If one is killed, the mark
is out of date and the run fails. The run also fails when the unmutated tests
fail.

When the matcher's source changes, a mutation whose `old` text no longer
occurs exactly once fails the run before any test starts. Update its `old` and
`new` in `mutations.toml`.

## Regenerate

**What it does.** Generates every clip again with Drama 3, recognizes it, and
records new fixtures and baselines. Use it to add or change a set.

**What it needs.** Drama 3 weights and a GPU. macOS 26 with the en_AU speech
assets. The Newsreader voice, exported from the Drama Studio library as a voice
JSON. Set `WORD_CHECK_VOICE_NEWSREADER` to its path. Items name the voice, so
the repository holds no library path.

**Command.**

```
WORD_CHECK_VOICE_NEWSREADER=/path/to/newsreader.json tools/word_check_eval/run.sh regenerate [set ...]
tools/word_check_eval/run.sh regenerate --dry-run
```

`--dry-run` prints the commands without running them. For each set the run
calls `word-check-lab gen` under `tools/with-gpu-lock`, writing WAVs to
`audio/<set>/`. It then calls `word-check-lab recog`, which writes
`audio/<set>/recog.json`, and `record.py`, which rewrites the set's fixture and
baseline. `gen` skips clips that already exist, so a stopped run resumes.

**Time.** Generation took about 4 seconds per clip on the machine that made
v1, about 47 minutes for all 704 clips.

**After a regeneration.** Clips differ from v1, so the floors and the archive
change too. Run `replay`, set each `replayFloor` from its output, and build a
new archive:

```
python3 -I tools/word_check_eval/pack_audio.py word-check-eval-audio-v2.zip \
    tuning=tools/word_check_eval/audio/tuning ...
```

Publish the zip as a release asset, and update `audio.json` with its name, URL,
size and SHA-256. Then run `audio` and set each `audioFloor`.
