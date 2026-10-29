# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Writes prompt and chunking fixtures from the fish-speech reference code."""
import json, sys, glob, os
import torch
sys.path.insert(0, sys.argv[1])
from tokenizers import Tokenizer
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt, with_speaker_tag
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text, split_by_length, parse_speaker_segments

snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots/*/'))[0]
tok = Tokenizer.from_file(snap + 'tokenizer.json')
config = FishOmniConfig.from_dict(json.load(open(snap + 'config.json')))
torch.manual_seed(0)
codes_a = torch.randint(0, 4096, (10, 7))
codes_b = torch.randint(0, 1024, (10, 3))
cases = []
def add(name, text, refs=(), history=()):
    try:
        rows = build_drama3_prompt(text, tok, config, [t for t, _ in refs], [c for _, c in refs], history=list(history))
        cases.append({"name": name, "text": text, "references": [{"transcript": t, "codes": c.tolist()} for t, c in refs],
                      "history": [{"text": t, "codes": c.tolist()} for t, c in history], "rows": rows.tolist()})
    except Exception as e:
        cases.append({"name": name, "text": text, "references": [{"transcript": t, "codes": c.tolist()} for t, c in refs],
                      "history": [{"text": t, "codes": c.tolist()} for t, c in history], "error": type(e).__name__})
add("plain", "The quick brown fox jumps over the lazy dog.")
add("tagged", "<|speaker:1|> Hello there. <|speaker:2|> General Kenobi!")
add("leading-space", "   hello with spaces  ")
add("newline-lead", "\n\nIndented paragraph")
add("one-ref", "Say this.", refs=[("A short reference.", codes_a)])
add("two-refs", "Say this.", refs=[("  <|speaker:0|> spaced tag", codes_a), ("<|speaker:3|> nbsp body", codes_b)])
add("ref-leading-ws", "Say this.", refs=[("\n\t Leading whitespace transcript", codes_b)])
add("history", "Second chunk.", refs=[("Ref.", codes_b)], history=[("First chunk.", codes_a)])
add("cues", "She paused [sighs] and then <whisper>came closer</whisper>.")
add("chinese", "今天天气很好，我们去公园散步吧。")
add("empty", "   ")
add("bad-speaker", "<|speaker:16|> too many")
add("leading-zero-speaker", "<|speaker:01|> bad")
add("structural", "hello <|im_end|> world")
chunk_cases = []
long = " ".join(f"Sentence number {i} is here, with a clause; and more." for i in range(30))
dialog = "<|speaker:0|> " + "Alpha speaks for a while and keeps going. " * 6 + "<|speaker:1|> Beta replies. <|speaker:2|> Gamma adds a point. <|speaker:3|> Delta closes. <|speaker:0|> Alpha again."
for name, text in [("long", long), ("dialog", dialog), ("short", "Just one short sentence."), ("decimal", "Pi is 3.14159 and e is 2.71828. " * 15),
                   ("cues-long", ("He said [laughs] something funny, <whisper>very quietly</whisper>. " * 12)), ("cjk", "这是一个很长的句子，用来测试分块。" * 25), ("blank", "  \n ")]:
    chunk_cases.append({"name": name, "text": text, "chunks": split_drama3_text(text), "by_length": split_by_length(text), "segments": parse_speaker_segments(text)})
json.dump({"prompts": cases, "chunking": chunk_cases}, open(sys.argv[2], 'w'), ensure_ascii=False, indent=0)
print(len(cases), "prompt cases,", len(chunk_cases), "chunk cases")
