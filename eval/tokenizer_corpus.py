import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = Path.home() / (
    ".cache/huggingface/hub/models--fishaudio--drama-3/snapshots/"
    "e0a9218aa48888270f06ef1ed9ea67622c0e17d2/tokenizer.json"
)
DEFAULT_OUT = REPO / "Tests/DramaEngineTests/Fixtures/tokenizer_cases.json"

PROMPT_LITERALS = [
    "<|im_start|>system\n",
    "convert the provided text to speech",
    "convert the provided text to speech reference to the following:",
    "\n\n[Reference 1]\nText: <|speaker:0|> hello\nSpeech: ",
    "\n\n[Reference 2]\nText: <|speaker:1|> 你好，很高兴认识你。\nSpeech: ",
    "<|im_end|>\n",
    "<|im_start|>user\n<|speaker:0|> text<|im_end|>\n",
    "<|im_start|>user\n<|speaker:0|> Hello there, how are you today?<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|>",
    "<|im_start|>assistant\n<|voice|><|im_end|>\n",
]

SPECIAL_INLINE = [
    "Then <|speaker:3|> said hello.",
    "<|speaker:3|>",
    "a<|speaker:3|>b",
    " <|speaker:3|> ",
    "<|im_start|><|im_end|>",
    "<|semantic:0|><|semantic:4095|>",
    "<|semantic:4096|>",
    "<|pad|><|endoftext|>",
    "<<|im_end|>>",
    "<|im_start",
    "im_start|>",
    "<|not_a_token|>",
    "<|IM_START|>",
    "<think>reasoning</think>",
    "<|speaker:0|>[laughs] no way",
]

ENGLISH = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "don't it's I'll",
    "We've, they're, she'd, I'm, you'll, he's, can't.",
    "DON'T IT'S I'LL WE'VE THEY'RE SHE'D I'M",
    "'s 't 're 've 'm 'll 'd",
    "rock'n'roll o'clock",
    "the dog's bone's color",
    "'ſ and 'K",
    "He said \"stop\" and she said 'go'.",
    "It was [laughs] really funny.",
    "[laughs]",
    "[sighs] Fine. [whispering] come here",
    "(softly) okay...",
    "Wait -- what?!",
    "!!!???...",
    "email me at someone@example.com",
    "https://example.com/path?q=1&r=two#frag",
    "C:\\Users\\name\\file.txt",
    "snake_case and camelCase and kebab-case",
    "def f(x):\n    return x + 1\n",
    "if (a && b) { c(); } else { d[0] = e; }",
    "<html><body>Hi</body></html>",
    "Mr. Smith arrived at 3:45 p.m. on Jan. 5th.",
    "It costs $19.99 (or €18,50).",
    "100% sure, 50/50 chance, 3.14159, 1,000,000",
    "2026-10-15T12:34:56Z",
    "12345678901234567890",
    "Call 555-0100 ext. 42",
    "1st 2nd 3rd 4th",
    "½ ¾ ² ³ Ⅻ ①",
    "١٢٣ ४५६ ７８９",
    "ALL CAPS SENTENCE HERE",
    "MiXeD CaSe WoRdS",
    "a",
    "I",
    ".",
    "  ",
    " ",
    "\n",
    "\t",
]

WHITESPACE = [
    "  leading spaces",
    "trailing spaces  ",
    "  both  ",
    "a  b   c    d",
    "\t\ttabs\tbetween\twords",
    "line1\nline2",
    "line1\r\nline2",
    "line1\rline2",
    "\n\n\n",
    "   \n  \n",
    " \t \n",
    "a \n b",
    "end\n",
    "\n start",
    "para one.\n\npara two.",
    "x\n\n\n\ny",
    "  \n\n  indented after blank",
    "trailing newline spaces \n ",
    "a\u000bb",
    "a\u000cb",
    "a\u0085b",
    "a\u00a0b",
    "a\u3000b",
    "a\u2028b",
    "a\u2029b",
    "a\u200bb",
    "a\ufeffb",
    "a\u001cb\u001dc\u001ed\u001ff",
    "a\u1680b",
    "a\u2003b\u2009c",
    " \u00a0 x",
    "\u3000\u3000全角空格",
    "word\u00a0\n",
    "   " * 30,
    "\n" * 20,
    "\t \t \t",
]

MULTILINGUAL = [
    "你好，世界！",
    "今天天气很好，我们去公园散步吧。",
    "他说：“这不是我的错。”然后就走了。",
    "我有3个苹果和12根香蕉。",
    "中文English混合text测试",
    "こんにちは、世界。",
    "カタカナとひらがなと漢字",
    "東京は日本の首都です。今日は2026年10月15日です。",
    "ｶﾀｶﾅ半角",
    "안녕하세요, 반갑습니다.",
    "Привет, мир! Как дела?",
    "Γειά σου Κόσμε",
    "مرحبا بالعالم",
    "שלום עולם",
    "क्षत्रिय नमस्ते",
    "สวัสดีครับ",
    "Tiếng Việt có dấu",
    "naïve café résumé",
    "Ame\u0301lie",
    "e\u0301",
    "\u0301 alone mark",
    " \u0301",
    "a\u0308\u0301",
    "a\u0301\u0323",
    "\u1e0b\u0323",
    "\u0628\u0651\u064e\u0628",
    "e\u0302\u0301 Vie\u0302\u0323t",
    "Ａｌｐｈａ１２３",
    "Straße ß ẞ",
    "İstanbul ıi",
    "ﬁ ligature ﬂ",
]

EMOJI = [
    "Hello 👋 world 🌍",
    "👨‍👩‍👧‍👦 family",
    "🏳️‍🌈",
    "❤️",
    "🇯🇵🇺🇸",
    "👍🏽👍🏿",
    "😀" * 30,
    "smile:) :-( <3",
    "text😀text",
    " 😀 ",
    "★☆♥♦",
    "1\ufe0f\u20e3 2\ufe0f\u20e3 #\ufe0f\u20e3",
    "1\u0301 2\u0308",
    "→ ← ↑ ↓",
    "™ © ®",
]

LONG = [
    "The old lighthouse keeper climbed the stairs every evening, counting each of the 217 steps, "
    "and when he reached the top he would light the lamp and watch the ships pass by. "
    "\"It's not much,\" he'd say, \"but it's mine.\"",
    "在很久很久以前，有一个小村庄，村庄里住着一位老人。他每天早上都会去河边钓鱼，"
    "不管天气怎么样，他从来没有停止过。村里的孩子们都很喜欢听他讲故事。",
    "a" * 500,
    "1" * 50,
    "ab" * 100,
    "-" * 80,
]

POOL = [
    *"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    *"0123456789",
    *"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    " ", " ", " ", "\t", "\n", "\r", "\r\n", "\u00a0", "\u3000", "\u000b", "\u0085", "\u2028",
    "é", "\u0301", "\u0300", "\u0308", "ñ", "ü",
    "中", "文", "字", "。", "，", "あ", "ア", "한", "ж",
    "😀", "👍🏽", "\u200d", "\ufe0f", "🇺🇸",
    "'s", "'LL", "'t", "'re", "'D",
    "<|im_start|>", "<|im_end|>", "<|voice|>", "<|speaker:3|>", "<|semantic:7|>", "<|", "|>",
    "٣", "１", "Ａ", "½", "ⅷ",
    "hello", "world", " the", " and", "ing",
]


def fuzz_cases(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(POOL) for _ in range(rng.randint(1, 40))) for _ in range(count)]


def padded_variants(texts: list[str]) -> list[str]:
    return [variant for t in texts for variant in (f" {t}", f"{t} ", f"{t}\n", f"\t{t}")]


def corpus() -> list[str]:
    base = PROMPT_LITERALS + SPECIAL_INLINE + ENGLISH + WHITESPACE + MULTILINGUAL + EMOJI + LONG
    texts = base + padded_variants(ENGLISH + MULTILINGUAL + EMOJI[:6]) + fuzz_cases(200, seed=3)
    return list(dict.fromkeys(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Swift tokenizer parity fixture.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = [{"text": t, "ids": tokenizer.encode(t, add_special_tokens=False).ids} for t in corpus()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ",\n".join(json.dumps(c, ensure_ascii=False) for c in cases)
    args.out.write_text(f"[\n{lines}\n]\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
