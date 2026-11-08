// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Written by tools/pin_model_plan.py; do not edit.

extension ModelDownloadPlan {
    /// `inclusionAI/Ming-3.0-tts-0.5B` at the commit the Ming engine was verified on, without
    /// the report and figures. Small files come first.
    package static let ming = ModelDownloadPlan(repo: "inclusionAI/Ming-3.0-tts-0.5B", revision: "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61", files: [
        .init(path: "README.md", size: 54759, sha256: "14dd4583ada608d61c86b9faf905dcf6d2b2363ec43f4139889a5636c50d7b55"),
        .init(path: "added_tokens.json", size: 1785, sha256: "34394b1e16d7da2db16618bd43a0f970902ade0075480bcd812fe62d559ec319"),
        .init(path: "config.json", size: 4754, sha256: "2fb0767361b3cd3f41dc9392709fc23fe5b40a39f19d70a38d04acabe991822c"),
        .init(path: "configuration.json", size: 51, sha256: "6720b8448f6ad5b4631689a60857bdeb238053a37f6cc6a5936eb50c2a3b9296"),
        .init(path: "merges.txt", size: 1671853, sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
        .init(path: "special_tokens_map.json", size: 2726, sha256: "2c2ec51b9da5db654ac20de8310350d0c9c1a65918ae0181b80dacd2919b4fc3"),
        .init(path: "tokenizer.json", size: 11431389, sha256: "87d54ca3b44ea2f822ba39a0a45cb1b61644bd6606d9d3536dea75367bffb06d"),
        .init(path: "tokenizer_config.json", size: 16260, sha256: "12d6162c815edef24f8606e65b60d43fe0d92b0a4e150a0db8fa01c523aaf36a"),
        .init(path: "vocab.json", size: 2776833, sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        .init(path: "campplus.onnx", size: 28303423, sha256: "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"),
        .init(path: "model.safetensors", size: 2959055504, sha256: "45c4d78d8ad6b1d942e3da86ba035fece02de569f7b2ca7e9d121dfaf18d674c"),
    ])
}
