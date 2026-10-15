"""Run the vendored fish-speech Drama 3 oracle and write the docs/EVAL.md dump."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from oracle.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX,
    CHECKPOINT_LM_HEAD_KEY,
    FishOmniBackbone,
    load_backbone_weights,
)
from oracle.fish_omni.config import FishOmniConfig
from oracle.fish_omni.generate import DRAMA3_PREFILL_CHUNK_SIZE, build_drama3_prompt
from oracle.fish_omni.model import AUDIO_DECODER_PREFIXES, iter_checkpoint_tensors
from oracle.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights
from oracle.sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P

DEFAULT_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--fishaudio--drama-3/snapshots"
    / "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_spec(path: Path) -> dict:
    """build_drama3_prompt keyword arguments; codes paths are relative to the spec."""
    spec = json.loads(path.read_text())

    def codes(name: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path.parent / name)).long()

    references = spec.get("references", [])
    return dict(
        text=spec["text"],
        prompt_texts=[r["transcript"] for r in references],
        prompt_tokens=[codes(r["codes"]) for r in references],
        history=[(h["text"], codes(h["codes"])) for h in spec.get("history", [])],
    )


def load_model(
    checkpoint: Path, config: FishOmniConfig, dtype: torch.dtype
) -> tuple[FishOmniBackbone, FishOmniAudioDecoder]:
    """FP8 experts stay FP8 buffers; MoEExperts dequantizes the selected ones per call."""
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=dtype)
        tower = FishOmniAudioDecoder(config, dtype=dtype)
    backbone.materialize(["cpu"])
    load_backbone_weights(
        backbone,
        iter_checkpoint_tensors(checkpoint, skip_prefixes=AUDIO_DECODER_PREFIXES),
    )
    tower.to_empty(device="cpu")
    load_audio_decoder_weights(
        tower,
        iter_checkpoint_tensors(
            checkpoint,
            skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY),
        ),
    )
    return backbone.eval(), tower.eval()


def capture_last_row(modules, store: list) -> list:
    """Hook each module so store[i] holds the last row of its latest output."""

    def hook(index: int):
        def save(_module, _inputs, out: torch.Tensor) -> None:
            store[index] = out.reshape(-1, out.shape[-1])[-1].float().clone()

        return save

    return [m.register_forward_hook(hook(i)) for i, m in enumerate(modules)]


def forced_tower_logits(
    tower: FishOmniAudioDecoder,
    hidden: torch.Tensor,
    position: torch.Tensor,
    frame_in: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """decode_step's tower pass with the codes of frame forced: head k logits [9, 4096]."""
    semantic = tower.is_semantic(frame[:, 0])
    h = tower.run_shared(hidden, frame[:, 1], semantic, position, frame_in)
    logits = []
    for k, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
        h = layer(h, frame[:, 1 + k], semantic, position)
        logits.append(head(h).float())
    return torch.cat(logits)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    dtype = DTYPES[args.dtype]
    config = FishOmniConfig.from_dict(
        json.loads((args.checkpoint / "config.json").read_text())
    )
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    prompt = build_drama3_prompt(
        tokenizer=tokenizer, config=config, **load_spec(args.spec)
    )
    forced = None if args.frames is None else torch.from_numpy(np.load(args.frames))
    steps = args.steps if forced is None else len(forced)
    rows = len(prompt)
    print(f"prompt: {rows} rows; {'forcing' if forced is not None else 'sampling'} {steps} frames")

    started = time.perf_counter()
    backbone, tower = load_model(args.checkpoint, config, dtype)
    load_s = time.perf_counter() - started
    print(f"load: {load_s:.1f}s", flush=True)

    capacity = rows + steps
    backbone.setup_caches(max_batch_size=1, max_seq_len=capacity)
    tower.setup_caches(capacity)
    frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long)
    frames[:rows] = prompt
    positions = torch.arange(capacity)

    started = time.perf_counter()
    layer_out = [None] * len(backbone.model.layers)
    handles = capture_last_row(backbone.model.layers, layer_out)
    # generate_drama3's chunked prefill, so the delta-rule chunks match the release.
    for start in range(0, rows, DRAMA3_PREFILL_CHUNK_SIZE):
        end = min(start + DRAMA3_PREFILL_CHUNK_SIZE, rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[start:end])[0]
        paired_end = min(end, rows - 1)
        if paired_end > start:
            tower.prefill(
                hidden[: paired_end - start],
                positions[start:paired_end],
                frames[start:paired_end],
                frames[start + 1 : paired_end + 1],
            )
    for handle in handles:
        handle.remove()
    prefill_s = time.perf_counter() - started
    print(f"prefill: {prefill_s:.1f}s", flush=True)

    eos = torch.tensor([config.eos_token_id])
    band = torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1)
    candidates = torch.cat((eos, band)).sort().values
    tower_layers = [*tower.shared_layers, *tower.residual_layers]
    tower_hidden = [None] * len(tower_layers)
    hiddens, cand_logits, tower_logits, step_s = [], [], [], []
    hidden = hidden[-1:]
    torch.manual_seed(args.seed)
    for step in range(steps + 1):
        started = time.perf_counter()
        position = rows - 1 + step
        here = positions[position : position + 1]
        logits = backbone.compute_logits(hidden)
        hiddens.append(hidden[0].float())
        cand_logits.append(logits[0, candidates].float())
        if step == steps:
            break
        if forced is None:
            frame = tower.decode_step(
                logits,
                hidden,
                here,
                frames[position : position + 1],
                frames[: position + 1, 0],
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            if frame[0, 0].item() == config.eos_token_id:
                print(f"step {step}: eos")
                break
        else:
            frame = forced[step : step + 1]
        handles = capture_last_row(tower_layers, tower_hidden) if step == 0 else []
        tower_logits.append(
            forced_tower_logits(tower, hidden, here, frames[position : position + 1], frame)
        )
        for handle in handles:
            handle.remove()
        frames[position + 1] = frame[0]
        embeddings = tower.embed_backbone(frame, backbone.embed(frame[:, 0]))
        hidden = backbone(
            inputs_embeds=embeddings[None], input_pos=positions[position + 1 : position + 2]
        )[0]
        step_s.append(time.perf_counter() - started)
        print(f"step {step}: {step_s[-1]:.2f}s  frame {frame[0].tolist()}", flush=True)

    produced = len(tower_logits)
    if produced == 0:
        sys.exit("eos at step 0: the prompt is wrong or the model is broken")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    meta = dict(
        producer="oracle",
        dtype=args.dtype,
        threads=torch.get_num_threads(),
        torch=torch.__version__,
        spec=str(args.spec),
        checkpoint=str(args.checkpoint),
        mode="sampled" if forced is None else "forced",
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        prompt_rows=rows,
        frames=produced,
        load_s=load_s,
        prefill_s=prefill_s,
        step_s=step_s,
        peak_rss_bytes=peak_rss,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        prompt_rows=prompt.numpy(),
        frames=frames[rows : rows + produced].numpy(),
        hidden=torch.stack(hiddens).numpy(),
        cand_logits=torch.stack(cand_logits).numpy(),
        tower_logits=torch.stack(tower_logits).numpy(),
        layer_out_step0=torch.stack(layer_out).numpy(),
        tower_hidden_step0=torch.stack(tower_hidden).numpy(),
        meta=np.array(json.dumps(meta)),
    )
    mean_step = sum(step_s) / len(step_s)
    print(
        f"wrote {args.out}: {produced} frames, {mean_step:.2f}s/step, "
        f"peak RSS {peak_rss / 2**30:.1f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=Path, help="[F, 11] int64 frames to force")
    parser.add_argument("--steps", type=int, default=12, help="frames to sample without --frames")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
