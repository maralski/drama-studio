"""Core ML friendly rewrites of the reference DAC, each equal to the op it replaces.

`make_exportable` folds weight norm and swaps modules in place; `CodecEncoder` and
`CodecDecoder` are the traced entry points. The reference modules in this package
stay untouched so they remain the oracle.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize, remove_weight_norm

from .dac_nn import ResidualVectorQuantize, Snake1d, VectorQuantize
from . import modded_dac, rvq
from .modded_dac import DAC, WindowLimitedTransformer

SAMPLES_PER_FRAME = 2048
# rvq.py carries its own copies of the causal conv classes
CAUSAL_CONVS = (modded_dac.CausalConvNet, rvq.CausalConvNet)
CAUSAL_TRANS_CONVS = (modded_dac.CausalTransConvNet, rvq.CausalTransConvNet)


class Snake(nn.Module):
    """x + sin(alpha x)^2 / (alpha + 1e-9), with the reciprocal precomputed and sin^2 as a product."""

    def __init__(self, snake: Snake1d):
        super().__init__()
        alpha = snake.alpha.detach()
        self.register_buffer("alpha", alpha)
        self.register_buffer("inv_alpha", (alpha + 1e-9).reciprocal())

    def forward(self, x):
        s = torch.sin(self.alpha * x)
        return x + self.inv_alpha * (s * s)


class LeftPadConv(nn.Module):
    """CausalConvNet with its right padding dropped: it is zero for every length that is a
    multiple of the stride, which all enumerated shapes are. The left zeros are a constant
    concatenated in front because the converter reads a traced F.pad as dynamic."""

    def __init__(self, conv: modded_dac.CausalConvNet):
        super().__init__()
        self.conv = conv.conv
        self.padded = conv.padding > 0
        self.register_buffer("zeros", torch.zeros(1, conv.conv.in_channels, conv.padding))

    def forward(self, x):
        if self.padded:
            x = torch.cat([self.zeros, x], dim=-1)
        return self.conv(x)


class PolyphaseTransConv(nn.Module):
    """CausalTransConvNet (kernel s or 2s, stride s, right-trimmed by kernel - s) as an
    ordinary causal conv. Output step n*s + r only sees inputs n and n-1, through kernel
    taps r and r + s, so one conv with s*Cout channels computes every phase r and a
    reshape interleaves them. The Neural Engine's own conv_transpose loses ~15 dB here."""

    def __init__(self, conv: modded_dac.CausalTransConvNet):
        super().__init__()
        tc = conv.conv
        s, k = tc.stride[0], tc.kernel_size[0]
        assert k in (s, 2 * s) and tc.dilation[0] == 1 and tc.padding[0] == 0 and tc.groups == 1
        cin, cout = tc.in_channels, tc.out_channels
        taps = k // s
        weight = tc.weight.detach()  # [cin, cout, k]
        phased = torch.zeros(cout, s, cin, taps)
        for lag in range(taps):
            phased[..., taps - 1 - lag] = weight[:, :, lag * s : (lag + 1) * s].permute(1, 2, 0)
        self.conv = nn.Conv1d(cin, cout * s, taps)
        self.conv.weight.data = phased.reshape(cout * s, cin, taps)
        self.conv.bias.data = tc.bias.detach().repeat_interleave(s)
        self.stride, self.out_channels = s, cout
        self.register_buffer("zeros", torch.zeros(1, cin, taps - 1))

    def forward(self, x):
        if self.zeros.shape[-1]:
            x = torch.cat([self.zeros, x], dim=-1)
        y = self.conv(x)
        n = y.shape[-1]
        return y.reshape(1, self.out_channels, self.stride, n).transpose(2, 3).reshape(1, self.out_channels, n * self.stride)


def _max_abs(x: torch.Tensor) -> torch.Tensor:
    # Activations reach 1.5e3 in the quantizer transformer, so x*x overflows fp16 unless
    # the row is first divided by its max; the norms are scale invariant once eps is
    # divided by the same factor squared.
    return x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-3)


class ScaledRMSNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.eps

    def forward(self, x):
        m = _max_abs(x)
        y = x / m
        return y * torch.rsqrt((y * y).mean(-1, keepdim=True) + self.eps / (m * m)) * self.weight


class ScaledLayerNorm(nn.Module):
    def __init__(self, norm: nn.LayerNorm):
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.eps = norm.eps

    def forward(self, x):
        m = _max_abs(x)
        y = x / m
        y = y - y.mean(-1, keepdim=True)
        y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + self.eps / (m * m))
        return y * self.weight + self.bias


def _rope_half_order(head_dim: int) -> torch.Tensor:
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


class WindowedTransformer(nn.Module):
    """WindowLimitedTransformer with the mask built from the traced length, RoPE on halves
    instead of interleaved pairs (q and k head dims permuted the same way, which leaves
    q.k unchanged), explicit softmax attention, and fp16-safe norms."""

    def __init__(self, t: WindowLimitedTransformer, max_len: int):
        super().__init__()
        assert isinstance(t.input_proj, nn.Identity) and isinstance(t.output_proj, nn.Identity)
        assert t.causal and t.channels_first
        cfg = t.config
        self.n_head, self.head_dim, self.dim = cfg.n_head, cfg.head_dim, cfg.dim
        self.window = t.window_size or max_len
        order = _rope_half_order(self.head_dim)
        per_head = torch.cat([h * self.head_dim + order for h in range(self.n_head)])
        self.layers = nn.ModuleList()
        for layer in t.layers:
            wq, wk, wv = layer.attention.wqkv.weight.detach().split(self.dim, dim=0)
            block = nn.Module()
            block.wq = nn.Linear(self.dim, self.dim, bias=False)
            block.wk = nn.Linear(self.dim, self.dim, bias=False)
            block.wv = nn.Linear(self.dim, self.dim, bias=False)
            block.wq.weight.data = wq[per_head].clone()
            block.wk.weight.data = wk[per_head].clone()
            block.wv.weight.data = wv.clone()
            block.wo = layer.attention.wo
            block.attention_norm = ScaledRMSNorm(layer.attention_norm)
            block.ffn_norm = ScaledRMSNorm(layer.ffn_norm)
            block.w1, block.w2, block.w3 = layer.feed_forward.w1, layer.feed_forward.w2, layer.feed_forward.w3
            block.attention_gamma = layer.attention_layer_scale.gamma
            block.ffn_gamma = layer.ffn_layer_scale.gamma
            self.layers.append(block)
        self.norm = ScaledRMSNorm(t.norm)
        freqs = t.freqs_cis[:max_len].float()
        self.register_buffer("cos", freqs[..., 0].clone())
        self.register_buffer("sin", freqs[..., 1].clone())

    def _rope(self, x, cos, sin):
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, x):
        x = x.transpose(1, 2)
        bsz, seqlen, _ = x.shape
        pos = torch.arange(seqlen, device=x.device)
        delta = pos[:, None] - pos[None, :]
        allowed = (delta >= 0) & (delta < self.window)
        bias = torch.where(allowed, 0.0, -1e4).to(x.dtype)
        cos = self.cos[:seqlen][None, None]
        sin = self.sin[:seqlen][None, None]
        scale = self.head_dim**-0.5
        for b in self.layers:
            h = b.attention_norm(x)
            q = b.wq(h).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
            k = b.wk(h).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
            v = b.wv(h).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
            attn = torch.softmax(q @ k.transpose(-1, -2) * scale + bias, dim=-1)
            y = (attn @ v).transpose(1, 2).reshape(bsz, seqlen, self.dim)
            x = x + b.wo(y) * b.attention_gamma
            h = b.ffn_norm(x)
            x = x + b.w2(F.silu(b.w1(h)) * b.w3(h)) * b.ffn_gamma
        return self.norm(x).transpose(1, 2)


def fold_weight_norm(model: nn.Module) -> None:
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        elif hasattr(m, "weight_g"):
            remove_weight_norm(m)


def make_exportable(model: DAC, max_frames: int) -> DAC:
    """Fold weight norm and swap every op that Core ML or fp16 cannot take as written."""
    fold_weight_norm(model)
    swaps = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, Snake1d):
                swaps.append((parent, name, Snake(child)))
            elif isinstance(child, CAUSAL_CONVS):
                swaps.append((parent, name, LeftPadConv(child)))
            elif isinstance(child, CAUSAL_TRANS_CONVS):
                swaps.append((parent, name, PolyphaseTransConv(child)))
            elif isinstance(child, nn.LayerNorm):
                swaps.append((parent, name, ScaledLayerNorm(child)))
            elif isinstance(child, WindowLimitedTransformer):
                # the encoder transformer runs at 4 positions per frame, the quantizer ones at 1
                per_frame = 1 if parent is model.quantizer else 4
                swaps.append((parent, name, WindowedTransformer(child, max_frames * per_frame)))
    for parent, name, new in swaps:
        setattr(parent, name, new)
    return model.eval()


class _Codebook(nn.Module):
    """One VectorQuantize: nearest code by cosine similarity, and its projection back."""

    def __init__(self, vq: VectorQuantize):
        super().__init__()
        self.in_proj = vq.in_proj
        self.out_proj = vq.out_proj
        self.codebook = vq.codebook.weight
        # dist = |e|^2 - 2 e.c + |c|^2 over unit vectors, so argmin dist = argmax e.c, and
        # normalising e (a positive per-row scale) cannot change that argmax.
        self.register_buffer("unit_codebook", F.normalize(vq.codebook.weight.detach())[:, :, None])

    def nearest(self, z):
        scores = F.conv1d(self.in_proj(z), self.unit_codebook)
        return scores.argmax(dim=1).to(torch.int32)

    def project(self, codes):
        return self.out_proj(F.embedding(codes, self.codebook).transpose(1, 2))


class CodecEncoder(nn.Module):
    """audio [1, 1, frames*2048] float -> codes [1, 10, frames] int32."""

    def __init__(self, model: DAC):
        super().__init__()
        self.encoder = model.encoder
        self.downsample = model.quantizer.downsample
        self.pre_module = model.quantizer.pre_module
        rvqs: list[ResidualVectorQuantize] = [model.quantizer.semantic_quantizer, model.quantizer.quantizer]
        self.codebooks = nn.ModuleList(_Codebook(vq) for rvq in rvqs for vq in rvq.quantizers)

    def forward(self, audio):
        residual = self.pre_module(self.downsample(self.encoder(audio)))
        codes = []
        for i, book in enumerate(self.codebooks):
            c = book.nearest(residual)
            codes.append(c)
            if i + 1 < len(self.codebooks):
                residual = residual - book.project(c)
        return torch.stack(codes, dim=1)


class CodecDecoder(nn.Module):
    """codes [1, 10, frames] int32 -> audio [1, frames*2048] float."""

    def __init__(self, model: DAC):
        super().__init__()
        q = model.quantizer
        self.semantic_size = q.semantic_quantizer.codebook_size
        self.residual_size = q.quantizer.codebook_size
        rvqs = [q.semantic_quantizer, q.quantizer]
        self.codebooks = nn.ModuleList(_Codebook(vq) for rvq in rvqs for vq in rvq.quantizers)
        self.post_module = q.post_module
        self.upsample = q.upsample
        self.decoder = model.decoder

    def forward(self, codes):
        z = 0
        for i, book in enumerate(self.codebooks):
            limit = self.semantic_size if i == 0 else self.residual_size
            z = z + book.project(codes[:, i].clamp(0, limit - 1))
        audio = self.decoder(self.upsample(self.post_module(z)))
        return audio.reshape(1, -1)
