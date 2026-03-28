"""
DiT-ECG-B: Diffusion Transformer for 1D ECG Generation
=======================================================
Based on DiT (Peebles & Xie, arXiv:2212.09748) adapted for 1D ECG signals.

Run 6 Architecture Changes:
  - Identity conditioning via DEDICATED CROSS-ATTENTION (IP-Adapter paradigm)
    instead of AdaLN-Zero, preventing identity from being diluted by timestep.
  - IdentityTokenizer: projects 512-dim embedding → 8 virtual tokens for
    rich cross-attention interaction.
  - AdaLN-Zero retains only timestep + HR conditioning.
  - ConditioningProjector now has 2 heads (t, hr) instead of 3.
  - Supports both noise prediction (DDPM) and velocity prediction (OT-CFM).

Previous (Run 5):
  x → AdaLN(t+hr+id) → SelfAttn → FFN → x'

Run 6:
  x → AdaLN(t+hr) → SelfAttn → CrossAttn(id_tokens) → FFN → x'

Architecture: DiT-ECG-B (~100M params)
- 24 Transformer blocks
- d_model = 768
- 12 attention heads
- Patch size = 10 (250 tokens for 2500-sample signal)
- 8 identity tokens via cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional/timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class HREmbedding(nn.Module):
    """
    Sinusoidal encoding of scalar heart rate (bpm) → dense vector.

    Research-confirmed (Option B): HR gets its own dedicated embedding
    with sinusoidal frequencies, normalized to [0, 1] over physiological
    range [30, 200] bpm.
    """

    def __init__(self, hidden_dim=256, hr_min=30.0, hr_max=200.0):
        super().__init__()
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(10000) * torch.arange(hidden_dim // 2) / (hidden_dim // 2))
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hr):
        # hr: (B,) bpm values
        hr_norm = ((hr - self.hr_min) / (self.hr_max - self.hr_min)).clamp(0, 1)
        args = hr_norm[:, None] * self.freqs[None]  # (B, dim/2)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)  # (B, hidden_dim)


class ConditioningProjector(nn.Module):
    """
    2 independent MLP heads → additive combination.

    Run 6: Only timestep + HR. Identity is now via cross-attention.
    HR head is zero-initialized for training stability.
    """

    def __init__(self, hidden_dim=256, d_model=768):
        super().__init__()

        def _head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.t_mlp = _head(hidden_dim, d_model)
        self.hr_mlp = _head(hidden_dim, d_model)

        # Zero-init HR head for training stability
        nn.init.zeros_(self.hr_mlp[-1].weight)
        nn.init.zeros_(self.hr_mlp[-1].bias)

    def forward(self, t_emb, hr_emb):
        """Both inputs: (B, hidden_dim). Output: (B, d_model)."""
        return self.t_mlp(t_emb) + self.hr_mlp(hr_emb)


class IdentityTokenizer(nn.Module):
    """
    Projects the 512-dim patient identity vector into multiple virtual tokens
    for cross-attention conditioning. This allows the cross-attention to attend
    to different aspects of the patient's cardiac signature.

    Inspired by IP-Adapter's image-prompt tokenization strategy.

    512-dim → (B, num_tokens, d_model)
    """

    def __init__(self, cond_dim=512, d_model=768, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model

        # Project to num_tokens * d_model, then reshape
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * num_tokens),
        )
        # LayerNorm on the output tokens for stable cross-attention
        self.norm = nn.LayerNorm(d_model)

        # Zero-init the final projection so identity starts as no-op
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, identity):
        """
        Args:
            identity: (B, cond_dim) — patient identity embedding
        Returns:
            tokens: (B, num_tokens, d_model) — virtual identity tokens
        """
        B = identity.shape[0]
        tokens = self.proj(identity)  # (B, num_tokens * d_model)
        tokens = tokens.view(B, self.num_tokens, self.d_model)
        tokens = self.norm(tokens)
        return tokens


class IdentityCrossAttention(nn.Module):
    """
    Dedicated cross-attention layer for identity conditioning.

    query = patch tokens (from self-attention output)
    key/value = identity tokens (from IdentityTokenizer)

    This is the IP-Adapter paradigm adapted for 1D ECG:
    identity info flows through a SEPARATE attention pathway,
    preventing it from being diluted by timestep/HR in AdaLN.

    Output is gated (zero-init) so training starts from pure
    timestep conditioning and gradually learns identity influence.
    """

    def __init__(self, d_model=768, n_heads=12, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Zero-init gate so identity cross-attention starts as no-op
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, id_tokens):
        """
        Args:
            x: (B, N, D) — patch tokens (after self-attention)
            id_tokens: (B, K, D) — identity tokens (K=8)
        Returns:
            x: (B, N, D) — patch tokens with identity info mixed in
        """
        # Cross-attention: patches attend to identity tokens
        h = self.norm(x)
        cross_out, _ = self.cross_attn(
            query=h,
            key=id_tokens,
            value=id_tokens,
        )
        # Gated residual — gate starts at 0, learned during training
        x = x + self.gate * cross_out
        return x


class PatchEmbed1D(nn.Module):
    """Patchify a 1D signal into non-overlapping patches via Conv1D."""

    def __init__(self, in_channels=1, patch_size=10, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, T) e.g. (B, 1, 2500)
        x = self.proj(x)  # (B, D, T/P)
        x = x.transpose(1, 2)  # (B, T/P, D)
        return x


class UnPatch1D(nn.Module):
    """Convert patch tokens back to 1D signal."""

    def __init__(self, out_channels=1, patch_size=10, embed_dim=768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, N, D) where N = T/P
        B, N, D = x.shape
        x = self.linear(x)  # (B, N, P*C)
        x = x.view(B, N, self.out_channels, self.patch_size)  # (B, N, C, P)
        x = x.permute(0, 2, 1, 3)  # (B, C, N, P)
        x = x.reshape(B, self.out_channels, N * self.patch_size)  # (B, C, T)
        return x


class ECGDiTBlock(nn.Module):
    """
    DiT Block with AdaLN-Zero modulation + Identity Cross-Attention.

    Run 6 Architecture:
    1. AdaLN-modulated Self-Attention (timestep + HR conditioning)
    2. Identity Cross-Attention (dedicated pathway for patient identity)
    3. AdaLN-modulated FFN

    The AdaLN conditioning now produces 6 modulation scalars from
    timestep + HR only. Identity flows through cross-attention.
    """

    def __init__(self, d_model=768, n_heads=12, mlp_ratio=4, dropout=0.0):
        super().__init__()

        # LayerNorm (no elementwise affine — modulated by AdaLN)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Identity Cross-Attention (Run 6)
        self.id_cross_attn = IdentityCrossAttention(d_model, n_heads, dropout)

        # Feed-Forward Network
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: MLP produces 6 × d_model scalars, zero-initialized
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-initialize the output layer for training stability
        nn.init.zeros_(self.adaLN_mlp[-1].weight)
        nn.init.zeros_(self.adaLN_mlp[-1].bias)

    def forward(self, x, cond, id_tokens=None):
        """
        Args:
            x: (B, N, D) — sequence of patch tokens
            cond: (B, D) — timestep + HR conditioning vector (NO identity)
            id_tokens: (B, K, D) — identity tokens for cross-attention
        """
        # Compute 6 modulation vectors from timestep+HR conditioning
        modulation = self.adaLN_mlp(cond)  # (B, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = modulation.chunk(6, dim=-1)

        # --- Self-Attention sub-block (timestep + HR modulated) ---
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + alpha1.unsqueeze(1) * attn_out

        # --- Identity Cross-Attention sub-block (Run 6) ---
        if id_tokens is not None:
            x = self.id_cross_attn(x, id_tokens)

        # --- FFN sub-block ---
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        x = x + alpha2.unsqueeze(1) * self.mlp(h)

        return x


class DiTECG(nn.Module):
    """
    DiT-ECG: Diffusion Transformer for 1D ECG Generation.

    Run 6: Identity via cross-attention, supports velocity prediction (OT-CFM).
    """

    def __init__(
        self,
        signal_length=2500,
        in_channels=1,
        patch_size=10,
        depth=24,
        dim=768,
        heads=12,
        mlp_ratio=4,
        cond_dim=512,
        cond_hidden=256,
        id_num_tokens=8,
        dropout=0.0,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size

        # --- Embeddings ---
        self.patch_embed = PatchEmbed1D(in_channels, patch_size, dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, dim)
        )

        # Timestep embedding → hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(cond_hidden),
            nn.Linear(cond_hidden, cond_hidden * 4),
            nn.SiLU(),
            nn.Linear(cond_hidden * 4, cond_hidden),
        )

        # HR embedding (sinusoidal encoding)
        self.hr_embed = HREmbedding(hidden_dim=cond_hidden)

        # Conditioning projector: 2 heads (t + hr) → d_model
        # Run 6: Identity is NO LONGER in AdaLN
        self.cond_proj = ConditioningProjector(
            hidden_dim=cond_hidden, d_model=dim
        )

        # Identity tokenizer: 512-dim → (B, num_tokens, d_model) for cross-attention
        self.id_tokenizer = IdentityTokenizer(
            cond_dim=cond_dim, d_model=dim, num_tokens=id_num_tokens
        )

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            ECGDiTBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # --- Output ---
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Final AdaLN modulation
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        self.unpatch = UnPatch1D(in_channels, patch_size, dim)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, t, cond, hr_bpm=None):
        """
        Args:
            x: (B, C, T) — noisy signal, e.g. (B, 1, 2500)
            t: (B,) — diffusion timesteps (continuous [0, 1])
            cond: (B, cond_dim) — identity conditioning vector (512-dim)
            hr_bpm: (B,) — heart rate in bpm (optional, defaults to 75.0)
        Returns:
            (B, C, T) — predicted noise ε (DDPM) or velocity v (OT-CFM)
        """
        B = x.shape[0]

        # Default HR if not provided (neutral = 75 bpm)
        if hr_bpm is None:
            hr_bpm = torch.full((B,), 75.0, device=x.device)

        # Compute conditioning embeddings
        t_emb = self.time_embed(t)              # (B, hidden_dim)
        hr_emb = self.hr_embed(hr_bpm)           # (B, hidden_dim)

        # Project timestep + HR through AdaLN heads (NO identity here)
        combined_cond = self.cond_proj(t_emb, hr_emb)  # (B, D)

        # Identity → virtual tokens for cross-attention
        id_tokens = self.id_tokenizer(cond)      # (B, K, D)

        # Patchify + positional embedding
        x = self.patch_embed(x)         # (B, N, D)
        x = x + self.pos_embed          # (B, N, D) + positional

        # Transformer blocks with cross-attention identity conditioning
        for block in self.blocks:
            x = block(x, combined_cond, id_tokens=id_tokens)

        # Final projection
        final_mod = self.final_adaLN(combined_cond)  # (B, 2*D)
        gamma_f, beta_f = final_mod.chunk(2, dim=-1)
        x = self.final_norm(x)
        x = x * (1 + gamma_f.unsqueeze(1)) + beta_f.unsqueeze(1)

        # Unpatchify back to signal
        x = self.unpatch(x)  # (B, C, T)
        return x

    def forward_with_cfg(self, x, t, cond, hr_bpm=None, guidance_scale=2.0):
        """
        Classifier-Free Guidance inference.
        Runs the model twice: once conditioned, once unconditioned.

        Run 6: Default guidance_scale lowered to 2.0 (from 3.0).
        """
        # Conditioned pass
        eps_cond = self.forward(x, t, cond, hr_bpm=hr_bpm)

        # Unconditioned pass (null identity + neutral HR)
        null_cond = torch.zeros_like(cond)
        null_hr = torch.full_like(hr_bpm, 75.0) if hr_bpm is not None else None
        eps_uncond = self.forward(x, t, null_cond, hr_bpm=null_hr)

        # Guided prediction
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps_guided


# ─── Factory functions ───────────────────────────────────────────────────────

def dit_ecg_s(cond_dim=512, **kwargs):
    """DiT-ECG-S: ~30M params — for quick experiments."""
    return DiTECG(depth=12, dim=384, heads=6, cond_dim=cond_dim, **kwargs)


def dit_ecg_b(cond_dim=512, **kwargs):
    """DiT-ECG-B: ~100M params — Run 6 baseline with cross-attention."""
    return DiTECG(depth=24, dim=768, heads=12, cond_dim=cond_dim, **kwargs)


def dit_ecg_l(cond_dim=512, **kwargs):
    """DiT-ECG-L: ~400M params — large scale."""
    return DiTECG(depth=24, dim=1024, heads=16, cond_dim=cond_dim, **kwargs)


if __name__ == "__main__":
    # Quick shape verification
    model = dit_ecg_b()
    x = torch.randn(2, 1, 2500)
    t = torch.rand(2)
    c = torch.randn(2, 512)
    hr = torch.tensor([72.0, 85.0])  # HR in bpm
    out = model(x, t, c, hr_bpm=hr)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DiT-ECG-B (Run 6): {params:.1f}M params")
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    # Test without HR (backward compat)
    out_no_hr = model(x, t, c)
    assert out_no_hr.shape == x.shape
    # Test CFG
    out_cfg = model.forward_with_cfg(x, t, c, hr_bpm=hr, guidance_scale=2.0)
    assert out_cfg.shape == x.shape
    # Test null identity (CFG unconditioned)
    null_id = torch.zeros(2, 512)
    out_null = model(x, t, null_id, hr_bpm=hr)
    assert out_null.shape == x.shape
    print("✅ Forward pass verified (Run 6: cross-attn identity + CFG)!")
