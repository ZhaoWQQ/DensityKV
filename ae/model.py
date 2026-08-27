"""Convolutional AutoEncoder for compressing [C, H, W] latent frames to a 1-D vector.

Designed for retrieval over WAN latent features with:
- GroupNorm (instance-independent, no train/eval distribution mismatch)
- Residual blocks for gradient flow
- Global Average Pooling instead of flatten (massive param reduction)
- Upsample+Conv decoder (no checkerboard artifacts from ConvTranspose2d)
- Optional L2 normalization for cosine-similarity retrieval

Note: No spatial attention — this AE operates on individual frames independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AEConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num_groups(channels: int, preferred: int = 32) -> int:
    """Pick a valid group count for GroupNorm."""
    for g in (preferred, 16, 8, 4, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-norm residual block: GN → SiLU → Conv → GN → SiLU → Conv + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_num_groups(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(_num_groups(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        # Zero-init last conv so the block starts as identity
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Progressively downsamples [C, H, W] → GAP → embedding vector."""

    def __init__(self, in_channels: int, hidden_dims: list[int],
                 latent_dim: int, spatial_h: int, spatial_w: int):
        super().__init__()
        # Build downsample + residual stack
        layers: list[nn.Module] = []
        ch = in_channels
        for hd in hidden_dims:
            layers.append(nn.Conv2d(ch, hd, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(_num_groups(hd), hd))
            layers.append(nn.SiLU(inplace=True))
            layers.append(ResBlock(hd))
            ch = hd
        self.conv_stack = nn.Sequential(*layers)

        # Final norm before pooling — stabilises feature scales across channels
        self.final_norm = nn.Sequential(
            nn.GroupNorm(_num_groups(ch), ch),
            nn.SiLU(inplace=True),
        )

        # Global Average Pooling → feature vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection to embedding space
        self.fc = nn.Linear(ch, latent_dim)

        self._bottleneck_channels = ch

        # Compute bottleneck spatial size (needed by decoder to reshape)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, spatial_h, spatial_w)
            out = self.conv_stack(dummy)
            self._feat_shape = out.shape[1:]   # (C', H', W')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, H, W] where N = B*S (flattened batch × sequence)
        Returns:
            embed: [N, latent_dim]
        """
        h = self.conv_stack(x)
        h = self.final_norm(h)
        h = self.gap(h).flatten(1)             # [N, C']
        return self.fc(h)


class LatentAE(nn.Module):
    """
    Encoder for video latent-frame retrieval.

    Input:  (B*S, C, H, W)  e.g. (B*S, 16, 60, 104)
    Latent: (B*S, latent_dim)  e.g. (B*S, 1024)
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            hidden_dims=cfg.hidden_dims,
            latent_dim=cfg.latent_dim,
            spatial_h=cfg.spatial_h,
            spatial_w=cfg.spatial_w,
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Re-apply zero-init for residual output projections
        # (the global kaiming/xavier init above overwrites their __init__ zero-init)
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.block[-1].weight)
                nn.init.zeros_(m.block[-1].bias)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

    # ── Encode only (for downstream retrieval) ────────────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor,
               normalize: bool = True) -> torch.Tensor:
        """Return the embedding, optionally L2-normalised for retrieval.

        Args:
            x: [N, C, H, W] where N = B*S
            normalize: L2-normalize for cosine similarity retrieval
        """
        embed = self.encoder(x)
        if normalize:
            embed = F.normalize(embed, dim=-1)
        return embed
