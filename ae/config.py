"""Inference-time configuration for the LongLive-RAG retrieval autoencoder."""

from dataclasses import dataclass, field


@dataclass
class AEConfig:
    in_channels: int = 16
    spatial_h: int = 60
    spatial_w: int = 104
    latent_dim: int = 1024
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256])
