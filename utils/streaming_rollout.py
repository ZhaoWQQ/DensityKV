"""Disk-backed latent shards and streaming causal VAE video encoding."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch


class LatentShardWriter:
    def __init__(self, output_dir: str | Path, shard_frames: int) -> None:
        if shard_frames <= 0:
            raise ValueError("shard_frames must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_frames = int(shard_frames)
        self._parts: list[torch.Tensor] = []
        self._buffered_frames = 0
        self._next_frame = 0
        self.paths: list[Path] = []

    def __call__(self, start_frame: int, latent: torch.Tensor) -> None:
        if int(start_frame) != self._next_frame:
            raise RuntimeError(
                f"non-contiguous latent stream: expected {self._next_frame}, got {start_frame}"
            )
        part = latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        self._parts.append(part)
        frames = int(part.shape[1])
        self._buffered_frames += frames
        self._next_frame += frames
        while self._buffered_frames >= self.shard_frames:
            self._flush_prefix(self.shard_frames)

    def _flush_prefix(self, frame_count: int) -> None:
        merged = torch.cat(self._parts, dim=1)
        shard = merged[:, :frame_count].contiguous()
        remainder = merged[:, frame_count:].contiguous()
        start = self._next_frame - self._buffered_frames
        end = start + frame_count
        path = self.output_dir / f"latent_{start:06d}_{end - 1:06d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(
            {"start_frame": start, "end_frame": end, "latents": shard},
            tmp_path,
        )
        os.replace(tmp_path, path)
        self.paths.append(path)
        self._parts = [remainder] if remainder.shape[1] else []
        self._buffered_frames -= frame_count

    def close(self) -> list[Path]:
        if self._buffered_frames:
            self._flush_prefix(self._buffered_frames)
        manifest = {
            "num_latent_frames": self._next_frame,
            "shard_frames": self.shard_frames,
            "shards": [path.name for path in self.paths],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return list(self.paths)


def _iter_latent_chunks(
    shard_paths: Iterable[str | Path], chunk_frames: int
) -> Iterable[torch.Tensor]:
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        latent = payload["latents"]
        for start in range(0, latent.shape[1], chunk_frames):
            yield latent[:, start : start + chunk_frames]


@torch.no_grad()
def decode_shards_to_mp4(
    *,
    vae: Any,
    shard_paths: Iterable[str | Path],
    output_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    chunk_frames: int = 12,
    fps: int = 16,
    crf: int = 18,
) -> dict[str, int]:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.mp4")
    ffmpeg_log_path = output_path.with_suffix(".ffmpeg.log")
    process = None
    log_file = ffmpeg_log_path.open("wb")
    decoded_frames = 0
    width = 0
    height = 0
    vae.begin_stream_decode()
    try:
        for latent_cpu in _iter_latent_chunks(shard_paths, chunk_frames):
            latent = latent_cpu.to(device=device, dtype=dtype)
            decoded = vae.decode_to_pixel_stream_chunk(latent)
            del latent
            pixels = (
                (decoded * 0.5 + 0.5)
                .clamp_(0, 1)
                .mul_(255)
                .to(torch.uint8)[0]
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            if process is None:
                height, width = int(pixels.shape[1]), int(pixels.shape[2])
                command = [
                    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial_path),
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log_file)
            assert process.stdin is not None
            process.stdin.write(pixels.numpy().tobytes())
            decoded_frames += int(pixels.shape[0])
            del decoded, pixels
            torch.cuda.empty_cache()
        if process is None:
            raise RuntimeError("no latent shards were decoded")
        assert process.stdin is not None
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}; see {ffmpeg_log_path}"
            )
        os.replace(partial_path, output_path)
    finally:
        vae.end_stream_decode()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        log_file.close()
    return {
        "decoded_frames": decoded_frames,
        "width": width,
        "height": height,
        "fps": int(fps),
    }
