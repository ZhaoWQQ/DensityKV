#!/usr/bin/env python3
"""Materialize and run one retained qualitative paper case.

Each case is evaluated with a matched backbone, prompt, seed, sampler, and
rollout length. Only the history mechanism changes across ``base``, ``rag``,
and ``densitykv``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "configs/examples/paper_cases.yaml"
METHODS = ("base", "rag", "densitykv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "figure1_panda",
            "figure3_kangaroo",
            "figure5_rabbit",
            "sunglasses_man",
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Matched methods to run (default: all three).",
    )
    parser.add_argument("--gpu", default="0", help="CUDA device id.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--attention-trace",
        action="store_true",
        help="Record the all-layer temporal-attention trace used by Figure 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write resolved configs and print commands without inference.",
    )
    return parser.parse_args()


def required_assets(config) -> list[Path]:
    assets = [
        REPO_ROOT / str(config.generator_ckpt),
        REPO_ROOT / "wan_models" / str(config.model_name),
    ]
    lora = getattr(config, "lora_ckpt", None)
    if lora:
        assets.append(REPO_ROOT / str(lora))
    ae_checkpoint = getattr(config.model_kwargs, "ae_ckpt", None)
    if ae_checkpoint:
        assets.append(REPO_ROOT / str(ae_checkpoint))
    return assets


def materialize(spec, case_id: str, method_id: str, *, attention_trace: bool):
    case = spec.cases[case_id]
    backbone = spec.backbones[str(case.backbone)]
    method = spec.methods[method_id]
    case_values = OmegaConf.to_container(case, resolve=True)
    assert isinstance(case_values, dict)
    case_values.pop("backbone")

    config = OmegaConf.merge(
        spec.common,
        backbone,
        method,
        OmegaConf.create(case_values),
    )
    output_root = REPO_ROOT / "outputs" / "paper_cases" / case_id
    run_id = f"{method_id}_trace" if attention_trace else method_id
    config.output_folder = str(output_root / run_id)
    config.data_path = str(REPO_ROOT / str(config.data_path))
    if attention_trace:
        config.skip_existing = False
        config.model_kwargs.temporal_attention_trace = {
            "enabled": True,
            "layers": "all",
            "heads": "all",
            "denoise_calls": [3],
            "query_chunk_size": 32,
        }

    config_dir = output_root / "resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    return config, config_path


def main() -> int:
    args = parse_args()
    spec = OmegaConf.load(args.spec)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for method_id in args.methods:
        config, config_path = materialize(
            spec,
            args.case,
            method_id,
            attention_trace=args.attention_trace,
        )
        missing = [path for path in required_assets(config) if not path.exists()]
        if missing and not args.dry_run:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing required assets:\n{formatted}")

        command = [
            sys.executable,
            str(REPO_ROOT / "inference.py"),
            "--config_path",
            str(config_path),
        ]
        print(f"[{args.case}/{method_id}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
