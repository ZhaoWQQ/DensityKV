from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/examples/paper_cases.yaml"


def test_retained_cases_are_exact() -> None:
    spec = OmegaConf.load(SPEC)
    assert set(spec.cases) == {
        "figure1_panda",
        "figure3_kangaroo",
        "figure5_rabbit",
        "sunglasses_man",
    }
    assert set(spec.methods) == {"base", "rag", "densitykv"}
    assert set(spec.backbones) == {
        "causal_forcing",
        "self_forcing",
        "longlive",
    }
    assert spec.cases.figure1_panda.seed == 13
    assert spec.cases.figure3_kangaroo.seed == 0
    assert spec.cases.figure5_rabbit.seed == 0
    assert spec.cases.sunglasses_man.seed == 0
    assert spec.cases.figure3_kangaroo.idx_offset == 12


def test_native_base_has_no_external_history() -> None:
    base = OmegaConf.load(SPEC).methods.base.model_kwargs
    assert "use_latentmem" not in base
    assert "density_kv" not in base


def test_densitykv_matches_final_paper_setting() -> None:
    density = OmegaConf.load(SPEC).methods.densitykv.model_kwargs.density_kv
    assert OmegaConf.to_container(density, resolve=True) == {
        "enabled": True,
        "capacity": 9360,
        "local_window_frames": 5,
        "logical_precommit": True,
        "density_scale": 8.0,
        "riesz_power": 2.0,
        "riesz_eps": 1.0,
        "density_growth_limit": 2.0,
        "density_baseline_floor": 1.0e-6,
        "work_chunk_size": 128,
        "compute_dtype": "bfloat16",
        "fast_impl": "auto",
    }
