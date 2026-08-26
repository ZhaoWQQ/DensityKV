#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
exec python "$ROOT/scripts/run_paper_case.py" --case figure3_kangaroo --gpu "$GPU" "$@"
