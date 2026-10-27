#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {360m-stage1|360m-stage2|1b} [Hydra overrides ...]" >&2
  exit 2
fi

case "$1" in
  360m-stage1) experiment="llops/continual_360m_stage1" ;;
  360m-stage2) experiment="llops/continual_360m_stage2" ;;
  1b)          experiment="llops/continual_1b" ;;
  *)
    echo "Unknown experiment: $1" >&2
    exit 2
    ;;
esac
shift

python -m train.run "experiment=${experiment}" "$@"
