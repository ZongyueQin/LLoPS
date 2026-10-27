#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [POWER] [USE_LINEAR_TERM] [extra lm-eval args ...]" >&2
  exit 2
fi

checkpoint=$1
power=${2:-}
use_linear_term=${3:-}
if [[ $# -ge 3 ]]; then shift 3; else shift "$#"; fi

model_args="checkpoint_name=${checkpoint},arch=mysketch"
if [[ -n "${power}" ]]; then
  model_args+=",power=${power}"
fi
if [[ -n "${use_linear_term}" ]]; then
  model_args+=",use_linear_term=${use_linear_term}"
fi

lm_eval \
  --model based_lm \
  --model_args "${model_args}" \
  --tasks winogrande,swde,squad_completion,piqa,lambada_standard,hellaswag,fda,arc_easy,arc_challenge \
  --num_fewshot 0 \
  --batch_size "${EVAL_BATCH_SIZE:-8}" \
  "$@"
