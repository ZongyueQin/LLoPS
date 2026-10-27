# LLoPS: Learnable Low-Rank Polynomial Sketches

This repository contains the language-model implementation and continual-
training pipeline for **Learnable Low-Rank Polynomial Sketch for Effective
Linear Attention (LLoPS)**. It converts pretrained BASED linear-attention
layers into trainable polynomial sketches, continues training on
SlimPajama-6B, saves Hugging Face-format checkpoints, and evaluates the
paper's nine zero-shot NLP tasks.

The release is derived from [HazyResearch/based](https://github.com/HazyResearch/based)
and includes a vendored, task-extended LM Evaluation Harness under
[`evaluation/`](evaluation/).

## What is implemented

- Learnable quadratic and quartic polynomial feature maps for causal linear
  attention (`power=1` and `power=2`).
- Optional constant and learned linear terms used by the paper's 1B model.
- Conversion from pretrained BASED attention projections to LLoPS modules.
- Continued pretraining on `DKYoon/SlimPajama-6B` at sequence length 2,048.
- HF-format checkpoint export with self-describing LLoPS architecture metadata.
- Zero-shot evaluation on WinoGrande, PIQA, LAMBADA, HellaSwag, ARC-Easy,
  ARC-Challenge, FDA, SWDE, and SQuAD-Completion.

The central implementation is
[`based/models/gpt_unified.py`](based/models/gpt_unified.py). The paper configs
are under [`train/configs/experiment/llops/`](train/configs/experiment/llops/).

## Installation

The recovered experiments used Python 3.10, PyTorch with ROCm, and AMD MI210
GPUs. Install a PyTorch build appropriate for the machine first, then:

```bash
python -m pip install -e '.[train]'
python -m pip install -e ./evaluation
```

The fused training path also requires compatible FlashAttention and
causal-conv1d builds. These packages are sensitive to the PyTorch/CUDA/ROCm
version, so install the versions provided by the target cluster image.

## Reproduce continual training

```bash
export DATA_DIR=/path/to/hf-cache
export CHECKPOINT_DIR=/path/to/checkpoints
export NUM_GPUS=8

# 360M: two recovered optimizer stages
scripts/train_continual.sh 360m-stage1
export INIT_MODEL_NAME="$CHECKPOINT_DIR/llops-360m-stage1/step_18000-hf"
scripts/train_continual.sh 360m-stage2

# 1B: set this to the released BASED-1B initializer
export INIT_MODEL_NAME=/path/to/based-1b-initializer
scripts/train_continual.sh 1b
```

See [the reproduction notes](docs/REPRODUCTION.md) for exact hyperparameters,
the known 1B-initializer provenance limitation, smoke-test overrides, and Slurm
usage considerations.

## Evaluate

```bash
# New self-describing checkpoint
scripts/evaluate_zero_shot.sh /path/to/llops-checkpoint

# Legacy paper checkpoints without metadata
scripts/evaluate_zero_shot.sh /path/to/360m-last-hf 2 false
scripts/evaluate_zero_shot.sh /path/to/1b-step_2000-hf 1 true
```

The recovered checkpoint mapping and measured numbers are recorded in
[docs/RESULTS.md](docs/RESULTS.md).

## Repository layout

```text
based/       model, attention, and kernel code
train/       PyTorch Lightning training pipeline and Hydra configs
evaluation/  vendored LM Evaluation Harness with BASED recall tasks
scripts/     stable training/evaluation entry points
docs/        experiment provenance and recovered results
```

## Citation and license

Please cite the LLoPS paper and the BASED paper when using this code. The root
code is released under Apache-2.0; the vendored evaluation harness retains its
own license in `evaluation/LICENSE.md`. See [NOTICE](NOTICE) for provenance.
