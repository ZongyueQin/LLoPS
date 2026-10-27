# Continual-training reproduction

The paper experiments replace each BASED global linear-attention layer with a
learnable low-rank polynomial sketch and continue language-model training on
`DKYoon/SlimPajama-6B`. The local sliding-attention and convolution layers are
left unchanged.

## Recovered paper runs

| Scale | Initializer | LLoPS kernel | Optimizer steps | Learning rate | Per-GPU batch | Grad accumulation |
|---|---|---|---:|---:|---:|---:|
| 360M, stage 1 | `hazyresearch/based-360m` | degree 4, no linear term | 18,000 | 2e-4 | 10 | 2 |
| 360M, stage 2 | stage 1 `step_18000-hf` | degree 4, no linear term | 33,000 after restart | 2e-4 | 10 | 2 |
| 1B | `hazyresearch/based-1b` | degree 2 plus linear term | 2,000 | 1e-5 | 2 | 16 |

The 360M result therefore follows two optimizer runs. Stage 2 intentionally
restarts optimizer and scheduler state; it is not a Lightning resume.

All runs used sequence length 2,048, AdamW (`betas=[0.9, 0.95]`, weight decay
0.1), 200 warmup steps, and the SlimPajama-6B Hugging Face dataset.

## Commands

Set cache/output locations before training:

```bash
export DATA_DIR=/path/to/hf-cache
export CHECKPOINT_DIR=/path/to/checkpoints
export NUM_GPUS=8
```

Run 360M stage 1, then point stage 2 at its HF checkpoint:

```bash
scripts/train_continual.sh 360m-stage1
export INIT_MODEL_NAME="$CHECKPOINT_DIR/llops-360m-stage1/step_18000-hf"
scripts/train_continual.sh 360m-stage2
```

Run 1B:

```bash
unset INIT_MODEL_NAME
scripts/train_continual.sh 1b
```

For a smoke test, append overrides such as
`trainer.devices=1 trainer.max_steps=2 datamodule.num_workers=2 logger=null`.

## Slurm / MI210 example

On the cluster used during recovery, the single-GPU partition assigns the GPU
at the partition level and rejects `--gres=gpu:1`. A smoke-test or evaluation
allocation can therefore be created as follows:

```bash
salloc --partition=mi2101x --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --mem=64G --time=12:00:00

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate YOUR_ENV
cd /path/to/llops
rocm-smi
```

If the shell disconnects while the allocation is still running, find it with
`squeue -u "$USER"` and reconnect with:

```bash
srun --jobid=JOB_ID --overlap --pty bash -l
```

Exact eight-GPU training requires a partition/allocation that exposes eight
devices; its Slurm flags are cluster-specific. Set `NUM_GPUS` to the number of
visible devices and override batch/accumulation only if intentionally changing
the effective batch size.

## Zero-shot evaluation

New checkpoints contain their LLoPS architecture metadata:

```bash
scripts/evaluate_zero_shot.sh /path/to/checkpoint
```

For legacy checkpoints that lack metadata, pass power and the linear-term flag:

```bash
# 360M
scripts/evaluate_zero_shot.sh /path/to/last-hf 2 false

# 1B
scripts/evaluate_zero_shot.sh /path/to/step_2000-hf 1 true
```

The nine tasks are WinoGrande, PIQA, LAMBADA, HellaSwag, ARC-Easy,
ARC-Challenge, FDA, SWDE, and SQuAD-Completion, all at zero shot.
