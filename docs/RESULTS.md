# Recovered checkpoint results

Published checkpoints: [LLoPS-360M](https://huggingface.co/zongyueq/LLoPS-360M)
and [LLoPS-1B](https://huggingface.co/zongyueq/LLoPS-1B).

These checkpoints were identified by running the repository's zero-shot suite.
Values are percentages except LAMBADA perplexity, which is omitted here in
favor of LAMBADA accuracy. Small differences from the paper are within the
reported evaluation variation and/or rounding.

| Model | Wino | PIQA | LAMBADA | HellaSwag | ARC-E | ARC-C | FDA | SWDE | SQuAD | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 360M paper | 53.04 | 66.70 | 29.23 | 37.07 | 43.94 | 25.51 | 7.53 | 31.41 | 31.94 | 36.26 |
| 360M recovered `last-hf` | 52.09 | 66.87 | 29.25 | 37.06 | 43.81 | 25.43 | 7.44 | 31.41 | 32.21 | 36.17 |
| 1B paper | 51.62 | 68.72 | 38.11 | 42.69 | 47.31 | 27.13 | 26.32 | 47.97 | 36.36 | 42.91 |
| 1B recovered `step_2000-hf` | 51.38 | 68.82 | 38.06 | 42.76 | 47.14 | 26.96 | 26.13 | 48.06 | 36.26 | 42.84 |

Recovered local paths (not intended for public documentation):

- 360M: `amd-0118-based-360m-sketch-ours-18000-00/last-hf`
- 1B: `amd-0124-slimpajama-based-1b-based-ours-02/step_2000-hf`

Before uploading legacy checkpoints, copy them to a release staging directory
and add architecture metadata:

```bash
scripts/add_llops_metadata.py /staging/llops-360m --power 2
scripts/add_llops_metadata.py /staging/llops-1b --power 1 --use-linear-term
```
