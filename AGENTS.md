# AGENTS.md

This file gives coding agents the local context needed to work in this repository safely and productively. It applies to the whole repo.

## Project Overview

PointGen is a PyTorch project for promptable/generative 3D dense matching and point-cloud registration. The current active path is a Flow Matching style generative model over coarse source points, conditioned on reference/source point-cloud encoder features.

Core areas:

- `train.py`: Hydra entry point for training. It uses `config/config_v0.yaml` by default and instantiates `src.engine.DiffusionTrainer`.
- `eval.py`: standalone evaluation entry point. It also uses `config/config_v0.yaml`.
- `src/engine/`: training loop, data preparation for diffusion/flow matching, and evaluation orchestration.
- `src/models/`: REGTR-style generative transformer, inference pipeline, encoders, KPConv, and Sonata integration.
- `src/data/`: dataset implementations and `DatasetFactory`.
- `config/`: Hydra configs, including encoder-specific configs under `config/encoder/`.
- `data/`: dataset pair lists and benchmark metadata. Raw point-cloud datasets are expected outside the repo.
- `outputs/`: generated experiment outputs. Treat as disposable/generated unless the user explicitly asks about logs or checkpoints.

## Environment

The README records the tested stack as Ubuntu 20.04, Python 3.7, PyTorch 1.9.0, CUDA 11.2, PyTorch3D 0.6.2, Open3D 0.16.0, MinkowskiEngine, Diffusers, Accelerate, Hydra, and OmegaConf. The code also imports `torch_scatter` and Sonata model code/checkpoints.

Before assuming dependencies are missing, inspect the active environment. Some machines may already have CUDA-specific packages installed outside `requirements.txt`.

## Common Commands

Run from the repository root.

```bash
python train.py
```

```bash
accelerate launch train.py
```

```bash
python eval.py resume=./outputs/<experiment>/<run>/<date>/ckpt/epoch-124
```

Useful Hydra overrides:

```bash
python train.py data.root=/path/to/3DMatch resume=null mixed_precision=no
python train.py defaults='[_self_,encoder:3dmatch/kpconv]'
python train.py num_train_epochs=1 val_epochs=1 ckpt_epochs=1 num_workers=0
```

For quick import/syntax checks, prefer targeted commands because full training/evaluation requires datasets, pretrained checkpoints, CUDA, and compiled extensions:

```bash
python -m py_compile train.py eval.py src/engine/*.py src/models/*.py src/data/*.py src/utils/*.py
```

## Configuration Notes

- Training and evaluation default to `config/config_v0.yaml`, not `config/config.yaml`.
- `output_dir` and `hydra.run.dir` both resolve to `./outputs/${experiment_name}/${runname}/${now:%Y-%m-%d}`.
- Checkpoints are saved by Accelerate under `ckpt/epoch-<N>`.
- TensorBoard logs go under the configured `log_with` directory, currently `tensorboard`.
- `data.dataset_type` is currently expected to be `3dmatch` or `kitti`.
- Encoder config is selected through Hydra defaults, e.g. `encoder: 3dmatch/sonata`.
- The Sonata config currently points at an absolute checkpoint path. Do not replace user-specific dataset or checkpoint paths unless asked.

## Data Contract

Dataset samples should return a dict with these keys:

- `ref_points`: tensor shaped `[N, 3]`
- `src_points`: tensor shaped `[M, 3]`
- `Tr`: rigid transform shaped `[4, 4]`, mapping source points into the reference frame
- `ref_overlap`, `src_overlap`: optional overlap masks used by processors/evaluation paths

The DataLoader batch size is intentionally `1` in `DiffusionTrainer.prepare_data()`. The diffusion training batch is created inside `DiffusionDataProcessor.prepare_noisy_data()` by expanding the coarse target to `cfg.train_batch_size`. Do not change the DataLoader batch size casually.

For `3dmatch`, `DatasetFactory.create(..., seqs="test")` returns two datasets: `3DMatch` and `3DLoMatch`. The trainer currently uses the first one for validation; standalone `eval.py` iterates both.

## Model And Processor Flow

The high-level training path is:

1. `train.py`
2. `DiffusionTrainer.fit()`
3. `DatasetFactory.create()`
4. `DiffusionDataProcessor.prepare_noisy_data()`
5. `RegTrGenerative.forward()`
6. `DiffusionDataProcessor.compute_loss()`
7. optional `DiffusionEvaluator.evaluate()`

Processor and encoder types must stay aligned:

- `processor.type == "sonata"` pairs with `encoder.type == "sonata"`.
- `processor.type == "kpconv"` pairs with `encoder.type == "kpconv"`.

Sonata processor output includes point dictionaries and an optional pooling cache. KPConv processor output includes point, neighbor, subsampling, and length lists. Preserve these structure contracts when editing the model or processors.

## Flow Matching Sign Convention

Training currently learns:

```python
v = target - noise
```

Inference in `PointGenPipeline.__call__()` passes `-v_pred` to the Diffusers `FlowMatchEulerDiscreteScheduler`:

```python
sample = self.scheduler.step(-v_pred, t, sample).prev_sample
```

Treat this sign convention as deliberate. `test_fm.py` is a small sanity script for Diffusers' expected velocity direction.

## Editing Guidelines

- Keep changes scoped. This repo often has active experiment edits; do not revert unrelated modifications.
- Do not commit generated files from `outputs/`, checkpoints, caches, or raw datasets.
- Prefer Hydra config changes over hard-coded constants when behavior is experiment-level.
- Preserve absolute dataset/checkpoint paths in configs unless the task is specifically to make them portable.
- Be careful with CPU/GPU assumptions. `src/utils/point_cloud_utils.py` uses CUDA-oriented utilities and `weighted_svd()` currently creates its transform with `.cuda()`.
- Avoid broad refactors in `src/models/sonata/` and `src/models/kpconv/` unless the user explicitly asks; these are dependency-heavy model components.
- When touching training stability, check non-finite loss/gradient handling in `DiffusionTrainer.train_step()`.

## Verification Expectations

Use the lightest verification that exercises the changed code:

- Config or documentation only: no runtime test required.
- Python surface changes: run `python -m py_compile` on touched modules.
- Dataset changes: instantiate the relevant dataset with a tiny local/configured path only if data is available.
- Training-loop/model changes: run a minimal one-epoch or import-level check when CUDA, data, and pretrained weights are available; otherwise state what could not be run.
- Evaluation changes: prefer `python eval.py resume=<ckpt>` only when the checkpoint, raw dataset, and GPU stack are available.

## Logging And Outputs

Training logs and Hydra output files live under `outputs/<experiment>/<run>/<date>/`. Evaluation appends JSONL metrics such as `metrics_3DMatch.jsonl` and `metrics_3DLoMatch.jsonl` under an `eval` directory derived from the checkpoint path.

When reading logs, prefer `rg`, `tail`, and small `sed` ranges. Some logs are large.
