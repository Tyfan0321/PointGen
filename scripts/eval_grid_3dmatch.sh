#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/py310/bin/python}"
RUN_DIR="${RUN_DIR:-outputs/mode_0_v_pred_v_loss_v0_global_std/3dmatch/2026-05-12}"
CKPTS=(${CKPTS:-epoch-124 epoch-119 epoch-134})
STEPS=(${STEPS:-5 10})
GPUS=(${GPUS:-0 1 2 3})
MAX_JOBS="${MAX_JOBS:-${#GPUS[@]}}"
NUM_GEN_SAMPLES="${NUM_GEN_SAMPLES:-null}"
USE_OVERLAP_METRICS="${USE_OVERLAP_METRICS:-false}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${RUN_DIR}/eval/logs"

running_jobs=0
job_index=0

for ckpt in "${CKPTS[@]}"; do
  ckpt_path="${RUN_DIR}/ckpt/${ckpt}"
  if [[ ! -d "${ckpt_path}" ]]; then
    echo "[skip] Missing checkpoint: ${ckpt_path}" >&2
    continue
  fi

  for steps in "${STEPS[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    overlap_part="overlap-off"
    if [[ "${USE_OVERLAP_METRICS}" == "true" ]]; then
      overlap_part="overlap-on"
    fi

    sample_part="full"
    if [[ "${NUM_GEN_SAMPLES}" != "null" ]]; then
      sample_part="samples-${NUM_GEN_SAMPLES}"
    fi

    config_name="steps-${steps}_${sample_part}_scheduler-fm-euler_${overlap_part}_seed-1_gt"
    log_path="${RUN_DIR}/eval/logs/${ckpt}_${config_name}.log"

    echo "[launch] gpu=${gpu} ckpt=${ckpt} steps=${steps} log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" eval.py \
      resume="${ckpt_path}" \
      num_inference_steps="${steps}" \
      num_gen_samples="${NUM_GEN_SAMPLES}" \
      eval.use_overlap_metrics="${USE_OVERLAP_METRICS}" \
      num_workers="${NUM_WORKERS}" \
      > "${log_path}" 2>&1 &

    running_jobs=$((running_jobs + 1))
    job_index=$((job_index + 1))

    if (( running_jobs >= MAX_JOBS )); then
      wait -n
      running_jobs=$((running_jobs - 1))
    fi
  done
done

wait
echo "[done] all evaluation jobs finished"
