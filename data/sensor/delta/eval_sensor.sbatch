#!/bin/bash
#SBATCH --job-name=sensor_eval
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=sensor_eval_%j.out

# Zero-shot (or LoRA) Qwen2.5-VL on sensor-based activity identification.
# Run from the repo root:
#   sbatch data/sensor/delta/eval_sensor.sbatch image
#   sbatch data/sensor/delta/eval_sensor.sbatch text
#   sbatch data/sensor/delta/eval_sensor.sbatch image sensor_work/lora/qwen2_5_vl_7b_image
# Quick check first with LIMIT=5: sbatch --export=ALL,LIMIT=5 data/sensor/delta/eval_sensor.sbatch image

set -eo pipefail
MODE=${1:-image}
ADAPTER=${2:-}
BASE=${BASE:-Qwen/Qwen2.5-VL-7B-Instruct}

module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"
conda activate cvfm4rehab
export STROKEREHAB_SENSOR_DIR=${STROKEREHAB_SENSOR_DIR:-sensor_work/trials_25hz}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl

MODEL_ARGS="pretrained=${BASE}"
RUN_NAME=$(basename "$BASE")_zeroshot
if [ -n "$ADAPTER" ]; then
  MODEL_ARGS="${MODEL_ARGS},adapter=${ADAPTER}"
  RUN_NAME=$(basename "$ADAPTER")
fi

TASK=strokerehab_sensor_identification_${MODE}
python -m lmms_eval \
  --model qwen2_5_vl_sensor \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --output_path "logs/${TASK}/${RUN_NAME}" \
  ${LIMIT:+--limit $LIMIT}
