#!/bin/bash
#SBATCH --job-name=sensor_lora
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=sensor_lora_%j.out

# LoRA fine-tune Qwen2.5-VL on the training split, then evaluate on the test split.
# Run from the repo root:
#   sbatch data/sensor/delta/train_sensor_lora.sbatch image
#   sbatch data/sensor/delta/train_sensor_lora.sbatch text
#   sbatch data/sensor/delta/train_sensor_lora.sbatch image_text
#   sbatch data/sensor/delta/train_sensor_lora.sbatch image_skeleton_text
#   sbatch --time=12:00:00 data/sensor/delta/train_sensor_lora.sbatch poses

set -eo pipefail
MODE=${1:-image}
BASE=${BASE:-Qwen/Qwen2.5-VL-7B-Instruct}
OUT=sensor_work/lora/$(basename "$BASE" | tr '[:upper:].-' '[:lower:]__')_${MODE}
# Pose images are 280x280; the default minimum (256*28*28) would upscale each to 448x448.
if [ "$MODE" = "poses" ]; then MIN_PIXELS=$((280 * 280)); else MIN_PIXELS=$((256 * 28 * 28)); fi

module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"
conda activate cvfm4rehab
export STROKEREHAB_SENSOR_DIR=${STROKEREHAB_SENSOR_DIR:-sensor_work/trials_25hz}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl

python -m data.sensor.train_lora \
  --train "sensor_work/sft/training_${MODE}.jsonl" \
  --output_dir "$OUT" \
  --base "$BASE" \
  --min_pixels "$MIN_PIXELS"

TASK=strokerehab_sensor_identification_${MODE}
python -m lmms_eval \
  --model qwen2_5_vl_sensor \
  --model_args "pretrained=${BASE},adapter=${OUT},min_pixels=${MIN_PIXELS}" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --output_path "logs/${TASK}/$(basename "$OUT")"
