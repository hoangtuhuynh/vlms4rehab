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

set -eo pipefail
MODE=${1:-image}
BASE=${BASE:-Qwen/Qwen2.5-VL-7B-Instruct}
OUT=sensor_work/lora/$(basename "$BASE" | tr '[:upper:].-' '[:lower:]__')_${MODE}

module load pytorch-conda/2.8
eval "$(conda shell.bash hook)"
conda activate cvfm4rehab
export STROKEREHAB_SENSOR_DIR=${STROKEREHAB_SENSOR_DIR:-sensor_work/trials_25hz}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl

python -m data.sensor.train_lora \
  --train "sensor_work/sft/training_${MODE}.jsonl" \
  --output_dir "$OUT" \
  --base "$BASE"

TASK=strokerehab_sensor_identification_${MODE}
python -m lmms_eval \
  --model qwen2_5_vl_sensor \
  --model_args "pretrained=${BASE},adapter=${OUT}" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --output_path "logs/${TASK}/$(basename "$OUT")"
