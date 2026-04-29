#!/usr/bin/env bash
set -o pipefail

export NCCL_IB_GID_INDEX=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/gemini/space/zyf/FG-CLIP:$PYTHONPATH
export WANDB_PROJECT="fgclip-stage1-caption"
export WANDB_MODE="offline"
export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"

LLM_MODEL_PATH="/gemini/space/zyf/models/Qwen/Qwen3-1.7B"

ROOT="/gemini/space/zyf/FG-CLIP"
DATA_ROOT="/gemini/space/gjx/FG-CLIP"
MODEL_DIR="$DATA_ROOT/siglip2-so400m-patch16-naflex"
DATA_WORK_DIR="$DATA_ROOT/data/TeleMM"
DENSE_SOURCE="$DATA_WORK_DIR/DenseFusion-1M_cleaned.jsonl"
PT_SAMPLE_PATH="$DATA_WORK_DIR/pt_llava-ov-mid-v1_sample1M_cleaned.jsonl"
DATA_PATH="${DATA_PATH:-$DATA_WORK_DIR/stage1_longonly_2M_cleaned_manifest.txt}"
IMG_ROOT="${IMG_ROOT:-$DATA_ROOT/data}"
LOG_DIR="${LOG_DIR:-$ROOT/output/stage1_siglip2_caption_pool_2x2}"
USE_SHORT_CAPTION_CONTRASTIVE_LOSS="${USE_SHORT_CAPTION_CONTRASTIVE_LOSS:-False}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-50000000}"
LONG_CAPTION_LOSS_WEIGHT="${LONG_CAPTION_LOSS_WEIGHT:-1.0}"
SHORT_CAPTION_LOSS_WEIGHT="${SHORT_CAPTION_LOSS_WEIGHT:-0.5}"
RUN_NAME="${RUN_NAME:-stage1_siglip2_caption_pool2x2_longonly_2M_cleaned_bs256_lr1e-6_proj1e-5_cap1.0_short${USE_SHORT_CAPTION_CONTRASTIVE_LOSS}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
mkdir -p "$DATA_WORK_DIR"
cd "$ROOT"

TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$TRAIN_LOG") 2>&1

echo "Training log: $TRAIN_LOG"
echo "Started at: $(date)"
echo "ROOT=$ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "USE_SHORT_CAPTION_CONTRASTIVE_LOSS=$USE_SHORT_CAPTION_CONTRASTIVE_LOSS"
echo "LONG_CAPTION_LOSS_WEIGHT=$LONG_CAPTION_LOSS_WEIGHT"
echo "SHORT_CAPTION_LOSS_WEIGHT=$SHORT_CAPTION_LOSS_WEIGHT"
echo "MAX_IMAGE_PIXELS=$MAX_IMAGE_PIXELS"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_MODE=$WANDB_MODE"
echo "RUN_NAME=$RUN_NAME"

printf "%s\n%s\n" "$DENSE_SOURCE" "$PT_SAMPLE_PATH" > "$DATA_PATH"
echo "Manifest:"
cat "$DATA_PATH"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  -l 2 > "$LOG_DIR/gpu_usage.csv" &
MON_PID=$!

cleanup() {
    status=$?
    kill $MON_PID 2>/dev/null
    echo "Finished at: $(date)"
    echo "Exit status: $status"
    echo "Training log: $TRAIN_LOG"
    exit $status
}

trap cleanup EXIT

deepspeed fgclip2/train/train.py \
    --deepspeed "$ROOT/scripts/zero2.json" \
    --base_model "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMG_ROOT" \
    --missing_image_log_path "$LOG_DIR/missing_images.jsonl" \
    --large_image_log_path "$LOG_DIR/large_images.jsonl" \
    --max_image_pixels "$MAX_IMAGE_PIXELS" \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --long_loss_weight "${LONG_LOSS_WEIGHT:-1.0}" \
    --short_loss_weight "${SHORT_LOSS_WEIGHT:-1.0}" \
    --from_siglip2 True \
    --naflex_train True \
    --max_num_patches 1024 \
    --output_dir "$LOG_DIR" \
    --train_use_word_size 8 \
    --add_box_loss False \
    --use_hard_neg False \
    --box_image_size 512 \
    --base_seq_length 64 \
    --max_seq_length 196 \
    --use_short_caption_contrastive_loss "$USE_SHORT_CAPTION_CONTRASTIVE_LOSS" \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --save_strategy "steps" \
    --save_steps 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name "$RUN_NAME" \
    --long_caption_loss_weight "$LONG_CAPTION_LOSS_WEIGHT" \
    --short_caption_loss_weight "$SHORT_CAPTION_LOSS_WEIGHT" \
    --llm_model_path "$LLM_MODEL_PATH" \
    --llm_gradient_checkpointing True \
    --projector_lr 1e-5 \
    --caption_pool_2x2_tokens True
