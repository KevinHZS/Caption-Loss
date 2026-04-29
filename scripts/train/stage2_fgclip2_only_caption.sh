export NCCL_IB_GID_INDEX=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/gemini/space/zyf/FG-CLIP:$PYTHONPATH

# INIT_MODEL_PATH="/hbox2dir"
LLM_MODEL_PATH="/gemini/space/zyf/models/Qwen/Qwen3-1.7B"

ROOT="/gemini/space/zyf/FG-CLIP"
MODEL_DIR="/gemini/space/gjx/FG-CLIP/models--qihoo360--fg-clip2-base"
DATA_PATH="/gemini/space/gjx/FG-CLIP/data/FineHARD/debug_coyo0_00000_exact_small.json"
IMG_ROOT="/gemini/space/gjx/FG-CLIP/data"
LOG_DIR="$ROOT/output/smoke_debug_8gpu_all_checkgpu_bs512_patch1024"
LONG_CAPTION_LOSS_WEIGHT="${LONG_CAPTION_LOSS_WEIGHT:-1.0}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  -l 2 > "$LOG_DIR/gpu_usage.csv" &
MON_PID=$!

trap 'kill $MON_PID 2>/dev/null' EXIT

deepspeed fgclip2/train/train.py \
    --deepspeed "$ROOT/scripts/zero2.json" \
    --base_model "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMG_ROOT" \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --long_loss_weight "${LONG_LOSS_WEIGHT:-1.0}" \
    --short_loss_weight "${SHORT_LOSS_WEIGHT:-1.0}" \
    --from_siglip2 False \
    --naflex_train True \
    --max_num_patches 1024 \
    --output_dir "$LOG_DIR" \
    --train_use_word_size 8 \
    --add_box_loss False \
    --use_hard_neg False \
    --box_image_size 512 \
    --base_seq_length 64 \
    --max_seq_length 196 \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none" \
    --long_caption_loss_weight "$LONG_CAPTION_LOSS_WEIGHT" \
    --llm_model_path $LLM_MODEL_PATH \
    --llm_gradient_checkpointing True \

kill $MON_PID 2>/dev/null
trap - EXIT
