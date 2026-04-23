export NCCL_IB_GID_INDEX=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT="/gemini/space/gjx/FG-CLIP"
MODEL_DIR="$ROOT/models--qihoo360--fg-clip2-base"
DATA_PATH="$ROOT/data/FineHARD/debug_coyo0_00000_exact_small.json"
IMG_ROOT="$ROOT/data"
LOG_DIR="$ROOT/output/smoke_debug_8gpu_all_checkgpu_bs128_patch1024_gc_false"

mkdir -p "$LOG_DIR"
cd "$ROOT"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  -l 2 > "$LOG_DIR/gpu_usage.csv" &
MON_PID=$!

trap 'kill $MON_PID 2>/dev/null' EXIT

deepspeed --num_gpus 8 fgclip2/train/train.py \
    --deepspeed "$ROOT/scripts/zero2.json" \
    --base_model "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMG_ROOT" \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --from_siglip2 False \
    --naflex_train True \
    --max_num_patches 1024 \
    --output_dir "$LOG_DIR" \
    --train_use_word_size 8 \
    --add_box_loss True \
    --use_hard_neg True \
    --box_image_size 512 \
    --base_seq_length 64 \
    --max_seq_length 196 \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --save_strategy epoch \
    --save_total_limit 5 \
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
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none"

kill $MON_PID 2>/dev/null
trap - EXIT
