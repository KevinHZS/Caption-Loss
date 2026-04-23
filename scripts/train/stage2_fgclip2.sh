export NCCL_IB_GID_INDEX=5

INIT_MODEL_PATH="/gemini/space/gjx/FG-CLIP"



name="models--qihoo360--fg-clip2-base"

en_data_path="en_pairs/"
en_img_root="en_images"
cn_root="cn_pairs/"
cn_img_root="cn_images"

data_path="/gemini/space/gjx/FG-CLIP/data/FineHARD/json_files"
img_root="/gemini/space/gjx/FG-CLIP/data/grit-12m"


deepspeed fgclip2/train/train.py \
    --deepspeed /gemini/space/gjx/FG-CLIP/scripts/zero2.json \
    --base_model $INIT_MODEL_PATH/$name \
    --model_name_or_path $INIT_MODEL_PATH/$name \
    --data_path $en_data_path \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --from_siglip2 True \
    --naflex_train True \
    --max_num_patches 1024 \
    --image_folder $img_root \
    --output_dir /gemini/space/gjx/FG-CLIP/output \
    --train_use_word_size 8 \
    --add_box_loss True \
    --use_hard_neg True \
    --box_image_size 512 \
    --base_seq_length 64 \
    --max_seq_length 196 \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 6 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 36 \
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
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none" \

