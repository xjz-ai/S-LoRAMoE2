# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
CUDA_VISIBLE_DEVICES=0  nohup python ../finetune.py  \
    --base_model '/Qwen2_5_3B' \
    --data_path '/commonsense_170k.json' \
    --output_dir  \
    --batch_size 8  --micro_batch_size 1 --num_epochs 1.2 \
    --learning_rate 5e-5 --cutoff_len 512 --val_set_size 0 \
    --eval_step 5000 --save_step 3000  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj","o_proj","gate_proj" ,"up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --use_gradient_checkpointing  > lora170ksenti.log 2>&1 &
                        
    # --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \