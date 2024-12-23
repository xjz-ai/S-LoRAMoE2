# # Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.
MODEL_NAME="qwen"
ADAPTER="LoRA"
BATCH_SIZE=64
BASE_MODEL="/home/model/Qwen2_5_3B"
LORA_WEIGHTS=""
# dataset = ['boolq', 'piqa','social_i_qa','winogrande','ARC-Easy','ARC-Challenge','openbookqa','hellaswag']
CUDA_VISIBLE_DEVICES=0 python ../commonsense_evaluate.py \
    --model "$MODEL_NAME" \
    --adapter "$ADAPTER" \
    --dataset boolq\
    --batch_size "$BATCH_SIZE" \
    --base_model "$BASE_MODEL" \
    --lora_weights "$LORA_WEIGHTS" | tee -a './result/boolq.txt'
