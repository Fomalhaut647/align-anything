#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

source /usr/local/Ascend/ascend-toolkit/set_env.sh

MODEL_NAME_OR_PATH="output/qwen_2_5_rm/slice_end" # 奖励模型路径

EVAL_DATASETS="scripts/qwen2_5/evaluation_results" # 数据集路径
EVAL_TEMPLATE="DPOEval" # 使用DPOEval模板
EVAL_SPLIT="evaluation_results.json" # 直接指定JSON文件名

OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR

if [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "OUTPUT_ROOT_DIR is not set"
    OUTPUT_ROOT_DIR="output"
fi

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5_rm_dpo_eval" # 输出目录

# For wandb online logging
export WANDB_API_KEY="94a37f280ad8b0c5600c5c4e12f4719f3150a6bc"

# Source the setup script
source scripts/setup.sh

echo "=== 使用奖励模型评估DPO数据 ==="
echo "奖励模型路径: ${MODEL_NAME_OR_PATH}"
echo "评估数据路径: ${EVAL_DATASETS}"
echo "评估模板: ${EVAL_TEMPLATE}"
echo "输出目录: ${OUTPUT_DIR}"

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 1 \
     --epochs 1 \
     --per_device_eval_batch_size 1 \
     --eval_strategy epoch

echo "=== 评估完成 ==="
echo "结果保存在: ${OUTPUT_DIR}" 