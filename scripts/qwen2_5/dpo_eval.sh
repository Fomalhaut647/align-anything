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

REWARD_MODEL_PATH="output/qwen_2_5_rm/slice_end" # 奖励模型路径
EVALUATION_DATA="scripts/qwen2_5/evaluation_results/evaluation_results.json" # 评估数据路径
OUTPUT_DIR="scripts/qwen2_5/reward_analysis" # 输出目录

# For wandb online logging
export WANDB_API_KEY="94a37f280ad8b0c5600c5c4e12f4719f3150a6bc"

# Source the setup script
source scripts/setup.sh

echo "=== 使用奖励模型评估DPO和基础模型的回答差异 ==="
echo "奖励模型路径: ${REWARD_MODEL_PATH}"
echo "评估数据路径: ${EVALUATION_DATA}"
echo "输出目录: ${OUTPUT_DIR}"

# 执行Python评估脚本
python scripts/qwen2_5/dpo_eval.py \
    --reward_model_path ${REWARD_MODEL_PATH} \
    --evaluation_data ${EVALUATION_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --device npu:0 \
    --max_samples 50

echo "=== 评估完成 ==="
echo "结果保存在: ${OUTPUT_DIR}" 