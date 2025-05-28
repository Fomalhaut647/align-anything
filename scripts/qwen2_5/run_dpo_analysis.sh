#!/usr/bin/env bash

echo "=== DPO模型与基础模型回答差异分析 ==="

# 方法1: 使用简化的文本分析方法
echo "执行文本差异分析..."
python scripts/qwen2_5/analyze_dpo_results.py \
    --evaluation_data scripts/qwen2_5/evaluation_results/evaluation_results.json \
    --output_dir scripts/qwen2_5/dpo_analysis

echo "文本分析完成！"

# 方法2: 使用奖励模型进行评分（需要较多资源）
# echo "执行奖励模型评估..."
# bash scripts/qwen2_5/qwen_2_5_rm_dpo_eval.sh

echo "=== 分析完成 ==="
echo "查看结果："
echo "  - 文本分析结果: scripts/qwen2_5/dpo_analysis/"
echo "  - 可视化图表: scripts/qwen2_5/dpo_analysis/dpo_base_comparison.png"
echo "  - 详细报告: scripts/qwen2_5/dpo_analysis/analysis_report.txt" 