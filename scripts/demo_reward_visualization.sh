#!/usr/bin/env bash
#
# Demo script for reward model preference dataset visualization
#

echo "=========================================="
echo "奖励模型偏好数据集可视化演示"
echo "=========================================="

# Check if required packages are installed
echo "检查依赖包..."
python -c "import matplotlib, seaborn, scipy, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告：缺少可视化依赖包。正在安装..."
    pip install -r requirements_visualization.txt
fi

# Set up paths
MODEL_PATH="output/qwen_2_5_rm/slice_end"
OUTPUT_DIR="output/qwen_2_5_rm_eval"

echo ""
echo "配置信息："
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误：找不到奖励模型，请确保模型路径正确：$MODEL_PATH"
    echo "请先训练奖励模型或修改MODEL_PATH变量"
    exit 1
fi

echo ""
echo "步骤 1: 运行奖励模型评估..."
echo "----------------------------------------"

# Run reward model evaluation
bash scripts/qwen2_5/rm_eval.sh

if [ $? -ne 0 ]; then
    echo "错误：奖励模型评估失败"
    exit 1
fi

echo ""
echo "步骤 2: 生成详细可视化分析..."
echo "----------------------------------------"

# Generate detailed visualizations
python scripts/analyze_reward_scores.py --data_dir "${OUTPUT_DIR}/reward_analysis"

if [ $? -ne 0 ]; then
    echo "错误：可视化生成失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "可视化完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "1. 数据文件："
echo "   - ${OUTPUT_DIR}/reward_analysis/chosen_responses.json"
echo "   - ${OUTPUT_DIR}/reward_analysis/rejected_responses.json"
echo ""
echo "2. 可视化图表："
echo "   - ${OUTPUT_DIR}/reward_analysis/reward_score_visualization.png"
echo "   - ${OUTPUT_DIR}/reward_analysis/comprehensive_reward_analysis.png"
echo "   - ${OUTPUT_DIR}/reward_analysis/text_length_analysis.png"
echo ""
echo "3. 统计报告："
echo "   - ${OUTPUT_DIR}/reward_analysis/detailed_statistics.txt"
echo ""
echo "您可以使用图片查看器打开PNG文件来查看可视化结果"
echo "或使用文本编辑器查看JSON和TXT文件获取详细数据"
echo ""
echo "==========================================" 