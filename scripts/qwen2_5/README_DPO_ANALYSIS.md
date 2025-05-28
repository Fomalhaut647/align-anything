# DPO模型与基础模型回答差异分析工具

本工具集提供了多种方法来分析DPO微调模型和基础模型在回答质量上的差异性。

## 文件说明

### 主要脚本

1. **`analyze_dpo_results.py`** - 文本差异分析脚本
   - 计算DPO和基础模型回答的文本相似度
   - 分析回答长度差异
   - 生成可视化图表和统计报告

2. **`qwen_2_5_rm_dpo_eval.py`** - 奖励模型评估脚本
   - 使用训练好的奖励模型对两种模型的回答进行打分
   - 计算奖励分数差异统计
   - 生成详细的评分分析报告

3. **`run_dpo_analysis.sh`** - 快速运行脚本
   - 一键执行分析流程

### 配置脚本

4. **`qwen_2_5_rm_dpo_eval.sh`** - 奖励模型评估Shell脚本
5. **`qwen_2_5_rm_dpo_deepspeed_eval.sh`** - 使用DeepSpeed的奖励模型评估脚本

## 使用方法

### 方法1: 快速文本分析（推荐）

```bash
# 直接运行分析脚本
bash scripts/qwen2_5/run_dpo_analysis.sh
```

或者手动运行：

```bash
python scripts/qwen2_5/analyze_dpo_results.py \
    --evaluation_data scripts/qwen2_5/evaluation_results/evaluation_results.json \
    --output_dir scripts/qwen2_5/dpo_analysis
```

### 方法2: 使用奖励模型评估

```bash
# 使用自定义Python脚本
bash scripts/qwen2_5/qwen_2_5_rm_dpo_eval.sh

# 或使用DeepSpeed版本
bash scripts/qwen2_5/qwen_2_5_rm_dpo_deepspeed_eval.sh
```

手动运行奖励模型评估：

```bash
python scripts/qwen2_5/qwen_2_5_rm_dpo_eval.py \
    --reward_model_path output/qwen_2_5_rm/slice_end \
    --evaluation_data scripts/qwen2_5/evaluation_results/evaluation_results.json \
    --output_dir scripts/qwen2_5/reward_analysis \
    --device npu:0 \
    --max_samples 50
```

## 输出结果

### 文本分析结果 (`scripts/qwen2_5/dpo_analysis/`)

- **`analysis_report.txt`** - 详细的文本分析报告
- **`dpo_analysis_results.json`** - 完整的分析数据和示例
- **`dpo_base_comparison.png`** - 可视化图表，包含：
  - 相似度分布图
  - 长度比例分布图
  - 长度对比散点图
  - 分类统计柱状图

### 奖励模型评估结果 (`scripts/qwen2_5/reward_analysis/`)

- **`evaluation_summary.txt`** - 奖励分数统计摘要
- **`reward_evaluation_results.json`** - 详细的评估结果
- **`score_comparison.png`** - 奖励分数对比图表
- **`improvement_analysis.png`** - 改进效果分析图

## 分析指标

### 文本分析指标

1. **文本相似度** - 基于词汇重叠度计算的相似度
2. **长度比例** - DPO回答长度 / 基础模型回答长度
3. **分类统计**：
   - 长度分类：更长、相似、更短等
   - 相似度分类：非常相似、相似、不同等

### 奖励模型指标

1. **奖励分数** - 奖励模型给出的质量评分
2. **分数差异** - DPO分数 - 基础模型分数
3. **改进统计**：
   - 改进率：得分提升的样本比例
   - 显著改进：分数提升>0.1的样本数
   - 退化样本：分数下降的样本数

## 参数配置

### 主要参数

- `--evaluation_data`: 评估数据文件路径（evaluation_results.json）
- `--output_dir`: 结果输出目录
- `--reward_model_path`: 奖励模型路径
- `--device`: 计算设备（cuda/npu）
- `--max_samples`: 最大评估样本数（用于测试）

### 数据格式要求

输入的evaluation_results.json应包含以下字段：
```json
[
  {
    "prompt": "用户问题",
    "base_response": "基础模型回答",
    "dpo_response": "DPO模型回答"
  }
]
```

## 注意事项

1. **资源需求**：奖励模型评估需要GPU/NPU资源，文本分析只需要CPU
2. **评估速度**：文本分析速度快，奖励模型评估相对较慢
3. **准确性**：奖励模型评估更准确，文本分析提供基础对比
4. **设备适配**：根据实际硬件环境调整device参数

## 故障排除

1. **模型加载失败**：检查奖励模型路径是否正确
2. **内存不足**：减少max_samples参数或使用更大的机器
3. **设备错误**：确认device参数与实际硬件匹配
4. **数据格式错误**：确认JSON文件格式正确

## 扩展功能

可以根据需要修改脚本来：
- 添加更多文本相似度算法
- 使用不同的奖励模型
- 添加更多可视化选项
- 自定义评估指标 