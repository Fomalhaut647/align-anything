# 奖励模型偏好数据集可视化

## 概述

本功能允许您使用训练好的奖励模型对偏好数据集进行评分，并通过多种可视化方法直观地展示chosen和rejected响应之间的差异。

## 功能特点

1. **自动保存评分数据**：在评估过程中自动保存所有文本和对应的奖励分值
2. **多维度可视化分析**：
   - 分数分布直方图
   - 箱线图和小提琴图
   - 核密度估计(KDE)
   - 分数差异分布
   - 累积分布函数
   - 散点图分析
   - 分数范围统计
3. **文本长度分析**：分析文本长度与奖励分值的关系
4. **统计检验**：Mann-Whitney U检验和Kolmogorov-Smirnov检验
5. **详细统计信息**：均值、标准差、分位数等统计指标

## 安装依赖

首先安装可视化所需的依赖包：

```bash
pip install -r requirements_visualization.txt
```

或者手动安装：

```bash
pip install matplotlib seaborn scipy pandas numpy
```

## 使用方法

### 1. 运行奖励模型评估

使用修改后的评估脚本：

```bash
bash scripts/qwen2_5/rm_eval.sh
```

该脚本会：
- 运行奖励模型评估（仅评估，不训练）
- 自动保存chosen和rejected响应的文本和分值
- 生成基础可视化图表

### 2. 生成详细分析报告

评估完成后，使用分析脚本生成更详细的可视化：

```bash
python scripts/analyze_reward_scores.py --data_dir output/qwen_2_5_rm_eval/reward_analysis
```

## 输出文件说明

运行完成后，会在 `{OUTPUT_DIR}/reward_analysis/` 目录下生成以下文件：

### 数据文件
- `chosen_responses.json`：chosen响应的文本和分值
- `rejected_responses.json`：rejected响应的文本和分值
- `detailed_statistics.txt`：详细统计信息

### 可视化文件
- `reward_score_visualization.png`：基础的4合1可视化图表
- `comprehensive_reward_analysis.png`：综合分析图表（3x3布局）
- `text_length_analysis.png`：文本长度与分值关系分析

## 可视化图表说明

### 1. 综合分析图表（comprehensive_reward_analysis.png）

包含9个子图：

1. **分布直方图**：chosen vs rejected分数的密度分布对比
2. **箱线图**：显示分数的中位数、四分位数和异常值
3. **小提琴图**：结合了箱线图和密度图的优势
4. **核密度估计**：平滑的概率密度曲线
5. **分数差异分布**：chosen - rejected的分数差异分布
6. **散点图**：chosen vs rejected分数的直接对比
7. **累积分布函数**：显示分数的累积概率
8. **分数范围统计**：按分数范围统计数量分布
9. **统计摘要**：详细的统计信息和假设检验结果

### 2. 文本长度分析图表（text_length_analysis.png）

包含6个子图：

1. **长度vs分数散点图**：文本长度与奖励分值的关系
2. **长度分布直方图**：chosen vs rejected的文本长度分布
3. **长度箱线图**：文本长度的统计对比
4. **相关性分析**：长度与分值的线性关系
5. **分段分析**：按长度分段的平均分值对比
6. **长度统计摘要**：文本长度的详细统计信息

## 统计指标解释

### 准确性指标
- **Accuracy**：chosen分数高于rejected分数的比例
- **Mean Difference**：chosen和rejected分数的平均差值

### 分布指标
- **Mean/Std**：平均值和标准差
- **Q25/Q75**：25%和75%分位数
- **Min/Max**：最小值和最大值

### 假设检验
- **Mann-Whitney U Test**：非参数检验，检验两组分数是否有显著差异
- **Kolmogorov-Smirnov Test**：检验两组分数分布是否相同

## 配置选项

### 评估脚本配置

在 `rm_eval.sh` 中可以修改以下参数：

```bash
MODEL_NAME_OR_PATH="path/to/your/reward/model"  # 奖励模型路径
EVAL_DATASETS="path/to/your/dataset"            # 评估数据集路径
EVAL_TEMPLATE="YOUR_TEMPLATE"                   # 数据集模板
EVAL_SPLIT="validation"                         # 数据集分割
OUTPUT_DIR="output/rm_eval"                     # 输出目录
```

### 可视化参数

可以通过修改 `analyze_reward_scores.py` 调整可视化参数：

- 图表尺寸和DPI
- 颜色方案
- 分数范围划分
- 统计检验方法

## 故障排除

### 常见问题

1. **导入错误**：确保安装了所有必要的依赖包
2. **内存不足**：对于大型数据集，可能需要增加系统内存
3. **权限错误**：确保对输出目录有写权限

### 错误处理

代码包含完善的错误处理机制：
- 缺少依赖时会显示安装提示
- 文件不存在时会给出明确错误信息
- 数据格式错误时会跳过处理并给出警告

## 扩展功能

### 自定义分析

您可以修改 `analyze_reward_scores.py` 来添加：

- 更多统计检验方法
- 自定义可视化样式
- 特定领域的分析指标
- 交互式图表（使用plotly等）

### 批量处理

对于多个模型或数据集的对比分析，可以编写脚本循环调用分析功能。

## 示例输出

运行成功后，您将看到类似以下的输出：

```
======================================================================
REWARD SCORE STATISTICS
======================================================================
Chosen responses:
  Mean: 0.2456
  Std:  0.1823
  Min:  -0.4521
  Max:  0.7834

Rejected responses:
  Mean: -0.1234
  Std:  0.1567
  Min:  -0.6789
  Max:  0.3456

Score Differences (Chosen - Rejected):
  Mean: 0.3690
  Std:  0.2145
  Accuracy: 78.45%
====================================================================== 