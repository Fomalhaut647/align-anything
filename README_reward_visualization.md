# 奖励模型偏好数据集可视化功能

## 功能概述

本项目为align-anything项目添加了奖励模型偏好数据集可视化功能，可以直观地展示chosen和rejected响应在奖励分值上的差异性。

## 主要修改文件

### 1. 核心文件修改

- **`align_anything/trainers/text_to_text/rm.py`**
  - 添加了 `save_reward_scores()` 方法保存文本和分值
  - 添加了 `visualize_reward_distribution()` 方法生成基础可视化
  - 修改了 `eval()` 方法收集文本和分值数据
  - 修改了 `train()` 方法支持仅评估模式

### 2. 脚本文件

- **`scripts/qwen2_5/rm_eval.sh`**
  - 修改为仅评估模式（epochs=0, eval_only）
  - 添加了结果提示信息

- **`scripts/analyze_reward_scores.py`** (新增)
  - 独立的可视化分析脚本
  - 提供9种不同的可视化图表
  - 包含文本长度分析功能
  - 统计检验和详细报告

- **`scripts/demo_reward_visualization.sh`** (新增)
  - 完整的演示脚本
  - 自动检查依赖和模型
  - 一键运行完整流程

### 3. 配置文件

- **`requirements_visualization.txt`** (新增)
  - 可视化所需的Python依赖包

- **`docs/reward_model_visualization.md`** (新增)
  - 详细的使用说明文档

## 功能特性

### 1. 自动数据保存
- 在评估过程中自动保存chosen和rejected响应的文本和奖励分值
- 输出格式为JSON，便于后续分析

### 2. 多维度可视化分析
- **分布对比**：直方图、KDE、小提琴图
- **统计对比**：箱线图、累积分布函数
- **关系分析**：散点图、相关性分析
- **差异分析**：分数差异分布、分数范围统计

### 3. 文本长度分析
- 文本长度与奖励分值的关系分析
- 长度分布对比
- 分段分析

### 4. 统计检验
- Mann-Whitney U检验
- Kolmogorov-Smirnov检验
- 准确率计算

## 使用方法

### 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements_visualization.txt
   ```

2. **运行演示**
   ```bash
   bash scripts/demo_reward_visualization.sh
   ```

### 分步执行

1. **运行评估**
   ```bash
   bash scripts/qwen2_5/rm_eval.sh
   ```

2. **生成详细分析**
   ```bash
   python scripts/analyze_reward_scores.py --data_dir output/qwen_2_5_rm_eval/reward_analysis
   ```

## 输出文件

执行完成后会生成以下文件：

### 数据文件
- `chosen_responses.json` - chosen响应的文本和分值
- `rejected_responses.json` - rejected响应的文本和分值
- `detailed_statistics.txt` - 详细统计信息

### 可视化文件
- `reward_score_visualization.png` - 基础4合1可视化
- `comprehensive_reward_analysis.png` - 综合分析(3×3)
- `text_length_analysis.png` - 文本长度分析

## 技术实现

### 1. 数据收集
在`eval()`方法中：
- 收集每个batch的文本和奖励分值
- 分别保存chosen和rejected数据
- 处理多GPU环境下的数据收集

### 2. 可视化生成
- 使用matplotlib和seaborn创建多种图表
- 兼容不同版本的seaborn样式设置
- 高质量PNG输出(300 DPI)

### 3. 统计分析
- numpy进行数值计算
- scipy进行统计检验
- pandas处理数据结构

## 配置选项

### 环境变量
- `OUTPUT_ROOT_DIR` - 设置输出根目录

### 可自定义参数
- 模型路径
- 数据集路径
- 输出目录
- 图表样式
- 统计检验方法

## 错误处理

- 依赖包缺失时自动提示安装
- 模型路径不存在时给出明确错误信息
- seaborn版本兼容性处理
- 文件权限和目录创建处理

## 扩展性

### 可添加的功能
- 更多统计检验方法
- 交互式图表（plotly）
- 多模型对比分析
- 自定义可视化样式
- 批量处理脚本

### 支持的数据格式
- 当前支持JSON格式
- 可扩展支持CSV、HDF5等格式

## 示例输出

典型的统计输出：
```
Chosen responses:
  Mean: 0.2456, Std: 0.1823
  Min: -0.4521, Max: 0.7834

Rejected responses:
  Mean: -0.1234, Std: 0.1567
  Min: -0.6789, Max: 0.3456

Accuracy: 78.45%
Mean Difference: 0.3690
```

## 注意事项

1. **内存使用**：大型数据集可能需要较多内存
2. **GPU支持**：自动处理多GPU环境下的数据收集
3. **文件权限**：确保对输出目录有写权限
4. **依赖版本**：建议使用requirements文件安装依赖

## 贡献指南

欢迎提交Issue和Pull Request来改进此功能：
- 新的可视化方法
- 性能优化
- 错误修复
- 文档改进

## 许可证

本功能遵循align-anything项目的Apache 2.0许可证。 