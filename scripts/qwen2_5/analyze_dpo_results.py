#!/usr/bin/env python3
"""
分析evaluation_results.json中DPO微调模型和基础模型的回答差异
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from collections import Counter
import re


def load_evaluation_data(file_path: str) -> List[Dict]:
    """加载评估数据"""
    print(f"Loading evaluation data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（简单的词重叠度）"""
    # 简单的词级别相似度计算
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def calculate_length_ratio(text1: str, text2: str) -> float:
    """计算两个文本的长度比例"""
    len1, len2 = len(text1), len(text2)
    if len2 == 0:
        return float('inf') if len1 > 0 else 1.0
    return len1 / len2


def analyze_response_differences(data: List[Dict]) -> Dict[str, Any]:
    """分析回答差异"""
    similarities = []
    length_ratios = []
    base_lengths = []
    dpo_lengths = []
    
    different_responses = 0
    identical_responses = 0
    
    quality_categories = {
        'dpo_much_longer': 0,  # DPO回答明显更长(>1.5倍)
        'dpo_longer': 0,       # DPO回答更长(1.1-1.5倍)
        'similar_length': 0,   # 长度相似(0.9-1.1倍)
        'dpo_shorter': 0,      # DPO回答更短(0.67-0.9倍)
        'dpo_much_shorter': 0, # DPO回答明显更短(<0.67倍)
    }
    
    similarity_categories = {
        'very_similar': 0,    # 相似度>0.8
        'similar': 0,         # 相似度0.6-0.8
        'somewhat_similar': 0, # 相似度0.4-0.6
        'different': 0,       # 相似度0.2-0.4
        'very_different': 0,  # 相似度<0.2
    }
    
    for item in data:
        base_response = item['base_response']
        dpo_response = item['dpo_response']
        
        # 计算相似度
        similarity = calculate_text_similarity(base_response, dpo_response)
        similarities.append(similarity)
        
        # 计算长度比例
        length_ratio = calculate_length_ratio(dpo_response, base_response)
        length_ratios.append(length_ratio)
        
        base_lengths.append(len(base_response))
        dpo_lengths.append(len(dpo_response))
        
        # 判断是否相同
        if base_response.strip() == dpo_response.strip():
            identical_responses += 1
        else:
            different_responses += 1
        
        # 长度分类
        if length_ratio > 1.5:
            quality_categories['dpo_much_longer'] += 1
        elif length_ratio > 1.1:
            quality_categories['dpo_longer'] += 1
        elif length_ratio >= 0.9:
            quality_categories['similar_length'] += 1
        elif length_ratio >= 0.67:
            quality_categories['dpo_shorter'] += 1
        else:
            quality_categories['dpo_much_shorter'] += 1
        
        # 相似度分类
        if similarity > 0.8:
            similarity_categories['very_similar'] += 1
        elif similarity > 0.6:
            similarity_categories['similar'] += 1
        elif similarity > 0.4:
            similarity_categories['somewhat_similar'] += 1
        elif similarity > 0.2:
            similarity_categories['different'] += 1
        else:
            similarity_categories['very_different'] += 1
    
    analysis = {
        'total_samples': len(data),
        'identical_responses': identical_responses,
        'different_responses': different_responses,
        'similarity_stats': {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'median': float(np.median(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        },
        'length_stats': {
            'base_avg_length': float(np.mean(base_lengths)),
            'dpo_avg_length': float(np.mean(dpo_lengths)),
            'length_ratio_mean': float(np.mean(length_ratios)),
            'length_ratio_std': float(np.std(length_ratios)),
            'length_ratio_median': float(np.median(length_ratios))
        },
        'quality_categories': quality_categories,
        'similarity_categories': similarity_categories
    }
    
    return analysis, similarities, length_ratios, base_lengths, dpo_lengths


def create_visualizations(analysis: Dict, similarities: List[float], length_ratios: List[float], 
                         base_lengths: List[int], dpo_lengths: List[int], output_dir: str):
    """创建可视化图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 相似度分布
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(similarities, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Text Similarity')
    plt.ylabel('Frequency')
    plt.title('DPO vs Base Response Similarity Distribution')
    plt.axvline(x=np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 长度比例分布
    plt.subplot(2, 3, 2)
    # 过滤极端值
    filtered_ratios = [r for r in length_ratios if r < 5]
    plt.hist(filtered_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Length Ratio (DPO/Base)')
    plt.ylabel('Frequency')
    plt.title('Response Length Ratio Distribution')
    plt.axvline(x=1, color='red', linestyle='--', label='Equal Length')
    plt.axvline(x=np.mean(filtered_ratios), color='orange', linestyle='--', 
               label=f'Mean: {np.mean(filtered_ratios):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 长度对比散点图
    plt.subplot(2, 3, 3)
    plt.scatter(base_lengths, dpo_lengths, alpha=0.6)
    max_len = max(max(base_lengths), max(dpo_lengths))
    plt.plot([0, max_len], [0, max_len], 'r--', label='y=x')
    plt.xlabel('Base Model Response Length')
    plt.ylabel('DPO Model Response Length')
    plt.title('Response Length Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 长度分类柱状图
    plt.subplot(2, 3, 4)
    categories = list(analysis['quality_categories'].keys())
    counts = list(analysis['quality_categories'].values())
    colors = ['darkred', 'red', 'gray', 'lightblue', 'darkblue']
    
    bars = plt.bar(range(len(categories)), counts, color=colors)
    plt.xlabel('Length Categories')
    plt.ylabel('Number of Samples')
    plt.title('Response Length Categories')
    plt.xticks(range(len(categories)), [cat.replace('_', '\n') for cat in categories], rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 5. 相似度分类柱状图
    plt.subplot(2, 3, 5)
    sim_categories = list(analysis['similarity_categories'].keys())
    sim_counts = list(analysis['similarity_categories'].values())
    sim_colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']
    
    bars = plt.bar(range(len(sim_categories)), sim_counts, color=sim_colors)
    plt.xlabel('Similarity Categories')
    plt.ylabel('Number of Samples')
    plt.title('Response Similarity Categories')
    plt.xticks(range(len(sim_categories)), [cat.replace('_', '\n') for cat in sim_categories], rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, sim_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # 6. 相似度vs长度比例散点图
    plt.subplot(2, 3, 6)
    filtered_indices = [i for i, r in enumerate(length_ratios) if r < 5]
    filtered_similarities = [similarities[i] for i in filtered_indices]
    filtered_length_ratios = [length_ratios[i] for i in filtered_indices]
    
    plt.scatter(filtered_similarities, filtered_length_ratios, alpha=0.6)
    plt.xlabel('Text Similarity')
    plt.ylabel('Length Ratio (DPO/Base)')
    plt.title('Similarity vs Length Ratio')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Length')
    plt.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Medium Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dpo_base_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def find_examples(data: List[Dict], analysis: Dict) -> Dict[str, List[Dict]]:
    """找出不同类型的示例"""
    examples = {
        'most_similar': [],
        'most_different': [],
        'dpo_much_longer': [],
        'dpo_much_shorter': [],
        'identical': []
    }
    
    # 计算所有相似度和长度比例
    for i, item in enumerate(data):
        base_response = item['base_response']
        dpo_response = item['dpo_response']
        
        similarity = calculate_text_similarity(base_response, dpo_response)
        length_ratio = calculate_length_ratio(dpo_response, base_response)
        
        item_with_metrics = {
            **item,
            'similarity': similarity,
            'length_ratio': length_ratio,
            'index': i
        }
        
        # 相同回答
        if base_response.strip() == dpo_response.strip():
            examples['identical'].append(item_with_metrics)
        
        # 最相似的回答
        if similarity > 0.9:
            examples['most_similar'].append(item_with_metrics)
        
        # 最不同的回答
        if similarity < 0.1:
            examples['most_different'].append(item_with_metrics)
        
        # DPO回答明显更长
        if length_ratio > 2:
            examples['dpo_much_longer'].append(item_with_metrics)
        
        # DPO回答明显更短
        if length_ratio < 0.5:
            examples['dpo_much_shorter'].append(item_with_metrics)
    
    # 限制每类示例数量
    for key in examples:
        if len(examples[key]) > 5:
            # 按相关指标排序并取前5个
            if key == 'most_similar':
                examples[key] = sorted(examples[key], key=lambda x: x['similarity'], reverse=True)[:5]
            elif key == 'most_different':
                examples[key] = sorted(examples[key], key=lambda x: x['similarity'])[:5]
            elif key == 'dpo_much_longer':
                examples[key] = sorted(examples[key], key=lambda x: x['length_ratio'], reverse=True)[:5]
            elif key == 'dpo_much_shorter':
                examples[key] = sorted(examples[key], key=lambda x: x['length_ratio'])[:5]
            else:
                examples[key] = examples[key][:5]
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Analyze DPO vs Base model response differences")
    parser.add_argument("--evaluation_data", type=str, 
                        default="scripts/qwen2_5/evaluation_results/evaluation_results.json",
                        help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="scripts/qwen2_5/dpo_analysis",
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载数据
    data = load_evaluation_data(args.evaluation_data)
    
    # 分析差异
    analysis, similarities, length_ratios, base_lengths, dpo_lengths = analyze_response_differences(data)
    
    # 找出示例
    examples = find_examples(data, analysis)
    
    # 保存详细分析结果
    results_file = os.path.join(args.output_dir, "dpo_analysis_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis": analysis,
            "examples": examples
        }, f, ensure_ascii=False, indent=2)
    
    # 保存文本报告
    report_file = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== DPO模型与基础模型回答差异分析报告 ===\n\n")
        
        f.write(f"总样本数: {analysis['total_samples']}\n")
        f.write(f"完全相同的回答: {analysis['identical_responses']} ({analysis['identical_responses']/analysis['total_samples']:.1%})\n")
        f.write(f"不同的回答: {analysis['different_responses']} ({analysis['different_responses']/analysis['total_samples']:.1%})\n\n")
        
        f.write("=== 文本相似度统计 ===\n")
        sim_stats = analysis['similarity_stats']
        f.write(f"平均相似度: {sim_stats['mean']:.3f}\n")
        f.write(f"相似度标准差: {sim_stats['std']:.3f}\n")
        f.write(f"相似度中位数: {sim_stats['median']:.3f}\n")
        f.write(f"最低相似度: {sim_stats['min']:.3f}\n")
        f.write(f"最高相似度: {sim_stats['max']:.3f}\n\n")
        
        f.write("=== 长度统计 ===\n")
        len_stats = analysis['length_stats']
        f.write(f"基础模型平均回答长度: {len_stats['base_avg_length']:.0f} 字符\n")
        f.write(f"DPO模型平均回答长度: {len_stats['dpo_avg_length']:.0f} 字符\n")
        f.write(f"平均长度比例(DPO/Base): {len_stats['length_ratio_mean']:.2f}\n")
        f.write(f"长度比例标准差: {len_stats['length_ratio_std']:.2f}\n")
        f.write(f"长度比例中位数: {len_stats['length_ratio_median']:.2f}\n\n")
        
        f.write("=== 长度分类统计 ===\n")
        for category, count in analysis['quality_categories'].items():
            percentage = count / analysis['total_samples'] * 100
            f.write(f"{category}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n=== 相似度分类统计 ===\n")
        for category, count in analysis['similarity_categories'].items():
            percentage = count / analysis['total_samples'] * 100
            f.write(f"{category}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\n=== 示例统计 ===\n")
        for example_type, example_list in examples.items():
            f.write(f"{example_type}: {len(example_list)} 个示例\n")
    
    # 创建可视化
    create_visualizations(analysis, similarities, length_ratios, base_lengths, dpo_lengths, args.output_dir)
    
    # 打印摘要
    print("\n=== 分析结果摘要 ===")
    print(f"总样本数: {analysis['total_samples']}")
    print(f"完全相同的回答: {analysis['identical_responses']} ({analysis['identical_responses']/analysis['total_samples']:.1%})")
    print(f"平均文本相似度: {analysis['similarity_stats']['mean']:.3f}")
    print(f"平均长度比例(DPO/Base): {analysis['length_stats']['length_ratio_mean']:.2f}")
    print(f"DPO回答更长的样本: {analysis['quality_categories']['dpo_longer'] + analysis['quality_categories']['dpo_much_longer']}")
    print(f"DPO回答更短的样本: {analysis['quality_categories']['dpo_shorter'] + analysis['quality_categories']['dpo_much_shorter']}")
    
    print(f"\n详细结果已保存到: {args.output_dir}")
    print(f"- 完整结果: {results_file}")
    print(f"- 分析报告: {report_file}")
    print(f"- 可视化图表: {args.output_dir}/dpo_base_comparison.png")


if __name__ == "__main__":
    main() 