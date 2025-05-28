#!/usr/bin/env python3
"""
使用奖励模型评估DPO微调模型和初始模型的回答，并分析得分差异性
"""

import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any
import os
import sys

# 添加align_anything的路径
sys.path.append('.')
from align_anything.models.pretrained_model import load_pretrained_models


def load_reward_model(model_path: str, device: str = "cuda"):
    """加载奖励模型"""
    print(f"Loading reward model from {model_path}")
    model, tokenizer, processor = load_pretrained_models(
        model_path,
        model_max_length=2048,
        padding_side='right',
        trust_remote_code=True,
        is_reward_model=True,
    )
    model.eval()
    
    # 确保模型移动到指定设备
    print(f"Moving model to device: {device}")
    model = model.to(device)
    
    return model, tokenizer


def format_conversation_for_reward_model(prompt: str, response: str, tokenizer) -> Dict:
    """格式化对话用于奖励模型评估"""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # 使用tokenizer格式化对话
    formatted_text = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # 将文本转换为tokens
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048
    )
    
    return inputs


def get_reward_score(model, tokenizer, prompt: str, response: str, device: str = "cuda") -> float:
    """使用奖励模型计算单个回答的奖励分数"""
    inputs = format_conversation_for_reward_model(prompt, response, tokenizer)
    
    # 移动到指定设备
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)
    
    # 验证设备一致性
    input_device = inputs['input_ids'].device
    model_device = next(model.parameters()).device
    
    # 如果设备不匹配，重新移动模型
    if model_device != input_device:
        print(f"Device mismatch detected. Moving model from {model_device} to {device}")
        model = model.to(device)
    
    with torch.no_grad():
        # 使用奖励模型进行推理
        outputs = model(**inputs)
        # 获取end_scores作为奖励分数
        reward_score = outputs.end_scores.squeeze(-1).item()
        
    return reward_score


def evaluate_responses(model, tokenizer, evaluation_data: List[Dict], device: str = "cuda") -> Dict[str, List[float]]:
    """评估所有回答并返回分数"""
    base_scores = []
    dpo_scores = []
    
    print("Evaluating responses with reward model...")
    for item in tqdm(evaluation_data):
        prompt = item["prompt"]
        base_response = item["base_response"]
        dpo_response = item["dpo_response"]
        
        # 计算基础模型的奖励分数
        base_score = get_reward_score(model, tokenizer, prompt, base_response, device)
        base_scores.append(base_score)
        
        # 计算DPO模型的奖励分数
        dpo_score = get_reward_score(model, tokenizer, prompt, dpo_response, device)
        dpo_scores.append(dpo_score)
    
    return {
        "base_scores": base_scores,
        "dpo_scores": dpo_scores
    }


def analyze_score_differences(base_scores: List[float], dpo_scores: List[float]) -> Dict[str, Any]:
    """分析分数差异性"""
    base_scores = np.array(base_scores)
    dpo_scores = np.array(dpo_scores)
    differences = dpo_scores - base_scores
    
    analysis = {
        "base_model_stats": {
            "mean": float(np.mean(base_scores)),
            "std": float(np.std(base_scores)),
            "median": float(np.median(base_scores)),
            "min": float(np.min(base_scores)),
            "max": float(np.max(base_scores))
        },
        "dpo_model_stats": {
            "mean": float(np.mean(dpo_scores)),
            "std": float(np.std(dpo_scores)),
            "median": float(np.median(dpo_scores)),
            "min": float(np.min(dpo_scores)),
            "max": float(np.max(dpo_scores))
        },
        "difference_stats": {
            "mean": float(np.mean(differences)),
            "std": float(np.std(differences)),
            "median": float(np.median(differences)),
            "min": float(np.min(differences)),
            "max": float(np.max(differences))
        },
        "improvement_metrics": {
            "improved_count": int(np.sum(differences > 0)),
            "degraded_count": int(np.sum(differences < 0)),
            "unchanged_count": int(np.sum(differences == 0)),
            "improvement_rate": float(np.mean(differences > 0)),
            "significant_improvement": int(np.sum(differences > 0.1)),  # 显著改进阈值
            "significant_degradation": int(np.sum(differences < -0.1))  # 显著退化阈值
        }
    }
    
    return analysis


def create_visualizations(base_scores: List[float], dpo_scores: List[float], output_dir: str):
    """创建可视化图表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_scores = np.array(base_scores)
    dpo_scores = np.array(dpo_scores)
    differences = dpo_scores - base_scores
    
    # 1. 分数分布对比
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(base_scores, bins=30, alpha=0.7, label='Base Model', color='blue')
    plt.hist(dpo_scores, bins=30, alpha=0.7, label='DPO Model', color='red')
    plt.xlabel('Reward Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 分数差异分布
    plt.subplot(2, 2, 2)
    plt.hist(differences, bins=30, alpha=0.7, color='green')
    plt.axvline(x=0, color='red', linestyle='--', label='No Change')
    plt.xlabel('Score Difference (DPO - Base)')
    plt.ylabel('Frequency')
    plt.title('Score Difference Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 散点图对比
    plt.subplot(2, 2, 3)
    plt.scatter(base_scores, dpo_scores, alpha=0.6)
    min_score = min(min(base_scores), min(dpo_scores))
    max_score = max(max(base_scores), max(dpo_scores))
    plt.plot([min_score, max_score], [min_score, max_score], 'r--', label='y=x')
    plt.xlabel('Base Model Scores')
    plt.ylabel('DPO Model Scores')
    plt.title('Score Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Box plot对比
    plt.subplot(2, 2, 4)
    data_to_plot = [base_scores, dpo_scores]
    plt.boxplot(data_to_plot, labels=['Base Model', 'DPO Model'])
    plt.ylabel('Reward Score')
    plt.title('Score Distribution (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 改进情况分析
    plt.figure(figsize=(10, 6))
    
    improvement_categories = ['Significant\nImprovement\n(>0.1)', 'Minor\nImprovement\n(0~0.1)', 
                            'No Change\n(=0)', 'Minor\nDegradation\n(-0.1~0)', 
                            'Significant\nDegradation\n(<-0.1)']
    
    significant_improvement = np.sum(differences > 0.1)
    minor_improvement = np.sum((differences > 0) & (differences <= 0.1))
    no_change = np.sum(differences == 0)
    minor_degradation = np.sum((differences < 0) & (differences >= -0.1))
    significant_degradation = np.sum(differences < -0.1)
    
    counts = [significant_improvement, minor_improvement, no_change, 
              minor_degradation, significant_degradation]
    colors = ['darkgreen', 'lightgreen', 'gray', 'lightcoral', 'darkred']
    
    plt.bar(improvement_categories, counts, color=colors)
    plt.ylabel('Number of Samples')
    plt.title('DPO Model Performance Changes')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO and base model responses using reward model")
    parser.add_argument("--reward_model_path", type=str, default="output/qwen_2_5_rm/slice_end",
                        help="Path to the reward model")
    parser.add_argument("--evaluation_data", type=str, 
                        default="scripts/qwen2_5/evaluation_results/evaluation_results.json",
                        help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="scripts/qwen2_5/reward_analysis",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载评估数据
    print(f"Loading evaluation data from {args.evaluation_data}")
    with open(args.evaluation_data, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)
    
    if args.max_samples:
        evaluation_data = evaluation_data[:args.max_samples]
        print(f"Using first {args.max_samples} samples for evaluation")
    
    print(f"Total samples to evaluate: {len(evaluation_data)}")
    
    # 加载奖励模型
    reward_model, reward_tokenizer = load_reward_model(args.reward_model_path, args.device)
    
    # 评估回答
    scores = evaluate_responses(reward_model, reward_tokenizer, evaluation_data, args.device)
    
    # 分析结果
    analysis = analyze_score_differences(scores["base_scores"], scores["dpo_scores"])
    
    # 保存详细结果
    detailed_results = []
    for i, item in enumerate(evaluation_data):
        detailed_results.append({
            "prompt": item["prompt"],
            "base_response": item["base_response"],
            "dpo_response": item["dpo_response"],
            "base_score": scores["base_scores"][i],
            "dpo_score": scores["dpo_scores"][i],
            "score_difference": scores["dpo_scores"][i] - scores["base_scores"][i]
        })
    
    # 保存结果
    results_file = os.path.join(args.output_dir, "reward_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis": analysis,
            "detailed_results": detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    # 保存统计摘要
    summary_file = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== DPO模型与基础模型奖励分数对比分析 ===\n\n")
        
        f.write("基础模型统计:\n")
        for key, value in analysis["base_model_stats"].items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nDPO模型统计:\n")
        for key, value in analysis["dpo_model_stats"].items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n分数差异统计:\n")
        for key, value in analysis["difference_stats"].items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n改进效果分析:\n")
        metrics = analysis["improvement_metrics"]
        f.write(f"  改进样本数: {metrics['improved_count']}\n")
        f.write(f"  退化样本数: {metrics['degraded_count']}\n")
        f.write(f"  不变样本数: {metrics['unchanged_count']}\n")
        f.write(f"  改进率: {metrics['improvement_rate']:.2%}\n")
        f.write(f"  显著改进样本数 (>0.1): {metrics['significant_improvement']}\n")
        f.write(f"  显著退化样本数 (<-0.1): {metrics['significant_degradation']}\n")
    
    # 创建可视化图表
    create_visualizations(scores["base_scores"], scores["dpo_scores"], args.output_dir)
    
    # 打印摘要
    print("\n=== 评估结果摘要 ===")
    print(f"评估样本数: {len(evaluation_data)}")
    print(f"基础模型平均分: {analysis['base_model_stats']['mean']:.4f}")
    print(f"DPO模型平均分: {analysis['dpo_model_stats']['mean']:.4f}")
    print(f"平均分数提升: {analysis['difference_stats']['mean']:.4f}")
    print(f"改进率: {analysis['improvement_metrics']['improvement_rate']:.2%}")
    print(f"显著改进样本数: {analysis['improvement_metrics']['significant_improvement']}")
    print(f"显著退化样本数: {analysis['improvement_metrics']['significant_degradation']}")
    
    print(f"\n详细结果已保存到: {args.output_dir}")
    print(f"- 完整结果: {results_file}")
    print(f"- 统计摘要: {summary_file}")
    print(f"- 可视化图表: {args.output_dir}/score_comparison.png")
    print(f"- 改进分析图: {args.output_dir}/improvement_analysis.png")


if __name__ == "__main__":
    main() 