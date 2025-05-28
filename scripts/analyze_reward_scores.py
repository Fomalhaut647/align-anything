#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze and visualize reward model scores for preference datasets.
This script reads the saved reward scores and creates comprehensive visualizations.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu, ks_2samp
from collections import Counter
import pandas as pd

def load_reward_data(chosen_file, rejected_file):
    """Load chosen and rejected response data with scores."""
    with open(chosen_file, 'r', encoding='utf-8') as f:
        chosen_data = json.load(f)
    
    with open(rejected_file, 'r', encoding='utf-8') as f:
        rejected_data = json.load(f)
    
    chosen_scores = [item['reward_score'] for item in chosen_data]
    rejected_scores = [item['reward_score'] for item in rejected_data]
    chosen_texts = [item['text'] for item in chosen_data]
    rejected_texts = [item['text'] for item in rejected_data]
    
    return chosen_scores, rejected_scores, chosen_texts, rejected_texts

def create_comprehensive_visualization(chosen_scores, rejected_scores, output_dir):
    """Create comprehensive visualizations of reward score distributions."""
    
    # Set plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    
    # Convert to numpy arrays
    chosen_scores = np.array(chosen_scores)
    rejected_scores = np.array(rejected_scores)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Histogram comparison
    ax1 = plt.subplot(3, 3, 1)
    plt.hist(chosen_scores, bins=50, alpha=0.7, label='Chosen', color='green', density=True)
    plt.hist(rejected_scores, bins=50, alpha=0.7, label='Rejected', color='red', density=True)
    plt.xlabel('Reward Score')
    plt.ylabel('Density')
    plt.title('Reward Score Distribution: Chosen vs Rejected')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = plt.subplot(3, 3, 2)
    box_data = [chosen_scores, rejected_scores]
    box_labels = ['Chosen', 'Rejected']
    box_colors = ['lightgreen', 'lightcoral']
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    plt.ylabel('Reward Score')
    plt.title('Box Plot: Reward Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. Violin plot
    ax3 = plt.subplot(3, 3, 3)
    df = pd.DataFrame({
        'scores': np.concatenate([chosen_scores, rejected_scores]),
        'type': ['Chosen'] * len(chosen_scores) + ['Rejected'] * len(rejected_scores)
    })
    sns.violinplot(data=df, x='type', y='scores', ax=ax3)
    plt.title('Violin Plot: Score Distribution')
    plt.ylabel('Reward Score')
    
    # 4. KDE plot
    ax4 = plt.subplot(3, 3, 4)
    if len(chosen_scores) > 1:
        kde_chosen = gaussian_kde(chosen_scores)
        x_range = np.linspace(min(min(chosen_scores), min(rejected_scores)), 
                            max(max(chosen_scores), max(rejected_scores)), 200)
        plt.plot(x_range, kde_chosen(x_range), 'g-', linewidth=2, label='Chosen')
    
    if len(rejected_scores) > 1:
        kde_rejected = gaussian_kde(rejected_scores)
        plt.plot(x_range, kde_rejected(x_range), 'r-', linewidth=2, label='Rejected')
    
    plt.xlabel('Reward Score')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Score difference distribution
    ax5 = plt.subplot(3, 3, 5)
    score_diff = chosen_scores - rejected_scores
    plt.hist(score_diff, bins=50, alpha=0.7, color='blue', density=True)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Difference')
    plt.axvline(x=np.mean(score_diff), color='red', linestyle='-', linewidth=2, 
              label=f'Mean: {np.mean(score_diff):.3f}')
    plt.xlabel('Score Difference (Chosen - Rejected)')
    plt.ylabel('Density')
    plt.title('Distribution of Score Differences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Scatter plot
    ax6 = plt.subplot(3, 3, 6)
    min_len = min(len(chosen_scores), len(rejected_scores))
    plt.scatter(chosen_scores[:min_len], rejected_scores[:min_len], alpha=0.6, s=20)
    plt.plot([min(min(chosen_scores), min(rejected_scores)), 
              max(max(chosen_scores), max(rejected_scores))], 
             [min(min(chosen_scores), min(rejected_scores)), 
              max(max(chosen_scores), max(rejected_scores))], 
             'r--', label='Equal scores')
    plt.xlabel('Chosen Scores')
    plt.ylabel('Rejected Scores')
    plt.title('Chosen vs Rejected Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Cumulative distribution
    ax7 = plt.subplot(3, 3, 7)
    chosen_sorted = np.sort(chosen_scores)
    rejected_sorted = np.sort(rejected_scores)
    chosen_cdf = np.arange(1, len(chosen_sorted) + 1) / len(chosen_sorted)
    rejected_cdf = np.arange(1, len(rejected_sorted) + 1) / len(rejected_sorted)
    
    plt.plot(chosen_sorted, chosen_cdf, 'g-', linewidth=2, label='Chosen')
    plt.plot(rejected_sorted, rejected_cdf, 'r-', linewidth=2, label='Rejected')
    plt.xlabel('Reward Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Score ranges comparison
    ax8 = plt.subplot(3, 3, 8)
    score_ranges = ['Very Low\n(<-1)', 'Low\n(-1 to -0.5)', 'Medium Low\n(-0.5 to 0)', 
                   'Medium High\n(0 to 0.5)', 'High\n(0.5 to 1)', 'Very High\n(>1)']
    
    def categorize_scores(scores):
        very_low = np.sum(scores < -1)
        low = np.sum((scores >= -1) & (scores < -0.5))
        med_low = np.sum((scores >= -0.5) & (scores < 0))
        med_high = np.sum((scores >= 0) & (scores < 0.5))
        high = np.sum((scores >= 0.5) & (scores < 1))
        very_high = np.sum(scores >= 1)
        return [very_low, low, med_low, med_high, high, very_high]
    
    chosen_counts = categorize_scores(chosen_scores)
    rejected_counts = categorize_scores(rejected_scores)
    
    x = np.arange(len(score_ranges))
    width = 0.35
    
    plt.bar(x - width/2, chosen_counts, width, label='Chosen', color='green', alpha=0.7)
    plt.bar(x + width/2, rejected_counts, width, label='Rejected', color='red', alpha=0.7)
    plt.xlabel('Score Range')
    plt.ylabel('Count')
    plt.title('Score Distribution by Range')
    plt.xticks(x, score_ranges, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Statistical summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Statistical tests
    statistic, p_value_mw = mannwhitneyu(chosen_scores, rejected_scores, alternative='greater')
    ks_stat, p_value_ks = ks_2samp(chosen_scores, rejected_scores)
    
    stats_text = f"""Statistical Summary:

Chosen Responses:
  Count: {len(chosen_scores)}
  Mean: {np.mean(chosen_scores):.4f}
  Std: {np.std(chosen_scores):.4f}
  Min: {np.min(chosen_scores):.4f}
  Max: {np.max(chosen_scores):.4f}
  Q25: {np.percentile(chosen_scores, 25):.4f}
  Q75: {np.percentile(chosen_scores, 75):.4f}

Rejected Responses:
  Count: {len(rejected_scores)}
  Mean: {np.mean(rejected_scores):.4f}
  Std: {np.std(rejected_scores):.4f}
  Min: {np.min(rejected_scores):.4f}
  Max: {np.max(rejected_scores):.4f}
  Q25: {np.percentile(rejected_scores, 25):.4f}
  Q75: {np.percentile(rejected_scores, 75):.4f}

Comparison:
  Mean Difference: {np.mean(chosen_scores) - np.mean(rejected_scores):.4f}
  Accuracy: {np.mean(chosen_scores > rejected_scores)*100:.2f}%
  
Statistical Tests:
  Mann-Whitney U (p-value): {p_value_mw:.4e}
  Kolmogorov-Smirnov (p-value): {p_value_ks:.4e}
"""
    
    plt.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'comprehensive_reward_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file, stats_text

def analyze_text_lengths(chosen_texts, rejected_texts, chosen_scores, rejected_scores, output_dir):
    """Analyze the relationship between text length and reward scores."""
    
    chosen_lengths = [len(text.split()) for text in chosen_texts]
    rejected_lengths = [len(text.split()) for text in rejected_texts]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Length vs Score scatter plot
    plt.subplot(2, 3, 1)
    plt.scatter(chosen_lengths, chosen_scores, alpha=0.6, color='green', label='Chosen', s=20)
    plt.scatter(rejected_lengths, rejected_scores, alpha=0.6, color='red', label='Rejected', s=20)
    plt.xlabel('Text Length (words)')
    plt.ylabel('Reward Score')
    plt.title('Text Length vs Reward Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Length distribution
    plt.subplot(2, 3, 2)
    plt.hist(chosen_lengths, bins=30, alpha=0.7, label='Chosen', color='green', density=True)
    plt.hist(rejected_lengths, bins=30, alpha=0.7, label='Rejected', color='red', density=True)
    plt.xlabel('Text Length (words)')
    plt.ylabel('Density')
    plt.title('Text Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot of lengths
    plt.subplot(2, 3, 3)
    plt.boxplot([chosen_lengths, rejected_lengths], labels=['Chosen', 'Rejected'])
    plt.ylabel('Text Length (words)')
    plt.title('Text Length Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Length correlation analysis
    plt.subplot(2, 3, 4)
    all_lengths = chosen_lengths + rejected_lengths
    all_scores = chosen_scores + rejected_scores
    correlation = np.corrcoef(all_lengths, all_scores)[0, 1]
    
    plt.scatter(all_lengths, all_scores, alpha=0.6, s=20)
    z = np.polyfit(all_lengths, all_scores, 1)
    p = np.poly1d(z)
    plt.plot(all_lengths, p(all_lengths), "r--", alpha=0.8)
    plt.xlabel('Text Length (words)')
    plt.ylabel('Reward Score')
    plt.title(f'Length vs Score Correlation (r={correlation:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 5. Length-based score analysis
    plt.subplot(2, 3, 5)
    # Bin by length and compare mean scores
    max_length = max(max(chosen_lengths), max(rejected_lengths))
    bins = np.linspace(0, max_length, 10)
    
    chosen_binned_scores = []
    rejected_binned_scores = []
    bin_centers = []
    
    for i in range(len(bins)-1):
        chosen_mask = (np.array(chosen_lengths) >= bins[i]) & (np.array(chosen_lengths) < bins[i+1])
        rejected_mask = (np.array(rejected_lengths) >= bins[i]) & (np.array(rejected_lengths) < bins[i+1])
        
        if np.any(chosen_mask):
            chosen_binned_scores.append(np.mean(np.array(chosen_scores)[chosen_mask]))
        else:
            chosen_binned_scores.append(np.nan)
            
        if np.any(rejected_mask):
            rejected_binned_scores.append(np.mean(np.array(rejected_scores)[rejected_mask]))
        else:
            rejected_binned_scores.append(np.nan)
            
        bin_centers.append((bins[i] + bins[i+1]) / 2)
    
    plt.plot(bin_centers, chosen_binned_scores, 'g-o', label='Chosen', linewidth=2)
    plt.plot(bin_centers, rejected_binned_scores, 'r-o', label='Rejected', linewidth=2)
    plt.xlabel('Text Length (words)')
    plt.ylabel('Mean Reward Score')
    plt.title('Mean Score by Length Bins')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    length_stats = f"""Length Analysis Summary:

Chosen Texts:
  Mean Length: {np.mean(chosen_lengths):.1f} words
  Std Length: {np.std(chosen_lengths):.1f} words
  Min Length: {np.min(chosen_lengths)} words
  Max Length: {np.max(chosen_lengths)} words

Rejected Texts:
  Mean Length: {np.mean(rejected_lengths):.1f} words
  Std Length: {np.std(rejected_lengths):.1f} words
  Min Length: {np.min(rejected_lengths)} words
  Max Length: {np.max(rejected_lengths)} words

Correlation Analysis:
  Length-Score Correlation: {correlation:.4f}
  
Length Difference:
  Mean Diff: {np.mean(chosen_lengths) - np.mean(rejected_lengths):.1f} words
"""
    
    plt.text(0.05, 0.95, length_stats, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'text_length_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    parser = argparse.ArgumentParser(description='Analyze reward model scores')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing chosen_responses.json and rejected_responses.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: same as data_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    # Load data
    chosen_file = os.path.join(args.data_dir, 'chosen_responses.json')
    rejected_file = os.path.join(args.data_dir, 'rejected_responses.json')
    
    if not os.path.exists(chosen_file) or not os.path.exists(rejected_file):
        print(f"Error: Could not find chosen_responses.json or rejected_responses.json in {args.data_dir}")
        return
    
    print("Loading reward data...")
    chosen_scores, rejected_scores, chosen_texts, rejected_texts = load_reward_data(chosen_file, rejected_file)
    
    print(f"Loaded {len(chosen_scores)} chosen and {len(rejected_scores)} rejected responses")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    plot_file, stats = create_comprehensive_visualization(chosen_scores, rejected_scores, args.output_dir)
    print(f"Comprehensive analysis saved to: {plot_file}")
    
    # Analyze text lengths
    print("Analyzing text lengths...")
    length_plot = analyze_text_lengths(chosen_texts, rejected_texts, chosen_scores, rejected_scores, args.output_dir)
    print(f"Text length analysis saved to: {length_plot}")
    
    # Save detailed statistics
    stats_file = os.path.join(args.output_dir, 'detailed_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(stats)
    print(f"Detailed statistics saved to: {stats_file}")
    
    print("\nAnalysis complete!")
    print(stats)

if __name__ == '__main__':
    main() 