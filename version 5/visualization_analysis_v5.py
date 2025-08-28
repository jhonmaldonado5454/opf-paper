"""
Advanced Visualization and Comparison Analysis Script - Version 5
For Texas Load Forecasting Framework
File: visualization_analysis_v5.py
This script creates additional comparative visualizations from the results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results_v5():
    """Load results from Julia JSON output"""
    try:
        with open('forecasting_results_julia_v5.json', 'r') as f:
            data = json.load(f)
        print("✓ Results loaded from forecasting_results_julia_v5.json")
        return data
    except FileNotFoundError:
        print("✗ File not found. Make sure to run Julia framework first!")
        return None

def create_comprehensive_comparison_v5(results):
    """Create comprehensive model comparison visualizations"""
    print("\n" + "="*60)
    print("Creating Comprehensive Model Comparison Plots...")
    print("="*60)
    
    # Extract results
    horizons = ['1h', '24h', '168h', '360h']
    models = ['linear', 'ridge', 'lasso', 'nn']
    metrics = ['rmse', 'mae', 'mape', 'r2']
    
    # Create data matrix
    data_dict = {metric: {model: [] for model in models} for metric in metrics}
    
    for horizon in horizons:
        for model in models:
            key = f"{model}_{horizon}_test"
            if key in results:
                for metric in metrics:
                    data_dict[metric][model].append(results[key][metric])
    
    # Figure 1: Side-by-side comparison for Neural Networks vs Linear
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Neural Networks vs Linear Regression - Comprehensive Comparison (v5)', 
                 fontsize=16, fontweight='bold')
    
    # Colors
    colors = {'linear': '#3498db', 'nn': '#e74c3c', 'ridge': '#2ecc71', 'lasso': '#f39c12'}
    
    # Plot 1: RMSE comparison
    ax = axes[0, 0]
    x = np.arange(len(horizons))
    width = 0.35
    ax.bar(x - width/2, data_dict['rmse']['linear'], width, label='Linear', color=colors['linear'], alpha=0.8)
    ax.bar(x + width/2, data_dict['rmse']['nn'], width, label='Neural Network', color=colors['nn'], alpha=0.8)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('RMSE (MW)')
    ax.set_title('Root Mean Square Error')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MAE comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, data_dict['mae']['linear'], width, label='Linear', color=colors['linear'], alpha=0.8)
    ax.bar(x + width/2, data_dict['mae']['nn'], width, label='Neural Network', color=colors['nn'], alpha=0.8)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('MAE (MW)')
    ax.set_title('Mean Absolute Error')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MAPE comparison
    ax = axes[0, 2]
    ax.bar(x - width/2, data_dict['mape']['linear'], width, label='Linear', color=colors['linear'], alpha=0.8)
    ax.bar(x + width/2, data_dict['mape']['nn'], width, label='Neural Network', color=colors['nn'], alpha=0.8)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Mean Absolute Percentage Error')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: R² Score comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, data_dict['r2']['linear'], width, label='Linear', color=colors['linear'], alpha=0.8)
    ax.bar(x + width/2, data_dict['r2']['nn'], width, label='Neural Network', color=colors['nn'], alpha=0.8)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('R² Score')
    ax.set_title('Coefficient of Determination')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Improvement Analysis
    ax = axes[1, 1]
    improvements = []
    for i in range(len(horizons)):
        if data_dict['rmse']['linear'][i] > 0:
            imp = ((data_dict['rmse']['linear'][i] - data_dict['rmse']['nn'][i]) / 
                   data_dict['rmse']['linear'][i]) * 100
            improvements.append(imp)
        else:
            improvements.append(0)
    
    bars = ax.bar(horizons, improvements, color=['green' if x > 0 else 'red' for x in improvements], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('NN Improvement over Linear (RMSE)')
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 6: All models comparison (average MAPE)
    ax = axes[1, 2]
    avg_mapes = {model: np.mean(data_dict['mape'][model]) for model in models}
    bars = ax.bar(models, list(avg_mapes.values()), 
                  color=[colors.get(m, '#95a5a6') for m in models], alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Average MAPE (%)')
    ax.set_title('Average MAPE Across All Horizons')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('nn_vs_linear_detailed_comparison_v5.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: nn_vs_linear_detailed_comparison_v5.png")
    
    # Figure 2: Performance Evolution Across Horizons
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Evolution Across Forecast Horizons (v5)', 
                 fontsize=16, fontweight='bold')
    
    horizon_values = [1, 24, 168, 360]  # Actual hours
    
    # Evolution plots for each metric
    for idx, (metric, ylabel, title) in enumerate([
        ('rmse', 'RMSE (MW)', 'RMSE Evolution'),
        ('mae', 'MAE (MW)', 'MAE Evolution'),
        ('mape', 'MAPE (%)', 'MAPE Evolution'),
        ('r2', 'R² Score', 'R² Score Evolution')
    ]):
        ax = axes[idx // 2, idx % 2]
        
        for model in models:
            values = data_dict[metric][model]
            ax.plot(horizon_values, values, marker='o', label=model.upper(), 
                   linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Forecast Horizon (hours)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(horizon_values)
        ax.set_xticklabels(horizons)
    
    plt.tight_layout()
    plt.savefig('performance_evolution_v5.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: performance_evolution_v5.png")
    
    # Figure 3: Heatmap Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Heatmaps (v5)', fontsize=16, fontweight='bold')
    
    for idx, (metric, title, cmap) in enumerate([
        ('rmse', 'RMSE Heatmap (MW)', 'YlOrRd'),
        ('mae', 'MAE Heatmap (MW)', 'YlOrBr'),
        ('mape', 'MAPE Heatmap (%)', 'RdYlBu_r'),
        ('r2', 'R² Score Heatmap', 'RdYlGn')
    ]):
        ax = axes[idx // 2, idx % 2]
        
        # Create matrix for heatmap
        matrix = []
        for model in models:
            matrix.append(data_dict[metric][model])
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(horizons)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(horizons)
        ax.set_yticklabels([m.upper() for m in models])
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Model')
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add value annotations
        for i in range(len(models)):
            for j in range(len(horizons)):
                text = ax.text(j, i, f'{matrix[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig('performance_heatmaps_v5.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: performance_heatmaps_v5.png")
    
    # Figure 4: Radar Chart Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Radar Charts (v5)', fontsize=16, fontweight='bold')
    
    # Prepare data for radar charts
    categories = ['RMSE\n(normalized)', 'MAE\n(normalized)', 'MAPE', 'R²\n(scaled)', 'Avg Performance']
    
    for idx, horizon in enumerate(['24h', '168h']):
        ax = axes[idx]
        ax = plt.subplot(1, 2, idx + 1, projection='polar')
        
        # Number of variables
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        for model in ['linear', 'nn']:
            key = f"{model}_{horizon}_test"
            if key in results:
                # Normalize values for radar chart (0-1 scale)
                rmse_norm = 1 - (results[key]['rmse'] / 5000)  # Assume max RMSE of 5000
                mae_norm = 1 - (results[key]['mae'] / 4000)    # Assume max MAE of 4000
                mape_val = 1 - (results[key]['mape'] / 10)     # Assume max MAPE of 10%
                r2_val = results[key]['r2']
                avg_perf = (rmse_norm + mae_norm + mape_val + r2_val) / 4
                
                values = [rmse_norm, mae_norm, mape_val, r2_val, avg_perf]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model.upper(), alpha=0.7)
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'{horizon} Forecast Horizon', size=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('radar_comparison_v5.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: radar_comparison_v5.png")

def create_summary_report_v5(results):
    """Create a visual summary report"""
    print("\nCreating Summary Report...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Texas Load Forecasting Framework v5 - Executive Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Key metrics summary (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    
    # Extract key metrics
    horizons = ['1h', '24h', '168h', '360h']
    models = ['linear', 'nn']
    
    summary_text = "KEY FINDINGS:\n\n"
    for horizon in ['24h', '360h']:
        linear_key = f"linear_{horizon}_test"
        nn_key = f"nn_{horizon}_test"
        
        if linear_key in results and nn_key in results:
            improvement = ((results[linear_key]['rmse'] - results[nn_key]['rmse']) / 
                          results[linear_key]['rmse']) * 100
            summary_text += f"• {horizon} Horizon: NN shows {improvement:.1f}% RMSE improvement\n"
            summary_text += f"  - Linear RMSE: {results[linear_key]['rmse']:.2f} MW\n"
            summary_text += f"  - NN RMSE: {results[nn_key]['rmse']:.2f} MW\n\n"
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Best model per horizon (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    models_all = ['linear', 'ridge', 'lasso', 'nn']
    best_models = []
    
    for horizon in horizons:
        best_rmse = float('inf')
        best_model = ''
        for model in models_all:
            key = f"{model}_{horizon}_test"
            if key in results and results[key]['rmse'] < best_rmse:
                best_rmse = results[key]['rmse']
                best_model = model
        best_models.append(best_model.upper())
    
    y_pos = np.arange(len(horizons))
    colors_map = {'LINEAR': '#3498db', 'NN': '#e74c3c', 'RIDGE': '#2ecc71', 'LASSO': '#f39c12'}
    colors = [colors_map.get(m, '#95a5a6') for m in best_models]
    
    ax2.barh(y_pos, [1]*len(horizons), color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(horizons)
    ax2.set_xlabel('Best Model')
    ax2.set_title('Best Model by Horizon')
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    
    for i, model in enumerate(best_models):
        ax2.text(0.5, i, model, ha='center', va='center', fontweight='bold')
    
    # MAPE comparison (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    mape_data = {'Linear': [], 'NN': []}
    
    for horizon in horizons:
        for model in ['linear', 'nn']:
            key = f"{model}_{horizon}_test"
            if key in results:
                model_name = 'Linear' if model == 'linear' else 'NN'
                mape_data[model_name].append(results[key]['mape'])
    
    x = np.arange(len(horizons))
    width = 0.35
    ax3.bar(x - width/2, mape_data['Linear'], width, label='Linear', color='#3498db', alpha=0.8)
    ax3.bar(x + width/2, mape_data['NN'], width, label='NN', color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Horizon')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('MAPE Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # R² Score comparison (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    r2_data = {'Linear': [], 'NN': []}
    
    for horizon in horizons:
        for model in ['linear', 'nn']:
            key = f"{model}_{horizon}_test"
            if key in results:
                model_name = 'Linear' if model == 'linear' else 'NN'
                r2_data[model_name].append(results[key]['r2'])
    
    ax4.bar(x - width/2, r2_data['Linear'], width, label='Linear', color='#3498db', alpha=0.8)
    ax4.bar(x + width/2, r2_data['NN'], width, label='NN', color='#e74c3c', alpha=0.8)
    ax4.set_xlabel('Horizon')
    ax4.set_ylabel('R² Score')
    ax4.set_title('R² Score Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(horizons)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Improvement percentages (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    improvements = []
    
    for horizon in horizons:
        linear_key = f"linear_{horizon}_test"
        nn_key = f"nn_{horizon}_test"
        
        if linear_key in results and nn_key in results:
            imp = ((results[linear_key]['rmse'] - results[nn_key]['rmse']) / 
                   results[linear_key]['rmse']) * 100
            improvements.append(imp)
        else:
            improvements.append(0)
    
    bars = ax5.bar(horizons, improvements, color=['green' if x > 0 else 'red' for x in improvements], alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Horizon')
    ax5.set_ylabel('Improvement (%)')
    ax5.set_title('NN Improvement over Linear')
    ax5.grid(True, alpha=0.3)
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Model complexity vs performance (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Calculate average MAPE for each model
    models_all = ['linear', 'ridge', 'lasso', 'nn']
    avg_mapes = []
    model_complexity = [1, 2, 2, 100]  # Relative complexity scores
    
    for model in models_all:
        mapes = []
        for horizon in horizons:
            key = f"{model}_{horizon}_test"
            if key in results:
                mapes.append(results[key]['mape'])
        avg_mapes.append(np.mean(mapes) if mapes else 0)
    
    scatter = ax6.scatter(model_complexity, avg_mapes, s=200, alpha=0.6,
                         c=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    
    for i, model in enumerate(models_all):
        ax6.annotate(model.upper(), (model_complexity[i], avg_mapes[i]),
                    ha='center', va='center', fontweight='bold')
    
    ax6.set_xlabel('Model Complexity (relative scale)', fontsize=12)
    ax6.set_ylabel('Average MAPE (%)', fontsize=12)
    ax6.set_title('Model Complexity vs Performance Trade-off', fontsize=13)
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    
    # Add trend line
    z = np.polyfit(np.log(model_complexity), avg_mapes, 1)
    p = np.poly1d(z)
    x_trend = np.logspace(0, 2.5, 100)
    ax6.plot(x_trend, p(np.log(x_trend)), "r--", alpha=0.5, label='Trend')
    ax6.legend()
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f'Generated: {timestamp} | Version 5.0', 
             ha='right', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig('executive_summary_v5.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: executive_summary_v5.png")

def main():
    """Main execution function"""
    print("="*80)
    print("ADVANCED VISUALIZATION ANALYSIS - VERSION 5")
    print("Texas Load Forecasting Framework")
    print("="*80)
    
    # Load results
    results = load_results_v5()
    
    if results and 'results' in results:
        results_data = results['results']
        
        # Generate all visualizations
        create_comprehensive_comparison_v5(results_data)
        create_summary_report_v5(results_data)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("1. nn_vs_linear_detailed_comparison_v5.png")
        print("2. performance_evolution_v5.png")
        print("3. performance_heatmaps_v5.png")
        print("4. radar_comparison_v5.png")
        print("5. executive_summary_v5.png")
        print("\nTotal: 5 additional comparison plots")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        horizons = ['1h', '24h', '168h', '360h']
        
        for horizon in horizons:
            linear_key = f"linear_{horizon}_test"
            nn_key = f"nn_{horizon}_test"
            
            if linear_key in results_data and nn_key in results_data:
                print(f"\n{horizon} Horizon:")
                print(f"  Linear - RMSE: {results_data[linear_key]['rmse']:.2f} MW, "
                      f"MAPE: {results_data[linear_key]['mape']:.2f}%")
                print(f"  NN     - RMSE: {results_data[nn_key]['rmse']:.2f} MW, "
                      f"MAPE: {results_data[nn_key]['mape']:.2f}%")
                
                improvement = ((results_data[linear_key]['rmse'] - results_data[nn_key]['rmse']) / 
                              results_data[linear_key]['rmse']) * 100
                print(f"  → Improvement: {improvement:.1f}%")
    else:
        print("\n✗ No results found. Please run the Julia framework first:")
        print("  julia load_forecasting_framework_v5.jl")

if __name__ == "__main__":
    main()