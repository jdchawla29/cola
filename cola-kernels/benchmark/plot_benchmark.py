import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_results(results_dir):
    """Load benchmark results and config from directory."""
    results_dir = Path(results_dir)
    
    with open(results_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    with open(results_dir / 'results.json', 'r') as f:
        results = json.load(f)
    
    return config, results

def results_to_dataframe(results, config):
    """Convert results to a pandas DataFrame for easier plotting."""
    data = []
    
    for size in results.keys():
        for matrix_type in results[size].keys():
            for method in results[size][matrix_type].keys():
                method_results = results[size][matrix_type][method]
                
                # Skip if no successful runs
                if 'mean_time' not in method_results:
                    continue
                
                row = {
                    'size': int(size),
                    'matrix_type': config['matrix_type_names'][matrix_type],
                    'method': method,
                    'mean_time': method_results['mean_time'],
                    'std_time': method_results['std_time'],
                    'mean_error': method_results['mean_error'],
                    'std_error': method_results['std_error'],
                    'successful_runs': method_results.get('successful_runs', 0)
                }
                
                if 'mean_peak_memory' in method_results:
                    row['mean_memory'] = method_results['mean_peak_memory'] / (1024 * 1024)  # Convert to MB
                    row['std_memory'] = method_results['std_peak_memory'] / (1024 * 1024)
                
                data.append(row)
    
    return pd.DataFrame(data)

def plot_time_scaling(df, output_dir):
    """Plot time scaling for each matrix type."""
    plt.figure(figsize=(15, 10))
    
    for i, matrix_type in enumerate(df['matrix_type'].unique()):
        plt.subplot(2, 2, i+1)
        data = df[df['matrix_type'] == matrix_type]
        
        sns.lineplot(data=data, x='size', y='mean_time', hue='method', marker='o')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Time Scaling: {matrix_type}')
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (s)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_scaling(df, output_dir):
    """Plot error scaling for each matrix type."""
    plt.figure(figsize=(15, 10))
    
    for i, matrix_type in enumerate(df['matrix_type'].unique()):
        plt.subplot(2, 2, i+1)
        data = df[df['matrix_type'] == matrix_type]
        
        sns.lineplot(data=data, x='size', y='mean_error', hue='method', marker='o')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Error Scaling: {matrix_type}')
        plt.xlabel('Matrix Size')
        plt.ylabel('Relative Error')
        plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_usage(df, output_dir):
    """Plot memory usage for each matrix type."""
    if 'mean_memory' not in df.columns:
        return
    
    plt.figure(figsize=(15, 10))
    
    for i, matrix_type in enumerate(df['matrix_type'].unique()):
        plt.subplot(2, 2, i+1)
        data = df[df['matrix_type'] == matrix_type]
        
        sns.lineplot(data=data, x='size', y='mean_memory', hue='method', marker='o')

        
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Memory Usage: {matrix_type}')
        plt.xlabel('Matrix Size')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(df, output_dir):
    """Create scatter plots comparing time vs error for each size."""
    unique_sizes = sorted(df['size'].unique())
    n_sizes = len(unique_sizes)
    n_cols = min(3, n_sizes)
    n_rows = (n_sizes + n_cols - 1) // n_cols
    
    plt.figure(figsize=(6*n_cols, 5*n_rows))
    
    for i, size in enumerate(unique_sizes):
        plt.subplot(n_rows, n_cols, i+1)
        data = df[df['size'] == size]
        
        sns.scatterplot(data=data, x='mean_time', y='mean_error', 
                       hue='method', style='matrix_type', s=100)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Performance Comparison (N={size})')
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Error')
        plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate(df, output_dir):
    """Plot success rate of different methods."""
    plt.figure(figsize=(12, 6))
    
    success_data = df.pivot_table(
        values='successful_runs',
        index=['method'],
        columns=['size'],
        aggfunc='mean'
    )
    
    sns.heatmap(success_data, annot=True, cmap='RdYlGn', vmin=0, vmax=5,
                fmt='.1f', cbar_kws={'label': 'Successful Runs (out of 5)'})
    
    plt.title('Method Success Rate by Matrix Size')
    plt.ylabel('Method')
    plt.xlabel('Matrix Size')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate a summary table of results."""
    summary = df.groupby(['method', 'matrix_type']).agg({
        'mean_time': ['mean', 'std'],
        'mean_error': ['mean', 'std'],
        'successful_runs': 'mean'
    }).round(4)
    
    summary.to_csv(output_dir / 'summary.csv')
    
    # Also save as a formatted markdown table
    with open(output_dir / 'summary.md', 'w') as f:
        f.write(summary.to_markdown())

def plot_all_results(results_dir='hutch_benchmark_results'):
    """Generate all plots from saved benchmark results."""
    print("Loading results...")
    config, results = load_results(results_dir)
    
    results_dir = Path(results_dir)
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("Converting results to DataFrame...")
    df = results_to_dataframe(results, config)
    
    print("Generating plots...")
    plot_time_scaling(df, plots_dir)
    plot_error_scaling(df, plots_dir)
    plot_memory_usage(df, plots_dir)
    plot_performance_comparison(df, plots_dir)
    plot_success_rate(df, plots_dir)
    
    print("Generating summary statistics...")
    generate_summary_table(df, plots_dir)
    
    print(f"All plots and statistics saved in {plots_dir}")

if __name__ == '__main__':
    plot_all_results()