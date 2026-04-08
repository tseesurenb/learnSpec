'''
Unified Filter Visualization for LearnSpec
Combines all filter analysis and visualization functionality
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from glob import glob
from filter import APSFilter, evaluate_polynomial_basis, normalize_eigenvalues_for_basis

def visualize_initial_filters():
    """Visualize the initial frequency response of the 4 subfilters"""
    
    # Create filter with ensemble mode (Case 1 configuration)
    aps_filter = APSFilter(
        filter_order=6,
        init_filter_name='uniform',
        mix=None,
        fix_filter_weights=False,
        poly_basis='ensemble'
    )
    
    # Create frequency range
    n_points = 1000
    eigenvals = torch.linspace(0, 1, n_points)
    
    # Get initial mixing weights
    initial_weights = aps_filter.mixing_weights.detach().numpy()
    softmax_weights = torch.softmax(aps_filter.mixing_weights, dim=0).detach().numpy()
    
    print("\n=== Initial Filter Configuration ===")
    print("Filter Names:", aps_filter.filter_names)
    print("Subfilter bases:", aps_filter.subfilter_bases)
    print("Initial weights (softmax):", softmax_weights)
    
    # Compute individual subfilter responses
    subfilter_responses = []
    subfilter_labels = []
    
    for i, (fname, basis) in enumerate(zip(aps_filter.filter_names, aps_filter.subfilter_bases)):
        filter_coeffs = getattr(aps_filter, f'filter_{i}').detach()
        x_basis = normalize_eigenvalues_for_basis(eigenvals, basis)
        response = evaluate_polynomial_basis(filter_coeffs, x_basis, basis)
        response = torch.exp(-torch.abs(response).clamp(max=10.0)) + 1e-6
        
        subfilter_responses.append(response.numpy())
        subfilter_labels.append(f'{fname} ({basis})')
    
    # Compute combined response
    combined_response = np.zeros_like(subfilter_responses[0])
    for i, response in enumerate(subfilter_responses):
        combined_response += softmax_weights[i] * response
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Individual subfilters
    colors = ['red', 'blue', 'green', 'orange']
    for i, (response, label) in enumerate(zip(subfilter_responses, subfilter_labels)):
        weight = softmax_weights[i]
        ax1.plot(eigenvals.numpy(), response, 
                color=colors[i], linewidth=2, 
                label=f'{label} (w={weight:.3f})', alpha=0.8)
    
    ax1.set_xlabel('Eigenvalue (Frequency)')
    ax1.set_ylabel('Filter Response')
    ax1.set_title('Individual Subfilter Responses (Before Learning)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Combined filter
    ax2.plot(eigenvals.numpy(), combined_response, 
             color='black', linewidth=3, label='Combined Filter')
    
    # Show weighted contributions
    for i, (response, fname) in enumerate(zip(subfilter_responses, aps_filter.filter_names)):
        weighted_response = softmax_weights[i] * response
        ax2.plot(eigenvals.numpy(), weighted_response, 
                color=colors[i], linewidth=1, alpha=0.6, linestyle='--',
                label=f'{fname} weighted')
    
    ax2.set_xlabel('Eigenvalue (Frequency)')
    ax2.set_ylabel('Filter Response')
    ax2.set_title('Combined Filter Response (Weighted Sum)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save to results directory outside src
    project_root = os.path.dirname(os.path.dirname(__file__))  # Go up from src to project root
    results_dir = os.path.join(project_root, 'results')
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    save_path = os.path.join(viz_dir, 'initial_filters.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analysis
    print(f"\n=== Initial Filter Analysis ===")
    print(f"Low-freq response (λ=0.1): {combined_response[int(0.1*n_points)]:.4f}")
    print(f"Mid-freq response (λ=0.5): {combined_response[int(0.5*n_points)]:.4f}")
    print(f"High-freq response (λ=0.9): {combined_response[int(0.9*n_points)]:.4f}")
    
    # Find cutoff
    cutoff_idx = np.where(combined_response <= 0.5)[0]
    if len(cutoff_idx) > 0:
        cutoff_freq = eigenvals[cutoff_idx[0]].item()
        print(f"Approximate cutoff frequency (0.5 response): λ={cutoff_freq:.3f}")
    
    print(f"\n💾 Saved to: {os.path.relpath(save_path)}")


def load_filter_params(filepath):
    """Load saved filter parameters"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    return params


def reconstruct_filter_response(state_dict, filter_names, subfilter_bases, n_points=1000):
    """Reconstruct filter response from saved parameters"""
    eigenvals = torch.linspace(0, 1, n_points)
    
    # Extract parameters
    mixing_weights_raw = state_dict['filter.mixing_weights']
    mixing_weights = torch.softmax(mixing_weights_raw, dim=0).numpy()
    
    refinement_coeffs = state_dict['filter.refinement_coeffs']
    refinement_scale = state_dict['filter.refinement_scale']
    transform_scale = state_dict['filter.transform_scale']
    transform_bias = state_dict['filter.transform_bias']
    
    # Get individual subfilter responses
    subfilter_responses = []
    
    for i, (fname, basis) in enumerate(zip(filter_names, subfilter_bases)):
        filter_coeffs = state_dict[f'filter.filter_{i}']
        refined_coeffs = filter_coeffs + refinement_scale * torch.tanh(refinement_coeffs)
        x_basis = normalize_eigenvalues_for_basis(eigenvals, basis)
        response = evaluate_polynomial_basis(refined_coeffs, x_basis, basis)
        response = transform_scale * response + transform_bias
        response = torch.exp(-torch.abs(response).clamp(max=10.0)) + 1e-6
        subfilter_responses.append(response.numpy())
    
    # Compute combined response
    combined_response = np.zeros_like(subfilter_responses[0])
    for i, response in enumerate(subfilter_responses):
        combined_response += mixing_weights[i] * response
    
    return eigenvals.numpy(), subfilter_responses, combined_response, mixing_weights


def visualize_evolution(params_file):
    """Visualize the filter evolution from initial to best"""
    
    # Load parameters
    params = load_filter_params(params_file)
    config = params['initial']['config']
    
    print(f"\n=== Filter Evolution Analysis ===")
    print(f"Dataset: {config['dataset']}")
    print(f"Filter: {config['filter']} with poly={config['poly']}")
    print(f"Best epoch: {params['best']['epoch']} with NDCG={params['best']['ndcg']:.4f}")
    
    if params['initial']['user_filter'] is None:
        print("No user filter found in saved parameters")
        return
    
    # Get metadata
    filter_names = params['initial']['filter_names']
    subfilter_bases = params['initial']['subfilter_bases'] if params['initial']['subfilter_bases'] else None
    
    # Handle non-ensemble case
    if subfilter_bases is None or config.get('poly') != 'ensemble':
        poly_basis = config.get('poly', 'bernstein')
        subfilter_bases = [poly_basis] * len(filter_names)
    
    # Reconstruct responses
    eigenvals, init_subs, init_combined, init_weights = reconstruct_filter_response(
        params['initial']['user_filter'], filter_names, subfilter_bases
    )
    
    _, best_subs, best_combined, best_weights = reconstruct_filter_response(
        params['best']['user_filter'], filter_names, subfilter_bases
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Initial state
    ax = axes[0, 0]
    for i, (response, fname, basis) in enumerate(zip(init_subs, filter_names, subfilter_bases)):
        ax.plot(eigenvals, response, colors[i], linewidth=2, alpha=0.7,
                label=f'{fname} ({basis}) w={init_weights[i]:.3f}')
    ax.plot(eigenvals, init_combined, 'k-', linewidth=3, label='Combined')
    ax.set_title('Initial Filter Response', fontsize=14, fontweight='bold')
    ax.set_xlabel('Eigenvalue (Frequency)')
    ax.set_ylabel('Filter Response')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Best state
    ax = axes[0, 1]
    for i, (response, fname, basis) in enumerate(zip(best_subs, filter_names, subfilter_bases)):
        ax.plot(eigenvals, response, colors[i], linewidth=2, alpha=0.7,
                label=f'{fname} ({basis}) w={best_weights[i]:.3f}')
    ax.plot(eigenvals, best_combined, 'k-', linewidth=3, label='Combined')
    ax.set_title(f'Learned Filter Response (Epoch {params["best"]["epoch"]})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Eigenvalue (Frequency)')
    ax.set_ylabel('Filter Response')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Direct comparison
    ax = axes[1, 0]
    ax.plot(eigenvals, init_combined, 'b-', linewidth=3, label='Initial', alpha=0.8)
    ax.plot(eigenvals, best_combined, 'r-', linewidth=3, label='Learned', alpha=0.8)
    ax.fill_between(eigenvals, init_combined, best_combined, 
                    where=(best_combined > init_combined), 
                    color='green', alpha=0.2, label='Increased')
    ax.fill_between(eigenvals, init_combined, best_combined, 
                    where=(best_combined <= init_combined), 
                    color='red', alpha=0.2, label='Decreased')
    ax.set_title('Filter Evolution: Initial → Learned', fontsize=14, fontweight='bold')
    ax.set_xlabel('Eigenvalue (Frequency)')
    ax.set_ylabel('Filter Response')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Weight changes
    ax = axes[1, 1]
    x_pos = np.arange(len(filter_names))
    width = 0.35
    
    ax.bar(x_pos - width/2, init_weights, width, label='Initial', alpha=0.8, color='blue')
    ax.bar(x_pos + width/2, best_weights, width, label='Learned', alpha=0.8, color='red')
    
    ax.set_xlabel('Subfilter')
    ax.set_ylabel('Mixing Weight')
    ax.set_title('Subfilter Weight Evolution', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{fname}\n({basis})' for fname, basis in zip(filter_names, subfilter_bases)], 
                       rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add performance text
    analysis_text = f"""
Performance: NDCG {params['best']['ndcg']:.4f} at epoch {params['best']['epoch']}
Test NDCG: {params['best']['test_ndcg']:.4f}
Test Recall: {params['best']['test_recall']:.4f}
    """
    ax.text(0.02, 0.98, analysis_text, transform=ax.transAxes, 
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    
    plt.suptitle(f"Filter Evolution: {config['dataset']} - {config['filter']} ({config['poly']})", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save to results directory outside src
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, 'results')
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    output_name = os.path.basename(params_file).replace('.pkl', '_evolution.png')
    output_path = os.path.join(viz_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analysis
    low_freq_idx = int(0.3 * len(eigenvals))
    mid_freq_idx = int(0.7 * len(eigenvals))
    
    init_low = np.mean(init_combined[:low_freq_idx])
    init_mid = np.mean(init_combined[low_freq_idx:mid_freq_idx])
    init_high = np.mean(init_combined[mid_freq_idx:])
    
    best_low = np.mean(best_combined[:low_freq_idx])
    best_mid = np.mean(best_combined[low_freq_idx:mid_freq_idx])
    best_high = np.mean(best_combined[mid_freq_idx:])
    
    print(f"\nFrequency Band Evolution:")
    print(f"Low-freq:  {init_low:.4f} → {best_low:.4f} (Δ={best_low-init_low:+.4f})")
    print(f"Mid-freq:  {init_mid:.4f} → {best_mid:.4f} (Δ={best_mid-init_mid:+.4f})")
    print(f"High-freq: {init_high:.4f} → {best_high:.4f} (Δ={best_high-init_high:+.4f})")
    
    print("\nWeight Evolution:")
    for i, fname in enumerate(filter_names):
        print(f"  {fname}: {init_weights[i]:.3f} → {best_weights[i]:.3f} (Δ={best_weights[i]-init_weights[i]:+.3f})")
    
    print(f"\n✓ Visualization saved to: {os.path.relpath(output_path)}")


def analyze_eigenvalue_distributions():
    """Analyze and compare eigenvalue distributions across datasets"""
    
    # Check what eigenvalue files are available
    cache_dir = os.path.join('..', 'cache')
    datasets = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
    
    eigenvalue_data = {}
    
    print("\n=== Loading Eigenvalue Data ===")
    
    # Try to load eigenvalue data for different datasets
    for dataset in datasets:
        dataset_cache_dir = os.path.join(cache_dir, dataset)
        if not os.path.exists(dataset_cache_dir):
            continue
            
        # Look for user and item eigenvalue files with flexible naming
        import glob
        user_files = glob.glob(os.path.join(dataset_cache_dir, '*user*Eigen*.pkl'))
        item_files = glob.glob(os.path.join(dataset_cache_dir, '*item*Eigen*.pkl'))
        
        if user_files or item_files:
            eigenvalue_data[dataset] = {}
            
            # Load user eigenvalues
            if user_files:
                try:
                    import pickle
                    with open(user_files[0], 'rb') as f:
                        data = pickle.load(f)
                        eigenvalue_data[dataset]['user'] = data.get('eigenvals', [])
                        print(f"✓ {dataset}: {len(eigenvalue_data[dataset]['user'])} user eigenvalues")
                except Exception as e:
                    print(f"✗ Failed to load {dataset} user eigenvalues: {e}")
            
            # Load item eigenvalues  
            if item_files:
                try:
                    with open(item_files[0], 'rb') as f:
                        data = pickle.load(f)
                        eigenvalue_data[dataset]['item'] = data.get('eigenvals', [])
                        print(f"✓ {dataset}: {len(eigenvalue_data[dataset]['item'])} item eigenvalues")
                except Exception as e:
                    print(f"✗ Failed to load {dataset} item eigenvalues: {e}")
    
    if not eigenvalue_data:
        print("No eigenvalue data found. Please run gen_eigen.py first.")
        return
    
    # Create separate plots for each dataset for clear comparison
    n_datasets = len(eigenvalue_data)
    
    # Determine grid size - aim for roughly square layout
    n_cols = min(3, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_datasets > 1 else axes
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Create individual plot for each dataset
    for i, (dataset, data) in enumerate(eigenvalue_data.items()):
        ax = axes_flat[i]
        
        if 'user' in data and len(data['user']) > 0:
            eigenvals = np.array(data['user'])
            
            # Plot histogram
            ax.hist(eigenvals, bins=50, alpha=0.8, color=colors[i % len(colors)], 
                   density=True, edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            stats_text = f"""Statistics:
n = {len(eigenvals)}
Range: [{eigenvals.min():.3f}, {eigenvals.max():.3f}]
Mean: {eigenvals.mean():.3f}
Std: {eigenvals.std():.3f}"""
            
            # Add frequency band analysis
            sorted_eigenvals = np.sort(eigenvals)
            n_vals = len(sorted_eigenvals)
            
            low_freq_vals = sorted_eigenvals[:int(0.3 * n_vals)]
            high_freq_vals = sorted_eigenvals[int(0.7 * n_vals):]
            
            total_energy = np.sum(sorted_eigenvals)
            low_freq_energy = np.sum(low_freq_vals) / total_energy
            high_freq_energy = np.sum(high_freq_vals) / total_energy
            
            freq_text = f"""
Spectral Energy:
Low-freq (global): {low_freq_energy:.1%}
High-freq (personal): {high_freq_energy:.1%}"""
            
            # Position text box
            ax.text(0.98, 0.98, stats_text + freq_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=9, family='monospace')
            
            # Highlight frequency regions with colored backgrounds
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Low frequency region (global patterns)
            low_freq_max = low_freq_vals[-1] if len(low_freq_vals) > 0 else eigenvals.min()
            ax.axvspan(xlim[0], low_freq_max, alpha=0.1, color='blue', label='Low-freq (Global)')
            
            # High frequency region (personal patterns)  
            high_freq_min = high_freq_vals[0] if len(high_freq_vals) > 0 else eigenvals.max()
            ax.axvspan(high_freq_min, xlim[1], alpha=0.1, color='red', label='High-freq (Personal)')
        
        ax.set_xlabel('Eigenvalue λ (Small=Global, Large=Personal)')
        ax.set_ylabel('Density')
        ax.set_title(f'{dataset.upper()}: User Eigenvalue Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Set reasonable x-axis limits based on data
        if 'user' in data and len(data['user']) > 0:
            eigenvals = np.array(data['user'])
            # Set xlim to show most of the data, with some padding
            q99 = np.percentile(eigenvals, 99)
            ax.set_xlim(0, min(q99 * 1.1, eigenvals.max() * 1.05))
    
    # Hide unused subplots
    for i in range(len(eigenvalue_data), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Individual Dataset Eigenvalue Distributions\n(Small λ = Global Popularity, Large λ = Personal Preferences)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save individual plots
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, 'results')
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    save_path_individual = os.path.join(viz_dir, 'eigenvalue_distributions_individual.png')
    plt.savefig(save_path_individual, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison plot with normalized eigenvalues
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Prepare data for comparison - focus on main 4 datasets
    target_datasets = ['ml-100k', 'gowalla', 'yelp2018', 'amazon-book']
    comparison_data = {}
    
    # Filter to only include target datasets that we have data for
    for dataset in target_datasets:
        if dataset in eigenvalue_data and 'user' in eigenvalue_data[dataset]:
            comparison_data[dataset] = eigenvalue_data[dataset]
    
    print(f"\n=== Creating Comparison Plots for {len(comparison_data)} datasets ===")
    
    # Plot 1: Normalized eigenvalue distributions overlay
    colors = {'ml-100k': 'red', 'gowalla': 'blue', 'yelp2018': 'green', 'amazon-book': 'orange'}
    
    for dataset, data in comparison_data.items():
        if 'user' in data and len(data['user']) > 0:
            eigenvals = np.array(data['user'])
            
            # Normalize eigenvalues to [0,1] for comparison
            eigenvals_norm = (eigenvals - eigenvals.min()) / (eigenvals.max() - eigenvals.min() + 1e-8)
            
            # Plot histogram
            ax1.hist(eigenvals_norm, bins=50, alpha=0.6, density=True, 
                    color=colors.get(dataset, 'gray'), 
                    label=f'{dataset.upper()} (n={len(eigenvals)})',
                    edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Normalized Eigenvalue (0=Global, 1=Personal)')
    ax1.set_ylabel('Density')
    ax1.set_title('Normalized Eigenvalue Distributions Comparison\n(All datasets scaled to [0,1] for direct comparison)', 
                  fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add frequency region markers
    ax1.axvspan(0, 0.3, alpha=0.1, color='blue', label='Low-freq (Global)')
    ax1.axvspan(0.7, 1.0, alpha=0.1, color='red', label='High-freq (Personal)')
    
    # Plot 2: Cumulative energy distribution
    for dataset, data in comparison_data.items():
        if 'user' in data and len(data['user']) > 0:
            eigenvals = np.array(data['user'])
            
            # Sort eigenvalues (small to large)
            sorted_eigenvals = np.sort(eigenvals)
            cumulative_energy = np.cumsum(sorted_eigenvals) / np.sum(sorted_eigenvals)
            
            # Create x-axis as percentage of eigenvalues
            x_percentage = np.linspace(0, 100, len(cumulative_energy))
            
            ax2.plot(x_percentage, cumulative_energy * 100, 
                    linewidth=3, color=colors.get(dataset, 'gray'),
                    label=f'{dataset.upper()}', alpha=0.8)
    
    ax2.set_xlabel('Eigenvalue Rank (% from smallest to largest)')
    ax2.set_ylabel('Cumulative Spectral Energy (%)')
    ax2.set_title('Cumulative Spectral Energy Distribution\n(Shows how energy is distributed across frequency spectrum)', 
                  fontweight='bold', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    
    # Add interpretation lines
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=30, color='blue', linestyle='--', alpha=0.5, label='Low-freq cutoff')
    ax2.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='High-freq cutoff')
    
    plt.suptitle('Dataset Spectral Signature Comparison\n(Normalized eigenvalues enable direct comparison across datasets)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save comparison plot
    save_path_comparison = os.path.join(viz_dir, 'eigenvalue_comparison.png')
    plt.savefig(save_path_comparison, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Individual plots saved to: {os.path.relpath(save_path_individual)}")
    print(f"💾 Comparison plot saved to: {os.path.relpath(save_path_comparison)}")
    
    # Print analysis
    print(f"\n=== Eigenvalue Distribution Analysis ===")
    for dataset, data in eigenvalue_data.items():
        print(f"\n{dataset.upper()}:")
        
        if 'user' in data:
            eigenvals = np.array(data['user'])
            print(f"  User eigenvalues: {len(eigenvals)} values")
            print(f"    Range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
            print(f"    Mean: {eigenvals.mean():.4f}, Std: {eigenvals.std():.4f}")
            
            # Frequency band energy (CORRECTED: small eigenvals = low-freq, large = high-freq)
            # Sort eigenvalues to get frequency ordering
            sorted_eigenvals = np.sort(eigenvals)
            n_vals = len(sorted_eigenvals)
            
            # Low frequency = smallest eigenvalues (global patterns)
            low_freq_vals = sorted_eigenvals[:int(0.3 * n_vals)]
            mid_freq_vals = sorted_eigenvals[int(0.3 * n_vals):int(0.7 * n_vals)]
            high_freq_vals = sorted_eigenvals[int(0.7 * n_vals):]
            
            total_energy = np.sum(sorted_eigenvals)
            low_freq_energy = np.sum(low_freq_vals) / total_energy
            mid_freq_energy = np.sum(mid_freq_vals) / total_energy  
            high_freq_energy = np.sum(high_freq_vals) / total_energy
            
            print(f"    Eigenvalue ranges:")
            print(f"      Low-freq (global): [{sorted_eigenvals[0]:.3f}, {low_freq_vals[-1]:.3f}] → Energy: {low_freq_energy:.3f}")
            print(f"      Mid-freq (mixed):  [{mid_freq_vals[0]:.3f}, {mid_freq_vals[-1]:.3f}] → Energy: {mid_freq_energy:.3f}")
            print(f"      High-freq (personal): [{high_freq_vals[0]:.3f}, {high_freq_vals[-1]:.3f}] → Energy: {high_freq_energy:.3f}")
        
        if 'item' in data:
            eigenvals = np.array(data['item'])
            print(f"  Item eigenvalues: {len(eigenvals)} values")
            print(f"    Range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")


def analyze_highpass_phenomenon():
    """Analyze why high-pass filtering works better than low-pass"""
    
    # Create filters
    aps_filter = APSFilter(filter_order=6, poly_basis='ensemble')
    eigenvals = torch.linspace(0, 1, 1000)
    
    # Get LearnSpec response
    with torch.no_grad():
        learnspec_response = aps_filter(eigenvals).numpy()
    
    # Traditional low-pass
    lowpass_response = torch.exp(-5 * eigenvals).numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter responses
    ax = axes[0, 0]
    ax.plot(eigenvals, learnspec_response, 'r-', linewidth=2, label='LearnSpec (High-pass)')
    ax.plot(eigenvals, lowpass_response, 'b-', linewidth=2, label='Traditional (Low-pass)')
    ax.set_xlabel('Eigenvalue λ')
    ax.set_ylabel('Filter Response')
    ax.set_title('Filter Frequency Responses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale
    ax = axes[0, 1]
    ax.semilogy(eigenvals, learnspec_response, 'r-', linewidth=2, label='LearnSpec (High-pass)')
    ax.semilogy(eigenvals, lowpass_response, 'b-', linewidth=2, label='Traditional (Low-pass)')
    ax.set_xlabel('Eigenvalue λ')
    ax.set_ylabel('Filter Response (log scale)')
    ax.set_title('Filter Responses (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Frequency content
    ax = axes[1, 0]
    freq_content = np.ones(1000)
    freq_content[:200] = 5.0  # Low frequency
    freq_content[200:800] = 1.0  # Mid frequency
    freq_content[800:] = 0.2  # High frequency
    
    filtered_learnspec = freq_content * learnspec_response
    filtered_lowpass = freq_content * lowpass_response
    
    ax.plot(eigenvals, freq_content, 'k-', alpha=0.5, linewidth=1, label='Original signal')
    ax.plot(eigenvals, filtered_learnspec, 'r-', linewidth=2, label='After LearnSpec')
    ax.plot(eigenvals, filtered_lowpass, 'b-', linewidth=2, label='After Low-pass')
    ax.set_xlabel('Eigenvalue λ')
    ax.set_ylabel('Signal Amplitude')
    ax.set_title('Signal Filtering Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Explanation
    ax = axes[1, 1]
    ax.text(0.05, 0.9, "Why High-Pass Works Better:", transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')
    
    explanation = """
1. User-Item Specificity:
   • Low frequencies = global popularity patterns
   • High frequencies = specific user preferences
   • LearnSpec preserves personalization
   
2. Noise vs Signal:
   • Traditional: assumes high-freq = noise
   • Reality: high-freq = valuable user uniqueness
   • Global patterns often less informative
   
3. Learned Adaptation:
   • Initial high-pass is just starting point
   • Learning adjusts to find optimal response
   • Easier to suppress than amplify
"""
    
    ax.text(0.05, 0.85, explanation, transform=ax.transAxes, 
            fontsize=10, va='top', ha='left', family='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save to results directory outside src
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, 'results')
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    save_path = os.path.join(viz_dir, 'highpass_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== High-Pass Analysis ===")
    print(f"LearnSpec preserves {np.sum(learnspec_response[800:]) / np.sum(learnspec_response) * 100:.1f}% of high frequencies")
    print(f"Traditional preserves {np.sum(lowpass_response[800:]) / np.sum(lowpass_response) * 100:.1f}% of high frequencies")
    print(f"\n💾 Saved to: {os.path.relpath(save_path)}")


def list_saved_params():
    """List all saved parameter files"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, 'results')
    params_dir = os.path.join(results_dir, 'filter_params')
    
    if not os.path.exists(params_dir):
        print("No filter_params directory found")
        return []
    
    files = []
    # Look in main directory and subdirectories
    for root, dirs, filenames in os.walk(params_dir):
        for filename in filenames:
            if filename.endswith('.pkl'):
                files.append(os.path.join(root, filename))
    
    if not files:
        print("No saved parameter files found")
        return []
    
    print("\nAvailable parameter files:")
    for i, f in enumerate(sorted(files)):
        # Show relative path from params_dir
        rel_path = os.path.relpath(f, params_dir)
        print(f"{i+1}. {rel_path}")
    
    return sorted(files)


def main():
    """Main entry point with menu"""
    
    print("🎨 LearnSpec Filter Visualization Tool\n")
    print("1. Visualize initial filter response")
    print("2. Visualize filter evolution (from saved parameters)")
    print("3. Analyze eigenvalue distributions across datasets")
    print("4. Analyze high-pass phenomenon")
    print("5. All visualizations")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        visualize_initial_filters()
        
    elif choice == '2':
        files = list_saved_params()
        if not files:
            print("\nPlease run main.py first to generate parameter files")
            return
        
        if len(files) == 1:
            params_file = files[0]
            print(f"\nUsing: {os.path.basename(params_file)}")
        else:
            file_choice = input("\nSelect file number (or press Enter for latest): ").strip()
            if file_choice == "":
                params_file = files[-1]
            else:
                try:
                    idx = int(file_choice) - 1
                    params_file = files[idx]
                except (ValueError, IndexError):
                    print("Invalid choice")
                    return
        
        visualize_evolution(params_file)
        
    elif choice == '3':
        analyze_eigenvalue_distributions()
        
    elif choice == '4':
        analyze_highpass_phenomenon()
        
    elif choice == '5':
        print("\n🎨 Running all visualizations...")
        
        # Initial filters
        print("\n1/4: Initial filter response...")
        visualize_initial_filters()
        
        # Eigenvalue distributions
        print("\n2/4: Eigenvalue distributions...")
        analyze_eigenvalue_distributions()
        
        # High-pass analysis
        print("\n3/4: High-pass phenomenon analysis...")
        analyze_highpass_phenomenon()
        
        # Evolution (if params exist)
        print("\n4/4: Filter evolution...")
        files = list_saved_params()
        if files:
            params_file = files[-1]  # Use latest
            print(f"Using latest: {os.path.basename(params_file)}")
            visualize_evolution(params_file)
        else:
            print("No saved parameters found for evolution visualization")
        
        print("\n✅ All visualizations complete!")
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    # If command line argument provided, visualize that file directly
    if len(sys.argv) > 1:
        params_file = sys.argv[1]
        if os.path.exists(params_file):
            visualize_evolution(params_file)
        else:
            print(f"File not found: {params_file}")
    else:
        # Interactive menu
        main()