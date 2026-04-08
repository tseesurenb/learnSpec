#!/usr/bin/env python3
"""
Mixed Dataset Visualization Tool for LearnSpec
Creates 4x2 comparison showing user view analysis for two different datasets side by side
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
from scipy.sparse.linalg import eigsh
from dataloader import Dataset
from filter import APSFilter, evaluate_polynomial_basis, normalize_eigenvalues_for_basis

# Set Times New Roman font for paper quality
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13


class MixedDatasetVisualizer:
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        # Initialize data loaders for both datasets
        self.data_generator1 = Dataset(path="../data/" + dataset1)
        self.data_generator2 = Dataset(path="../data/" + dataset2)
        
        print(f"Dataset 1: {dataset1}")
        print(f"Users: {self.data_generator1.n_users}, Items: {self.data_generator1.m_items}")
        print(f"Interactions: {self.data_generator1.UserItemNet.nnz}")
        
        print(f"\nDataset 2: {dataset2}")
        print(f"Users: {self.data_generator2.n_users}, Items: {self.data_generator2.m_items}")
        print(f"Interactions: {self.data_generator2.UserItemNet.nnz}")
    
    def get_eigenvalues(self, dataset, degNorm=0.5):
        """Load cached eigenvalues for user similarity matrix of given dataset"""
        cache_dir = os.path.join('..', 'cache', dataset)
        
        # Look for appropriate eigenvalue file
        eigen_file = None
        degNorm_str = f"{degNorm}".replace('.', 'p')
        
        # Try to find matching file for user view
        candidates = [
            f'full_{dataset}_user_largestEigen_n2000_degNorm_{degNorm_str}.pkl',
            f'full_{dataset}_user_largestEigen_n3000_degNorm_{degNorm_str}.pkl',
            f'partial_{dataset}_user_largestEigen_n2000_degNorm_{degNorm_str}_seed_42_ratio_70.pkl'
        ]
        
        # Find first existing file
        for candidate in candidates:
            full_path = os.path.join(cache_dir, candidate)
            if os.path.exists(full_path):
                eigen_file = full_path
                break
        
        # If not found with exact degNorm, try common values
        if eigen_file is None:
            common_degNorms = ['0p5', '0p45', '0p4', '0p3']
            for deg in common_degNorms:
                test_file = f'full_{dataset}_user_largestEigen_n2000_degNorm_{deg}.pkl'
                full_path = os.path.join(cache_dir, test_file)
                if os.path.exists(full_path):
                    eigen_file = full_path
                    print(f"Using cached eigenvalues with degNorm={deg.replace('p', '.')} for {dataset}")
                    break
        
        if eigen_file is None:
            raise FileNotFoundError(f"No cached eigenvalues found for {dataset} user similarity")
        
        # Load eigenvalues
        print(f"Loading cached eigenvalues from: {os.path.basename(eigen_file)}")
        with open(eigen_file, 'rb') as f:
            data = pickle.load(f)
        
        if 'eigenvals' in data:
            eigenvalues = data['eigenvals']
        else:
            raise ValueError("No eigenvalues found in cached file")
        
        return eigenvalues, os.path.basename(eigen_file)
    
    def load_filter_params(self, dataset):
        """Load latest filter parameters for the dataset"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        params_dir = os.path.join(project_root, 'results', 'filter_params')
        
        if not os.path.exists(params_dir):
            return None
        
        # Find latest file for this dataset
        files = []
        for filename in os.listdir(params_dir):
            if filename.startswith(dataset) and filename.endswith('.pkl'):
                files.append(os.path.join(params_dir, filename))
        
        if not files:
            return None
        
        # Get latest file
        latest_file = max(files, key=os.path.getmtime)
        print(f"Loading filter params from: {os.path.basename(latest_file)} for {dataset}")
        
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    
    def create_mixed_visualization(self):
        """Create mixed dataset comparison visualization"""
        
        # Load filter parameters for both datasets
        filter_data1 = self.load_filter_params(self.dataset1)
        filter_data2 = self.load_filter_params(self.dataset2)
        
        # Set up figure layout: 4x2 layout
        fig = plt.figure(figsize=(12, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.12)
        
        # Load eigenvalues for both datasets
        eigenvals1, eigen_file1 = self.get_eigenvalues(self.dataset1, degNorm=0.5)
        eigenvals2, eigen_file2 = self.get_eigenvalues(self.dataset2, degNorm=0.5)
        
        max_eigen1 = eigenvals1.max()
        max_eigen2 = eigenvals2.max()
        
        # Column 1: Dataset 1 (yelp2018)
        self.plot_spectral_distribution(fig.add_subplot(gs[0, 0]), eigenvals1, self.dataset1.upper(), eigen_file1)
        self.plot_initial_filter(fig.add_subplot(gs[1, 0]), self.dataset1.upper(), max_eigen1)
        self.plot_learned_filter(fig.add_subplot(gs[2, 0]), self.dataset1.upper(), filter_data1, max_eigen1)
        self.plot_filtered_distribution(fig.add_subplot(gs[3, 0]), eigenvals1, self.dataset1.upper(), filter_data1)
        
        # Column 2: Dataset 2 (gowalla)
        self.plot_spectral_distribution(fig.add_subplot(gs[0, 1]), eigenvals2, self.dataset2.upper(), eigen_file2)
        self.plot_initial_filter(fig.add_subplot(gs[1, 1]), self.dataset2.upper(), max_eigen2)
        self.plot_learned_filter(fig.add_subplot(gs[2, 1]), self.dataset2.upper(), filter_data2, max_eigen2)
        self.plot_filtered_distribution(fig.add_subplot(gs[3, 1]), eigenvals2, self.dataset2.upper(), filter_data2)
        
        # Save visualization
        project_root = os.path.dirname(os.path.dirname(__file__))
        viz_dir = os.path.join(project_root, 'results', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        save_path = os.path.join(viz_dir, f'{self.dataset1}_vs_{self.dataset2}_user_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Mixed visualization saved to: {os.path.relpath(save_path)}")
        return save_path
    
    def plot_spectral_distribution(self, ax, eigenvals, dataset_name, eigen_filename):
        """Plot eigenvalue distribution with normalized x-axis"""
        
        min_eigen = eigenvals.min()
        max_eigen = eigenvals.max()
        
        # Normalize eigenvalues to [0,1] for plotting
        normalized_eigenvals = (eigenvals - min_eigen) / (max_eigen - min_eigen)
        
        # Histogram on normalized scale
        ax.hist(normalized_eigenvals, bins=50, alpha=0.7, color='steelblue', 
                density=True, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for frequency regions (on normalized scale)
        n_vals = len(eigenvals)
        sorted_eigenvals = np.sort(eigenvals)
        
        # 30-70 split for frequency bands
        low_cutoff_abs = sorted_eigenvals[int(0.3 * n_vals)]
        high_cutoff_abs = sorted_eigenvals[int(0.7 * n_vals)]
        
        # Convert to normalized scale
        low_cutoff_norm = (low_cutoff_abs - min_eigen) / (max_eigen - min_eigen)
        high_cutoff_norm = (high_cutoff_abs - min_eigen) / (max_eigen - min_eigen)
        
        # Minimal frequency region markers
        ax.axvline(low_cutoff_norm, color='green', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(high_cutoff_norm, color='red', linestyle='--', linewidth=1, alpha=0.6)
        
        # Custom x-axis with both normalized and absolute values
        ax.set_xlim([0, 1])
        
        # Set custom tick labels showing norm (abs) - single line format
        tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        tick_labels = []
        for pos in tick_positions:
            abs_val = pos * max_eigen + (1-pos) * min_eigen
            tick_labels.append(f'{pos:.1f}({abs_val:.1f})')
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        
        ax.set_xlabel('Norm. Eigenvalue (Abs.)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        
        ax.set_title(f'{dataset_name} User Eigenvalue Dist.', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_initial_filter(self, ax, dataset_name, max_eigenval):
        """Plot initial filter response"""
        
        # Create filter
        filter_order = 8
        aps_filter = APSFilter(
            filter_order=filter_order,
            init_filter_name='uniform',
            fix_filter_weights=False,
            poly_basis='bernstein'
        )
        
        # Evaluate on frequency range
        freqs = torch.linspace(0, 1, 1000)
        with torch.no_grad():
            response = aps_filter(freqs).numpy()
        
        # Main combined plot only - cleaner visualization
        ax.plot(freqs.numpy(), response, 'k-', linewidth=3, label='Initial Filter')
        
        # Frequency region indicators with labels
        ax.axvspan(0, 0.3, alpha=0.15, color='lightblue', label='Low-freq')
        ax.axvspan(0.3, 0.7, alpha=0.15, color='lightgreen', label='Mid-freq')
        ax.axvspan(0.7, 1.0, alpha=0.15, color='lightcoral', label='High-freq')
        
        ax.set_xlabel('Norm. Frequency', fontsize=14)
        ax.set_ylabel('Filter Response', fontsize=14)
        ax.set_title(f'{dataset_name} User Initial Filter', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    def plot_learned_filter(self, ax, dataset_name, filter_data, max_eigenval):
        """Plot learned filter response"""
        
        if filter_data is None or 'best' not in filter_data:
            # No learned filter available
            ax.text(0.5, 0.5, 'No learned filter data available\nRun training first', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax.set_title(f'{dataset_name} User Learned Filter', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            return
        
        # Get user filter state
        if filter_data['best'].get('user_filter'):
            state_dict = filter_data['best']['user_filter']
        else:
            ax.text(0.5, 0.5, f'No user filter in saved data', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{dataset_name} User Learned Filter', fontsize=16, fontweight='bold')
            return
        
        # Reconstruct filter
        freqs = torch.linspace(0, 1, 1000)
        
        # Get mixing weights
        mixing_weights_raw = state_dict['filter.mixing_weights']
        mixing_weights = torch.softmax(mixing_weights_raw, dim=0).numpy()
        
        # Get filter info
        filter_names = filter_data['initial']['filter_names']
        poly_basis = filter_data['initial']['config'].get('poly', 'bernstein')
        
        # Compute combined response only - cleaner visualization
        combined_response = np.zeros(1000)
        
        for i, fname in enumerate(filter_names):
            if f'filter.filter_{i}' in state_dict:
                coeffs = state_dict[f'filter.filter_{i}']
                x_basis = normalize_eigenvalues_for_basis(freqs, poly_basis)
                response = evaluate_polynomial_basis(coeffs, x_basis, poly_basis)
                response = torch.exp(-torch.abs(response).clamp(max=10.0)) + 1e-6
                response_np = response.detach().numpy()
                
                combined_response += mixing_weights[i] * response_np
        
        # Combined response only
        ax.plot(freqs.numpy(), combined_response, 'k-', linewidth=3, 
                label='Learned Filter', zorder=10)
        
        # Frequency regions
        ax.axvspan(0, 0.3, alpha=0.15, color='lightblue', label='Low-freq')
        ax.axvspan(0.3, 0.7, alpha=0.15, color='lightgreen', label='Mid-freq')  
        ax.axvspan(0.7, 1.0, alpha=0.15, color='lightcoral', label='High-freq')
        
        ax.set_xlabel('Norm. Frequency', fontsize=14)
        ax.set_ylabel('Filter Response', fontsize=14)
        
        # Add filter file info to title if available
        filter_file = filter_data.get('filename', '') if filter_data else ''
        if filter_file:
            # Extract short info from filename
            base_name = os.path.basename(filter_file)
            parts = base_name.split('_')
            poly_type = next((p for p in parts if p in ['bernstein', 'cheby', 'ensemble']), '')
            title_extra = f'\n({poly_type})' if poly_type else ''
        else:
            title_extra = ''
        
        ax.set_title(f'{dataset_name} User Learned Filter{title_extra}', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    def plot_filtered_distribution(self, ax, eigenvals, dataset_name, filter_data):
        """Plot eigenvalue distribution after applying learned filter"""
        
        min_eigen = eigenvals.min()
        max_eigen = eigenvals.max()
        
        # Normalize eigenvalues to [0,1]
        normalized_eigenvals = (eigenvals - min_eigen) / (max_eigen - min_eigen)
        
        if filter_data and 'best' in filter_data:
            # Apply learned filter
            filtered_eigenvals = self.apply_filter_to_eigenvals(eigenvals, filter_data)
            if filtered_eigenvals is not None:
                # Only keep eigenvalues that survive filtering (above threshold)
                filter_threshold = 0.1 * filtered_eigenvals.max()
                surviving_eigenvals = filtered_eigenvals[filtered_eigenvals > filter_threshold]
                
                if len(surviving_eigenvals) > 0:
                    # Normalize surviving eigenvalues for plotting
                    normalized_surviving = (surviving_eigenvals - min_eigen) / (max_eigen - min_eigen)
                    
                    # Calculate key stats for legend
                    filtered_energy = np.sum(filtered_eigenvals)
                    original_energy = np.sum(eigenvals)
                    energy_ratio = filtered_energy / original_energy
                    survival_rate = len(surviving_eigenvals) / len(eigenvals)
                    
                    # Plot with compact legend containing key stats
                    legend_text = f'Survived: {len(surviving_eigenvals)}/{len(eigenvals)} ({survival_rate:.1%}), Energy: {energy_ratio:.1%}'
                    ax.hist(normalized_surviving, bins=30, alpha=0.8, color='darkred', 
                           density=True, label=legend_text, edgecolor='black', linewidth=0.5)
                else:
                    # No eigenvalues survive - show message
                    ax.text(0.5, 0.5, 'All eigenvalues suppressed by filter', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                
                # Add frequency region markers
                n_vals = len(eigenvals)
                sorted_eigenvals = np.sort(eigenvals)
                low_cutoff_abs = sorted_eigenvals[int(0.3 * n_vals)]
                high_cutoff_abs = sorted_eigenvals[int(0.7 * n_vals)]
                low_cutoff_norm = (low_cutoff_abs - min_eigen) / (max_eigen - min_eigen)
                high_cutoff_norm = (high_cutoff_abs - min_eigen) / (max_eigen - min_eigen)
                
                # Minimal frequency markers
                ax.axvline(low_cutoff_norm, color='green', linestyle='--', linewidth=1, alpha=0.4)
                ax.axvline(high_cutoff_norm, color='red', linestyle='--', linewidth=1, alpha=0.4)
            else:
                # No filter available - show original only
                ax.hist(normalized_eigenvals, bins=50, alpha=0.7, color='gray', 
                       density=True, label='Original (No filter)', edgecolor='black', linewidth=0.5)
        else:
            # No filter data
            ax.hist(normalized_eigenvals, bins=50, alpha=0.7, color='gray', 
                   density=True, label='Original (No filter)', edgecolor='black', linewidth=0.5)
        
        # Custom x-axis with both normalized and absolute values - single line
        ax.set_xlim([0, 1])
        tick_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        tick_labels = []
        for pos in tick_positions:
            abs_val = pos * max_eigen + (1-pos) * min_eigen
            tick_labels.append(f'{pos:.1f}({abs_val:.1f})')
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        
        ax.set_xlabel('Norm. Eigenvalue (Abs.)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'{dataset_name} User Filtered Eigenval. Dist.', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def apply_filter_to_eigenvals(self, eigenvals, filter_data):
        """Apply learned filter to eigenvalues"""
        
        if not filter_data or 'best' not in filter_data:
            return None
        
        # Get user filter state
        if filter_data['best'].get('user_filter'):
            state_dict = filter_data['best']['user_filter']
        else:
            return None
        
        try:
            # Normalize eigenvalues to [0,1] for filter application
            min_eigen = eigenvals.min()
            max_eigen = eigenvals.max()
            normalized_eigenvals = (eigenvals - min_eigen) / (max_eigen - min_eigen)
            
            # Convert to tensor
            eigenvals_tensor = torch.from_numpy(normalized_eigenvals).float()
            
            # Get filter parameters
            mixing_weights_raw = state_dict['filter.mixing_weights']
            mixing_weights = torch.softmax(mixing_weights_raw, dim=0).numpy()
            
            filter_names = filter_data['initial']['filter_names']
            poly_basis = filter_data['initial']['config'].get('poly', 'bernstein')
            
            # Compute filter response
            combined_response = np.zeros_like(normalized_eigenvals)
            
            for i, fname in enumerate(filter_names):
                if f'filter.filter_{i}' in state_dict:
                    coeffs = state_dict[f'filter.filter_{i}']
                    x_basis = normalize_eigenvalues_for_basis(eigenvals_tensor, poly_basis)
                    response = evaluate_polynomial_basis(coeffs, x_basis, poly_basis)
                    response = torch.exp(-torch.abs(response).clamp(max=10.0)) + 1e-6
                    response_np = response.detach().numpy()
                    
                    combined_response += mixing_weights[i] * response_np
            
            # Apply filter: multiply eigenvalues by filter response
            filtered_eigenvals = eigenvals * combined_response
            
            return filtered_eigenvals
            
        except Exception as e:
            print(f"Error applying filter: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Mixed dataset spectral comparison visualization')
    parser.add_argument('--dataset1', type=str, default='yelp2018',
                        choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                        help='First dataset name (left column)')
    parser.add_argument('--dataset2', type=str, default='gowalla',
                        choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                        help='Second dataset name (right column)')
    
    args = parser.parse_args()
    
    # Create visualizer and generate plots
    visualizer = MixedDatasetVisualizer(args.dataset1, args.dataset2)
    visualizer.create_mixed_visualization()


if __name__ == "__main__":
    main()