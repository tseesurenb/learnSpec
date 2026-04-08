#!/usr/bin/env python3
"""
Comprehensive Visualization Tool for LearnSpec
Creates 6-panel analysis showing spectral distribution, initial filters, and learned filters
for both user and item views
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


class SpectralVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_generator = Dataset(path="../data/" + dataset)
        self.n_users = self.data_generator.n_users
        self.n_items = self.data_generator.m_items
        self.train_mat = self.data_generator.UserItemNet
        
        print(f"Dataset: {dataset}")
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        print(f"Interactions: {self.train_mat.nnz}")
    
    def get_eigenvalues(self, sim_type='u', degNorm=0.5):
        """Load cached eigenvalues for user or item similarity matrix"""
        cache_dir = os.path.join('..', 'cache', self.dataset)
        
        # Look for appropriate eigenvalue file
        # Prefer full over partial, and match degree normalization
        eigen_file = None
        degNorm_str = f"{degNorm}".replace('.', 'p')
        
        # Try to find matching file
        if sim_type == 'u':
            # For user, look for files with reasonable n (2000-3000)
            candidates = [
                f'full_{self.dataset}_user_largestEigen_n2000_degNorm_{degNorm_str}.pkl',
                f'full_{self.dataset}_user_largestEigen_n3000_degNorm_{degNorm_str}.pkl',
                f'partial_{self.dataset}_user_largestEigen_n2000_degNorm_{degNorm_str}_seed_42_ratio_70.pkl'
            ]
        else:
            # For item, look for files with reasonable n (1000-2000)
            candidates = [
                f'full_{self.dataset}_item_largestEigen_n2000_degNorm_{degNorm_str}.pkl',
                f'full_{self.dataset}_item_largestEigen_n1000_degNorm_{degNorm_str}.pkl',
                f'partial_{self.dataset}_item_largestEigen_n2000_degNorm_{degNorm_str}_seed_42_ratio_70.pkl'
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
                if sim_type == 'u':
                    test_file = f'full_{self.dataset}_user_largestEigen_n2000_degNorm_{deg}.pkl'
                else:
                    test_file = f'full_{self.dataset}_item_largestEigen_n2000_degNorm_{deg}.pkl'
                
                full_path = os.path.join(cache_dir, test_file)
                if os.path.exists(full_path):
                    eigen_file = full_path
                    print(f"Using cached eigenvalues with degNorm={deg.replace('p', '.')}")
                    break
        
        if eigen_file is None:
            raise FileNotFoundError(f"No cached eigenvalues found for {sim_type} similarity")
        
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
        print(f"Loading filter params from: {os.path.basename(latest_file)}")
        
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    
    def create_visualization(self, views='ui'):
        """Create spectral visualization for specified views"""
        
        # Determine which views to show
        show_user = 'u' in views
        show_item = 'i' in views
        
        # Load filter parameters
        filter_data = self.load_filter_params(self.dataset)
        
        # Set up figure layout based on views
        if show_user and show_item:
            # Both views: 4x2 layout - optimized for half-page display
            fig = plt.figure(figsize=(12, 14))
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.12)
            n_cols = 2
        else:
            # Single view: 4x1 layout
            fig = plt.figure(figsize=(6, 14))
            gs = fig.add_gridspec(4, 1, hspace=0.4, wspace=0.12)
            n_cols = 1
        
        col_idx = 0
        
        # User analysis
        if show_user:
            user_eigenvals, user_eigen_file = self.get_eigenvalues('u', degNorm=0.5)
            user_max = user_eigenvals.max()
            
            self.plot_spectral_distribution(fig.add_subplot(gs[0, col_idx]), user_eigenvals, 'User', user_eigen_file)
            self.plot_initial_filter(fig.add_subplot(gs[1, col_idx]), 'User', user_max)
            self.plot_learned_filter(fig.add_subplot(gs[2, col_idx]), 'User', filter_data, user_max)
            self.plot_filtered_distribution(fig.add_subplot(gs[3, col_idx]), user_eigenvals, 'User', filter_data)
            
            if n_cols == 2:
                col_idx += 1
        
        # Item analysis
        if show_item:
            item_eigenvals, item_eigen_file = self.get_eigenvalues('i', degNorm=0.5)
            item_max = item_eigenvals.max()
            
            self.plot_spectral_distribution(fig.add_subplot(gs[0, col_idx]), item_eigenvals, 'Item', item_eigen_file)
            self.plot_initial_filter(fig.add_subplot(gs[1, col_idx]), 'Item', item_max)
            self.plot_learned_filter(fig.add_subplot(gs[2, col_idx]), 'Item', filter_data, item_max)
            self.plot_filtered_distribution(fig.add_subplot(gs[3, col_idx]), item_eigenvals, 'Item', filter_data)
        
        # No main title for paper
        
        # Save with view suffix
        project_root = os.path.dirname(os.path.dirname(__file__))
        viz_dir = os.path.join(project_root, 'results', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        view_suffix = views if views != 'ui' else 'both'
        save_path = os.path.join(viz_dir, f'{self.dataset}_spectral_analysis_{view_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visualization saved to: {os.path.relpath(save_path)}")
        return save_path
    
    def plot_spectral_distribution(self, ax, eigenvals, view_type, eigen_filename):
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
        
        ax.set_title(f'{view_type} Eigenvalue Dist.', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_initial_filter(self, ax, view_type, max_eigenval):
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
        
        # Minimal annotations removed for clean paper version
        
        ax.set_xlabel('Norm. Frequency', fontsize=14)
        ax.set_ylabel('Filter Response', fontsize=14)
        ax.set_title(f'{view_type} Initial Filter', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    def plot_learned_filter(self, ax, view_type, filter_data, max_eigenval):
        """Plot learned filter response"""
        
        if filter_data is None or 'best' not in filter_data:
            # No learned filter available
            ax.text(0.5, 0.5, 'No learned filter data available\nRun training first', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax.set_title(f'{view_type} Learned Filter', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            return
        
        # Get appropriate filter state
        if view_type.lower() == 'user' and filter_data['best'].get('user_filter'):
            state_dict = filter_data['best']['user_filter']
        elif view_type.lower() == 'item' and filter_data['best'].get('item_filter'):
            state_dict = filter_data['best']['item_filter']
        else:
            ax.text(0.5, 0.5, f'No {view_type.lower()} filter in saved data', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{view_type} Learned Filter', fontsize=14, fontweight='bold')
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
        # Frequency region indicators with labels
        ax.axvspan(0, 0.3, alpha=0.15, color='lightblue', label='Low-freq')
        ax.axvspan(0.3, 0.7, alpha=0.15, color='lightgreen', label='Mid-freq')  
        ax.axvspan(0.7, 1.0, alpha=0.15, color='lightcoral', label='High-freq')
        
        # No automatic plateau detection - let the visualization speak for itself
        
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
        
        ax.set_title(f'{view_type} Learned Filter{title_extra}', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    def plot_filtered_distribution(self, ax, eigenvals, view_type, filter_data):
        """Plot eigenvalue distribution after applying learned filter"""
        
        min_eigen = eigenvals.min()
        max_eigen = eigenvals.max()
        
        # Normalize eigenvalues to [0,1]
        normalized_eigenvals = (eigenvals - min_eigen) / (max_eigen - min_eigen)
        
        if filter_data and 'best' in filter_data:
            # Apply learned filter
            filtered_eigenvals = self.apply_filter_to_eigenvals(eigenvals, view_type, filter_data)
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
        ax.set_title(f'{view_type} Filtered Eigenval. Dist.', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def apply_filter_to_eigenvals(self, eigenvals, view_type, filter_data):
        """Apply learned filter to eigenvalues"""
        
        if not filter_data or 'best' not in filter_data:
            return None
        
        # Get appropriate filter state
        if view_type.lower() == 'user' and filter_data['best'].get('user_filter'):
            state_dict = filter_data['best']['user_filter']
        elif view_type.lower() == 'item' and filter_data['best'].get('item_filter'):
            state_dict = filter_data['best']['item_filter']
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
    parser = argparse.ArgumentParser(description='Comprehensive spectral visualization')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book'],
                        help='Dataset name')
    parser.add_argument('--views', type=str, default='ui', choices=['u', 'i', 'ui'],
                        help='Which views to show: u (user only), i (item only), ui (both)')
    
    args = parser.parse_args()
    
    # Create visualizer and generate plots
    visualizer = SpectralVisualizer(args.dataset)
    visualizer.create_visualization(views=args.views)


if __name__ == "__main__":
    main()