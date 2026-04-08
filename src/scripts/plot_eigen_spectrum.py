import os
import sys
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_eigenvals(filepath):
    if filepath.endswith('.npz'):
        return np.load(filepath)['eigenvals']
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)['eigenvals']


def find_eigen_file(cache_dir, dataset, view):
    view_name = 'user' if view == 'u' else 'item'
    for ext in ('npz', 'pkl'):
        pattern = os.path.join(cache_dir, dataset, f'full_{dataset}_{view_name}_largestEigen_n*_degNorm_*.{ext}')
        files = glob.glob(pattern)
        if files:
            return sorted(files, key=os.path.getsize)[0]
    return None


def main():
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cache'))
    datasets = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']

    # Load all available eigenvalues
    data = {}
    for ds in datasets:
        u_file = find_eigen_file(cache_dir, ds, 'u')
        i_file = find_eigen_file(cache_dir, ds, 'i')
        if u_file and i_file:
            data[ds] = {
                'user': load_eigenvals(u_file),
                'item': load_eigenvals(i_file),
            }
            print(f"{ds}: user={len(data[ds]['user'])}, item={len(data[ds]['item'])}")
        else:
            print(f"{ds}: skipped (files not found)")

    if not data:
        print("No eigen files found!")
        return

    colors = {'ml-100k': '#e74c3c', 'lastfm': '#3498db', 'gowalla': '#2ecc71',
              'yelp2018': '#9b59b6', 'amazon-book': '#e67e22'}

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Eigenvalue spectrum (log scale) — User view
    ax1 = fig.add_subplot(gs[0, 0])
    for ds, d in data.items():
        evals = np.sort(d['user'])[::-1]
        ax1.semilogy(range(len(evals)), evals, label=ds, color=colors.get(ds), alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue (log)')
    ax1.set_title('User Eigenvalue Spectrum')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Eigenvalue spectrum (log scale) — Item view
    ax2 = fig.add_subplot(gs[0, 1])
    for ds, d in data.items():
        evals = np.sort(d['item'])[::-1]
        ax2.semilogy(range(len(evals)), evals, label=ds, color=colors.get(ds), alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue (log)')
    ax2.set_title('Item Eigenvalue Spectrum')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Normalized eigenvalue spectrum (compare shapes)
    ax3 = fig.add_subplot(gs[0, 2])
    for ds, d in data.items():
        evals = np.sort(d['user'])[::-1]
        x = np.linspace(0, 1, len(evals))
        ax3.semilogy(x, evals / evals[0], label=ds, color=colors.get(ds), alpha=0.8, linewidth=1.5)
    ax3.set_xlabel('Normalized Index')
    ax3.set_ylabel('λ / λ_max')
    ax3.set_title('Normalized User Spectrum')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative energy — User
    ax4 = fig.add_subplot(gs[1, 0])
    for ds, d in data.items():
        evals = np.sort(d['user'])[::-1]
        energy = np.cumsum(evals ** 2) / np.sum(evals ** 2)
        ax4.plot(range(len(energy)), energy, label=ds, color=colors.get(ds), alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('Number of Eigenvalues')
    ax4.set_ylabel('Cumulative Energy')
    ax4.set_title('User Cumulative Energy')
    ax4.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax4.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Cumulative energy — Item
    ax5 = fig.add_subplot(gs[1, 1])
    for ds, d in data.items():
        evals = np.sort(d['item'])[::-1]
        energy = np.cumsum(evals ** 2) / np.sum(evals ** 2)
        ax5.plot(range(len(energy)), energy, label=ds, color=colors.get(ds), alpha=0.8, linewidth=1.5)
    ax5.set_xlabel('Number of Eigenvalues')
    ax5.set_ylabel('Cumulative Energy')
    ax5.set_title('Item Cumulative Energy')
    ax5.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax5.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Summary table: eigenvalues needed for 80/90/95% energy
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    headers = ['Dataset', 'View', '80%', '90%', '95%']
    rows = []
    for ds, d in data.items():
        for view_name, key in [('user', 'user'), ('item', 'item')]:
            evals = np.sort(d[key])[::-1]
            energy = np.cumsum(evals ** 2) / np.sum(evals ** 2)
            n80 = np.searchsorted(energy, 0.80) + 1
            n90 = np.searchsorted(energy, 0.90) + 1
            n95 = np.searchsorted(energy, 0.95) + 1
            rows.append([ds, view_name, str(n80), str(n90), str(n95)])

    table = ax6.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax6.set_title('Eigenvalues for X% Energy', fontsize=10, pad=20)

    plt.suptitle('Eigenvalue Spectrum Analysis Across Datasets', fontsize=14, fontweight='bold', y=0.98)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'eigen_spectrum_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
