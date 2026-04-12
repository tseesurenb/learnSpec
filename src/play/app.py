"""
LearnSpec Filter Viewer
=======================
Visualize initial vs learned spectral filter responses from training logs.

Launch: cd src && streamlit run play/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

st.set_page_config(page_title="LearnSpec Filter Viewer", layout="wide")

# =============================================================================
# Data loading
# =============================================================================

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))


@st.cache_data(ttl=30)
def get_available_runs():
    """Scan logs directory for available training runs."""
    runs = {}
    if not os.path.exists(LOGS_DIR):
        return runs
    for dataset in sorted(os.listdir(LOGS_DIR)):
        dataset_dir = os.path.join(LOGS_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        dataset_runs = []
        for run_dir in sorted(os.listdir(dataset_dir), reverse=True):
            run_path = os.path.join(dataset_dir, run_dir)
            config_path = os.path.join(run_path, 'config.json')
            summary_path = os.path.join(run_path, 'summary.json')
            epochs_path = os.path.join(run_path, 'epochs.json')
            if os.path.isdir(run_path) and os.path.exists(config_path) and os.path.exists(epochs_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    summary = {}
                    if os.path.exists(summary_path):
                        with open(summary_path) as f:
                            summary = json.load(f)
                    poly = config.get('poly', config.get('f_poly', 'bernstein'))
                    init = config.get('f_init', 'uniform')
                    act = config.get('f_act', 'sigmoid')
                    order = config.get('f_order', 32)
                    lr = config.get('lr', 0.001)
                    decay = config.get('decay', 0.0)
                    best_epoch = summary.get('best_epoch', '?')
                    baseline = summary.get('baseline_ndcg', 0)
                    final = summary.get('final_ndcg', 0)
                    improv = summary.get('ndcg_improvement_pct', 0)

                    # Extract timestamp from dir name (last part after last _)
                    parts = run_dir.rsplit('_', 2)
                    timestamp = parts[-2] + '_' + parts[-1] if len(parts) >= 3 else ''
                    date_str = timestamp[:8] if len(timestamp) >= 8 else ''
                    time_str = timestamp[9:] if len(timestamp) >= 15 else ''
                    ts_label = f"{date_str[4:6]}/{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}" if date_str and time_str else run_dir[-15:]

                    poly_label = poly if poly == 'direct' else f"{poly}(K={order})"
                    label = f"[{ts_label}] {poly_label} | {init} | {baseline:.4f}→{final:.4f} ({improv:+.1f}%)"

                    dataset_runs.append({
                        'path': run_path,
                        'label': label,
                        'config': config,
                        'summary': summary,
                        'dir_name': run_dir,
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        if dataset_runs:
            runs[dataset] = dataset_runs
    return runs


@st.cache_data
def load_epochs(run_path):
    """Load epoch snapshots from a training run."""
    epochs_path = os.path.join(run_path, 'epochs.json')
    with open(epochs_path) as f:
        return json.load(f)


def get_filter_response(epoch_data, view='user'):
    """Extract filter response from epoch data."""
    key = f'{view}_filter'
    if key not in epoch_data or epoch_data[key] is None:
        return None
    filt = epoch_data[key]
    if 'response' in filt:
        return np.array(filt['response'])
    return None


def get_coefficients(epoch_data, view='user'):
    """Extract polynomial coefficients from epoch data."""
    key = f'{view}_filter'
    if key not in epoch_data or epoch_data[key] is None:
        return None
    filt = epoch_data[key]
    if 'coefficients' in filt and isinstance(filt['coefficients'], dict):
        if 'coeffs' in filt['coefficients']:
            return np.array(filt['coefficients']['coeffs'])
    return None


# =============================================================================
# Visualization functions
# =============================================================================

def plot_filter_comparison(initial_response, learned_response, best_response,
                           title="Filter Response", view_label="", best_epoch=None):
    """Plot initial vs learned filter response."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    n = len(initial_response)
    x = np.linspace(0, 1, n)

    # Plot 1: Initial
    ax = axes[0]
    ax.fill_between(x, 0, initial_response, alpha=0.3, color='steelblue')
    ax.plot(x, initial_response, color='steelblue', linewidth=2)
    ax.set_title(f'{view_label} Initial (before training)', fontsize=11)
    ax.set_xlabel('Normalized eigenvalue')
    ax.set_ylabel('h(λ)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # Plot 2: Best epoch
    ax = axes[1]
    ax.fill_between(x, 0, best_response, alpha=0.3, color='red')
    ax.plot(x, best_response, color='red', linewidth=2)
    epoch_label = f'Best (epoch {best_epoch})' if best_epoch else 'Learned'
    ax.set_title(f'{view_label} {epoch_label}', fontsize=11)
    ax.set_xlabel('Normalized eigenvalue')
    ax.set_ylabel('h(λ)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2)

    # Plot 3: Overlay + difference
    ax = axes[2]
    ax.plot(x, initial_response, color='steelblue', linewidth=2, label='Initial', alpha=0.7)
    ax.plot(x, best_response, color='red', linewidth=2, label=epoch_label, alpha=0.7)
    diff = best_response - initial_response
    ax.fill_between(x, initial_response, best_response,
                     where=diff > 0, color='green', alpha=0.15, label='Increased')
    ax.fill_between(x, initial_response, best_response,
                     where=diff < 0, color='red', alpha=0.15, label='Decreased')
    ax.set_title(f'{view_label} Overlay', fontsize=11)
    ax.set_xlabel('Normalized eigenvalue')
    ax.set_ylabel('h(λ)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


def plot_training_trajectory(epochs_data, best_epoch=None):
    """Plot validation NDCG over training epochs."""
    epoch_nums = [e['epoch'] for e in epochs_data]
    val_ndcgs = [e.get('val_ndcg', 0) for e in epochs_data]
    losses = [e.get('loss', 0) for e in epochs_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(epoch_nums, val_ndcgs, color='steelblue', linewidth=1.5)
    if best_epoch is not None:
        ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (ep {best_epoch})')
        ax1.legend(fontsize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation NDCG@20')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.2)

    ax2.plot(epoch_nums, losses, color='orange', linewidth=1.5)
    if best_epoch is not None:
        ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


def plot_filter_evolution(epochs_data, view='user', n_snapshots=8):
    """Plot filter response evolution over training."""
    responses = []
    epoch_nums = []
    for e in epochs_data:
        resp = get_filter_response(e, view)
        if resp is not None:
            responses.append(resp)
            epoch_nums.append(e['epoch'])

    if len(responses) < 2:
        return None

    # Sample evenly
    indices = np.linspace(0, len(responses) - 1, min(n_snapshots, len(responses)), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.cm.viridis
    for i, idx in enumerate(indices):
        color = cmap(i / max(len(indices) - 1, 1))
        n = len(responses[idx])
        x = np.linspace(0, 1, n)
        alpha = 0.4 if idx != indices[-1] else 1.0
        lw = 1 if idx != indices[-1] else 2.5
        ax.plot(x, responses[idx], color=color, linewidth=lw, alpha=alpha,
                label=f'Epoch {epoch_nums[idx]}')

    ax.set_xlabel('Normalized eigenvalue')
    ax.set_ylabel('h(λ)')
    ax.set_title(f'{view.capitalize()} Filter Evolution')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_coefficient_comparison(initial_coeffs, best_coeffs, view_label=""):
    """Plot initial vs learned polynomial coefficients."""
    if initial_coeffs is None or best_coeffs is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 3))
    n = len(initial_coeffs)
    x = np.arange(n)
    width = 0.35

    ax.bar(x - width/2, initial_coeffs, width, color='steelblue', alpha=0.7, label='Initial')
    ax.bar(x + width/2, best_coeffs, width, color='red', alpha=0.7, label='Learned')
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Value')
    ax.set_title(f'{view_label} Polynomial Coefficients')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    return fig


# =============================================================================
# Main app
# =============================================================================

st.title("LearnSpec Filter Viewer")
st.caption("Visualize initial vs learned spectral filter responses from training runs")

runs = get_available_runs()

if not runs:
    st.error(f"No training logs found in `{LOGS_DIR}`. Run training with `--log` flag first.")
    st.stop()

# Dataset selection
dataset = st.sidebar.selectbox("Dataset", list(runs.keys()))
dataset_runs = runs[dataset]

# Run selection
run_labels = [r['label'] for r in dataset_runs]
selected_idx = st.sidebar.selectbox("Training Run", range(len(run_labels)),
                                     format_func=lambda i: run_labels[i])
selected_run = dataset_runs[selected_idx]

# Load data
config = selected_run['config']
summary = selected_run['summary']
epochs_data = load_epochs(selected_run['path'])

if not epochs_data:
    st.error("No epoch data found in this run.")
    st.stop()

# Display config
st.sidebar.markdown("---")
st.sidebar.markdown("**Configuration**")
poly = config.get('poly', config.get('f_poly', 'bernstein'))
st.sidebar.markdown(f"- Filter: `{poly}`")
if poly != 'direct':
    st.sidebar.markdown(f"- Order K: `{config.get('f_order', '?')}`")
st.sidebar.markdown(f"- Init: `{config.get('f_init', '?')}`")
st.sidebar.markdown(f"- Activation: `{config.get('f_act', 'sigmoid')}`")
st.sidebar.markdown(f"- u_eigen: `{config.get('u_eigen', '?')}`, i_eigen: `{config.get('i_eigen', '?')}`")
st.sidebar.markdown(f"- beta: `{config.get('beta', '?')}`")
st.sidebar.markdown(f"- lr: `{config.get('lr', '?')}`, decay: `{config.get('decay', '?')}`")
st.sidebar.markdown(f"- Optimizer: `{config.get('opt', '?')}`")

# Summary metrics
best_epoch = summary.get('best_epoch', len(epochs_data))
baseline_ndcg = summary.get('baseline_ndcg', 0)
final_ndcg = summary.get('final_ndcg', 0)
improv = summary.get('ndcg_improvement_pct', 0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline NDCG", f"{baseline_ndcg:.4f}")
col2.metric("Final NDCG", f"{final_ndcg:.4f}")
col3.metric("Improvement", f"{improv:+.1f}%")
col4.metric("Best Epoch", best_epoch)

# Epoch slider
st.markdown("---")
epoch_options = [e['epoch'] for e in epochs_data]
best_snap_idx = 0
for i, ep in enumerate(epoch_options):
    if ep == best_epoch:
        best_snap_idx = i
        break

selected_snap = st.slider("Epoch snapshot", 0, len(epochs_data) - 1, best_snap_idx,
                            help="Slide to see filter at different training stages")
current_epoch = epochs_data[selected_snap]
current_epoch_num = current_epoch['epoch']
is_best = (current_epoch_num == best_epoch)

st.caption(f"Showing epoch **{current_epoch_num}**{' (BEST)' if is_best else ''} | "
           f"val NDCG: {current_epoch.get('val_ndcg', 0):.4f} | "
           f"loss: {current_epoch.get('loss', 0):.4f}")

# =============================================================================
# Filter visualizations
# =============================================================================

initial_epoch = epochs_data[0]
best_epoch_data = epochs_data[best_snap_idx]

# User filter
st.subheader("User View Filter")
user_init = get_filter_response(initial_epoch, 'user')
user_current = get_filter_response(current_epoch, 'user')
user_best = get_filter_response(best_epoch_data, 'user')

if user_init is not None and user_best is not None:
    fig = plot_filter_comparison(user_init, user_current, user_best,
                                  view_label="User", best_epoch=best_epoch)
    st.pyplot(fig)
    plt.close()

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.caption(f"Initial: min={user_init.min():.3f}, max={user_init.max():.3f}, mean={user_init.mean():.3f}")
    c2.caption(f"Best: min={user_best.min():.3f}, max={user_best.max():.3f}, mean={user_best.mean():.3f}")
    diff = user_best - user_init
    c3.caption(f"Change: max_increase={diff.max():.3f}, max_decrease={diff.min():.3f}")
else:
    st.info("No user filter response data available in this run.")

# Item filter
st.subheader("Item View Filter")
item_init = get_filter_response(initial_epoch, 'item')
item_current = get_filter_response(current_epoch, 'item')
item_best = get_filter_response(best_epoch_data, 'item')

if item_init is not None and item_best is not None:
    fig = plot_filter_comparison(item_init, item_current, item_best,
                                  view_label="Item", best_epoch=best_epoch)
    st.pyplot(fig)
    plt.close()

    c1, c2, c3 = st.columns(3)
    c1.caption(f"Initial: min={item_init.min():.3f}, max={item_init.max():.3f}, mean={item_init.mean():.3f}")
    c2.caption(f"Best: min={item_best.min():.3f}, max={item_best.max():.3f}, mean={item_best.mean():.3f}")
    diff = item_best - item_init
    c3.caption(f"Change: max_increase={diff.max():.3f}, max_decrease={diff.min():.3f}")
else:
    st.info("No item filter response data available in this run.")

# =============================================================================
# Training trajectory
# =============================================================================

st.markdown("---")
st.subheader("Training Trajectory")

fig = plot_training_trajectory(epochs_data, best_epoch)
st.pyplot(fig)
plt.close()

# =============================================================================
# Filter evolution
# =============================================================================

st.markdown("---")
st.subheader("Filter Evolution Over Training")

col1, col2 = st.columns(2)

with col1:
    fig = plot_filter_evolution(epochs_data, 'user')
    if fig:
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Insufficient user filter snapshots.")

with col2:
    fig = plot_filter_evolution(epochs_data, 'item')
    if fig:
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Insufficient item filter snapshots.")

# =============================================================================
# Polynomial coefficients (if applicable)
# =============================================================================

if poly != 'direct':
    st.markdown("---")
    st.subheader("Polynomial Coefficients")

    col1, col2 = st.columns(2)

    with col1:
        init_coeffs = get_coefficients(initial_epoch, 'user')
        best_coeffs = get_coefficients(best_epoch_data, 'user')
        fig = plot_coefficient_comparison(init_coeffs, best_coeffs, "User")
        if fig:
            st.pyplot(fig)
            plt.close()

    with col2:
        init_coeffs = get_coefficients(initial_epoch, 'item')
        best_coeffs = get_coefficients(best_epoch_data, 'item')
        fig = plot_coefficient_comparison(init_coeffs, best_coeffs, "Item")
        if fig:
            st.pyplot(fig)
            plt.close()

# =============================================================================
# Fusion weights
# =============================================================================

st.markdown("---")
st.subheader("Fusion Weights")

fusion_data = []
for e in epochs_data:
    if 'fusion' in e and e['fusion'] is not None:
        f = e['fusion']
        if isinstance(f, dict) and 'user_weight' in f:
            fusion_data.append({
                'epoch': e['epoch'],
                'user': f['user_weight'],
                'item': f['item_weight'],
            })

if fusion_data:
    fig, ax = plt.subplots(figsize=(10, 3))
    epochs_f = [d['epoch'] for d in fusion_data]
    user_w = [d['user'] for d in fusion_data]
    item_w = [d['item'] for d in fusion_data]
    ax.plot(epochs_f, user_w, color='steelblue', linewidth=1.5, label='User weight')
    ax.plot(epochs_f, item_w, color='orange', linewidth=1.5, label='Item weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title('Fusion Weight Evolution')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    last = fusion_data[-1]
    st.caption(f"Final weights: User={last['user']:.3f}, Item={last['item']:.3f}")
else:
    st.info("No fusion weight data available.")

# =============================================================================
# Raw epoch data
# =============================================================================

with st.expander("Raw Epoch Data"):
    st.json(current_epoch)
