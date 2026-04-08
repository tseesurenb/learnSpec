"""
LearnSpec Interactive Playground
================================
Explore eigendecomposition, spectral filters, and predictions visually.

Launch: cd src && streamlit run play/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.special import comb as scipy_comb
import sys
import os
import pickle

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import Dataset

# =============================================================================
# Data loading (cached)
# =============================================================================

@st.cache_data
def load_item_names(dataset_name):
    """Load item names for display. Returns dict: encoded_item_id -> name."""
    if dataset_name == 'ml-100k':
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'data', 'ml-100k')

        # Load original_id -> name from u.item
        id_to_name = {}
        item_file = os.path.join(data_dir, 'u.item')
        if os.path.exists(item_file):
            with open(item_file, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        id_to_name[int(parts[0])] = parts[1]

        # Load original_id -> encoded_id mapping
        mapping_file = os.path.join(data_dir, 'item_id_mapping.csv')
        encoded_to_name = {}
        if os.path.exists(mapping_file):
            import csv
            with open(mapping_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig = int(row['original_item_id'])
                    enc = int(row['encoded_item_id'])
                    encoded_to_name[enc] = id_to_name.get(orig, f"Item {orig}")
        else:
            # Fallback: assume 1-indexed offset
            for orig, name in id_to_name.items():
                encoded_to_name[orig - 1] = name

        return encoded_to_name
    return {}


@st.cache_data
def load_dataset(name):
    return Dataset(path=f"../data/{name}")


@st.cache_data
def compute_similarity_and_eigen(name, view, beta):
    """Load precomputed eigendecomposition from cache, fall back to computing if not found."""
    dataset = load_dataset(name)
    R = dataset.UserItemNet.toarray().astype(np.float64)

    # Try loading from cache first
    # Try multiple possible cache locations
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(src_dir)
    cache_dir = os.path.join(project_root, 'cache', name)
    if not os.path.exists(cache_dir):
        # Fallback: relative to cwd
        cache_dir = os.path.join(os.getcwd(), '..', 'cache', name)
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(os.getcwd(), 'cache', name)
    beta_str = str(beta).replace('.', 'p')
    view_name = 'user' if view == 'user' else 'item'

    eigenvals = None
    eigenvecs = None

    if os.path.exists(cache_dir):
        # Find matching eigen file (largest available)
        pattern = f"full_{name}_{view_name}_largestEigen_n"
        candidates = []
        for f in os.listdir(cache_dir):
            if f.startswith(pattern) and f"degNorm_{beta_str}.pkl" in f:
                try:
                    n = int(f[len(pattern):].split('_')[0])
                    candidates.append((n, f))
                except ValueError:
                    continue
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_n, best_file = candidates[0]
            filepath = os.path.join(cache_dir, best_file)
            with open(filepath, 'rb') as fh:
                data = pickle.load(fh)
            eigenvals = data['eigenvals']
            eigenvecs = data['eigenvecs']
            # Sort descending
            if len(eigenvals) > 1 and eigenvals[0] < eigenvals[-1]:
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
            eigenvals = np.maximum(eigenvals, 0)

    if eigenvals is None:
        # Fallback: compute (slow for large datasets)
        st.warning(f"No cached eigen files found for {name}/{view_name}. Computing from scratch...")
        from sklearn.utils.extmath import randomized_svd
        from scipy.sparse import diags, csr_matrix

        d_u = np.array(dataset.UserItemNet.sum(axis=1)).flatten().astype(np.float64)
        d_i = np.array(dataset.UserItemNet.sum(axis=0)).flatten().astype(np.float64)
        d_u_inv = np.where(d_u > 0, d_u ** (-beta), 0)
        d_i_inv = np.where(d_i > 0, d_i ** (-(1 - beta)), 0)
        A_n = diags(d_u_inv) @ dataset.UserItemNet.astype(np.float64) @ diags(d_i_inv)

        k = min(500, min(A_n.shape) - 1)
        U, sigma, Vt = randomized_svd(A_n, n_components=k, n_iter=10, random_state=42)
        eigenvals = sigma ** 2
        if view == 'user':
            eigenvecs = U
        else:
            eigenvecs = Vt.T

    # all_eigenvals = the full set we have (for energy analysis)
    all_eigenvals = eigenvals.copy()

    return all_eigenvals, eigenvals, eigenvecs, R, None


# =============================================================================
# Filter functions
# =============================================================================

def bernstein_basis(k, K, x):
    """Bernstein basis polynomial b_k^K(x)."""
    return scipy_comb(K, k, exact=True) * (x ** k) * ((1 - x) ** (K - k))


def evaluate_filter_real(eigenvals, filter_name, order, poly_basis='bernstein'):
    """Evaluate a filter initialization shape using current model logic."""
    import torch
    from filter import get_init_coefficients, evaluate_polynomial_basis, normalize_eigenvalues_for_basis

    # Map playground names to init types
    name_map = {'golden': 'bandpass', 'smooth': 'lowpass', 'butterworth': 'highpass',
                'gaussian': 'lowpass', 'chebyshev': 'highpass',
                'lowpass': 'lowpass', 'highpass': 'highpass', 'bandpass': 'bandpass', 'uniform': 'uniform'}
    init_type = name_map.get(filter_name, 'uniform')

    coeffs = get_init_coefficients(init_type, order)
    coeffs = torch.tensor(coeffs, dtype=torch.float32)
    x = torch.tensor(eigenvals, dtype=torch.float32)
    x_norm = normalize_eigenvalues_for_basis(x, poly_basis)
    response_raw = evaluate_polynomial_basis(coeffs, x_norm, poly_basis)

    return response_raw.detach().numpy()


def compute_user_metrics(top_items, test_items, k):
    """Compute NDCG@k, Recall@k, HitRate@k for one user."""
    if not test_items:
        return {'ndcg': 0, 'recall': 0, 'hr': 0}

    test_set = set(test_items)
    hits = [1 if item in test_set else 0 for item in top_items[:k]]

    # NDCG@k
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sorted(hits, reverse=True)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0

    # Recall@k
    recall = sum(hits) / len(test_set) if test_set else 0

    # Hit Rate (1 if any hit)
    hr = 1 if sum(hits) > 0 else 0

    return {'ndcg': ndcg, 'recall': recall, 'hr': hr}


def compute_global_metrics(R, eigenvecs, eigenvals, filter_response, test_dict, view, k=20):
    """Compute average NDCG@k and Recall@k across all test users."""
    all_ndcg = []
    all_recall = []

    for user_id, test_items in test_dict.items():
        if not test_items:
            continue

        interacted = np.where(R[user_id] > 0)[0]

        if view == 'user':
            u_coeffs = eigenvecs[user_id]
            sim = eigenvecs @ (filter_response * u_coeffs)
            scores = sim @ R
        else:
            spectral = R[user_id] @ eigenvecs
            scores = (filter_response * spectral) @ eigenvecs.T

        scores[interacted] = -np.inf
        top_items = np.argsort(scores)[::-1][:k]

        m = compute_user_metrics(top_items, test_items, k)
        all_ndcg.append(m['ndcg'])
        all_recall.append(m['recall'])

    return {
        'ndcg': np.mean(all_ndcg) if all_ndcg else 0,
        'recall': np.mean(all_recall) if all_recall else 0,
    }


def _apply_activation(x, act_type='sigmoid'):
    """Apply activation function to filter response tensor."""
    import torch
    if act_type == 'softplus':
        return torch.nn.functional.softplus(x)
    elif act_type == 'tanh':
        return (torch.tanh(x) + 1) / 2
    elif act_type == 'none':
        return x
    else:  # sigmoid
        return torch.sigmoid(x)


def evaluate_saved_filter(eigenvals, filter_state_dict, poly_basis='bernstein', activation='sigmoid'):
    """Evaluate a saved polynomial filter from its state dict.

    Handles both new format (single filter.coeffs) and old format (multi-filter with mixing weights).
    Returns activated filter response.
    """
    import torch
    from filter import evaluate_polynomial_basis, normalize_eigenvalues_for_basis

    x = torch.tensor(eigenvals, dtype=torch.float32)

    # Adaptive or polynomial format with filter.coeffs
    if 'filter.coeffs' in filter_state_dict:
        coeffs = filter_state_dict['filter.coeffs']
        if not isinstance(coeffs, torch.Tensor):
            coeffs = torch.tensor(coeffs, dtype=torch.float32)
        # Adaptive and bernstein both use bernstein basis internally
        basis = 'bernstein' if poly_basis == 'adaptive' else poly_basis
        x_norm = normalize_eigenvalues_for_basis(x, basis)
        response = evaluate_polynomial_basis(coeffs, x_norm, basis).detach().numpy()

        # Adaptive: add per-eigenvalue corrections (truncate/pad to match eigenvals size)
        if 'filter.corrections' in filter_state_dict:
            corrections = filter_state_dict['filter.corrections']
            if not isinstance(corrections, torch.Tensor):
                corrections = torch.tensor(corrections, dtype=torch.float32)
            corr = corrections.numpy()
            n = len(response)
            if len(corr) >= n:
                response = response + corr[:n]
            else:
                response[:len(corr)] = response[:len(corr)] + corr

        combined = _apply_activation(torch.tensor(response, dtype=torch.float32), activation).numpy()
        return combined

    # Old format: multi-filter with mixing weights
    if 'filter.mixing_weights' in filter_state_dict:
        mixing_logits = filter_state_dict['filter.mixing_weights']
        if not isinstance(mixing_logits, torch.Tensor):
            mixing_logits = torch.tensor(mixing_logits)
        mixing_weights = torch.softmax(mixing_logits, dim=0).numpy()

        basis = 'bernstein' if poly_basis in ('adaptive', 'bernstein') else poly_basis
        combined_raw = np.zeros_like(eigenvals, dtype=np.float64)
        i = 0
        while f'filter.filter_{i}' in filter_state_dict:
            coeffs = filter_state_dict[f'filter.filter_{i}']
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, dtype=torch.float32)
            x_norm = normalize_eigenvalues_for_basis(x, basis)
            resp_raw = evaluate_polynomial_basis(coeffs, x_norm, basis).detach().numpy()
            if i < len(mixing_weights):
                combined_raw += mixing_weights[i] * resp_raw
            i += 1

        combined = _apply_activation(torch.tensor(combined_raw, dtype=torch.float32), activation).numpy()
        return combined

    # Fallback
    return np.ones_like(eigenvals) * 0.5


def evaluate_direct_filter(filter_state_dict, activation='sigmoid'):
    """Evaluate a saved DirectFilter from its state dict.

    DirectFilter stores one learnable parameter per eigenvalue in 'filter.filter_values'.
    Returns the activated filter response.
    """
    import torch
    filter_values = filter_state_dict['filter.filter_values']
    if not isinstance(filter_values, torch.Tensor):
        filter_values = torch.tensor(filter_values, dtype=torch.float32)
    response = _apply_activation(filter_values, activation).detach().numpy()
    return response


def evaluate_reference_filter(eigenvals, filter_name, steepness=5.0, center=0.5):
    """Evaluate classic reference filters (low-pass, high-pass, band-pass, uniform)."""
    lam = eigenvals / eigenvals.max() if eigenvals.max() > 0 else eigenvals

    if filter_name == 'low-pass':
        return np.exp(-steepness * lam)
    elif filter_name == 'high-pass':
        return 1 - np.exp(-steepness * lam)
    elif filter_name == 'band-pass':
        return np.exp(-steepness * (lam - center) ** 2)
    elif filter_name == 'uniform':
        return np.ones_like(lam)
    return np.ones_like(lam)


def evaluate_combined_filter(eigenvals, order, weights, active_filters, poly_basis='bernstein'):
    """Evaluate weighted combination using REAL model logic, including exp(-|x|) activation."""
    all_names = ['golden', 'smooth', 'butterworth', 'gaussian', 'chebyshev']
    raw_responses = {}
    for name in all_names:
        raw_responses[name] = evaluate_filter_real(eigenvals, name, order, poly_basis)

    # Weighted combination (same as APSFilter.forward)
    # Use softmax weights to match model behavior
    import torch
    logits = torch.tensor([weights.get(name, 0.0) for name in active_filters])
    softmax_weights = torch.softmax(logits, dim=0).numpy()

    combined_raw = np.zeros_like(eigenvals, dtype=np.float64)
    for i, name in enumerate(active_filters):
        combined_raw += softmax_weights[i] * raw_responses[name]

    # Apply the model's activation: exp(-|x|) + epsilon
    final = np.log1p(np.exp(combined_raw)) + 1e-6

    # Apply activation to individual responses for display
    responses_activated = {}
    for name in all_names:
        responses_activated[name] = np.log1p(np.exp(raw_responses[name])) + 1e-6

    return final, responses_activated


# =============================================================================
# App
# =============================================================================

st.set_page_config(page_title="LearnSpec Playground", layout="wide")
st.title("LearnSpec Playground")

# Sidebar controls
st.sidebar.header("Dataset & Eigen")
ALL_DATASETS = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
SMALL_DATASETS = ['ml-100k', 'lastfm']  # datasets small enough for full interactive analysis
dataset_name = st.sidebar.selectbox("Dataset", ALL_DATASETS, index=0)
is_large_dataset = dataset_name not in SMALL_DATASETS
view = st.sidebar.selectbox("View", ['user', 'item'], index=0)
beta = st.sidebar.slider("Beta (degree norm)", 0.0, 1.0, 0.5, 0.1)
n_eigen = st.sidebar.slider("n_eigen", 10, 500, 100, 10)

st.sidebar.header("Filter")
order = st.sidebar.slider("Polynomial order (K)", 2, 16, 8)
poly_basis = st.sidebar.selectbox("Polynomial basis", ['bernstein', 'cheby'], index=0)

st.sidebar.subheader("APSF Subfilters")
active_filters = []
weights = {}
for name, default_on in [('golden', True), ('gaussian', True), ('butterworth', True), ('smooth', True)]:
    col1, col2 = st.sidebar.columns([1, 2])
    on = col1.checkbox(name, value=default_on)
    if on:
        active_filters.append(name)
        weights[name] = col2.slider(f"w_{name}", 0.0, 1.0, 0.25, 0.05, key=f"w_{name}")

st.sidebar.subheader("Reference Filters")
active_refs = []
for name, default_on in [('low-pass', False), ('high-pass', False), ('band-pass', False), ('uniform', False)]:
    if st.sidebar.checkbox(name, value=default_on, key=f"ref_{name}"):
        active_refs.append(name)

ref_steepness = st.sidebar.slider("Steepness (low/high-pass)", 1.0, 20.0, 5.0, 0.5) if ('low-pass' in active_refs or 'high-pass' in active_refs) else 5.0
ref_center = st.sidebar.slider("Center (band-pass)", 0.0, 1.0, 0.5, 0.05) if 'band-pass' in active_refs else 0.5

# Load data — for large datasets, only load eigenvalues (skip R and eigenvectors)
eigenvals_used = None
eigenvecs_used = None
R = None
item_names = {}

if is_large_dataset:
    # Large dataset: only load eigenvalues from cache for Section 5
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'cache', dataset_name)
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(os.getcwd(), '..', 'cache', dataset_name)
    beta_str = str(beta).replace('.', 'p')
    view_name = 'user' if view == 'user' else 'item'

    # Find eigen file
    eigenvals_used = None
    if os.path.exists(cache_dir):
        pattern = f"full_{dataset_name}_{view_name}_largestEigen_n"
        for f in sorted(os.listdir(cache_dir), reverse=True):
            if f.startswith(pattern) and f"degNorm_{beta_str}.pkl" in f:
                with open(os.path.join(cache_dir, f), 'rb') as fh:
                    data = pickle.load(fh)
                eigenvals_loaded = data['eigenvals']
                if eigenvals_loaded[0] < eigenvals_loaded[-1]:
                    eigenvals_loaded = eigenvals_loaded[::-1]
                eigenvals_loaded = np.maximum(eigenvals_loaded, 0)
                eigenvals_used = eigenvals_loaded[:n_eigen]
                all_eigenvals = eigenvals_loaded
                total_dim = len(all_eigenvals)
                st.sidebar.success(f"Loaded {len(all_eigenvals)} eigenvalues from cache")
                break

    if eigenvals_used is None:
        st.warning(f"No cached eigenvalues found for {dataset_name}. Sections 1-4 unavailable.")
        all_eigenvals = np.array([1.0])
        eigenvals_used = np.array([1.0])
        total_dim = 0

    st.info(f"Large dataset ({dataset_name}): Sections 1-4 use eigenvalues only. Section 5 (Before vs After Training) fully available.")
else:
    # Small dataset: full interactive analysis
    with st.spinner("Loading eigendecomposition..."):
        all_eigenvals, eigenvals, eigenvecs, R, S = compute_similarity_and_eigen(dataset_name, view, beta)

    eigenvals_used = eigenvals[:n_eigen]
    eigenvecs_used = eigenvecs[:, :n_eigen]
    total_dim = len(all_eigenvals)
    item_names = load_item_names(dataset_name)

# =============================================================================
# Section 1: Eigenvalue Explorer
# =============================================================================
st.header("1. Eigenvalue Explorer")

total_energy = np.sum(all_eigenvals)
selected_energy = np.sum(eigenvals_used)
energy_pct = selected_energy / total_energy * 100 if total_energy > 0 else 0
cumulative_all = np.cumsum(all_eigenvals) / total_energy if total_energy > 0 else np.zeros_like(all_eigenvals)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    # Show all eigenvalues, highlight selected
    ax.bar(range(n_eigen), eigenvals_used, color='steelblue', alpha=0.8, label=f'Selected (n={n_eigen})')
    if len(all_eigenvals) > n_eigen:
        remaining = all_eigenvals[n_eigen:]
        ax.bar(range(n_eigen, len(all_eigenvals)), remaining, color='lightgray', alpha=0.5, label=f'Remaining ({len(remaining)})')
    ax.axvline(x=n_eigen - 0.5, color='red', linestyle='--', linewidth=1.5, label=f'n_eigen={n_eigen}')
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalue Distribution ({view}, total={total_dim})")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(cumulative_all)), cumulative_all * 100, color='darkorange', linewidth=2)
    ax.axvline(x=n_eigen, color='red', linestyle='--', linewidth=1.5, label=f'n_eigen={n_eigen} ({energy_pct:.1f}%)')
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90%')
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.3, label='95%')
    # Shade selected region
    ax.fill_between(range(n_eigen), 0, cumulative_all[:n_eigen] * 100, alpha=0.15, color='steelblue')
    ax.set_xlabel(f"Number of eigenvalues (total: {total_dim})")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Cumulative Spectral Energy (full spectrum)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    st.pyplot(fig)
    plt.close()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total energy", f"{total_energy:.2f}")
col2.metric("Selected energy", f"{selected_energy:.2f}")
col3.metric("Energy captured", f"{energy_pct:.1f}%")
n_90 = np.searchsorted(cumulative_all, 0.9) + 1
col4.metric("n for 90% energy", f"{n_90} / {total_dim}")

# =============================================================================
# Section 2: Filter Playground
# =============================================================================
st.header("2. Filter Playground")

lam_norm = eigenvals_used / eigenvals_used.max() if eigenvals_used.max() > 0 else eigenvals_used

# Compute reference filter responses
ref_responses = {}
ref_colors = {'low-pass': '#1565C0', 'high-pass': '#C62828', 'band-pass': '#2E7D32', 'uniform': '#757575'}
for name in active_refs:
    ref_responses[name] = evaluate_reference_filter(eigenvals_used, name, ref_steepness, ref_center)

if not active_filters and not active_refs:
    st.warning("Enable at least one filter in the sidebar.")
else:
    combined = None
    individual_responses = {}
    if active_filters:
        combined, individual_responses = evaluate_combined_filter(eigenvals_used, order, weights, active_filters, poly_basis)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        apsf_colors = {'golden': '#e6a817', 'smooth': '#2196F3', 'butterworth': '#4CAF50', 'gaussian': '#9C27B0'}
        for name in active_filters:
            resp = individual_responses[name]
            ax.plot(lam_norm, resp, label=f"{name} (w={weights.get(name, 0.25):.2f})",
                    color=apsf_colors.get(name, 'gray'), linewidth=1.5, alpha=0.7)
        for name in active_refs:
            ax.plot(lam_norm, ref_responses[name], label=name,
                    color=ref_colors[name], linewidth=2, linestyle='--', alpha=0.8)
        ax.set_xlabel("Normalized eigenvalue (0=low freq, 1=high freq)")
        ax.set_ylabel("Filter response h(lambda)")
        ax.set_title("Individual Filter Responses")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, None)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        if combined is not None:
            ax.plot(lam_norm, combined, color='red', linewidth=2.5, label='Combined APSF')
            ax.fill_between(lam_norm, combined, alpha=0.1, color='red')
        for name in active_refs:
            ax.plot(lam_norm, ref_responses[name], label=name,
                    color=ref_colors[name], linewidth=2, linestyle='--', alpha=0.8)
        ax.set_xlabel("Normalized eigenvalue (0=low freq, 1=high freq)")
        ax.set_ylabel("Filter response h(lambda)")
        ax.set_title(f"Combined APSF (K={order}) vs Reference Filters")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, None)
        st.pyplot(fig)
        plt.close()

# =============================================================================
# Section 3: Filtered Spectrum
# =============================================================================
st.header("3. Filtered Spectrum")

# Pick active filter for spectrum/prediction: APSF if available, else first reference
active_filter_response = None
active_filter_label = None
if combined is not None:
    active_filter_response = combined
    active_filter_label = "APSF"
elif active_refs:
    active_filter_response = ref_responses[active_refs[0]]
    active_filter_label = active_refs[0]

if active_filter_response is not None:
    filtered_eigenvals = eigenvals_used * active_filter_response

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(eigenvals_used))
        ax.bar(x - 0.2, eigenvals_used, width=0.4, color='steelblue', alpha=0.6, label='Original')
        ax.bar(x + 0.2, filtered_eigenvals, width=0.4, color='red', alpha=0.6, label='Filtered')
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Original vs Filtered ({active_filter_label})")
        ax.legend()
        if len(eigenvals_used) > 50:
            ax.set_xlim(-1, 50)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 1, 30)
        ax.hist(lam_norm, bins=bins, alpha=0.5, color='steelblue', label='Original', density=True)
        filtered_norm = filtered_eigenvals / eigenvals_used.max() if eigenvals_used.max() > 0 else filtered_eigenvals
        ax.hist(filtered_norm, bins=bins, alpha=0.5, color='red', label='Filtered', density=True)
        ax.set_xlabel("Normalized eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title("Spectral Distribution (Before/After)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # Energy comparison
    original_energy = np.sum(eigenvals_used)
    filtered_energy = np.sum(filtered_eigenvals)
    col1, col2, col3 = st.columns(3)
    col1.metric("Original energy", f"{original_energy:.2f}")
    col2.metric("Filtered energy", f"{filtered_energy:.2f}")
    col3.metric("Energy retained", f"{filtered_energy/original_energy*100:.1f}%" if original_energy > 0 else "N/A")

# =============================================================================
# Section 4: Prediction Preview
# =============================================================================
st.header("4. Prediction Preview")

if is_large_dataset:
    st.info("Prediction preview not available for large datasets (requires loading full interaction matrix).")
elif active_filter_response is not None:
    # Find users with interactions
    user_interactions = np.array(R.sum(axis=1)).flatten()
    valid_users = np.where(user_interactions > 0)[0]

    if len(valid_users) == 0:
        st.warning("No users with interactions found.")
    else:
        user_id = st.selectbox("Select user", valid_users[:100].tolist(),
                               format_func=lambda u: f"User {u} ({int(user_interactions[u])} interactions)")

        n_recs = st.slider("Top-N recommendations", 5, 50, 10)

        # Compute scores for all three: uniform, APSF, reference
        def compute_scores(filter_response):
            if view == 'user':
                u_coeffs = eigenvecs_used[user_id]
                sim = eigenvecs_used @ (filter_response * u_coeffs)
                scores = sim @ R
            else:
                r_u = R[user_id]
                spectral = r_u @ eigenvecs_used
                scores = (filter_response * spectral) @ eigenvecs_used.T
            scores[interacted] = -np.inf
            return scores

        interacted = np.where(R[user_id] > 0)[0]

        # 1. Uniform h(λ)=1
        uniform_scores = compute_scores(np.ones_like(eigenvals_used))
        top_uniform = np.argsort(uniform_scores)[::-1][:n_recs]

        # 2. APSF (if active)
        apsf_scores = None
        top_apsf = None
        if combined is not None:
            apsf_scores = compute_scores(combined)
            top_apsf = np.argsort(apsf_scores)[::-1][:n_recs]

        # 3. Reference filter (if active)
        ref_score_dict = {}
        top_ref_dict = {}
        for rname in active_refs:
            ref_sc = compute_scores(ref_responses[rname])
            ref_score_dict[rname] = ref_sc
            top_ref_dict[rname] = np.argsort(ref_sc)[::-1][:n_recs]

        # Get test items for this user
        dataset = load_dataset(dataset_name)
        test_items = dataset.testDict.get(user_id, [])

        # Build display data: (label, top_items, scores, filter_response)
        entries = [('Uniform h(λ)=1', top_uniform, uniform_scores, np.ones_like(eigenvals_used))]
        if combined is not None:
            entries.append(('APSF', top_apsf, apsf_scores, combined))
        for rname in active_refs:
            entries.append((rname, top_ref_dict[rname], ref_score_dict[rname], ref_responses[rname]))

        cols = st.columns(len(entries))

        for col_idx, (label, top_items, scores, _) in enumerate(entries):
            with cols[col_idx]:
                st.subheader(label)
                test_set = set(test_items)
                n_test = len(test_items)
                for rank, item in enumerate(top_items):
                    score = scores[item]
                    if score > -np.inf:
                        name = item_names.get(item, f"Item {item}")
                        name_safe = name.replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')
                        if item in test_set:
                            st.markdown(f":green[`#{rank+1:>2d}` {name_safe}  `{score:.4f}`]")
                        else:
                            st.markdown(f":red[`#{rank+1:>2d}` {name_safe}  `{score:.4f}`]")

                # Per-user metrics
                m = compute_user_metrics(top_items, test_items, n_recs)
                st.caption(f"NDCG@{n_recs}: **{m['ndcg']:.4f}** | Recall@{n_recs}: **{m['recall']:.4f}**")

        st.caption(f"User {user_id} has {len(test_items)} test items")

        # Overlap
        st.subheader("Overlap")
        all_tops = {e[0]: set(e[1]) for e in entries}
        names = list(all_tops.keys())
        if len(names) >= 2:
            metric_cols = st.columns(len(names) * (len(names) - 1) // 2)
            mc = 0
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    overlap = len(all_tops[names[i]] & all_tops[names[j]])
                    metric_cols[mc].metric(
                        f"{names[i]} vs {names[j]}",
                        f"{overlap}/{n_recs}"
                    )
                    mc += 1

        # Global metrics across all test users
        st.subheader("Global Metrics (all test users)")
        if st.button("Compute global metrics (may take a few seconds)"):
            global_cols = st.columns(len(entries))
            for col_idx, (label, _, _, filt_resp) in enumerate(entries):
                gm = compute_global_metrics(R, eigenvecs_used, eigenvals_used,
                                            filt_resp, dataset.testDict, view, k=n_recs)
                global_cols[col_idx].metric(f"{label}", f"NDCG={gm['ndcg']:.4f} | R={gm['recall']:.4f}")

# =============================================================================
# Section 5: Before vs After Training
# =============================================================================
st.header("5. APSF Before vs After Training")

# Try multiple paths for filter params
params_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'filter_params')
if not os.path.exists(params_dir):
    params_dir = os.path.join(os.getcwd(), '..', 'results', 'filter_params')
if not os.path.exists(params_dir):
    params_dir = os.path.join(os.getcwd(), 'results', 'filter_params')

if os.path.exists(params_dir):
    # Sort by modification time (newest first)
    matching = [f for f in os.listdir(params_dir) if f.endswith('.pkl') and dataset_name in f]
    pkl_files = sorted(matching, key=lambda f: os.path.getmtime(os.path.join(params_dir, f)), reverse=True)
else:
    pkl_files = []

if not pkl_files:
    st.info(f"No saved filter params found for {dataset_name}. Run `python main.py --dataset {dataset_name} ...` first.")
else:
    selected_file = st.selectbox("Saved filter params", pkl_files,
                                 format_func=lambda f: f.replace('.pkl', ''))

    with open(os.path.join(params_dir, selected_file), 'rb') as f:
        saved_data = pickle.load(f)

    initial = saved_data.get('initial', {})
    best = saved_data.get('best', {})

    # Detect filter type and activation from saved config
    saved_config = initial.get('config', {})
    is_direct = saved_config.get('f_poly', saved_config.get('poly', '')) == 'direct'
    saved_act = saved_config.get('f_act', 'sigmoid')

    has_user_filter = best.get('user_filter') or initial.get('user_filter')
    is_apsf = (not is_direct) and has_user_filter
    is_direct_filter = is_direct and has_user_filter

    if is_apsf or is_direct_filter:
        st.caption(f"Best epoch: {best.get('epoch')} | val NDCG: {best.get('ndcg', 0):.4f} | "
                   f"test NDCG: {best.get('test_ndcg', 0):.4f} | test Recall: {best.get('test_recall', 0):.4f}")

        # Build epoch options for slider
        epoch_snapshots = saved_data.get('epochs', [])
        best_epoch = best.get('epoch', len(epoch_snapshots))

        if epoch_snapshots:
            # Build list of available epoch numbers from snapshots
            snapshot_epochs = [s.get('epoch', i+1) for i, s in enumerate(epoch_snapshots)]
            # Find which snapshot index corresponds to best_epoch
            best_snap_idx = 0
            for i, ep in enumerate(snapshot_epochs):
                if ep == best_epoch:
                    best_snap_idx = i + 1  # +1 because 0 = initial
                    break

            options = ["Initial"] + [f"Epoch {e}" for e in snapshot_epochs]
            selected_idx = st.slider("Snapshot", 0, len(epoch_snapshots), best_snap_idx,
                                     help="0 = initial (before training)")

            if selected_idx == 0:
                current_label = "Initial (before training)"
                current_state = initial
                current_user_filter = initial.get('user_filter', {})
                current_item_filter = initial.get('item_filter')
                current_ndcg = 0
                selected_epoch = 0
            else:
                snap = epoch_snapshots[selected_idx - 1]
                selected_epoch = snap.get('epoch', selected_idx)
                is_best = (selected_epoch == best_epoch)
                current_label = f"Epoch {selected_epoch}" + (" (BEST)" if is_best else "")
                current_state = snap
                current_user_filter = snap['user_filter']
                current_item_filter = snap.get('item_filter')
                current_ndcg = snap.get('val_ndcg', 0)

            st.caption(f"Showing: **{current_label}** | val NDCG: {current_ndcg:.4f}")
        else:
            selected_epoch = best_epoch
            current_label = f"Best (epoch {best_epoch})"
            current_user_filter = best.get('user_filter', {})
            current_item_filter = best.get('item_filter')
            current_ndcg = best.get('ndcg', 0)

    if is_direct_filter and not is_apsf:
        # === DirectFilter visualization ===
        n_eigen = len(initial.get('user_filter', {}).get('filter.filter_values', []))

        init_response = evaluate_direct_filter(initial.get('user_filter', {}), saved_act)

        if epoch_snapshots:
            sel_response = evaluate_direct_filter(current_user_filter, saved_act)
        else:
            sel_response = evaluate_direct_filter(best.get('user_filter', {}), saved_act)

        # X-axis: eigenvalue index (evenly spaced, 1=largest/low-freq, n=smallest/high-freq)
        # Eigenvalues are sorted descending, so index 1 = largest eigenvalue = lowest frequency
        x_plot = np.arange(1, n_eigen + 1)
        x_label = "Eigenvalue index (1=largest/low-freq → N=smallest/high-freq)"
        bar_width = 0.8

        # Shared y-axis range across all plots
        y_max = max(init_response.max(), sel_response.max()) * 1.1

        # Row 1: Initial vs Selected filter response
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x_plot, init_response, width=bar_width, color='steelblue', alpha=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel("h(λ)")
            ax.set_title("Initial (before training)")
            ax.set_ylim(0, y_max)
            st.pyplot(fig)
            plt.close()
            st.caption(f"Direct filter: {n_eigen} learnable parameters (1 per eigenvalue)")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x_plot, sel_response, width=bar_width, color='red', alpha=0.8)
            ax.set_xlabel(x_label)
            ax.set_ylabel("h(λ)")
            ax.set_title(current_label)
            ax.set_ylim(0, y_max)
            st.pyplot(fig)
            plt.close()

        # Row 2: Overlay + diff
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x_plot - 0.2, init_response, width=0.4, color='steelblue', alpha=0.7, label='Initial')
            ax.bar(x_plot + 0.2, sel_response, width=0.4, color='red', alpha=0.7, label=current_label)
            ax.set_xlabel(x_label)
            ax.set_ylabel("h(λ)")
            ax.set_title("Direct Filter: Initial vs Selected")
            ax.legend(fontsize=9)
            ax.set_ylim(0, y_max)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[1, 1], sharex=True)
            fig.subplots_adjust(hspace=0.3)

            diff = sel_response - init_response
            ax1.bar(x_plot, diff, width=bar_width,
                    color=['green' if d > 0 else 'red' for d in diff], alpha=0.6)
            ax1.axhline(y=0, color='black', linewidth=0.5)
            ax1.set_ylabel("Change")
            ax1.set_title(f"Learning effect ({current_label} - initial)", fontsize=10)

            ax2.bar(x_plot - 0.2, init_response, width=0.4,
                    color='steelblue', alpha=0.6, label='Initial')
            ax2.bar(x_plot + 0.2, sel_response, width=0.4,
                    color='red', alpha=0.6, label='Learned')
            ax2.set_xlabel(x_label)
            ax2.set_ylabel("h(λ)")
            ax2.legend(fontsize=7, loc='upper left')
            ax2.set_ylim(0, y_max)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, y_max)

            st.pyplot(fig)
            plt.close()

        # NDCG progression
        if epoch_snapshots:
            st.subheader("Training Progression")
            epoch_nums = [s['epoch'] for s in epoch_snapshots]
            val_ndcgs = [s.get('val_ndcg', 0) for s in epoch_snapshots]

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(epoch_nums, val_ndcgs, color='steelblue', linewidth=1.5)
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (ep {best_epoch})')
            if selected_epoch > 0:
                ax.axvline(x=selected_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Selected (ep {selected_epoch})')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("val NDCG")
            ax.set_title("Validation NDCG over epochs")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()

        # Item view for direct filter
        if current_item_filter and initial.get('item_filter'):
            st.subheader("Item View")
            i_init_response = evaluate_direct_filter(initial['item_filter'], saved_act)
            i_sel_response = evaluate_direct_filter(current_item_filter, saved_act)
            n_i_eigen = len(i_init_response)
            i_x_plot = np.arange(1, n_i_eigen + 1)
            i_y_max = max(i_init_response.max(), i_sel_response.max()) * 1.1

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(i_x_plot - 0.2, i_init_response, width=0.4, color='steelblue', alpha=0.7, label='Initial')
                ax.bar(i_x_plot + 0.2, i_sel_response, width=0.4, color='red', alpha=0.7, label=current_label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("h(λ)")
                ax.set_title("Item View: Initial vs Selected")
                ax.legend(fontsize=9)
                ax.set_ylim(0, i_y_max)
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[1, 1], sharex=True)
                fig.subplots_adjust(hspace=0.3)
                i_diff = i_sel_response - i_init_response
                ax1.bar(i_x_plot, i_diff, width=0.8,
                        color=['green' if d > 0 else 'red' for d in i_diff], alpha=0.6)
                ax1.axhline(y=0, color='black', linewidth=0.5)
                ax1.set_ylabel("Change")
                ax1.set_title("Item View: Learning effect", fontsize=10)
                ax2.bar(i_x_plot - 0.2, i_init_response, width=0.4,
                        color='steelblue', alpha=0.6, label='Initial')
                ax2.bar(i_x_plot + 0.2, i_sel_response, width=0.4,
                        color='red', alpha=0.6, label='Learned')
                ax2.set_xlabel(x_label)
                ax2.set_ylabel("h(λ)")
                ax2.legend(fontsize=7, loc='upper left')
                ax2.set_ylim(0, i_y_max)
                st.pyplot(fig)
                plt.close()

    elif is_apsf:
        # === Polynomial filter visualization ===
        lam_norm_plot = eigenvals_used / eigenvals_used.max() if eigenvals_used.max() > 0 else eigenvals_used
        saved_poly = saved_config.get('f_poly', saved_config.get('poly', 'bernstein'))

        # Evaluate initial and selected filter
        init_combined = evaluate_saved_filter(
            eigenvals_used, initial.get('user_filter', {}), saved_poly, saved_act)

        if epoch_snapshots:
            sel_combined = evaluate_saved_filter(
                eigenvals_used, current_user_filter, saved_poly, saved_act)
        else:
            sel_combined = evaluate_saved_filter(
                eigenvals_used, best['user_filter'], saved_poly, saved_act)

        # Row 1: Initial vs Selected filter
        col1, col2 = st.columns(2)

        y_max = max(init_combined.max(), sel_combined.max()) * 1.1

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(lam_norm_plot, init_combined, color='steelblue', linewidth=2.5, label='Initial')
            ax.set_xlabel("Normalized eigenvalue")
            ax.set_ylabel("h(λ)")
            ax.set_title("Initial (before training)")
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_max)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(lam_norm_plot, sel_combined, color='red', linewidth=2.5, label=current_label)
            ax.set_xlabel("Normalized eigenvalue")
            ax.set_ylabel("h(λ)")
            ax.set_title(current_label)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_max)
            st.pyplot(fig)
            plt.close()

        # Row 2: Overlay + diff
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(lam_norm_plot, init_combined, color='blue', linewidth=2.5, label='Initial')
            ax.plot(lam_norm_plot, sel_combined, color='red', linewidth=2.5, label=current_label)
            ax.fill_between(lam_norm_plot, init_combined, sel_combined, alpha=0.1, color='purple')
            ax.set_xlabel("Normalized eigenvalue")
            ax.set_ylabel("h(λ)")
            ax.set_title("Filter: Initial vs Selected")
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, y_max)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[1, 1], sharex=True)
            fig.subplots_adjust(hspace=0.3)

            diff = sel_combined - init_combined
            ax1.bar(lam_norm_plot, diff, width=0.01,
                    color=['green' if d > 0 else 'red' for d in diff], alpha=0.6)
            ax1.axhline(y=0, color='black', linewidth=0.5)
            ax1.set_ylabel("Change")
            ax1.set_title(f"Learning effect ({current_label} - initial)", fontsize=10)

            ax2.bar(lam_norm_plot - 0.005, init_combined, width=0.008,
                    color='steelblue', alpha=0.6, label='Initial')
            ax2.bar(lam_norm_plot + 0.005, sel_combined, width=0.008,
                    color='red', alpha=0.6, label='Learned')
            ax2.set_xlabel("Normalized eigenvalue")
            ax2.set_ylabel("h(λ)")
            ax2.legend(fontsize=7, loc='upper left')
            ax2.set_xlim(0, 1)

            st.pyplot(fig)
            plt.close()

        # Row 3: NDCG progression over epochs
        if epoch_snapshots:
            st.subheader("Training Progression")
            epoch_nums = [s['epoch'] for s in epoch_snapshots]
            val_ndcgs = [s.get('val_ndcg', 0) for s in epoch_snapshots]

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(epoch_nums, val_ndcgs, color='steelblue', linewidth=1.5)
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (ep {best_epoch})')
            if selected_epoch > 0:
                ax.axvline(x=selected_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Selected (ep {selected_epoch})')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("val NDCG")
            ax.set_title("Validation NDCG over epochs")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()

        # Item view (if available)
        if current_item_filter and initial.get('item_filter'):
            st.subheader("Item View")
            i_init_combined = evaluate_saved_filter(
                eigenvals_used, initial['item_filter'], saved_poly, saved_act)
            i_sel_combined = evaluate_saved_filter(
                eigenvals_used, current_item_filter, saved_poly, saved_act)

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(lam_norm_plot, i_init_combined, color='blue', linewidth=2.5, label='Initial')
                ax.plot(lam_norm_plot, i_sel_combined, color='red', linewidth=2.5, label=current_label)
                ax.fill_between(lam_norm_plot, i_init_combined, i_sel_combined, alpha=0.1, color='purple')
                ax.set_xlabel("Normalized eigenvalue")
                ax.set_ylabel("h(lambda)")
                ax.set_title("Item View: Initial vs Selected")
                ax.legend(fontsize=9)
                ax.set_xlim(0, 1)
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[1, 1], sharex=True)
                fig.subplots_adjust(hspace=0.3)

                # Top: change
                i_diff = i_sel_combined - i_init_combined
                ax1.bar(lam_norm_plot, i_diff, width=0.01,
                        color=['green' if d > 0 else 'red' for d in i_diff], alpha=0.6)
                ax1.axhline(y=0, color='black', linewidth=0.5)
                ax1.set_ylabel("Change")
                ax1.set_title("Item View: Learning effect", fontsize=10)

                # Bottom: initial vs learned
                ax2.bar(lam_norm_plot - 0.005, i_init_combined, width=0.008,
                        color='steelblue', alpha=0.6, label='Initial')
                ax2.bar(lam_norm_plot + 0.005, i_sel_combined, width=0.008,
                        color='red', alpha=0.6, label='Learned')
                ax2.set_xlabel("Normalized eigenvalue (0=high freq, 1=low freq)")
                ax2.set_ylabel("h(λ)")
                ax2.legend(fontsize=7, loc='upper left')
                ax2.set_xlim(0, 1)

                st.pyplot(fig)
                plt.close()
    else:
        st.warning("Saved file doesn't contain filter names or user filter data.")
