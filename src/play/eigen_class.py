"""
Spectral User Classification Explorer
======================================
Analyze how users cluster based on their spectral profiles.

Launch: cd src && streamlit run play/eigen_class.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Spectral User Classification", layout="wide")
st.title("Spectral User Classification")

# =============================================================================
# Sidebar
# =============================================================================
dataset_name = st.sidebar.selectbox("Dataset", ['ml-100k', 'lastfm', 'gowalla', 'yelp2018'])
beta = st.sidebar.slider("Beta (degree norm)", 0.0, 1.0, 0.5, 0.1)

# Find available eigen files
cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'cache', dataset_name)
if not os.path.exists(cache_dir):
    cache_dir = os.path.join(os.getcwd(), '..', 'cache', dataset_name)

beta_str = str(beta).replace('.', 'p')

# Load eigenvectors
@st.cache_data
def load_eigenvectors(cache_dir, dataset_name, beta_str):
    pattern = f"full_{dataset_name}_user_largestEigen_n"
    for f in sorted(os.listdir(cache_dir), reverse=True):
        if f.startswith(pattern) and f"degNorm_{beta_str}.pkl" in f:
            with open(os.path.join(cache_dir, f), 'rb') as fh:
                data = pickle.load(fh)
            eigenvecs = data.get('eigenvecs', data.get('all_eigenvecs'))
            eigenvals = data.get('eigenvals', data.get('all_eigenvals'))
            if eigenvals[0] < eigenvals[-1]:
                eigenvals = eigenvals[::-1]
                eigenvecs = eigenvecs[:, ::-1]
            return np.array(eigenvals), np.array(eigenvecs), f
    return None, None, None

if not os.path.exists(cache_dir):
    st.error(f"Cache directory not found: {cache_dir}")
    st.stop()

eigenvals, eigenvecs, loaded_file = load_eigenvectors(cache_dir, dataset_name, beta_str)

if eigenvecs is None:
    st.error(f"No eigenvectors found for {dataset_name} with beta={beta}")
    st.stop()

n_users, n_eigen = eigenvecs.shape
st.sidebar.success(f"Loaded: {loaded_file}")
st.sidebar.info(f"Users: {n_users}, Eigenvalues: {n_eigen}")

n_eigen_use = st.sidebar.slider("Eigenvalues to use", 10, min(n_eigen, 500), min(100, n_eigen))
k_low_ratio = st.sidebar.slider("Low-freq ratio (k_low / k)", 0.1, 0.9, 0.5, 0.05)

eigenvecs_used = eigenvecs[:, :n_eigen_use]
eigenvals_used = eigenvals[:n_eigen_use]
k_low = max(1, int(n_eigen_use * k_low_ratio))

# =============================================================================
# Section 1: Spectral Profiles
# =============================================================================
st.header("1. Spectral Energy Distribution")

# Compute spectral profiles
energy_low = (eigenvecs_used[:, :k_low] ** 2).sum(axis=1)
energy_high = (eigenvecs_used[:, k_low:] ** 2).sum(axis=1)
energy_total = energy_low + energy_high + 1e-8
spectral_ratio = energy_high / energy_total  # r(u) in [0, 1]

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spectral_ratio, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(x=np.median(spectral_ratio), color='red', linestyle='--', label=f'Median={np.median(spectral_ratio):.3f}')
    ax.axvline(x=np.mean(spectral_ratio), color='orange', linestyle='--', label=f'Mean={np.mean(spectral_ratio):.3f}')
    ax.set_xlabel("Spectral ratio r(u) = E_high / E_total")
    ax.set_ylabel("Number of users")
    ax.set_title("Distribution of Spectral Ratio")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    # Sort users by spectral ratio
    sorted_idx = np.argsort(spectral_ratio)
    ax.scatter(range(n_users), spectral_ratio[sorted_idx], s=1, alpha=0.5, color='steelblue')
    ax.axhline(y=np.median(spectral_ratio), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("User (sorted by spectral ratio)")
    ax.set_ylabel("Spectral ratio r(u)")
    ax.set_title("Per-user Spectral Ratio (sorted)")
    st.pyplot(fig)
    plt.close()

st.caption(f"k_low={k_low}, k_high={n_eigen_use - k_low} | "
           f"Low-freq users (r < 0.5): {(spectral_ratio < 0.5).sum()} | "
           f"High-freq users (r >= 0.5): {(spectral_ratio >= 0.5).sum()}")

# =============================================================================
# Section 2: Optimal Number of Clusters
# =============================================================================
st.header("2. Optimal Number of Clusters")

from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Use full spectral profile for clustering (or PCA-reduced)
profiles = eigenvecs_used ** 2  # energy per component
if profiles.shape[1] > 20:
    pca = PCA(n_components=min(20, profiles.shape[1]))
    profiles_reduced = pca.fit_transform(profiles)
    st.caption(f"PCA: {profiles.shape[1]} -> {profiles_reduced.shape[1]} components "
               f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance explained)")
else:
    profiles_reduced = profiles

# 2a: K-Means + Silhouette
max_k = min(10, n_users // 10)
k_range = range(2, max_k + 1)
silhouette_scores = []
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(profiles_reduced)
    silhouette_scores.append(silhouette_score(profiles_reduced, labels))
    inertias.append(km.inertia_)

best_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]

# 2b: GMM + BIC
bic_scores = []
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
    gmm.fit(profiles_reduced)
    bic_scores.append(gmm.bic(profiles_reduced))

best_k_bic = list(k_range)[np.argmin(bic_scores)]

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(k_range), silhouette_scores, 'o-', color='steelblue')
    ax.axvline(x=best_k_silhouette, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"K-Means: Best k={best_k_silhouette}")
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(k_range), bic_scores, 'o-', color='green')
    ax.axvline(x=best_k_bic, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title(f"GMM: Best k={best_k_bic}")
    st.pyplot(fig)
    plt.close()

with col3:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(k_range), inertias, 'o-', color='purple')
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)
    plt.close()

# 2c: DBSCAN
from sklearn.preprocessing import StandardScaler
profiles_scaled = StandardScaler().fit_transform(profiles_reduced)

eps_val = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 1.5, 0.1)
min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)

dbscan = DBSCAN(eps=eps_val, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(profiles_scaled)
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = (dbscan_labels == -1).sum()

# 2d: Mean Shift
try:
    ms = MeanShift()
    ms_labels = ms.fit_predict(profiles_scaled)
    n_meanshift_clusters = len(set(ms_labels))
except:
    ms_labels = np.zeros(n_users, dtype=int)
    n_meanshift_clusters = 1

st.markdown(f"""
**Automatic cluster detection:**
| Method | Optimal k |
|--------|-----------|
| K-Means + Silhouette | **{best_k_silhouette}** |
| GMM + BIC | **{best_k_bic}** |
| DBSCAN | **{n_dbscan_clusters}** clusters ({n_noise} noise points) |
| Mean Shift | **{n_meanshift_clusters}** |
""")

# =============================================================================
# Section 3: Cluster Visualization
# =============================================================================
st.header("3. Cluster Visualization")

cluster_method = st.selectbox("Clustering method", [
    f"K-Means (k={best_k_silhouette}, silhouette)",
    f"GMM (k={best_k_bic}, BIC)",
    "DBSCAN",
    "Mean Shift",
    "Manual (2 groups by median)",
])

if "K-Means" in cluster_method:
    k_use = best_k_silhouette
    km = KMeans(n_clusters=k_use, random_state=42, n_init=10)
    labels = km.fit_predict(profiles_reduced)
elif "GMM" in cluster_method:
    k_use = best_k_bic
    gmm = GaussianMixture(n_components=k_use, random_state=42)
    labels = gmm.fit_predict(profiles_reduced)
elif "DBSCAN" in cluster_method:
    labels = dbscan_labels
    k_use = n_dbscan_clusters
elif "Mean Shift" in cluster_method:
    labels = ms_labels
    k_use = n_meanshift_clusters
else:
    # Manual 2-group split
    labels = (spectral_ratio > np.median(spectral_ratio)).astype(int)
    k_use = 2

# PCA for 2D visualization
pca_2d = PCA(n_components=2)
profiles_2d = pca_2d.fit_transform(profiles_reduced)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 2)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = f"Noise ({mask.sum()})" if label == -1 else f"Group {label} ({mask.sum()})"
        ax.scatter(profiles_2d[mask, 0], profiles_2d[mask, 1],
                   s=3, alpha=0.5, color=colors[i % len(colors)], label=name)
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Users in PCA Spectral Space")
    ax.legend(fontsize=7, markerscale=3)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = labels == label
        ax.hist(spectral_ratio[mask], bins=30, alpha=0.6,
                color=colors[i % len(colors)], label=f"Group {label}", edgecolor='white')
    ax.set_xlabel("Spectral ratio r(u)")
    ax.set_ylabel("Count")
    ax.set_title("Spectral Ratio Distribution per Cluster")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close()

# =============================================================================
# Section 4: Cluster Statistics
# =============================================================================
st.header("4. Cluster Statistics")

stats_data = []
for label in sorted(set(labels)):
    if label == -1:
        continue
    mask = labels == label
    stats_data.append({
        'Group': label,
        'Users': mask.sum(),
        'Mean r(u)': f"{spectral_ratio[mask].mean():.4f}",
        'Std r(u)': f"{spectral_ratio[mask].std():.4f}",
        'Min r(u)': f"{spectral_ratio[mask].min():.4f}",
        'Max r(u)': f"{spectral_ratio[mask].max():.4f}",
        'Mean E_low': f"{energy_low[mask].mean():.4f}",
        'Mean E_high': f"{energy_high[mask].mean():.4f}",
        'Label': 'Low-freq (mainstream)' if spectral_ratio[mask].mean() < 0.5 else 'High-freq (niche)',
    })

if stats_data:
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# =============================================================================
# Section 5: Per-eigenvalue Energy by Cluster
# =============================================================================
st.header("5. Per-eigenvalue Energy by Cluster")

fig, ax = plt.subplots(figsize=(10, 4))
unique_labels_clean = [l for l in sorted(set(labels)) if l != -1]
colors_clean = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels_clean), 2)))

for i, label in enumerate(unique_labels_clean):
    mask = labels == label
    mean_energy = (eigenvecs_used[mask] ** 2).mean(axis=0)
    ax.plot(range(n_eigen_use), mean_energy, linewidth=1.5, alpha=0.8,
            color=colors_clean[i], label=f"Group {label} ({mask.sum()} users)")

ax.axvline(x=k_low, color='gray', linestyle=':', alpha=0.5, label=f'k_low={k_low}')
ax.set_xlabel("Eigenvalue index (0=largest/low-freq)")
ax.set_ylabel("Mean squared eigenvector entry")
ax.set_title("Average Spectral Energy Profile per Cluster")
ax.legend(fontsize=8)
st.pyplot(fig)
plt.close()

st.caption(f"Each line shows how much spectral energy the cluster's users have at each eigenvalue. "
           f"Clusters with more energy on the right side are 'high-frequency' (niche preference) users.")
