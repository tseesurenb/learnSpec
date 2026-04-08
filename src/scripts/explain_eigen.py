"""
Understanding Eigendecomposition in LearnSpec
=============================================

This script walks through the math step by step using ml-100k.

The key idea: we decompose user-item interactions into frequency components
(like Fourier transform for graphs), so we can filter them.

Pipeline:
  R (interactions) → A_n (normalized) → S (similarity) → U,Λ (eigen) → filter → predict
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import Dataset

# =============================================================================
# Step 0: Load data
# =============================================================================
print("="*60)
print("Step 0: Load ml-100k")
print("="*60)

dataset = Dataset(path=\"../data/ml-100k\")
R = dataset.UserItemNet  # sparse (944 x 1683), R[u,i] = 1 if user u interacted with item i

print(f"R shape: {R.shape} (users x items)")
print(f"Non-zero entries: {R.nnz} (interactions)")
print(f"Sparsity: {1 - R.nnz / (R.shape[0] * R.shape[1]):.4f}")
print(f"\nExample: User 0 interacted with items: {R[0].nonzero()[1][:10]}...")

# =============================================================================
# Step 1: Degree normalization
# =============================================================================
print(f"\n{'='*60}")
print("Step 1: Degree-normalized adjacency matrix")
print("="*60)
print("""
Formula: A_n = D_u^{-β} · R · D_i^{-(1-β)}

Where:
  D_u = diagonal matrix of user degrees (how many items each user interacted with)
  D_i = diagonal matrix of item degrees (how many users interacted with each item)
  β = 0.5 (symmetric normalization, standard choice)

This normalizes by popularity: a user who rated 1000 items gets downweighted,
an item rated by 1 user gets downweighted. Prevents popular items from dominating.
""")

beta = 0.5
R_dense = R.toarray().astype(np.float64)

# Degree matrices
d_u = np.array(R.sum(axis=1)).flatten()  # user degrees
d_i = np.array(R.sum(axis=0)).flatten()  # item degrees

print(f"User degrees: min={d_u.min():.0f}, max={d_u.max():.0f}, mean={d_u.mean():.1f}")
print(f"Item degrees: min={d_i.min():.0f}, max={d_i.max():.0f}, mean={d_i.mean():.1f}")

# Avoid division by zero
d_u_inv = np.where(d_u > 0, d_u ** (-beta), 0)
d_i_inv = np.where(d_i > 0, d_i ** (-(1 - beta)), 0)

# A_n = D_u^{-β} · R · D_i^{-(1-β)}
A_n = (d_u_inv[:, None] * R_dense) * d_i_inv[None, :]

print(f"\nA_n shape: {A_n.shape}")
print(f"A_n range: [{A_n.min():.4f}, {A_n.max():.4f}]")

# =============================================================================
# Step 2: Similarity matrices
# =============================================================================
print(f"\n{'='*60}")
print("Step 2: Similarity matrices")
print("="*60)
print("""
User similarity:  S_u = A_n · A_n^T   (944 x 944)
Item similarity:  S_i = A_n^T · A_n   (1683 x 1683)

S_u[i,j] measures how similar users i and j are based on shared items.
S_i[i,j] measures how similar items i and j are based on shared users.

These are symmetric positive semi-definite matrices (key property for eigen).
""")

S_u = A_n @ A_n.T  # (944 x 944)
S_i = A_n.T @ A_n  # (1683 x 1683)

print(f"S_u shape: {S_u.shape}")
print(f"S_u is symmetric: {np.allclose(S_u, S_u.T)}")
print(f"S_u diagonal (self-similarity): min={np.diag(S_u).min():.4f}, max={np.diag(S_u).max():.4f}")
print(f"\nS_i shape: {S_i.shape}")
print(f"S_i is symmetric: {np.allclose(S_i, S_i.T)}")

# =============================================================================
# Step 3: Eigendecomposition
# =============================================================================
print(f"\n{'='*60}")
print("Step 3: Eigendecomposition")
print("="*60)
print("""
S = U · Λ · U^T

Where:
  Λ = diagonal matrix of eigenvalues (λ_1, λ_2, ..., λ_k)
  U = matrix of eigenvectors (columns are orthonormal basis vectors)

Each eigenvalue λ_k represents a "frequency":
  - Large λ → low frequency → captures global/popular patterns
  - Small λ → high frequency → captures local/personalized patterns

We only keep the top-k largest eigenvalues (truncated decomposition).
This is like keeping only the most important frequency components.
""")

k = 50  # number of eigenvalues to compute
print(f"Computing top-{k} eigenvalues of S_u ({S_u.shape[0]}x{S_u.shape[0]})...")

eigenvals_u, eigenvecs_u = eigsh(S_u, k=k, which='LM')  # LM = Largest Magnitude

# eigsh returns in ascending order, reverse to descending
eigenvals_u = eigenvals_u[::-1]
eigenvecs_u = eigenvecs_u[:, ::-1]

print(f"\nUser eigenvalues (top 10): {eigenvals_u[:10].round(4)}")
print(f"User eigenvalues (last 5): {eigenvals_u[-5:].round(4)}")
print(f"Eigenvector matrix shape: {eigenvecs_u.shape} (users x components)")
print(f"Eigenvectors are orthonormal: U^T·U ≈ I ? {np.allclose(eigenvecs_u.T @ eigenvecs_u, np.eye(k), atol=1e-6)}")

# =============================================================================
# Step 4: What eigenvalues mean
# =============================================================================
print(f"\n{'='*60}")
print("Step 4: What eigenvalues mean (frequency interpretation)")
print("="*60)
print("""
Think of eigenvalues as frequencies in a Fourier transform:

  λ_1 (largest)  = lowest frequency  = "everyone likes popular items"
  λ_2            = next frequency    = "broad genre preferences"
  ...
  λ_k (smallest) = highest frequency = "this specific user likes this niche item"

The eigenvalue magnitude tells you how much "energy" is in that frequency.
""")

total_energy = np.sum(eigenvals_u)
cumulative = np.cumsum(eigenvals_u) / total_energy

print(f"Total spectral energy (sum of eigenvalues): {total_energy:.2f}")
print(f"\nCumulative energy captured:")
for pct in [10, 20, 50, 80, 90, 95]:
    n_needed = np.searchsorted(cumulative, pct/100) + 1
    print(f"  {pct}% energy captured by top {n_needed} eigenvalues")

# =============================================================================
# Step 5: Spectral filtering (the core of LearnSpec)
# =============================================================================
print(f"\n{'='*60}")
print("Step 5: Spectral filtering")
print("="*60)
print("""
This is where LearnSpec's innovation happens.

Without filter (baseline):
  Y = U · Λ · U^T · R
  (reconstruct predictions using all frequencies equally)

With filter h(λ):
  Y = U · h(Λ) · U^T · R
  (scale each frequency by a learned weight)

Example filters:
  h(λ) = 1          → keep all frequencies (baseline)
  h(λ) = λ          → emphasize low frequencies (popularity)
  h(λ) = exp(-λ)    → smooth low-pass filter
  h(λ) = learned    → LearnSpec's APSF adapts per dataset
""")

# Normalize eigenvalues to [0, 1]
eigenvals_norm = eigenvals_u / eigenvals_u.max()

# Try different filters
filters = {
    'no filter (h=1)':     np.ones_like(eigenvals_u),
    'identity (h=λ)':      eigenvals_u,
    'low-pass exp(-2λ)':   np.exp(-2 * eigenvals_norm),
    'high-pass (1-e^-2λ)': 1 - np.exp(-2 * eigenvals_norm),
}

print("Filter responses at different frequencies:")
print(f"{'Filter':<25} {'λ_1 (low)':<12} {'λ_25 (mid)':<12} {'λ_50 (high)':<12}")
print("-" * 60)
for name, h in filters.items():
    print(f"{name:<25} {h[0]:<12.4f} {h[k//2]:<12.4f} {h[-1]:<12.4f}")

# =============================================================================
# Step 6: Making predictions
# =============================================================================
print(f"\n{'='*60}")
print("Step 6: Making predictions")
print("="*60)
print("""
For a user u, the prediction for all items is:

  y_u = U · diag(h(Λ)) · U^T · r_u

Where r_u is user u's interaction row from R.

Step by step:
  1. Project user into spectral domain:  ĥ = U^T · r_u     (k coefficients)
  2. Apply filter:                       ĥ_filtered = h(Λ) * ĥ
  3. Project back to item domain:        y_u = U · ĥ_filtered

The result y_u has a score for every item — higher = stronger recommendation.
""")

user_id = 0
r_u = R_dense[user_id]  # user 0's interactions (1683,)

# Using identity filter (h = eigenvalues)
h = eigenvals_u

# User-view prediction: U · h(Λ) · U^T gives filtered user similarity (944 x 944)
# Then multiply by R to get item scores
#
# Step by step for user u:
#   1. Get user u's similarity to all users:  s_u = U · h(Λ) · U^T[u, :]   (944,)
#   2. Weighted sum of all users' items:      y_u = s_u · R                  (1683,)

# Extract user u's row from eigenvector matrix
u_coeffs = eigenvecs_u[user_id]                      # (k,) — user u in spectral basis
filtered_coeffs = h * u_coeffs                        # (k,) — apply filter
user_similarities = eigenvecs_u @ filtered_coeffs     # (944,) — similarity to all users

# Use similarities as weights over all users' interactions
item_scores = user_similarities @ R_dense             # (1683,) — item scores

# Mask already-interacted items
interacted = R[user_id].nonzero()[1]
item_scores[interacted] = -np.inf

# Top recommendations
top_items = np.argsort(item_scores)[::-1][:10]
print(f"User {user_id} has interacted with {len(interacted)} items")
print(f"Top 10 recommended items: {top_items}")
print(f"Their scores: {item_scores[top_items].round(4)}")

# =============================================================================
# Step 7: Why truncation matters
# =============================================================================
print(f"\n{'='*60}")
print("Step 7: Why truncation (n_eigen) matters")
print("="*60)
print("""
Using all eigenvalues = perfect reconstruction of S (but includes noise).
Using fewer = keeps only dominant patterns, acts as denoising.

This is why n_eigen is a critical hyperparameter:
  Too few  → loses important patterns
  Too many → includes noise, overfits
""")

for n in [10, 20, 50, 100, 200]:
    # Reconstruction error with top-n eigenvalues
    S_approx = eigenvecs_u[:, :n] @ np.diag(eigenvals_u[:n]) @ eigenvecs_u[:, :n].T
    error = np.linalg.norm(S_u - S_approx, 'fro') / np.linalg.norm(S_u, 'fro')
    energy = np.sum(eigenvals_u[:n]) / total_energy * 100
    print(f"  n_eigen={n:>4d}: reconstruction error={error:.4f}, energy captured={energy:.1f}%")

print(f"\n{'='*60}")
print("Summary")
print("="*60)
print("""
LearnSpec's pipeline:
  1. R → A_n → S_u, S_i     (interaction → similarity)
  2. S → U, Λ               (eigendecomposition)
  3. h(Λ) = APSF(Λ; θ)      (learnable spectral filter)
  4. Y = U·h(Λ)·U^T·R       (filtered prediction)

What's learned: the filter function h(λ; θ) with ~82 parameters
What's precomputed: eigendecomposition (expensive, done once)
What's searched: n_eigen (how many components to keep)
""")
