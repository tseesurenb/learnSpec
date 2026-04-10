"""
LearnSpec Toy Demo — Interactive step-by-step spectral collaborative filtering.
Run: streamlit run src/toy/app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

st.set_page_config(page_title="LearnSpec Toy Demo", layout="wide")
st.title("LearnSpec: Spectral Collaborative Filtering — Step by Step")

# ============================================================
# STEP 0: Define toy dataset
# ============================================================
st.header("Step 0: User-Item Interactions")
st.markdown("""
**8 users × 6 items.** Two user groups:
- Users 0-3 prefer items 0-2 (e.g., action movies)
- Users 4-7 prefer items 3-5 (e.g., romance movies)
- Some cross-group interactions add complexity
""")

# Default interaction matrix (editable)
default_R = np.array([
    # Items:  0  1  2  3  4  5
    [1, 1, 1, 0, 0, 0],  # User 0 — action fan
    [1, 1, 0, 1, 0, 0],  # User 1 — action + crossover
    [1, 0, 1, 0, 0, 0],  # User 2 — action fan
    [0, 1, 1, 0, 1, 0],  # User 3 — action + crossover
    [0, 0, 0, 1, 1, 1],  # User 4 — romance fan
    [0, 0, 1, 1, 1, 0],  # User 5 — romance + crossover
    [0, 0, 0, 1, 0, 1],  # User 6 — romance fan
    [0, 1, 0, 0, 1, 1],  # User 7 — romance + crossover
], dtype=np.float32)

n_users, n_items = default_R.shape

st.markdown("**Interaction Matrix R** (1 = interacted, 0 = not)")
R_df = pd.DataFrame(default_R,
                     index=[f"User {i}" for i in range(n_users)],
                     columns=[f"Item {j}" for j in range(n_items)])
edited_R = st.data_editor(R_df, use_container_width=True, key="R_editor")
R = edited_R.values.astype(np.float32)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total interactions", int(R.sum()))
    st.metric("Sparsity", f"{(1 - R.sum() / (n_users * n_items)) * 100:.1f}%")
with col2:
    st.metric("Users", n_users)
    st.metric("Items", n_items)

# ============================================================
# STEP 1: Train/Test Split
# ============================================================
st.header("Step 1: Train/Test Split")
st.markdown("Hold out 1 interaction per user for testing (the last positive item).")

R_train = R.copy()
R_test = np.zeros_like(R)
test_items = {}
for u in range(n_users):
    pos_items = np.where(R[u] > 0)[0]
    if len(pos_items) > 1:
        test_item = pos_items[-1]  # Last positive as test
        R_train[u, test_item] = 0
        R_test[u, test_item] = 1
        test_items[u] = test_item

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Training Data**")
    train_df = pd.DataFrame(R_train,
                            index=[f"User {i}" for i in range(n_users)],
                            columns=[f"Item {j}" for j in range(n_items)])
    st.dataframe(train_df.style.highlight_max(axis=1))
with col2:
    st.markdown("**Test Data** (held-out items)")
    test_df = pd.DataFrame(R_test,
                           index=[f"User {i}" for i in range(n_users)],
                           columns=[f"Item {j}" for j in range(n_items)])
    st.dataframe(test_df.style.highlight_max(axis=1))

st.info(f"Test items: {test_items}")

# ============================================================
# STEP 2: Sub-Eigenspace Split
# ============================================================
st.header("Step 2: Sub-Eigenspace Split (for training)")

split_ratio = st.slider("Split ratio (ρ)", 0.5, 0.9, 0.7, 0.1)
st.markdown(f"""
Split training data into **{int(split_ratio*100)}% sub-training** and **{int((1-split_ratio)*100)}% validation**.
The sub-training builds the eigenspace. Validation is used to train the filter.
""")

np.random.seed(42)
R_sub = R_train.copy()
R_val = np.zeros_like(R_train)
val_items = {}
for u in range(n_users):
    pos_items = np.where(R_train[u] > 0)[0]
    if len(pos_items) > 1:
        np.random.shuffle(pos_items)
        n_keep = max(1, int(len(pos_items) * split_ratio))
        val_set = pos_items[n_keep:]
        for v in val_set:
            R_sub[u, v] = 0
            R_val[u, v] = 1
        val_items[u] = val_set.tolist()

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Sub-Training Data ({int(split_ratio*100)}%)**")
    st.dataframe(pd.DataFrame(R_sub,
                              index=[f"User {i}" for i in range(n_users)],
                              columns=[f"Item {j}" for j in range(n_items)]))
with col2:
    st.markdown(f"**Validation Data ({int((1-split_ratio)*100)}%)**")
    st.dataframe(pd.DataFrame(R_val,
                              index=[f"User {i}" for i in range(n_users)],
                              columns=[f"Item {j}" for j in range(n_items)]))

# ============================================================
# STEP 3: Degree Normalization
# ============================================================
st.header("Step 3: Degree Normalization")

beta = st.slider("Beta (β)", 0.0, 1.0, 0.5, 0.05)
st.latex(r"A_n = D_u^{-\beta} \cdot R_{sub} \cdot D_i^{-\beta}")

d_u = R_sub.sum(axis=1)  # User degrees
d_i = R_sub.sum(axis=0)  # Item degrees

d_u_inv = np.where(d_u > 0, d_u ** (-beta), 0)
d_i_inv = np.where(d_i > 0, d_i ** (-beta), 0)

A_n = np.diag(d_u_inv) @ R_sub @ np.diag(d_i_inv)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**User degrees (d_u)**")
    st.write(pd.DataFrame({'degree': d_u, f'd_u^(-{beta})': d_u_inv},
                          index=[f"User {i}" for i in range(n_users)]))
with col2:
    st.markdown("**Item degrees (d_i)**")
    st.write(pd.DataFrame({'degree': d_i, f'd_i^(-{beta})': d_i_inv},
                          index=[f"Item {j}" for j in range(n_items)]))
with col3:
    st.markdown("**Normalized A_n**")
    st.dataframe(pd.DataFrame(np.round(A_n, 3),
                              index=[f"U{i}" for i in range(n_users)],
                              columns=[f"I{j}" for j in range(n_items)]))

# ============================================================
# STEP 4: Similarity Matrices
# ============================================================
st.header("Step 4: Similarity Matrices")
st.latex(r"S_u = A_n \cdot A_n^T \quad \text{(user-user similarity)}")
st.latex(r"S_i = A_n^T \cdot A_n \quad \text{(item-item similarity)}")

S_u = A_n @ A_n.T
S_i = A_n.T @ A_n

col1, col2 = st.columns(2)
with col1:
    st.markdown("**User Similarity S_u** (8×8)")
    st.dataframe(pd.DataFrame(np.round(S_u, 3),
                              index=[f"U{i}" for i in range(n_users)],
                              columns=[f"U{j}" for j in range(n_users)]))
with col2:
    st.markdown("**Item Similarity S_i** (6×6)")
    st.dataframe(pd.DataFrame(np.round(S_i, 3),
                              index=[f"I{i}" for i in range(n_items)],
                              columns=[f"I{j}" for j in range(n_items)]))

# ============================================================
# STEP 5: Eigendecomposition
# ============================================================
st.header("Step 5: Eigendecomposition")

k_u = st.slider("User eigenvalues (k_u)", 1, n_users, min(4, n_users))
k_i = st.slider("Item eigenvalues (k_i)", 1, n_items, min(4, n_items))

# Eigendecomposition
eigenvals_u, eigenvecs_u = eigh(S_u)
eigenvals_i, eigenvecs_i = eigh(S_i)

# Sort descending (largest first)
idx_u = np.argsort(eigenvals_u)[::-1]
eigenvals_u = eigenvals_u[idx_u]
eigenvecs_u = eigenvecs_u[:, idx_u]

idx_i = np.argsort(eigenvals_i)[::-1]
eigenvals_i = eigenvals_i[idx_i]
eigenvecs_i = eigenvecs_i[:, idx_i]

# Truncate
eigenvals_u_k = eigenvals_u[:k_u]
eigenvecs_u_k = eigenvecs_u[:, :k_u]
eigenvals_i_k = eigenvals_i[:k_i]
eigenvecs_i_k = eigenvecs_i[:, :k_i]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**User Eigenvalues (top {k_u})**")
    ev_df = pd.DataFrame({
        'eigenvalue': np.round(eigenvals_u_k, 4),
        'cumulative %': np.round(np.cumsum(eigenvals_u_k) / eigenvals_u.sum() * 100, 1)
    }, index=[f"λ_{i}" for i in range(k_u)])
    st.dataframe(ev_df)

    st.markdown(f"**User Eigenvectors U** ({n_users}×{k_u})")
    st.dataframe(pd.DataFrame(np.round(eigenvecs_u_k, 3),
                              index=[f"U{i}" for i in range(n_users)],
                              columns=[f"e_{j}" for j in range(k_u)]))

with col2:
    st.markdown(f"**Item Eigenvalues (top {k_i})**")
    ev_df = pd.DataFrame({
        'eigenvalue': np.round(eigenvals_i_k, 4),
        'cumulative %': np.round(np.cumsum(eigenvals_i_k) / eigenvals_i.sum() * 100, 1)
    }, index=[f"λ_{i}" for i in range(k_i)])
    st.dataframe(ev_df)

    st.markdown(f"**Item Eigenvectors V** ({n_items}×{k_i})")
    st.dataframe(pd.DataFrame(np.round(eigenvecs_i_k, 3),
                              index=[f"I{i}" for i in range(n_items)],
                              columns=[f"e_{j}" for j in range(k_i)]))

# ============================================================
# STEP 6: Spectral Filter
# ============================================================
st.header("Step 6: Spectral Filter h(λ)")
st.markdown("""
The filter assigns a weight to each eigenvalue. Different initializations emphasize different frequencies.
After sigmoid: values in (0, 1). Higher = keep more of that spectral component.
""")

filter_init = st.selectbox("Filter initialization",
                           ['uniform', 'lowpass', 'highpass', 'bandpass', 'rise', 'butterworth'])
filter_act = st.selectbox("Activation function", ['sigmoid', 'softplus'])

def get_init_values(init_type, n):
    t = np.linspace(0, 1, n)
    if init_type == 'lowpass':
        return np.array([1.0 * (0.5 ** i) for i in range(n)])
    elif init_type == 'highpass':
        return np.array([1.0 * (0.5 ** (n - 1 - i)) for i in range(n)])
    elif init_type == 'bandpass':
        mid = (n - 1) / 2.0
        return np.array([np.exp(-((i - mid) ** 2) / max(1, n / 3)) for i in range(n)])
    elif init_type == 'rise':
        x = np.clip(0.40 + 0.25 * t, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    elif init_type == 'butterworth':
        x = np.clip(0.40 + 0.35 / (1.0 + (2.0 * t) ** 4), 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    else:  # uniform
        return np.zeros(n)

def apply_act(x, act):
    if act == 'softplus':
        return np.log(1 + np.exp(x))
    else:  # sigmoid
        return 1 / (1 + np.exp(-x))

# User filter
coeffs_u = get_init_values(filter_init, k_u)
h_u = apply_act(coeffs_u, filter_act)

# Item filter
coeffs_i = get_init_values(filter_init, k_i)
h_i = apply_act(coeffs_i, filter_act)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**User Filter h_u(λ)**")
    filter_df = pd.DataFrame({
        'eigenvalue': np.round(eigenvals_u_k, 4),
        'coefficients': np.round(coeffs_u, 4),
        f'h(λ) [{filter_act}]': np.round(h_u, 4),
    }, index=[f"λ_{i}" for i in range(k_u)])
    st.dataframe(filter_df)

with col2:
    st.markdown("**Item Filter h_i(λ)**")
    filter_df = pd.DataFrame({
        'eigenvalue': np.round(eigenvals_i_k, 4),
        'coefficients': np.round(coeffs_i, 4),
        f'h(λ) [{filter_act}]': np.round(h_i, 4),
    }, index=[f"λ_{i}" for i in range(k_i)])
    st.dataframe(filter_df)

# ============================================================
# STEP 7: Score Computation
# ============================================================
st.header("Step 7: Score Computation")

st.latex(r"\text{score}_u(u,:) = [U[u,:] \cdot h_u(\Lambda)] \cdot U^T \cdot R_{sub}")
st.latex(r"\text{score}_i(u,:) = [R_{sub}[u,:] \cdot V \cdot h_i(\Lambda)] \cdot V^T")

# User view scores
U_k = eigenvecs_u_k  # (n_users, k_u)
V_k = eigenvecs_i_k  # (n_items, k_i)

# Precompute spectral projections
UtR = U_k.T @ R_sub  # (k_u, n_items)
RV = R_sub @ V_k      # (n_users, k_i)

# User view: score_u = (U * h_u) @ U^T @ R
scores_u = (U_k * h_u[np.newaxis, :]) @ UtR  # (n_users, n_items)

# Item view: score_i = (R @ V * h_i) @ V^T
scores_i = (RV * h_i[np.newaxis, :]) @ V_k.T  # (n_users, n_items)

# Fusion
fusion_weight = st.slider("User view weight (α)", 0.0, 1.0, 0.5, 0.05)
scores = fusion_weight * scores_u + (1 - fusion_weight) * scores_i

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**User View Scores**")
    st.dataframe(pd.DataFrame(np.round(scores_u, 3),
                              index=[f"U{i}" for i in range(n_users)],
                              columns=[f"I{j}" for j in range(n_items)]))
with col2:
    st.markdown("**Item View Scores**")
    st.dataframe(pd.DataFrame(np.round(scores_i, 3),
                              index=[f"U{i}" for i in range(n_users)],
                              columns=[f"I{j}" for j in range(n_items)]))
with col3:
    st.markdown(f"**Fused Scores** (α={fusion_weight:.2f})")
    st.dataframe(pd.DataFrame(np.round(scores, 3),
                              index=[f"U{i}" for i in range(n_users)],
                              columns=[f"I{j}" for j in range(n_items)]))

# ============================================================
# STEP 8: Ranking and Evaluation
# ============================================================
st.header("Step 8: Ranking and Evaluation")
st.markdown("Mask training items (set to -inf), then rank remaining items.")

# Mask training items
masked_scores = scores.copy()
masked_scores[R_train > 0] = -np.inf

st.markdown("**Masked Scores** (training items = -inf)")
masked_df = pd.DataFrame(np.where(R_train > 0, '—', np.round(masked_scores, 3)),
                         index=[f"U{i}" for i in range(n_users)],
                         columns=[f"I{j}" for j in range(n_items)])
st.dataframe(masked_df)

# Rank and evaluate
K = st.slider("Top-K for evaluation", 1, n_items, min(3, n_items))

results = []
for u in range(n_users):
    if u not in test_items:
        continue
    user_scores = masked_scores[u]
    ranked = np.argsort(user_scores)[::-1][:K]
    hit = 1 if test_items[u] in ranked else 0
    rank_pos = np.where(ranked == test_items[u])[0]
    ndcg = 1 / np.log2(rank_pos[0] + 2) if len(rank_pos) > 0 else 0

    results.append({
        'User': f"U{u}",
        'Test item': f"I{test_items[u]}",
        f'Top-{K} ranked': [f"I{r}" for r in ranked],
        'Hit': '✓' if hit else '✗',
        f'NDCG@{K}': round(ndcg, 4),
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

avg_ndcg = np.mean([r[f'NDCG@{K}'] for r in results])
avg_recall = np.mean([1 if r['Hit'] == '✓' else 0 for r in results])
st.success(f"**NDCG@{K} = {avg_ndcg:.4f}** | **Recall@{K} = {avg_recall:.4f}**")

# ============================================================
# Summary
# ============================================================
st.header("Summary")
st.markdown(f"""
| Step | What | Shape |
|------|------|-------|
| R (interactions) | Binary matrix | {n_users}×{n_items} |
| A_n (normalized) | D_u^(-β) R D_i^(-β) | {n_users}×{n_items} |
| S_u (user sim) | A_n A_n^T | {n_users}×{n_users} |
| S_i (item sim) | A_n^T A_n | {n_items}×{n_items} |
| U (user eigenvecs) | Top-{k_u} of S_u | {n_users}×{k_u} |
| V (item eigenvecs) | Top-{k_i} of S_i | {n_items}×{k_i} |
| h_u (user filter) | {filter_init} + {filter_act} | {k_u} values |
| h_i (item filter) | {filter_init} + {filter_act} | {k_i} values |
| Scores | α·score_u + (1-α)·score_i | {n_users}×{n_items} |
| **Params** | **{k_u + k_i + 2} total** | filter coeffs + fusion |
""")
