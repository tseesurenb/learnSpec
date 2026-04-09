# Mathematical Improvements for LearnSpec

## Current Model

The current model applies a global spectral filter to precomputed eigendecompositions:

**User view**: `score_u(u, :) = [U[u,:] * h(L)] . U^T . R`

**Item view**: `score_i(u, :) = [R[u,:] . V * g(L)] . V^T`

where:
- U, L = eigenvectors/values of S_u = A_n . A_n^T
- V, L = eigenvectors/values of S_i = A_n^T . A_n
- h(l) = s(sum_k c_k . B_k(l)) = polynomial filter with sigmoid, same for all users
- g(l) = same structure for item view
- Final: score = a . score_u + (1-a) . score_i (learnable fusion)

**Limitation**: One filter for all users. A heavy metal fan and a jazz lover get the same spectral weighting.

---

## Idea 1: User-Adaptive Spectral Filter

**Motivation**: Different users have different spectral profiles. A user who interacts with popular items lives in the low-frequency (smooth) part of the spectrum. A niche user with unique tastes needs high-frequency components.

**Formulation**: Instead of global coefficients c_k, make them user-dependent:

    h_u(l_i) = s(sum_k c_k(u) . B_k(l_i))

where the coefficients are a function of the user's spectral embedding:

    c(u) = W . z_u + b

- z_u = U[u,:] * L — user's eigenvalue-weighted spectral profile (k-dimensional)
- W in R^{(K+1) x k} — maps spectral profile to polynomial coefficients
- b in R^{K+1} — bias (acts as the global filter baseline)

**Parameter count**: With K=32 polynomial order and k=200 eigenvalues:
- Current: K+1 = 33 parameters (global filter)
- Proposed: (K+1) x k + (K+1) = 33 x 200 + 33 = 6,633 parameters
- Still tiny compared to embedding-based methods (millions of params)

**Computation**: The per-user filter adds one matrix multiply per batch:

    C = Z_batch @ W^T + b    # (batch, K+1) — user-specific coefficients
    h_u = s(C @ B(L))        # (batch, k) — user-specific filter responses
    scores = (U[users] * h_u) @ (U^T @ R)  # same precomputed projection

**Why this is novel**: Existing spectral CF methods (GF-CF, BSPM) all use global filters. This is the first to personalize the spectral filter per user while maintaining the efficiency of precomputed eigendecompositions.

---

## Idea 2: Cross-View Spectral Interaction

**Motivation**: The current model treats user and item spectral domains independently, then fuses with a scalar weight. But the spectral components from the two views may interact — certain user spectral components may align with certain item spectral components.

**Formulation**: Learn a joint spectral kernel:

    score(u, i) = sum_j sum_l K(l_j^u, l_l^i) . (U[u,j] . (U^T R)_{j,:}) . (V[i,l] . (R^T V)_{:,l})

The kernel K(l^u, l^i) can be parameterized as low-rank:

    K(l_j^u, l_l^i) = s(a_j . b_l)

where a in R^{k_u} and b in R^{k_i} are learnable vectors.

**Simplified efficient form**:

    score(u, :) = [U[u,:] * a]^T . U^T . R . V . diag(b) . V^T

This is still efficient because U^T . R . V can be precomputed as a (k_u, k_i) matrix.

**Parameter count**: k_u + k_i parameters (e.g., 200 + 700 = 900).

---

## Idea 3: Multi-Resolution Spectral Filtering

**Motivation**: A single polynomial captures one "scale" of spectral information. Like wavelets, multiple scales can capture both global structure and local patterns.

**Formulation**: S filters at different resolutions, combined via attention:

    h(l) = sum_s a_s . h_s(l)

where each h_s has different polynomial order K_s:
- h_1: order 4 (coarse, captures broad trends)
- h_2: order 16 (medium, captures clusters)
- h_3: order 64 (fine, captures individual patterns)

Attention weights can be global or user-specific:

    a_s(u) = softmax(w_s . z_u)    # user-adaptive scale selection

**Parameter count**: sum(K_s + 1) + S x k for user-adaptive attention.

---

## Idea 4: Spectral Contrastive Learning

**Motivation**: MSE loss on validation interactions is a weak signal. Contrastive learning can provide a stronger self-supervised signal by enforcing consistency under spectral perturbations.

**Formulation**:

1. **Spectral augmentation**: Create two views by randomly masking eigenvalue responses:

       h'(l_i) = h(l_i) . m_i,    m_i ~ Bernoulli(p)

2. **Contrastive loss**: For each user, their spectral embeddings under two augmentations should be similar:

       L_cl = -log(exp(sim(z_u, z_u') / t) / sum_v exp(sim(z_u, z_v') / t))

3. **Total loss**: L = L_mse + w . L_cl

**Why it's principled**: Unlike random edge/node dropout (SGL), spectral dropout directly perturbs the frequency domain — it forces the model to not over-rely on any single spectral component. This is a natural regularizer for spectral methods.

---

## Idea 5: Optimal Filter Theory

**Theorem sketch**: For the MSE objective min_h ||R - U diag(h(L)) U^T R||_F^2, the optimal filter has closed form:

    h*(l_i) = l_i / (l_i + u)

where u is a regularization parameter related to noise level. This is Tikhonov regularization in the spectral domain.

**Implications**:
- The optimal filter is a low-pass filter with smooth rolloff
- The learned Bernstein polynomial should converge to this shape
- Provides theoretical justification for why the method works
- The deviation of learned filter from h* reveals dataset-specific spectral structure

**Paper value**: A theorem that characterizes the optimal filter gives reviewers something concrete. You can show empirically that learned filters approximate h* and analyze the residual.

---

## Recommendation

**For the strongest paper, combine Ideas 1 + 5:**

- **Idea 1** (user-adaptive filter) = the architectural contribution. Clean, novel, efficient
- **Idea 5** (optimal filter theory) = the theoretical contribution. Provides a theorem

**Experiment plan**:
1. Show global filter (current) vs user-adaptive filter across all 5 datasets
2. Visualize learned filters per user cluster — show they differ meaningfully
3. Compare learned filter shape to theoretical optimum h* = l/(l+u)
4. Ablation: polynomial order, number of eigenvalues, with/without adaptation









============================================================================================================================================

Idea 1 — Spectral Consistency Learning:
To address the potential mismatch between sub-eigenspace training and full-spectrum inference, we introduce a spectral consistency objective that enforces alignment between predictions obtained in the truncated eigenspace and those projected from the full spectral space. Specifically, the model is encouraged to produce consistent outputs when operating under different spectral resolutions, ensuring that the learned filter generalizes beyond the training subspace. This regularization stabilizes training and reduces bias introduced by truncated representations, making the learned spectral response more robust and transferable.

Idea 2 — Frequency-Aware Regularization:
We incorporate a frequency-aware regularization mechanism that explicitly accounts for the role of different spectral components during learning. Instead of treating all eigenvalues equally, the model is guided to learn smoother and more structured spectral responses by penalizing abrupt changes across neighboring frequencies or by weighting frequency components based on their contribution to global versus local signals. This encourages the filter to maintain a coherent shape in the spectral domain, preventing overfitting to noisy high-frequency components while preserving meaningful variations necessary for personalization.

Idea 3 — Spectral Dropout for Robust Learning:
To improve generalization and robustness, we introduce spectral dropout, a data augmentation strategy applied directly in the eigenspace. During training, a subset of eigencomponents is randomly masked or perturbed, forcing the model to learn stable spectral patterns that do not rely on specific components. This stochastic regularization mimics variations in graph structure and encourages the model to distribute information more evenly across frequencies, leading to improved resilience against noise and better performance under different spectral configurations.
