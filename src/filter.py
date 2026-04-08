import torch
import torch.nn as nn
import numpy as np


def apply_activation(x, act_type='sigmoid'):
    if act_type == 'softplus':
        return torch.nn.functional.softplus(x)
    elif act_type == 'tanh':
        return (torch.tanh(x) + 1) / 2
    elif act_type == 'none':
        return x
    else:
        return torch.sigmoid(x)


def get_init_coefficients(init_type, order):
    n = order + 1
    if init_type == 'lowpass':
        return [1.0 * (0.5 ** i) for i in range(n)]
    elif init_type == 'highpass':
        return [1.0 * (0.5 ** (n - 1 - i)) for i in range(n)]
    elif init_type == 'bandpass':
        mid = (n - 1) / 2.0
        return [np.exp(-((i - mid) ** 2) / max(1, n / 3)) for i in range(n)]
    else:  # uniform
        return [0.0] * n


def precompute_bernstein_binomials(n):
    binomials = torch.ones(n + 1)
    for i in range(1, n + 1):
        binomials[i] = binomials[i - 1] * (n - i + 1) / i
    return binomials


def evaluate_polynomial_basis(coeffs, x, basis_type='cheby', binomials=None):
    if basis_type == 'cheby':
        result = coeffs[0] * torch.ones_like(x)
        if len(coeffs) > 1:
            T_prev, T_curr = torch.ones_like(x), x
            result += coeffs[1] * T_curr
            for i in range(2, len(coeffs)):
                T_next = 2 * x * T_curr - T_prev
                result += coeffs[i] * T_next
                T_prev, T_curr = T_curr, T_next

    elif basis_type == 'bernstein':
        n = len(coeffs) - 1
        result = torch.zeros_like(x)
        if binomials is None:
            binomials = precompute_bernstein_binomials(n)
        for i in range(len(coeffs)):
            basis = binomials[i] * (x ** i) * ((1 - x) ** (n - i))
            result += coeffs[i] * basis
    else:
        raise ValueError(f"Unknown basis: {basis_type}")

    return result


def normalize_eigenvalues_for_basis(eigenvalues, basis_type='cheby'):
    min_val, max_val = eigenvalues.min(), eigenvalues.max()
    if basis_type == 'cheby':
        return 2 * (eigenvalues - min_val) / (max_val - min_val + 1e-8) - 1
    elif basis_type == 'bernstein':
        return (eigenvalues - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown basis: {basis_type}")


class APSFilter(nn.Module):
    def __init__(self, filter_order=8, init_filter_name='uniform', poly_basis='bernstein', dropout=0.0, activation='sigmoid'):
        super().__init__()
        self.filter_order = filter_order
        self.poly_basis = poly_basis
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.activation = activation
        self.coeffs = nn.Parameter(torch.tensor(get_init_coefficients(init_filter_name, filter_order), dtype=torch.float32))

        if poly_basis == 'bernstein':
            self.register_buffer('_binomials', precompute_bernstein_binomials(filter_order))
        else:
            self._binomials = None

        self._cached_eigenvals_id = None
        self._cached_x_normalized = None

    def forward(self, eigenvals):
        batch_shape = eigenvals.shape
        eigenvals_flat = eigenvals.view(-1)
        coeffs = self.dropout(self.coeffs) if self.dropout is not None and self.training else self.coeffs

        ev_id = eigenvals_flat.data_ptr()
        if self._cached_eigenvals_id != ev_id:
            self._cached_x_normalized = normalize_eigenvalues_for_basis(eigenvals_flat, self.poly_basis)
            self._cached_eigenvals_id = ev_id

        response = evaluate_polynomial_basis(coeffs, self._cached_x_normalized, self.poly_basis, self._binomials)
        return apply_activation(response, self.activation).view(batch_shape)

    def get_parameter_groups(self, config):
        return [{'params': [self.coeffs], 'name': 'filter_coeffs'}]

    def get_filter_values(self, n_points=100):
        with torch.no_grad():
            x = torch.linspace(0, 1, n_points)
            return x.numpy(), self.forward(x).numpy()


class DirectFilter(nn.Module):
    def __init__(self, n_eigen, init_type='uniform', dropout=0.0, activation='sigmoid'):
        super().__init__()
        self.n_eigen = n_eigen
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.activation = activation

        if init_type == 'lowpass':
            init_vals = torch.linspace(2.0, -2.0, n_eigen)
        elif init_type == 'highpass':
            init_vals = torch.linspace(-2.0, 2.0, n_eigen)
        elif init_type == 'bandpass':
            mid = (n_eigen - 1) / 2.0
            init_vals = 2.0 * torch.exp(-((torch.arange(n_eigen).float() - mid) ** 2) / max(1, n_eigen / 3)) - 1.0
        else:  # uniform
            init_vals = torch.zeros(n_eigen)

        self.filter_values = nn.Parameter(init_vals)

    def forward(self, eigenvals=None):
        response = apply_activation(self.filter_values, self.activation)
        if self.dropout is not None:
            response = self.dropout(response)
        return response

    def get_parameter_groups(self, config):
        return [{'params': [self.filter_values], 'name': 'direct_filter_values'}]

    def get_filter_values(self, n_points=100):
        with torch.no_grad():
            x = torch.linspace(0, 1, self.n_eigen)
            return x.numpy(), self.forward().cpu().numpy()


class UserAdaptiveFilter(nn.Module):
    """Per-user adaptive spectral filter: c(u) = W @ z_u + b, then polynomial evaluation."""

    def __init__(self, n_eigen, filter_order=8, init_type='uniform', poly_basis='bernstein', activation='sigmoid'):
        super().__init__()
        self.n_eigen = n_eigen
        self.filter_order = filter_order
        self.poly_basis = poly_basis
        self.activation = activation

        # Bias acts as global baseline filter
        self.bias = nn.Parameter(torch.tensor(get_init_coefficients(init_type, filter_order), dtype=torch.float32))
        # W maps user spectral profile (n_eigen) to polynomial coefficients (K+1)
        self.W = nn.Parameter(torch.randn(filter_order + 1, n_eigen) * 0.01)

        if poly_basis == 'bernstein':
            self.register_buffer('_binomials', precompute_bernstein_binomials(filter_order))
        else:
            self._binomials = None

        self._cached_eigenvals_id = None
        self._cached_x_normalized = None
        self._cached_basis = None

    def _get_normalized(self, eigenvals):
        ev_id = eigenvals.data_ptr()
        if self._cached_eigenvals_id != ev_id:
            self._cached_x_normalized = normalize_eigenvalues_for_basis(eigenvals, self.poly_basis)
            self._cached_basis = self._precompute_basis(self._cached_x_normalized)
            self._cached_eigenvals_id = ev_id
        return self._cached_x_normalized, self._cached_basis

    def _precompute_basis(self, x):
        """Precompute basis functions: (K+1, k) matrix."""
        K = self.filter_order
        basis = torch.zeros(K + 1, len(x), device=x.device)
        if self.poly_basis == 'bernstein':
            for i in range(K + 1):
                basis[i] = self._binomials[i] * (x ** i) * ((1 - x) ** (K - i))
        elif self.poly_basis == 'cheby':
            basis[0] = torch.ones_like(x)
            if K >= 1:
                basis[1] = x
                for i in range(2, K + 1):
                    basis[i] = 2 * x * basis[i - 1] - basis[i - 2]
        return basis

    def forward(self, eigenvals, user_spectral_embed=None):
        """
        eigenvals: (k,)
        user_spectral_embed: (batch, k) — user eigenvecs weighted by eigenvals
        Returns: (batch, k) per-user filter responses, or (k,) global if no embed given
        """
        _, basis = self._get_normalized(eigenvals)  # basis: (K+1, k)

        if user_spectral_embed is None:
            # Global mode: just use bias
            response = self.bias @ basis  # (k,)
            return apply_activation(response, self.activation)

        # Per-user coefficients: (batch, K+1)
        coeffs = user_spectral_embed @ self.W.T + self.bias.unsqueeze(0)
        # Per-user filter response: (batch, k)
        response = coeffs @ basis
        return apply_activation(response, self.activation)

    def get_parameter_groups(self, config):
        return [
            {'params': [self.bias], 'name': 'filter_bias'},
            {'params': [self.W], 'name': 'filter_adapt_W'},
        ]

    def get_filter_values(self, n_points=100):
        with torch.no_grad():
            x = torch.linspace(0, 1, n_points)
            return x.numpy(), self.forward(x).numpy()


class AdaptiveFilter(nn.Module):
    def __init__(self, n_eigen, filter_order=8, init_type='uniform', dropout=0.0, activation='sigmoid'):
        super().__init__()
        self.n_eigen = n_eigen
        self.filter_order = filter_order
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.activation = activation
        self.coeffs = nn.Parameter(torch.tensor(get_init_coefficients(init_type, filter_order), dtype=torch.float32))
        self.corrections = nn.Parameter(torch.zeros(n_eigen))
        self.register_buffer('_binomials', precompute_bernstein_binomials(filter_order))
        self._cached_eigenvals_id = None
        self._cached_x_normalized = None

    def forward(self, eigenvals):
        coeffs = self.dropout(self.coeffs) if self.dropout is not None and self.training else self.coeffs
        ev_id = eigenvals.data_ptr()
        if self._cached_eigenvals_id != ev_id:
            self._cached_x_normalized = normalize_eigenvalues_for_basis(eigenvals, 'bernstein')
            self._cached_eigenvals_id = ev_id
        poly_response = evaluate_polynomial_basis(coeffs, self._cached_x_normalized, 'bernstein', self._binomials)
        return apply_activation(poly_response + self.corrections, self.activation)

    def get_parameter_groups(self, config):
        return [
            {'params': [self.coeffs], 'name': 'poly_coeffs'},
            {'params': [self.corrections], 'name': 'eigen_corrections'},
        ]

    def get_filter_values(self, n_points=100):
        with torch.no_grad():
            x = torch.linspace(0, 1, self.n_eigen)
            return x.numpy(), self.forward(x).cpu().numpy()


def create_filter(order=8, init_type='uniform', config=None):
    poly_basis = config.get('poly', 'bernstein') if config else 'bernstein'
    activation = config.get('f_act', 'sigmoid') if config else 'sigmoid'
    dropout = config.get('f_dropout', 0.0) if config else 0.0

    if config.get('puf', False):
        # UserAdaptiveFilter uses polynomial basis; map 'adaptive'/'direct' to 'bernstein'
        puf_basis = poly_basis if poly_basis in ('bernstein', 'cheby') else 'bernstein'
        return UserAdaptiveFilter(n_eigen=config.get('n_eigen', order), filter_order=order, init_type=init_type, poly_basis=puf_basis, activation=activation)
    elif poly_basis == 'direct':
        return DirectFilter(n_eigen=config.get('n_eigen', order), init_type=init_type, dropout=dropout, activation=activation)
    elif poly_basis == 'adaptive':
        return AdaptiveFilter(n_eigen=config.get('n_eigen', order), filter_order=order, init_type=init_type, dropout=dropout, activation=activation)
    else:
        return APSFilter(filter_order=order, init_filter_name=init_type, poly_basis=poly_basis, dropout=dropout, activation=activation)
