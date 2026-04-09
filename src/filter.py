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


def _logit(x):
    """Inverse sigmoid: maps (0,1) -> (-inf, inf). Coefficients are pre-sigmoid."""
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return float(np.log(x / (1 - x)))


def get_init_coefficients(init_type, order):
    n = order + 1
    t = np.linspace(0, 1, n)  # control points in [0,1]
    if init_type == 'lowpass':
        return [1.0 * (0.5 ** i) for i in range(n)]
    elif init_type == 'highpass':
        return [1.0 * (0.5 ** (n - 1 - i)) for i in range(n)]
    elif init_type == 'bandpass':
        mid = (n - 1) / 2.0
        return [np.exp(-((i - mid) ** 2) / max(1, n / 3)) for i in range(n)]
    elif init_type == 'butterworth':
        # Mild low-pass with sharper rolloff than lowpass
        vals = 0.40 + 0.35 / (1.0 + (2.0 * t) ** 4)
        return [_logit(v) for v in vals]
    elif init_type == 'bandreject':
        # Opposite of bandpass: high at edges, dip in middle (notch filter)
        mid = (n - 1) / 2.0
        return [-np.exp(-((i - mid) ** 2) / max(1, n / 3)) for i in range(n)]
    elif init_type == 'decay':
        # Smooth exponential decay — gentle rolloff
        vals = 0.25 * np.exp(-1.5 * t) + 0.50
        return [_logit(v) for v in vals]
    elif init_type == 'rise':
        # Smooth linear rise — gentle low-to-high
        vals = 0.40 + 0.25 * t
        return [_logit(v) for v in vals]
    elif init_type == 'plateau':
        # Wide flat plateau in middle, rolls off at edges
        vals = 0.45 + 0.30 * np.exp(-8.0 * (t - 0.5) ** 2)
        return [_logit(v) for v in vals]
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


class GroupFilter(nn.Module):
    """Group-adaptive spectral filter: G group filters mixed by soft user-group assignment."""

    def __init__(self, n_eigen, n_groups=5, filter_order=8, init_type='uniform', poly_basis='bernstein', activation='sigmoid'):
        super().__init__()
        self.n_eigen = n_eigen
        self.n_groups = n_groups
        self.filter_order = filter_order
        self.poly_basis = poly_basis
        self.activation = activation

        # G sets of polynomial coefficients, each initialized differently
        init_offsets = torch.linspace(-0.5, 0.5, n_groups)
        base_init = torch.tensor(get_init_coefficients(init_type, filter_order), dtype=torch.float32)
        group_coeffs = base_init.unsqueeze(0).repeat(n_groups, 1) + init_offsets.unsqueeze(1) * 0.1
        self.group_coeffs = nn.Parameter(group_coeffs)  # (G, K+1)

        # Assignment matrix: maps spectral profile (n_eigen) to group logits (G)
        self.V = nn.Parameter(torch.randn(n_groups, n_eigen) * 0.01)

        if poly_basis == 'bernstein':
            self.register_buffer('_binomials', precompute_bernstein_binomials(filter_order))
        else:
            self._binomials = None

        self._cached_eigenvals_id = None
        self._cached_basis = None

    def _get_basis(self, eigenvals):
        ev_id = eigenvals.data_ptr()
        if self._cached_eigenvals_id != ev_id:
            x = normalize_eigenvalues_for_basis(eigenvals, self.poly_basis)
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
            self._cached_basis = basis
            self._cached_eigenvals_id = ev_id
        return self._cached_basis

    def forward(self, eigenvals, user_spectral_embed=None):
        """
        eigenvals: (k,)
        user_spectral_embed: (batch, k) — L2-normalized spectral profile
        Returns: (batch, k) per-group filter responses, or (k,) global if no embed
        """
        basis = self._get_basis(eigenvals)  # (K+1, k)
        # All group filter responses: (G, k)
        group_responses = apply_activation(self.group_coeffs @ basis, self.activation)

        if user_spectral_embed is None:
            # Global mode: equal-weight average of all groups
            return group_responses.mean(dim=0)

        # Soft assignment: (batch, G)
        group_weights = torch.softmax(user_spectral_embed @ self.V.T, dim=1)
        # Weighted mix of group responses: (batch, k)
        return group_weights @ group_responses

    def get_parameter_groups(self, config):
        return [
            {'params': [self.group_coeffs], 'name': 'group_filter_coeffs'},
            {'params': [self.V], 'name': 'group_assign_V'},
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

    n_groups = config.get('n_groups', 5) if config else 5
    if config.get('guf', False):
        guf_basis = poly_basis if poly_basis in ('bernstein', 'cheby') else 'bernstein'
        return GroupFilter(n_eigen=config.get('n_eigen', order), n_groups=n_groups, filter_order=order, init_type=init_type, poly_basis=guf_basis, activation=activation)
    elif poly_basis == 'direct':
        return DirectFilter(n_eigen=config.get('n_eigen', order), init_type=init_type, dropout=dropout, activation=activation)
    elif poly_basis == 'adaptive':
        return AdaptiveFilter(n_eigen=config.get('n_eigen', order), filter_order=order, init_type=init_type, dropout=dropout, activation=activation)
    else:
        return APSFilter(filter_order=order, init_filter_name=init_type, poly_basis=poly_basis, dropout=dropout, activation=activation)
