import torch
import torch.nn as nn
import numpy as np


def apply_activation(x, act_type='sigmoid'):
    if act_type == 'softplus':
        return torch.nn.functional.softplus(x)
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
    elif init_type == 'decay':
        # Smooth exponential decay — gentle rolloff
        vals = 0.25 * np.exp(-1.5 * t) + 0.50
        return [_logit(v) for v in vals]
    elif init_type == 'rise':
        vals = 0.25 + 0.35 * t
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
    def __init__(self, filter_order=8, init_filter_name='uniform', poly_basis='bernstein', activation='sigmoid', n_jitter=0):
        super().__init__()
        self.filter_order = filter_order
        self.poly_basis = poly_basis
        self.activation = activation
        self.n_jitter = n_jitter
        self.coeffs = nn.Parameter(torch.tensor(get_init_coefficients(init_filter_name, filter_order), dtype=torch.float32))

        if poly_basis == 'bernstein':
            self.register_buffer('_binomials', precompute_bernstein_binomials(filter_order))
        else:
            self._binomials = None

        # Fourier refinement: learnable oscillations on top of polynomial
        if n_jitter > 0:
            self.jitter_cos = nn.Parameter(torch.zeros(n_jitter))
            self.jitter_sin = nn.Parameter(torch.zeros(n_jitter))

        self._cached_eigenvals_id = None
        self._cached_x_normalized = None

    def _fourier_jitter(self, x):
        """Evaluate Fourier refinement: Σ aₖcos(kπx) + bₖsin(kπx)."""
        jitter = torch.zeros_like(x)
        for k in range(self.n_jitter):
            freq = (k + 1) * np.pi * x
            jitter = jitter + self.jitter_cos[k] * torch.cos(freq) + self.jitter_sin[k] * torch.sin(freq)
        return jitter

    def forward(self, eigenvals):
        batch_shape = eigenvals.shape
        eigenvals_flat = eigenvals.view(-1)

        ev_id = eigenvals_flat.data_ptr()
        if self._cached_eigenvals_id != ev_id:
            self._cached_x_normalized = normalize_eigenvalues_for_basis(eigenvals_flat, self.poly_basis)
            self._cached_eigenvals_id = ev_id

        response = evaluate_polynomial_basis(self.coeffs, self._cached_x_normalized, self.poly_basis, self._binomials)

        if self.n_jitter > 0:
            response = response + self._fourier_jitter(self._cached_x_normalized)

        return apply_activation(response, self.activation).view(batch_shape)

    def get_parameter_groups(self, config):
        groups = [{'params': [self.coeffs], 'name': 'filter_coeffs'}]
        if self.n_jitter > 0:
            groups.append({'params': [self.jitter_cos, self.jitter_sin], 'name': 'filter_jitter'})
        return groups

    def get_filter_values(self, n_points=100):
        with torch.no_grad():
            x = torch.linspace(0, 1, n_points)
            return x.numpy(), self.forward(x).numpy()


class DirectFilter(nn.Module):
    """One learnable parameter per eigenvalue. Uses actual eigenvalue positions."""

    def __init__(self, n_eigen, init_type='uniform', activation='sigmoid'):
        super().__init__()
        self.n_eigen = n_eigen
        self.activation = activation
        t = np.linspace(0, 1, n_eigen)

        if init_type == 'lowpass':
            init_vals = np.linspace(2.0, -2.0, n_eigen)
        elif init_type == 'highpass':
            init_vals = np.linspace(-2.0, 2.0, n_eigen)
        elif init_type == 'bandpass':
            mid = (n_eigen - 1) / 2.0
            init_vals = 2.0 * np.exp(-((np.arange(n_eigen) - mid) ** 2) / max(1, n_eigen / 3)) - 1.0
        elif init_type == 'rise':
            vals = 0.25 + 0.35 * t
            init_vals = np.array([_logit(v) for v in vals])
        elif init_type == 'decay':
            vals = 0.25 * np.exp(-1.5 * t) + 0.50
            init_vals = np.array([_logit(v) for v in vals])
        elif init_type == 'butterworth':
            vals = 0.40 + 0.35 / (1.0 + (2.0 * t) ** 4)
            init_vals = np.array([_logit(v) for v in vals])
        else:  # uniform
            init_vals = np.zeros(n_eigen)

        self.filter_values = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))

    def forward(self, eigenvals=None):
        return apply_activation(self.filter_values, self.activation)

    def get_parameter_groups(self, config):
        return [{'params': [self.filter_values], 'name': 'direct_filter'}]

    def get_filter_values(self, n_points=None):
        with torch.no_grad():
            x = torch.linspace(0, 1, self.n_eigen)
            return x.numpy(), self.forward().cpu().numpy()


def create_filter(order=8, init_type='uniform', config=None):
    poly_basis = config.get('poly', 'bernstein') if config else 'bernstein'
    activation = config.get('f_act', 'sigmoid') if config else 'sigmoid'
    n_jitter = config.get('f_jitter', 0) if config else 0
    if poly_basis == 'direct':
        return DirectFilter(n_eigen=config.get('n_eigen', order), init_type=init_type, activation=activation)
    return APSFilter(filter_order=order, init_filter_name=init_type, poly_basis=poly_basis, activation=activation, n_jitter=n_jitter)
