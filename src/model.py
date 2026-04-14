import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import os
import pickle
from filter import create_filter
from utils import get_cache_prefix_and_suffix, format_beta_string, deep_copy_state_dict


class LearnSpecCF(nn.Module):

    def __init__(self, adj_mat, config, use_cache=True, split_seed=None, split_ratio=None, verbose=True):
        super().__init__()
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.dataset = config.get('dataset', 'unknown')
        self.verbose = verbose
        self.view = config.get('view', 'ui')

        if sp.issparse(adj_mat):
            self.adj_mat = adj_mat.tocsr()
        else:
            self.adj_mat = sp.csr_matrix(adj_mat)

        self.n_users, self.n_items = self.adj_mat.shape

        base_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cache'))
        self.cache_dir = os.path.join(base_cache_dir, self.dataset)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.u_eigen = config.get('u_eigen', 25)
        self.beta = config.get('beta', 0.5)
        self.i_eigen = config.get('i_eigen', 200)
        self.lr = config.get('lr', 0.01)
        self.decay = config.get('decay', 1e-5)

        f_order = config.get('f_order', 8)
        f_init = config.get('f_init', 'uniform')

        self.user_filter = self._create_filter(f_order, f_init, 'u') if 'u' in self.view else None
        self.item_filter = self._create_filter(f_order, f_init, 'i') if 'i' in self.view else None

        fusion_weights = []
        if 'u' in self.view: fusion_weights.append(1.0)
        if 'i' in self.view: fusion_weights.append(1.0)
        if len(fusion_weights) > 1:
            self.fusion_logits = nn.Parameter(torch.tensor(fusion_weights, dtype=torch.float32, device=self.device))

        self.use_cache = use_cache
        self.split_seed = split_seed
        self.split_ratio = split_ratio

        self._precompute_spectral_features()
        self._precompute_spectral_projections()

    def _precompute_spectral_projections(self):
        R = self.adj_mat.tocsr()
        R_csr = torch.sparse_csr_tensor(
            torch.tensor(R.indptr, dtype=torch.int64),
            torch.tensor(R.indices, dtype=torch.int64),
            torch.tensor(R.data, dtype=torch.float32),
            size=(self.n_users, self.n_items)).to(self.device)

        if 'u' in self.view and hasattr(self, 'user_eigenvecs'):
            R_coo = R_csr.to_sparse_coo()
            Rt = torch.sparse_coo_tensor(
                torch.stack([R_coo.indices()[1], R_coo.indices()[0]]),
                R_coo.values(), size=(self.n_items, self.n_users)).to_sparse_csr()
            spectral_R = (Rt @ self.user_eigenvecs).T
            self.register_buffer('user_spectral_R', spectral_R.contiguous())
            del R_coo, Rt
            if self.verbose:
                print(f"    Precomputed user spectral projection: U^T@R {spectral_R.shape}")

        if 'i' in self.view and hasattr(self, 'item_eigenvecs'):
            n_users = R_csr.shape[0]
            n_eigen = self.item_eigenvecs.shape[1]
            # Estimate memory: result is n_users * n_eigen * 4 bytes
            result_bytes = n_users * n_eigen * 4
            # Use chunked computation if result would exceed ~1 GiB
            if result_bytes > 1e9:
                # Pre-allocate result, fill column-chunks in-place to avoid peak memory spike
                col_chunk = max(1, int(1e9 / (n_users * 4)))
                spectral_R = torch.empty(n_users, n_eigen, dtype=torch.float32, device=self.device)
                for start in range(0, n_eigen, col_chunk):
                    end = min(start + col_chunk, n_eigen)
                    spectral_R[:, start:end] = R_csr @ self.item_eigenvecs[:, start:end]
                if self.verbose:
                    print(f"    Precomputed item spectral projection: R@V {spectral_R.shape} (chunked)")
            else:
                spectral_R = R_csr @ self.item_eigenvecs
                if self.verbose:
                    print(f"    Precomputed item spectral projection: R@V {spectral_R.shape}")
            self.register_buffer('item_spectral_R', spectral_R.contiguous())

        del R_csr

    def _create_filter(self, f_order, init_type, view='u'):
        view_config = self.config.copy()
        view_config['n_eigen'] = self.u_eigen if view == 'u' else self.i_eigen
        return create_filter(order=f_order, init_type=init_type, config=view_config)

    def _apply_f_drop(self, response):
        """Spectral dropout: randomly mask eigencomponents during training."""
        f_drop = self.config.get('f_drop', 0.0)
        if f_drop > 0 and self.training:
            mask = torch.bernoulli(torch.full_like(response, 1.0 - f_drop))
            return response * mask / (1.0 - f_drop)
        return response

    def get_user_spectral_filtering(self, users, target_items=None):
        user_in_matrix_mask = users < self.user_eigenvecs.shape[0]
        output_size = len(target_items) if target_items is not None else self.n_items
        user_filtered = torch.zeros(len(users), output_size, device=self.device)

        if user_in_matrix_mask.any() and self.user_filter is not None:
            valid_users = users[user_in_matrix_mask]
            batch_user_vecs = self.user_eigenvecs[valid_users]
            user_response = self._apply_f_drop(self.user_filter(self.user_eigenvals))
            weighted_vecs = batch_user_vecs * user_response.unsqueeze(0)

            if target_items is not None:
                result = weighted_vecs @ self.user_spectral_R[:, target_items]
            else:
                result = weighted_vecs @ self.user_spectral_R

            user_filtered[user_in_matrix_mask] = result

        return user_filtered

    def get_item_spectral_filtering(self, users, target_items=None):
        spectral_profiles = self.item_spectral_R[users]

        if self.item_filter is not None:
            item_response = self._apply_f_drop(self.item_filter(self.item_eigenvals))
            filtered = spectral_profiles * item_response
            if target_items is not None:
                return filtered @ self.item_eigenvecs[target_items].T
            return filtered @ self.item_eigenvecs.T

        if target_items is not None:
            return torch.zeros(len(users), len(target_items), device=self.device)
        return torch.zeros(len(users), self.n_items, device=self.device)

    def _fuse_views(self, view_embeddings):
        if len(view_embeddings) > 1 and hasattr(self, 'fusion_logits'):
            weights = torch.softmax(self.fusion_logits, dim=0)
            result = torch.zeros_like(view_embeddings[0])
            for i, embed in enumerate(view_embeddings):
                result += weights[i] * embed
            return result
        elif len(view_embeddings) > 1:
            return torch.stack(view_embeddings, dim=0).sum(dim=0)
        return view_embeddings[0]

    def forward(self, users):
        if not isinstance(users, torch.Tensor):
            users = torch.as_tensor(users, dtype=torch.long, device=self.device)
        else:
            users = users.to(self.device)

        views = []
        if 'u' in self.view:
            views.append(self.get_user_spectral_filtering(users))
        if 'i' in self.view:
            views.append(self.get_item_spectral_filtering(users))
        return self._fuse_views(views)

    def forward_selective(self, users, target_items):
        if not isinstance(users, torch.Tensor):
            users = torch.as_tensor(users, dtype=torch.long, device=self.device)
        else:
            users = users.to(self.device)
        if not isinstance(target_items, torch.Tensor):
            target_items = torch.as_tensor(target_items, dtype=torch.long, device=self.device)
        else:
            target_items = target_items.to(self.device)

        views = []
        if 'u' in self.view:
            views.append(self.get_user_spectral_filtering(users, target_items))
        if 'i' in self.view:
            views.append(self.get_item_spectral_filtering(users, target_items))
        return self._fuse_views(views)

    def _precompute_spectral_features(self):
        if 'u' in self.view:
            if not self._load_precomputed_eigen('u', self.u_eigen):
                print("FATAL: Required user eigendecomposition not found!")
                import sys; sys.exit(1)
        if 'i' in self.view:
            if not self._load_precomputed_eigen('i', self.i_eigen):
                print("FATAL: Required item eigendecomposition not found!")
                import sys; sys.exit(1)

    def _get_cache_prefix_and_suffix(self):
        return get_cache_prefix_and_suffix(self.split_seed, self.split_ratio)

    def _format_beta_string(self, view):
        return format_beta_string(self.beta)

    def _get_eigen_cache_config(self, view):
        cache_prefix, cache_suffix = self._get_cache_prefix_and_suffix()
        view_name = 'user' if view == 'u' else 'item'
        beta_str = self._format_beta_string(view)
        return {
            'cache_suffix': cache_suffix, 'cache_prefix': cache_prefix,
            'eigen_type': 'LM', 'view_name': view_name,
            'eigen_type_name': 'largestEigen', 'beta_str': beta_str
        }

    def _find_available_eigen_files(self, config, n_eigen):
        available_files = []
        eigen_dir = self.cache_dir
        expected_pattern = f"{config['cache_prefix']}{self.dataset}_{config['view_name']}_{config['eigen_type_name']}_n"

        if not os.path.exists(eigen_dir):
            return available_files

        valid_extensions = ('.npz', '.pkl')

        for filename in os.listdir(eigen_dir):
            if not filename.endswith(valid_extensions):
                continue

            if config['cache_prefix'] == "partial_":
                for ext in valid_extensions:
                    expected_full = f"{expected_pattern}{n_eigen}_degNorm_{config['beta_str']}{config['cache_suffix']}{ext}"
                    if filename == expected_full:
                        available_files.append((n_eigen, filename))
                        break
                else:
                    for ext in valid_extensions:
                        if (filename.startswith(expected_pattern) and
                            filename.endswith(f"_degNorm_{config['beta_str']}{config['cache_suffix']}{ext}")):
                            try:
                                pattern_end = len(expected_pattern)
                                degNorm_start = filename.find('_degNorm_')
                                if degNorm_start > pattern_end:
                                    file_n = int(filename[pattern_end:degNorm_start])
                                    if file_n >= n_eigen:
                                        available_files.append((file_n, filename))
                            except (ValueError, IndexError):
                                continue
                            break
            else:
                for ext in valid_extensions:
                    if filename.startswith(expected_pattern) and filename.endswith(f"_degNorm_{config['beta_str']}{ext}"):
                        try:
                            remaining = filename[len(expected_pattern):]
                            parts = remaining.split('_')
                            if len(parts) >= 3 and parts[1] == 'degNorm':
                                file_n = int(parts[0])
                                file_beta = parts[2].replace(ext, '')
                                if file_beta == config['beta_str'] and file_n >= n_eigen:
                                    available_files.append((file_n, filename))
                        except (ValueError, IndexError):
                            continue
                        break

        return available_files

    def _show_eigen_file_not_found_error(self, config, n_eigen):
        if not self.verbose:
            return
        cache_type = "partial" if config['cache_prefix'] == "partial_" else "full"
        pattern = f"{config['cache_prefix']}{self.dataset}_{config['view_name']}_{config['eigen_type_name']}_n"
        print(f"No {cache_type} {config['view_name']}-view eigen found for n>={n_eigen} beta={config['beta_str'].replace('p', '.')}")
        print(f"   Expected: {pattern}*_degNorm_{config['beta_str']}.npz")
        if os.path.exists(self.cache_dir):
            files = [f for f in os.listdir(self.cache_dir) if 'eigen' in f.lower()]
            for f in sorted(files):
                print(f"   - {f}")

    def _load_and_process_eigen_data(self, filepath, view, n_eigen):
        try:
            if filepath.endswith('.npz'):
                npz = np.load(filepath)
                eigen_data = {
                    'dataset': str(npz['dataset']), 'view': str(npz['view']),
                    'eigenvals': npz['eigenvals'], 'eigenvecs': npz['eigenvecs'],
                    'which': str(npz['which']) if 'which' in npz else 'LM',
                }
                if self.split_seed is not None and 'split_seed' in npz:
                    file_seed = int(npz['split_seed'])
                    file_ratio = float(npz['split_ratio'])
                    assert file_seed == self.split_seed and abs(file_ratio - self.split_ratio) < 1e-6, (
                        f"Split mismatch! File: seed={file_seed}, ratio={file_ratio}. "
                        f"Expected: seed={self.split_seed}, ratio={self.split_ratio}. "
                        f"Regenerate with: python precompute_eigen.py --dataset {self.dataset} "
                        f"--split_ratio {self.split_ratio} --seed {self.split_seed}")
            else:
                with open(filepath, 'rb') as f:
                    eigen_data = pickle.load(f)
                if self.split_seed is not None and 'split_seed' in eigen_data:
                    file_seed = eigen_data['split_seed']
                    file_ratio = eigen_data['split_ratio']
                    assert file_seed == self.split_seed and abs(file_ratio - self.split_ratio) < 1e-6, (
                        f"Split mismatch! File: seed={file_seed}, ratio={file_ratio}. "
                        f"Expected: seed={self.split_seed}, ratio={self.split_ratio}.")

            if (eigen_data.get('dataset') != self.dataset or
                eigen_data.get('view') != view or
                eigen_data.get('eigenvals') is None or
                eigen_data.get('eigenvecs') is None):
                if self.verbose:
                    print(f"  Invalid eigen file: {os.path.basename(filepath)}")
                return False

            all_eigenvals = eigen_data['eigenvals']
            all_eigenvecs = eigen_data['eigenvecs']
            which_type = eigen_data.get('which', 'LM')

            if which_type == 'LM':
                if len(all_eigenvals) > 1 and all_eigenvals[0] >= all_eigenvals[-1]:
                    eigenvals = all_eigenvals[:n_eigen]
                    eigenvecs = all_eigenvecs[:, :n_eigen]
                else:
                    eigenvals = all_eigenvals[-n_eigen:]
                    eigenvecs = all_eigenvecs[:, -n_eigen:]
            else:
                if len(all_eigenvals) > 1 and all_eigenvals[0] <= all_eigenvals[-1]:
                    eigenvals = all_eigenvals[:n_eigen]
                    eigenvecs = all_eigenvecs[:, :n_eigen]
                else:
                    eigenvals = all_eigenvals[-n_eigen:]
                    eigenvecs = all_eigenvecs[:, -n_eigen:]

            if view == 'u':
                self.register_buffer('user_eigenvals', torch.tensor(eigenvals, dtype=torch.float32).to(self.device))
                self.register_buffer('user_eigenvecs', torch.tensor(eigenvecs, dtype=torch.float32).to(self.device))
            elif view == 'i':
                self.register_buffer('item_eigenvals', torch.tensor(eigenvals, dtype=torch.float32).to(self.device))
                self.register_buffer('item_eigenvecs', torch.tensor(eigenvecs, dtype=torch.float32).to(self.device))

            return True

        except Exception as e:
            if self.verbose:
                print(f"  Failed to load {os.path.basename(filepath)}: {e}")
            return False

    def _load_precomputed_eigen(self, view, n_eigen):
        if not self.use_cache:
            return False

        config = self._get_eigen_cache_config(view)
        available_files = self._find_available_eigen_files(config, n_eigen)

        if not available_files:
            self._show_eigen_file_not_found_error(config, n_eigen)
            import sys; sys.exit(1)

        available_files.sort(key=lambda x: x[0])
        file_n_eigen, filename = available_files[0]
        filepath = os.path.join(self.cache_dir, filename)

        if self.verbose:
            print(f"Loading {view}-view: {filename} ({n_eigen}/{file_n_eigen})")

        return self._load_and_process_eigen_data(filepath, view, n_eigen)

    def get_optimizer_groups(self):
        param_groups = []

        for prefix, filt in [('user', self.user_filter), ('item', self.item_filter)]:
            if filt is None:
                continue
            if hasattr(filt, 'get_parameter_groups'):
                groups = filt.get_parameter_groups(self.config)
                for g in groups:
                    g['lr'] = self.lr
                    g['weight_decay'] = self.decay
                    g['name'] = f"{prefix}_{g.get('name', 'filter')}"
                param_groups.extend(groups)
            else:
                param_groups.append({
                    'params': filt.parameters(), 'lr': self.lr,
                    'weight_decay': self.decay, 'name': f'{prefix}_filter'
                })

        if hasattr(self, 'fusion_logits'):
            param_groups.append({
                'params': [self.fusion_logits], 'lr': self.lr,
                'weight_decay': self.decay, 'name': 'fusion_weights'
            })

        return param_groups

    def get_filter_snapshot(self):
        snapshot = {}
        if self.user_filter is not None:
            snapshot['user_filter'] = deep_copy_state_dict(self.user_filter.state_dict())
        if self.item_filter is not None:
            snapshot['item_filter'] = deep_copy_state_dict(self.item_filter.state_dict())
        if hasattr(self, 'fusion_logits'):
            snapshot['fusion_logits'] = self.fusion_logits.data.clone().detach()
        return snapshot

    def load_filter_snapshot(self, snapshot):
        if snapshot is None:
            return
        if 'user_filter' in snapshot and self.user_filter is not None:
            self.user_filter.load_state_dict(snapshot['user_filter'])
        if 'item_filter' in snapshot and self.item_filter is not None:
            self.item_filter.load_state_dict(snapshot['item_filter'])
        if 'fusion_logits' in snapshot and hasattr(self, 'fusion_logits'):
            self.fusion_logits.data.copy_(snapshot['fusion_logits'])

    def getUsersRating(self, users):
        return self.forward(users)
