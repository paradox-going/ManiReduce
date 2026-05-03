import numpy as np
import torch as th
import torch.nn as nn
from time import time

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.manifold._utils import _binary_search_perplexity

MACHINE_EPSILON = np.finfo(np.double).eps


def init_points_low(k: int, n: int = 3, eps: float = 1e-6, seed: int = None) -> th.Tensor:

    if seed is not None:
        th.manual_seed(seed)

    # Sample random lower-triangular matrices and form covariance matrices
    L = th.tril(th.randn(k, n, n, dtype=th.float64))
    diag_idx = th.arange(n)
    L[:, diag_idx, diag_idx] = th.abs(L[:, diag_idx, diag_idx]) + 0.1

    cov = L @ L.transpose(1, 2)  # (k, n, n)

    # Normalize to obtain Correlation matrices
    diag_std = th.sqrt(th.diagonal(cov, dim1=1, dim2=2))  # (k, n)
    corr = cov / (diag_std.unsqueeze(2) * diag_std.unsqueeze(1))

    # Numerical regularization and scale damping
    eye = th.eye(n).expand(k, n, n)
    corr = (1 - eps) * corr + eps * eye
    corr = corr * 0.01

    # Enforce unit diagonal
    corr[:, diag_idx, diag_idx] = 1.0

    return corr


def make_LT0_param(C: th.Tensor) -> th.Tensor:

    return th.tril(th.linalg.cholesky(C), diagonal=-1)


def LT0_to_LT1(raw: th.Tensor) -> th.Tensor:

    k = raw.shape[0]
    n = raw.shape[1]
    eye = th.eye(n).expand(k, n, n)
    return eye + th.tril(raw, diagonal=-1)


def chol_to_C(Gamma: th.Tensor) -> th.Tensor:

    A = Gamma @ th.transpose(Gamma, -1, -2)
    d = th.sqrt(th.diagonal(A, dim1=1, dim2=2))
    C = A / (d[:, :, None] * d[:, None, :])
    return C


class Cor_SNE:

    @_deprecate_positional_args
    def __init__(
        self,
        max_time: int = 1200,
        learning_rate: float = 10.0,
        max_iter: int = 1000,
        t_gamma: float = 1.0,
        verbose: int = 0,
        manifold=None,
        perplexity: int = None,
    ):
        self.t_gamma = t_gamma
        self.learning_rate = learning_rate
        self.manifold = manifold
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_time = max_time
        self.perplexity = perplexity

        # Internal state
        self.loss_fun = []
        self.res_opti = None
        self.initial_point = None
        self.dist = None

        self.alpha = None



    def _compute_high_dim_P(self, X: np.ndarray):

        X_th = th.from_numpy(X)
        n = X_th.shape[0]

        dist = self.manifold.poly_hyperbolic_Cholesky_distance(
            X_th, alpha=self.alpha, squared=False
        ).numpy()
        dist = (1 - np.eye(n)) * dist

        conditional_P = th.Tensor(
            _binary_search_perplexity(
                (dist ** 2).astype(np.float32), self.perplexity, 0
            )
        )
        P = (conditional_P + conditional_P.T) / (2 * n)
        return P, dist

    def _compute_low_dim_Q(self, Y: th.Tensor):

        n = Y.shape[0]
        dist = self.manifold.poly_hyperbolic_Cholesky_distance(
            Y, alpha=self.alpha, squared=False
        )
        dist = (1 - th.eye(n)) * dist

        # Cauchy kernel: t_gamma / (t_gamma^2 + dist^2)
        kernel = self.t_gamma / (self.t_gamma ** 2 + dist ** 2)
        # Exclude diagonal when computing the normalizing constant
        diag_correction = self.t_gamma / (self.t_gamma ** 2 + dist.diagonal() ** 2)
        denominator = kernel.sum() - diag_correction.sum()

        Q = (1 - th.eye(n)) * kernel / denominator
        return Q, dist

    def _kl_divergence(self, P: th.Tensor, Q: th.Tensor) -> th.Tensor:
        eye = th.eye(P.shape[0])
        return th.sum(P * th.log((P + eye) / (Q + eye)))


    def _run_minimization(self, P: th.Tensor) -> np.ndarray:
        # Convert initial point to LT0 parameterization
        L1_param = self.manifold.correlation_to_LT1(self.initial_point.clone().detach())
        raw_param = nn.Parameter(self.manifold.LT1_to_LT0(L1_param))

        optimizer = th.optim.Adam([raw_param], lr=self.learning_rate)

        initial_time = time()
        start = initial_time
        best_loss = float('inf')
        best_iter = -1
        n_iter_without_progress = 200
        iter_interval = 10

        for step in range(self.max_iter):
            optimizer.zero_grad()

            # Reconstruct Correlation matrices from LT0 parameters
            Gamma = self.manifold.LT0_to_LT1(raw_param)
            C = self.manifold.LT1_to_correlation(Gamma)

            Q, _ = self._compute_low_dim_Q(C)
            loss = self._kl_divergence(P, Q)
            self.loss_fun.append(loss.item())

            loss.backward()
            optimizer.step()

            # Logging and early stopping
            if (step + 1) % iter_interval == 0:
                duration = time() - start
                start = time()

                if self.verbose >= 2:
                    print(
                        f"Iteration {step}: Loss = {self.loss_fun[-1]:.7f} "
                        f"({iter_interval} iterations in {duration:.3f}s)"
                    )

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_iter = step
                elif step - best_iter > n_iter_without_progress:
                    if self.verbose >= 2:
                        print(
                            f"Iteration {step + 1}: no progress in last "
                            f"{n_iter_without_progress} iterations. Finished."
                        )
                    break

            if time() - initial_time >= self.max_time:
                print(f"Time limit reached after {step} iterations.")
                break

        if self.verbose >= 2:
            print(f"KL divergence after {step} iterations: {self.loss_fun[-1]:.7f}")

        return self.manifold.LT1_to_correlation(
            self.manifold.LT0_to_LT1(raw_param)
        ).detach().numpy()


    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit the Cor-SNE model and return low-dimensional embeddings.

        Args:
            X: Array of shape (n, d, d) containing high-dimensional Correlation matrices.

        Returns:
            Array of shape (n, 3, 3) containing 3x3 Correlation matrix embeddings.
        """
        n_samples = X.shape[0]

        # Set defaults for unspecified hyperparameters
        if self.initial_point is None:
            self.initial_point = init_points_low(n_samples, n=3, seed=44)

        if self.perplexity is None:
            self.perplexity = int(0.75 * n_samples)

        if self.alpha is None:
            self.alpha = th.ones(X.shape[-1] - 1)

        # Compute high-dimensional similarities
        P, dist = self._compute_high_dim_P(X)
        self.dist = dist

        # Optimize low-dimensional embeddings
        self.res_opti = self._run_minimization(P)
        return self.res_opti