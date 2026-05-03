import numpy as np
import torch as th
import torch.nn as nn
from time import time

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.manifold._utils import _binary_search_perplexity

MACHINE_EPSILON = np.finfo(np.double).eps

def init_points_low(
    n: int,
    k: int,
    p: int,
    q: int,
    seed: int = 42,
) -> th.Tensor:

    if seed is not None:
        th.manual_seed(seed)

    U = th.randn(n, k, q)  # (n, k, q)
    V = th.randn(n, p, q)  # (n, p, q)

    # Low-rank factorization: F = U @ V^T, shape (n, k, p)
    F = th.matmul(U, V.transpose(-1, -2)) * 0.01
    return F


class SPSD_SNE:

    @_deprecate_positional_args
    def __init__(
        self,
        max_time: int = 600,
        learning_rate: float = 0.01,
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
        self.initial_param = None
        self.dist = None


    def _compute_high_dim_P(self, X: np.ndarray):

        X_th = th.from_numpy(X)
        n = X_th.shape[0]

        dist = self.manifold.spsd_wasserstein_distance(X_th, squared=False).numpy()
        dist = dist * (1 - np.eye(n))

        conditional_P = th.Tensor(
            _binary_search_perplexity(
                (dist ** 2).astype(np.float32), self.perplexity, 0
            )
        )
        P = (conditional_P + conditional_P.T) / (2 * n)
        return P, dist

    def _compute_low_dim_Q(self, Y: th.Tensor):

        n = Y.shape[0]
        dist = self.manifold.spsd_wasserstein_distance(Y, squared=False)
        dist = dist * (1 - th.eye(n))

        # Cauchy kernel: t_gamma / (t_gamma^2 + dist^2)
        kernel = self.t_gamma / (self.t_gamma ** 2 + dist ** 2)
        # Exclude diagonal entries from the normalizing constant
        diag_correction = self.t_gamma / (self.t_gamma ** 2 + dist.diagonal() ** 2)
        denominator = kernel.sum() - diag_correction.sum()

        Q = (1 - th.eye(n)) * kernel / denominator
        return Q, dist

    def _kl_divergence(self, P: th.Tensor, Q: th.Tensor) -> th.Tensor:
        eye = th.eye(P.shape[0])
        return th.sum(P * th.log((P + eye) / (Q + eye)))

    def _run_minimization(self, P: th.Tensor) -> np.ndarray:

        raw_param = nn.Parameter(self.initial_param.clone().detach())

        optimizer = th.optim.Adam([raw_param], lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )

        initial_time = time()
        start = initial_time
        best_loss = float('inf')
        best_iter = -1
        n_iter_without_progress = 200
        iter_interval = 10

        for step in range(self.max_iter):
            optimizer.zero_grad()

            # Reconstruct SPSD matrix from factor: S = F @ F^T
            cur_point = raw_param @ raw_param.transpose(-1, -2)

            Q, _ = self._compute_low_dim_Q(cur_point)
            loss = self._kl_divergence(P, Q)
            self.loss_fun.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

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

        # Return final SPSD matrices reconstructed from optimized factors
        F = raw_param.detach()
        return (F @ F.transpose(-1, -2)).numpy()


    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit the SPSD-SNE model and return low-dimensional embeddings.

        Args:
            X: Array of shape (n, d, d) containing high-dimensional SPSD matrices.

        Returns:
            Array of shape (n, 2, 2) containing 2x2 SPSD matrix embeddings.
        """
        n_samples = X.shape[0]

        # Set defaults for unspecified hyperparameters
        if self.initial_param is None:
            self.initial_param = init_points_low(n_samples, k=2, p=1, q=1, seed=42)

        if self.perplexity is None:
            self.perplexity = int(0.5 * n_samples)

        # Compute high-dimensional similarities
        P, dist = self._compute_high_dim_P(X)
        self.dist = dist

        # Optimize low-dimensional embeddings
        self.res_opti = self._run_minimization(P)
        return self.res_opti