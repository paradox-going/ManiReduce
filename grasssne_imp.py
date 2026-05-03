import numpy as np
import torch as th
from time import time

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.manifold._utils import _binary_search_perplexity

MACHINE_EPSILON = np.finfo(np.double).eps


def init_points_low(count: int, seed: int = None) -> np.ndarray:

    if seed is not None:
        np.random.seed(seed)

    alpha = 0.1 * np.random.rand(count) * 2 * np.pi
    beta = 0.1 * np.random.rand(count) * 0.5 * np.pi

    x = np.cos(alpha) * np.cos(beta)
    y = np.sin(alpha) * np.cos(beta)
    z = np.sin(beta)

    points = np.array([x, y, z]).T           # (count, 3)
    return np.expand_dims(points, axis=-1)    # (count, 3, 1)


class Grass_SNE:
    @_deprecate_positional_args
    def __init__(
        self,
        max_time: int = 1200,
        learning_rate: float = 1.0,
        max_iter: int = 10000,
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
        self.dist_h = None
        self.dist_l = None


    def _compute_high_dim_P(self, X: np.ndarray):

        X_th = th.from_numpy(X)
        n = X_th.shape[0]

        dist = self.manifold.distance(X_th).numpy()
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
        dist = self.manifold.distance(Y)
        dist = (1 - th.eye(n)) * dist

        kernel = self.t_gamma / (self.t_gamma ** 2 + dist ** 2)
        # Exclude diagonal entries from the normalizing constant
        diag_correction = self.t_gamma / (self.t_gamma ** 2 + dist.diagonal() ** 2)
        denominator = kernel.sum() - diag_correction.sum()

        Q = (1 - th.eye(n)) * kernel / denominator
        return Q, dist

    def _kl_divergence(self, P: th.Tensor, Q: th.Tensor) -> th.Tensor:

        eye = th.eye(P.shape[0])
        return th.sum(P * th.log((P + eye) / (Q + eye)))

    def _armijo_line_search(
        self,
        current_sol: th.Tensor,
        rie_grad: th.Tensor,
        loss: th.Tensor,
        grad_norm: float,
        P: th.Tensor,
        alpha_init: float,
    ):

        tau = 0.5    # Backtracking factor
        r = 1e-4     # Sufficient decrease constant
        max_ls_iter = 25

        alpha = alpha_init
        retracted = None

        for _ in range(max_ls_iter):
            retracted = self.manifold.expmap(current_sol, -alpha * rie_grad)
            Q_new, _ = self._compute_low_dim_Q(retracted)
            loss_new = self._kl_divergence(P, Q_new).item()

            if loss.item() - loss_new > r * alpha * grad_norm ** 2:
                return retracted, alpha, True
            alpha *= tau

        return retracted, alpha, False



    def _run_minimization(self, P: th.Tensor) -> np.ndarray:
        current_sol = th.tensor(
            self.initial_point, requires_grad=True, dtype=th.float64
        )

        initial_time = time()
        start = initial_time
        best_loss = float('inf')
        best_iter = -1
        n_iter_without_progress = 100
        min_grad_norm = 1e-5
        iter_interval = 10
        alpha = self.learning_rate

        for step in range(self.max_iter):
            if current_sol.grad is not None:
                current_sol.grad.zero_()

            Q, _ = self._compute_low_dim_Q(current_sol)
            loss = self._kl_divergence(P, Q)
            self.loss_fun.append(loss.item())

            loss.backward()

            grad = current_sol.grad
            rie_grad = self.manifold.grassman_grad(current_sol, grad)
            grad_norm = self.manifold.grassmann_grad_norm(current_sol, grad)

            # Step size initialization
            if step == 0:
                alpha = 1.0 / grad_norm
            else:
                alpha = 4 * (self.loss_fun[-2] - loss.item()) / (grad_norm ** 2)

            # Armijo backtracking line search
            retracted, alpha, success = self._armijo_line_search(
                current_sol, rie_grad, loss, grad_norm, P, alpha
            )

            if not success and self.verbose >= 2:
                print(
                    f"Step {step}: line search did not converge, "
                    f"proceeding with alpha = {alpha:.6f}"
                )

            with th.no_grad():
                current_sol.copy_(retracted)

            # Logging and early stopping
            if (step + 1) % iter_interval == 0:
                duration = time() - start
                start = time()

                if self.verbose >= 2:
                    print(
                        f"Iteration {step}: Loss = {self.loss_fun[-1]:.7f}, "
                        f"grad norm = {grad_norm:.7f}, lr = {alpha:.7f} "
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

                if grad_norm <= min_grad_norm:
                    if self.verbose >= 2:
                        print(
                            f"Iteration {step + 1}: gradient norm {grad_norm:.7f}. Finished."
                        )
                    break

            if time() - initial_time >= self.max_time:
                print(f"Time limit reached after {step} iterations.")
                break

        if self.verbose >= 2:
            print(f"KL divergence after {step} iterations: {self.loss_fun[-1]:.7f}")

        return current_sol.detach().numpy()


    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit the Grass-SNE model and return low-dimensional embeddings.

        Args:
            X: Array of shape (n, d, q) containing high-dimensional Grassmannian points.

        Returns:
            Array of shape (n, 3, 1) containing low-dimensional Gr(3,1) embeddings.
        """
        n_samples = X.shape[0]

        # Set defaults for unspecified hyperparameters
        if self.initial_point is None:
            self.initial_point = init_points_low(n_samples, seed=42)

        if self.perplexity is None:
            self.perplexity = int(0.75 * n_samples)

        # Compute high-dimensional similarities
        P, dist = self._compute_high_dim_P(X)
        self.dist_h = dist

        # Optimize low-dimensional embeddings
        self.res_opti = self._run_minimization(P)
        return self.res_opti