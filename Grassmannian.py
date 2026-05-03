import torch as th
import numpy as np
from geoopt.manifolds import Manifold

from typing import Optional, Tuple, Union
from .utilities import aux_svd_logmap, gr_identity_batch

saved_grads = {}


class Mymanifold(Manifold):
    """  Base class: Mymanifold """

    def retr(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        pass

    name = "Mymanifold"

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def tensors_are_close(self, tensor1, tensor2, eps=None):
        """ Check if two tensors are the same within a specified tolerance. """
        if eps is None:
            eps = self.eps
        return th.allclose(tensor1, tensor2, atol=eps, rtol=0)


class Grassmannian(Mymanifold):
    """
    Stiefle perspective of the Grassmannian Computation for Grassmannian raw with size of [...,n,p]:
        [Bendokat,2024] A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects
    """
    # __scaling__ = Manifold.__scaling__.copy()
    name = "Grassmannian ONB perspective"
    ndim = 2
    reversible = False

    def __init__(self, n, p, eps=1e-8):
        super().__init__(eps=eps)
        self.n = n
        self.p = p
        self.register_buffer('identity', gr_identity_batch(n, p))

    # QR分解
    def qr_flipped(self, X):
        """ Compute qr with flipping, which might be useful for standard Riem log."""
        Q, R = th.linalg.qr(X)
        # flipping
        output = th.matmul(Q, th.diag_embed(th.sign(th.sign(th.diagonal(R, dim1=-2, dim2=-1)) + 0.5)))
        return output

    @staticmethod
    # 在 Grassmannian 流形上生成一个基于均值的随机点
    def random(*shape, mean=None, std=0.1):
        """ Generates a random point on the Grassmannian manifold with dimensions [..., n, p]. """
        *leading_dims, n, p = shape

        if mean is None:
            # np.random.randn(42)
            # 生成一个指定形状的随机矩阵
            random_matrix = np.random.randn(*shape)
            # 对随机矩阵进行QR分解
            mean, _ = np.linalg.qr(random_matrix)

        # Ensure p is not greater than n
        if p >= n:
            raise ValueError("p should not be greater than n.")

        shape = (*leading_dims, n, p) if leading_dims else (n, p)

        # 围绕均值生成随机点
        random_matrix = mean + np.random.randn(*shape) * std
        # 使用奇异值分解（SVD）对随机点正交化处理
        u, s, vt = np.linalg.svd(random_matrix, full_matrices=False)
        q = u @ vt

        return q

    @staticmethod
    def grassmann_random_sample(mean_Q, std=0.1):
        n, p = mean_Q.shape
        Z = np.random.randn(n, p) * std
        Delta = Z - mean_Q @ (mean_Q.T @ Z)
        U, S, Vt = np.linalg.svd(Delta, full_matrices=False)
        new_point = mean_Q @ Vt.T @ np.diag(np.cos(S)) @ Vt + U @ np.diag(np.sin(S)) @ Vt
        return new_point



    def logmap_standard(self, x, y):
        """
            Perform a logarithmic map :math:`\operatorname{Log}_{x}(y)`.
            Note that z=self.expmap(x,self.log(x,y)), then z != y, but self.dist(z,y) is almost 0
        """
        ytx = y.transpose(-1, -2).matmul(x)
        At = y.transpose(-1, -2).subtract(ytx.matmul(x.transpose(-1, -2)))
        Bt = th.linalg.pinv(ytx).matmul(At)
        # Bt = th.linalg.solve(ytx,At)
        u, s, vh = th.linalg.svd(Bt.transpose(-1, -2), full_matrices=False)
        s_atan = th.atan(s)

        return u.mul(s_atan.unsqueeze(-2)).matmul(vh)

    # 黎曼对数映射
    def logmap(self, U0, U1, is_return_tuple=False):
        """
        Compute the Grassmann Riemannian logarithm \riemlog_{U0}(U1) by single svd [Alg. 5.3, Bendokat,2024].
        Parameters:
            U0, U1: Stiefel representatives of subspaces, shape = [..., n, p]
            is_return_tuple: needed for efficient geodesic
        intermediate:
            U1star: Adapted Stiefel representative of U1
        Returns:
            Delta: Tangent vector in horizontal space at U0, from U0 to U1star
            or U, arctan(\Sigma), Vh with [...,n,p], [...,p] and [...,n,p]
        Note that
            1. Different from the offcicial Matlab code, we solve several singular cases for the forward computation
            2. U might fail to be orthonormal, if arctan(\Sigma_i) = 0, but this do not affect the computation of geodeisc
            3. This code might fail to deal with the BP under U0=U1, but we don't need it.
            4. We deal with the BP of the case of (\bar{\Sigma_i}) = 0
            5. In the calculation of geodesic, arctan(\Sigma_i) = 0 might affect BP of singvals, but we don't need it
            :param is_return_tuple:
            :param U0:
            :param U1:
        """
        # Check if U1 and U0 are essentially the same
        if th.allclose(U0, U1, atol=self.eps):
            # If singvals is zero, then U1 and U0 are the same and Delta should be zero
            # this case might undermine the auto differentiation, but we do not need to care about this issue currently.
            if is_return_tuple:
                n, p = U1.shape[-2], U1.shape[-1]
                asin_singvals = th.zeros_like(U1)[..., -1, :] if len(U1.shape) > 2 else th.zeros_like(U1[-1, :])
                Q_hat = th.eye(n, p, dtype=U0.dtype, device=U0.device)
                R1_vh = th.eye(p, p, dtype=U0.dtype, device=U0.device)
                return Q_hat, asin_singvals, R1_vh
            else:
                # note that we use U1 here, as U0 might be [c,n,p]
                return th.zeros_like(U1)
        else:
            # Step 1: Procrustes, svd is efficiently calculated on q \times q matrices
            Q1_ascending, S1_ascending, R1_vh_ascending = aux_svd_logmap(th.matmul(U1.transpose(-2, -1), U0))

            # Calculate new rep
            U1star = th.matmul(U1, Q1_ascending)
            # Step 3: SVD without actual SVD
            H = U1star - th.matmul(U0, th.matmul(U0.transpose(-2, -1), U1star))
            singvals = th.sqrt(1 - S1_ascending ** 2)
            # this is the sigma of \delta_U0
            asin_singvals = th.asin(singvals)

            # Resolve 0/0 ambiguity by ensuring orthogonality of Q2
            # note  that it is okay for singvals approching 0, but cannot be 0
            # RHS to make sure when 0/0 happens, we should have \partial{L} / \partial{sigma_i} = \partial{L} / \partial{th.asin(singvals)}
            asin_div_singvals = th.where(singvals != 0, asin_singvals / singvals, asin_singvals)
            # Step 3: Return tuple or tangent vector
            if is_return_tuple:
                singvals_expanded = singvals.unsqueeze(-2)  # Adds a new dimension to match H's shape for division
                condition = singvals_expanded != 0
                # Note that in geodesic, this might affect BP of singvals, which is not our case
                Q_hat = th.where(condition, H.div(singvals_expanded), H)
                return Q_hat, asin_singvals, R1_vh_ascending
            else:
                Delta = th.matmul(th.mul(H, asin_div_singvals.unsqueeze(-2)), R1_vh_ascending)
                return Delta

    # 指数映射
    def expmap(self, U0, Delta):
        """
        Compute the Grassmann exponential \rieexp_{U0}(Delta).
        """
        Q, Sigma, Vh = th.linalg.svd(Delta, full_matrices=False)
        cosSigma = th.cos(Sigma).unsqueeze(-2)
        sinSigma = th.sin(Sigma).unsqueeze(-2)
        U1 = (U0.matmul(Vh.transpose(-2, -1)).mul(cosSigma) + Q.mul(sinSigma)).matmul(Vh)
        # Our exp on log_exp indicates this is minor. But manopt indicates that re-orth might be important, possiblily for optimization
        return self.qr_flipped(U1)
        # return U1

    @staticmethod
    def principal_angle(U0, U1=None):
        """
        Compute the principal angles between two sets of orthonormal basis matrices U0 and U1.
        U0: Tensor of shape (N, d, k)
        U1: Tensor of shape (N, d, k) or None (if None, computes pairwise within U0)
        Returns:
            principal_angle: Tensor of shape (N, N, k) if U1 is None, else (N, k)
        """
        eps = 1e-6

        if U1 is None:
            # Pairwise computation within U0
            N = U0.shape[0]
            d, k = U0.shape[1:]
            # (N, 1, d, k) @ (1, N, d, k) -> (N, N, k, k)
            U0_t = U0.transpose(-1, -2).unsqueeze(1)  # (N, 1, k, d)
            U1_expand = U0.unsqueeze(0)  # (1, N, d, k)
            prod = th.matmul(U0_t, U1_expand)  # (N, N, k, k)
            # Flatten for batch SVD
            prod_flat = prod.reshape(-1, k, k)  # (N*N, k, k)
            prod_flat = prod_flat + 1e-6 * th.eye(k, device=prod_flat.device)  # 正则化
            _, S, _ = th.linalg.svd(prod_flat)  # S: (N*N, k)
            S_clamped = th.clamp(S, -1 + eps, 1 - eps)
            angles = th.acos(S_clamped)  # (N*N, k)
            angles = angles.reshape(N, N, k)  # (N, N, k)
            # Make symmetric
            angles = (angles + angles.transpose(0, 1)) / 2
            return angles

        else:
            # Elementwise computation between U0 and U1
            U0_t = U0.transpose(-1, -2)  # (N, k, d)
            prod = th.matmul(U0_t, U1)  # (N, k, k)
            _, S, _ = th.linalg.svd(prod)  # (N, k)
            S_clamped = th.clamp(S, -1 + eps, 1 - eps)
            angles = th.acos(S_clamped)  # (N, k)
            return angles

    @classmethod
    def distance(cls, x, y=None, squared=False):
        """ geodesic distance between x,y with input [...,n,p] and output [...] """
        xy_pangle = cls.principal_angle(x, y)
        return th.norm(xy_pangle, dim=-1)

    def distance_Asimov(self, x, y=None):
        dist, _ = th.max(self.principal_angle(x, y), dim=-1)
        return dist
    #
    # def distance_Binet_Cauchy(self, x, y=None):
    #     cos_theta_prod = th.prod(th.cos(self.principal_angle(x, y)), dim=-1)
    #     cos_theta_prod = th.clamp(cos_theta_prod, -1 + 1e-6, 1 - 1e-6)
    #     return (1-cos_theta_prod)**0.5

    def distance_Binet_Cauchy(self, x, y=None):
        cos_theta = th.cos(self.principal_angle(x, y))  # 形状 (..., k)
        cos_sq_prod = th.prod(cos_theta ** 2, dim=-1)  # 连乘余弦平方
        cos_sq_prod = th.clamp(cos_sq_prod, 0, 1 - 1e-6)  # 限制在[0, 1)
        return th.sqrt(1 - cos_sq_prod)  # 标准定义

    def distance_Chordal(self, x, y=None):
        theta = self.principal_angle(x, y)
        theta = theta.clamp(1e-6, th.pi / 2 - 1e-6)  # 避免边界值
        return th.norm(th.sin(theta), dim=-1)

    def distance_Fubini_Study(self, x, y=None):
        cos_theta_prod = th.prod(th.cos(self.principal_angle(x, y)), dim=-1)
        cos_theta_prod = th.clamp(cos_theta_prod, -1 + 1e-6, 1 - 1e-6)  # 避免arccos边界
        return th.acos(cos_theta_prod)  # 正确距离度量

    def distance_Martin(self, x, y=None):
        cos_theta = th.cos(self.principal_angle(x, y))  # 形状 (..., k)
        cos_theta = th.clamp(cos_theta, min=1e-6, max=1 - 1e-6)  # 避免边界值
        log_cos_sq = -2 * th.log(cos_theta)  # 计算 -2*log(cos θ)
        return th.sqrt(th.sum(log_cos_sq, dim=-1))  # 开方求和

    def distance_Procrustes(self, x, y=None):
        # 计算角度并裁剪
        angles = self.principal_angle(x, y)
        angles = angles.clamp(max=3.1415926 - 1e-6)

        # 稳定计算
        sin_half_theta = th.sin(angles / 2)
        sin_half_theta = sin_half_theta.clamp(min=1e-6)  # 避免零梯度

        return 2 * th.norm(sin_half_theta, dim=-1)

    def distance_Projection(self, x, y=None):
        dist, _ = th.max(self.principal_angle(x, y), dim=-1)
        return th.sin(dist)

    def distance_Spectral(self, x, y=None):
        dist, _ = th.max(self.principal_angle(x, y), dim=-1)
        dist = 2 * th.sin(dist / 2)
        return dist

    def geodesic(self, U0, U1, t):
        """ The geodesic from U0 to U1 with single svd"""
        # Note that we don't re-orth Q_i when arctan(\Sigma_i) = 0, but this does not affect the computation of geodesic
        # this might cause problem in BP, but we don't need it
        Q, simga, Vh = self.logmap(U0, U1, is_return_tuple=True)
        singma_new = t * simga
        costSigma = th.cos(singma_new).unsqueeze(-2)
        sintSigma = th.sin(singma_new).unsqueeze(-2)
        U_new = (U0.matmul(Vh.transpose(-2, -1)).mul(costSigma) + Q.mul(sintSigma)).matmul(Vh)
        return U_new

    # 检查给定的张量 x 是否位于 Grassmannian 流形上
    def PP2ONB(self, P):
        S, U = th.linalg.eigh(P)
        S_desc, indices = th.sort(S, descending=True)

        # Sort eigenvectors to correspond to sorted eigenvalues
        U_desc = th.gather(U, -1, indices.unsqueeze(-2).expand_as(U))

        output = U_desc[..., :self.subspace_dim]

        return output

    # 检查给定的张量 x 是否位于 Grassmannian 流形上
    def _check_point_on_manifold(
            self, x: th.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        # Calculate the dot product of x and its transpose, expecting an identity matrix for orthonormal columns
        x_dot_xT = th.matmul(x.transpose(-2, -1), x)
        identity = th.eye(x.shape[-1], device=x.device, dtype=x.dtype)

        # Check if the result is close to the identity matrix
        if th.allclose(x_dot_xT, identity, atol=atol, rtol=rtol):
            return True
        else:
            reason = "The columns of the input matrix are not orthonormal."
            return False, reason

    # 检查给定的向量 u 是否是点 x 在 Grassmannian 流形上的切空间中的水平向量（即是否满足正交条件 x^T * u = 0）
    def _check_vector_on_tangent(
            self, x: th.Tensor, u: th.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """x^\top u = 0"""
        orthogonality_condition = th.matmul(x.transpose(-2, -1), u)
        if th.all(th.allclose(orthogonality_condition, th.zeros_like(orthogonality_condition), atol=atol, rtol=rtol, dim=(-2, -1))):
            return True, None
        else:
            return False, "At least one vector does not satisfy the orthogonality condition within the given tolerances."

    # 将给定点 x 在切空间中的欧氏梯度 u 投影到点 x 的水平子空间中，从而得到该点的黎曼梯度
    def egrad2rgrad(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self.proju(x, u)

    # 计算切向量 u 和 v 在点 x 的切空间上的内积
    def inner(self, x: th.Tensor, u: th.Tensor, v: th.Tensor = None, *, keepdim=False) -> th.Tensor:
        if v is None:
            v = u
        # Flatten u and v, then compute their dot product.
        # For higher-dimensional batch processing, use element-wise multiplication followed by summation.
        return th.sum(u * v, dim=(-2, -1), keepdim=keepdim)
        # inner_prod = th.sum(u * v, dim=(-2, -1), keepdim=keepdim)
        # inner_prod2 = th.einsum('...ij,...ij->...', u, v)

    # 将一个环境向量 U 正交投影到 Grassmannian 流形上的点 X 的水平切空间中
    def proju(self, x: th.Tensor, u: th.Tensor) -> th.Tensor:
        """Orthogonal projection of an ambient vector U to the horizontal space at X."""

        xtu = th.matmul(x.transpose(-2, -1).contiguous(), u)
        tangent_vec = u - th.matmul(x, xtu)
        return tangent_vec

    # 将输入的矩阵 x 正交投影到 Grassmannian 流形上
    def projx(self, x: th.Tensor) -> th.Tensor:
        # Perform QR decomposition on x. Following manopt, we do not need to worry about flipping signs of columns here
        q, _ = th.linalg.qr(x)
        return q

    def grassman_grad(self, x, v):
        xt = x.transpose(-2, -1)
        xx_t = th.matmul(x, xt)

        I = th.eye(x.shape[1]).unsqueeze(0)
        I = I.expand(x.shape[0], -1, -1)
        proj = I - xx_t

        Rie_grad = th.matmul(proj, v)

        return Rie_grad

    # 计算切空间上梯度的范数
    def grassmann_grad_norm(self, x, v):
        # xt = x.transpose(-2, -1)
        # xx_t = th.matmul(x, xt)
        #
        # I = th.eye(x.shape[1]).unsqueeze(0)
        # I = I.expand(x.shape[0], -1, -1)
        # proj = I - xx_t
        #
        # Rie_grad = th.matmul(proj, v)
        Rie_grad = self.grassman_grad(x, v)
        fro_norm = th.norm(Rie_grad, p='fro')
        grassmann_norm = fro_norm / (2 ** 0.5)
        return grassmann_norm
