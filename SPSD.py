import torch as th


# from scipy.linalg import cholesky, logm, norm


class SPSD:
    def __init__(self, n):
        self.n = n

    @classmethod
    def spsd_wasserstein_distance(cls, A: th.Tensor, B: th.Tensor=None, squared=False, eps=1e-8):
        """
        高效、可微地计算成对的 Bures-Wasserstein 距离矩阵
        输入:
            A: (m, n, n) batch SPSD 矩阵
            B: (m, n, n) batch SPSD 矩阵
        输出:
            dist: (m, m) 成对距离矩阵
        """
        # assert th.allclose(A, A.transpose(-1, -2), atol=1e-6), "Input must be symmetric."
        # eigvals = th.linalg.eigvalsh(A)
        # assert th.all(eigvals >= -1e-6), "Input must be positive semi-definite."

        m, n, _ = A.shape

        if B is None:
            B = A

        # 计算 trace(A) 和 trace(B)
        tr_A = A.diagonal(dim1=-2, dim2=-1).sum(-1)  # (m,)
        tr_B = B.diagonal(dim1=-2, dim2=-1).sum(-1)  # (m,)

        # (m, m) 展开
        tr_A_expand = tr_A[:, None]
        tr_B_expand = tr_B[None, :]

        # 计算 cross term: tr( (A^{1/2} B A^{1/2})^{1/2} )
        # 使用 Cholesky + sqrtm 或对称特征分解实现可微版本

        # 对 A 进行特征分解并取 sqrt
        eigvals_A, eigvecs_A = th.linalg.eigh(A)  # (m, n), (m, n, n)
        eigvals_A_sqrt = eigvals_A.clamp_min(eps).sqrt()  # 防止负值 sqrt 出 nan
        A_sqrt = eigvecs_A @ th.diag_embed(eigvals_A_sqrt) @ eigvecs_A.transpose(-1, -2)  # (m, n, n)

        # 构造 (m, m, n, n) 的中间矩阵
        A_sqrt_expand = A_sqrt[:, None, :, :]  # (m, 1, n, n)
        B_expand = B[None, :, :, :]  # (1, m, n, n)
        C = A_sqrt_expand @ B_expand @ A_sqrt_expand  # (m, m, n, n)

        # 对 C 计算 sqrt 并取 trace
        eigvals_C, eigvecs_C = th.linalg.eigh(C)  # (m, m, n), (m, m, n, n)
        eigvals_C_sqrt = eigvals_C.clamp_min(eps).sqrt()  # 避免零特征值 sqrt 导致 grad NaN
        # C_sqrt = eigvecs_C @ th.diag_embed(eigvals_C_sqrt) @ eigvecs_C.transpose(-1, -2)  # (m, m, n, n)
        # tr_C_sqrt = C_sqrt.diagonal(dim1=-2, dim2=-1).sum(-1)  # (m, m)
        tr_C_sqrt = eigvals_C_sqrt.sum(-1)

        # 按照公式组合
        dist_squared = tr_A_expand + tr_B_expand - 2 * tr_C_sqrt
        dist_squared = dist_squared.clamp_min(eps)
        dist = dist_squared.sqrt()

        return dist ** 2 if squared else dist

