import torch as th


class Correlation:
    def __init__(self, n):
        self.n = n

    # 相关矩阵分解为单位对角下三角矩阵 cor+->LT1(n)
    @classmethod
    def correlation_to_LT1(cls, C):
        """
        计算相关矩阵 C 的 log-Cholesky 表示。
        """
        D_inv = th.zeros_like(C)
        L = th.linalg.cholesky(C)

        # 归一化对角为1，得到单位对角下三角矩阵 Gamma ∈ LT1(n)
        reciprocal_diag = 1 / th.diagonal(L, dim1=1, dim2=2)
        batch_indices = th.arange(L.shape[0]).unsqueeze(1)
        diag_indices = th.arange(C.shape[1])
        D_inv[batch_indices, diag_indices, diag_indices] = reciprocal_diag

        Gamma = D_inv @ L

        return Gamma

    # 单位对角下三角矩阵映射回相关矩阵 LT1(n)->cor+
    @classmethod
    def LT1_to_correlation(cls, Gamma: th.Tensor) -> th.Tensor:
        """
        将单位对角下三角矩阵 Gamma (LT^1(n)) 映射到满秩相关矩阵 Cor^+(n)。

        参数:
            Gamma (torch.Tensor): 形状为 (n, n) 的单位对角下三角矩阵（Gamma[i,i] = 1，Gamma[i,j] = 0 当 i < j）。

        返回:
            C (torch.Tensor): 形状为 (n, n) 的相关矩阵（对称正定，对角线为1）。
        """
        A = Gamma @ th.transpose(Gamma, -1, -2)

        diag = th.diagonal(A, dim1=1, dim2=2)
        diag_inv_sqrt = th.diag_embed(1.0 / th.sqrt(diag))

        C = diag_inv_sqrt @ A @ diag_inv_sqrt

        C = 0.5 * (C + th.transpose(C, -1, -2))

        return C

    @classmethod
    def LT1_to_LT0(cls, LT):
        """将严格下三角矩阵变为单位对角下三角矩阵"""
        LT0 = LT - th.diag_embed(th.diagonal(LT, dim1=1, dim2=2))

        return LT0

    @classmethod
    def LT0_to_LT1(cls, LT):
        """将严格下三角矩阵变为单位对角下三角矩阵"""
        n = LT.shape[0]
        # 创建单位矩阵
        eye_matrix = th.eye(LT.shape[1])
        # 在第 0 维上重复 10 次
        LT1 = th.tile(eye_matrix, (n, 1, 1)) + th.tril(LT, diagonal=-1)

        return LT1

    @classmethod
    def log_map(cls, C):
        """
        Compute the off-log map: Log(C) = Off(log(C)).

        Args:
            C: (..., n, n) batch of correlation matrices (symmetric, positive definite, diagonal=1).

        Returns:
            L: (..., n, n) batch of hollow symmetric matrices (diagonal=0).
        """
        # Compute matrix logarithm via eigenvalue decomposition
        C = (C + C.transpose(-2, -1)) / 2  # 确保对称
        # C = C + th.eye(C.shape[-1], device=C.device) * 1e-6  # Add small diagonal bias

        eigvals, eigvecs = th.linalg.eigh(C)
        # assert th.all(eigvals > 0), "Input not positive definite!"
        eigvals = eigvals.clamp(min=1e-8)  # C = Q diag(λ) Q^T

        log_eigvals = th.log(eigvals)  # log(λ)
        log_C = eigvecs @ (log_eigvals.unsqueeze(-1) * eigvecs.transpose(-2, -1))

        return log_C

    def scaling_matrix(self, S, max_iter=50, tol=1e-6):
        """
        Find Δ such that Δ exp(S) Δ has unit row sums (via Newton's method).

        Args:
            S: (..., n, n) batch of symmetric matrices with null row sums (S 1 = 0).
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Δ: (..., n) batch of diagonal scaling factors.
        """
        batch_shape = S.shape[:-2]
        n = S.shape[-1]
        device = S.device

        # Initialize Δ = 1
        delta = th.ones(*batch_shape, n, device=device)

        for _ in range(max_iter):
            # Compute Σ = exp(S) and scaled Σ̃ = Δ Σ Δ
            Sigma = th.matrix_exp(S)
            Sigma_scaled = delta.unsqueeze(-1) * Sigma * delta.unsqueeze(-2)

            # Compute gradient: ∇F = Σ̃ 1 - Δ^{-1}
            grad = th.sum(Sigma_scaled, dim=-1) - 1.0 / delta.clamp(min=1e-8)

            # Check convergence
            if th.max(th.abs(grad)) < tol:
                break

            # Hessian approximation: H ≈ Σ̃ + Δ^{-2}
            hess = Sigma_scaled + th.diag_embed(1.0 / delta.clamp(min=1e-8) ** 2)
            # Newton step: Δ ← Δ - H^{-1} ∇F
            delta_update = th.linalg.solve(hess, grad.unsqueeze(-1)).squeeze(-1)
            delta = delta - delta_update

        return delta

    def log_scaled_map(self, C):
        """
        Compute the log-scaled map: Log*(C) = log(Δ C Δ), where Δ scales C to unit row sums.

        Args:
            C: (..., n, n) batch of correlation matrices.

        Returns:
            S: (..., n, n) batch of symmetric matrices with null row sums (S 1 = 0).
        """
        # Compute S = log(Δ C Δ) via scaling
        S_init = self.log_map(C)  # Initial guess (ignoring scaling)
        delta = self.scaling_matrix(S_init)
        Sigma_scaled = delta.unsqueeze(-1) * C * delta.unsqueeze(-2)
        S = self.log_map(Sigma_scaled)
        return S

    @classmethod
    def euclidean_cholesky_distance(cls, X, Y=None, squared=False):
        """
        计算两个相关矩阵之间的 log-Euclidean-Cholesky 距离，支持自动求导。
        """
        G1 = cls.correlation_to_LT1(X)

        if Y is None:
            G2 = G1
            # G1: [N, D, D]
            # expand -> [N, 1, D, D], [1, N, D, D] -> [N, N, D, D]
            diff = G1[:, None, :, :] - G1[None, :, :, :]
            dist = th.linalg.norm(diff, ord='fro', dim=(2, 3))  # [N, N]
            dist = dist * (1 - th.eye(X.shape[0], device=X.device))  # 去掉对角线
        else:
            G2 = cls.correlation_to_LT1(Y)
            # G1: [N1, D, D], G2: [N2, D, D]
            diff = G1[:, None, :, :] - G2[None, :, :, :]  # [N1, N2, D, D]
            dist = th.linalg.norm(diff, ord='fro', dim=(2, 3))  # [N1, N2]

        return dist ** 2 if squared else dist

    @classmethod
    def log_euclidean_cholesky_distance(cls, X, Y=None, squared=False) -> th.Tensor:
        """
        计算两个相关矩阵集合在对数欧几里得 Cholesky 距离下的距离矩阵。
        """
        G1 = cls.correlation_to_LT1(X)  # [N1, D, D]
        N1, D, _ = G1.shape

        if Y is None:
            G2 = G1
            N2 = N1
            XisY = True
        else:
            G2 = cls.correlation_to_LT1(Y)  # [N2, D, D]
            N2 = G2.shape[0]
            XisY = False

        # 构造单位阵 [N, D, D]
        I1 = th.eye(D, device=X.device).expand(N1, D, D)
        I2 = I1 if XisY else th.eye(D, device=G2.device).expand(N2, D, D)

        # 利用幂级数计算 log(Θ(C))，前 D-1 项
        log_Theta_C1 = th.zeros_like(G1)
        log_Theta_C2 = th.zeros_like(G2)

        A1 = G1 - I1
        A2 = G2 - I2

        for k in range(1, D):  # D 次幂足够收敛（或更小）
            coeff = ((-1) ** (k - 1)) / k
            log_Theta_C1 = log_Theta_C1 + coeff * th.matrix_power(A1, k)
            log_Theta_C2 = log_Theta_C2 + coeff * th.matrix_power(A2, k)

        # 计算 Frobenius 距离矩阵
        if XisY:
            diff = log_Theta_C1[:, None, :, :] - log_Theta_C1[None, :, :, :]  # [N1, N1, D, D]
            dist = th.linalg.norm(diff, ord='fro', dim=(2, 3))  # [N1, N1]
            dist = dist * (1 - th.eye(N1, device=X.device))  # 清除对角线
        else:
            diff = log_Theta_C1[:, None, :, :] - log_Theta_C2[None, :, :, :]  # [N1, N2, D, D]
            dist = th.linalg.norm(diff, ord='fro', dim=(2, 3))  # [N1, N2]

        return dist ** 2 if squared else dist

    @classmethod
    def poly_hyperbolic_Cholesky_distance(cls, C1: th.Tensor, C2: th.Tensor = None, alpha: th.Tensor = None,
                                          squared=False) -> th.Tensor:
        n, m = C1.shape[0], C1.shape[1]

        # 默认权重
        if alpha is None:
            alpha = th.ones(m - 1)
            # alpha = 1 / th.arange(1, m)
            # alpha = th.arange(1, m)

        # 1. Cholesky 分解
        L1 = th.linalg.cholesky(C1)  # (n, m, m)
        # 2. 提取 L 的第 1 到第 m-1 行（不含第 0 行）
        L1_rows = L1[:, 1:, :]  # (n, m-1, m)

        if C2 is None:
            L2_rows = L1_rows
        else:
            L2 = th.linalg.cholesky(C2)
            L2_rows = L2[:, 1:, :]  # (n, m-1, m)

        # 3. 初始化距离张量
        dist2 = th.zeros(n, n)

        for i in range(m - 1):
            k = i + 1  # 第i行

            # 取出 L1 和 L2 的第 i 行，保留前 k+1 个元素
            L1_rows_i = L1_rows[:, i, :k + 1]  # (n, k+1)
            L2_rows_i = L2_rows[:, i, :k + 1]  # (n, k+1)

            # 映射到双曲面上
            L1_norm_i = L1_rows_i[:, :-1] / L1_rows_i[:, -1].unsqueeze(-1) + 1e-8
            L2_norm_i = L2_rows_i[:, :-1] / L2_rows_i[:, -1].unsqueeze(-1) + 1e-8
            L1_norm_i_last_col = 1 / L1_rows_i[:, -1] + 1e-8
            L2_norm_i_last_col = 1 / L2_rows_i[:, -1] + 1e-8

            L1_i = th.cat([L1_norm_i, L1_norm_i_last_col.unsqueeze(-1)], dim=1)
            L2_i = th.cat([L2_norm_i, L2_norm_i_last_col.unsqueeze(-1)], dim=1)

            # 计算内积（L1_i @ L2_i.T）中用于双曲空间的双内积：前 d-1 和最后一维
            x1, y1 = L1_i[:, :-1], L1_i[:, -1]
            x2, y2 = L2_i[:, :-1], L2_i[:, -1]

            # 计算 dot_x 和 dot_y 的广播乘积
            dot_x = x1 @ x2.T  # (n, n)
            dot_y = y1[:, None] * y2[None, :]  # (n, n)

            Q = dot_x - dot_y  # (n, n)
            Q = Q.clamp(max=-1 - 1e-6)
            d2 = th.arccosh(-Q) ** 2
            mean_value = d2.mean()
            # 双曲距离平方
            dist2 = dist2 + alpha[i] * d2

        dist = th.sqrt(dist2 + 1e-8)
        Dis = dist - th.diag_embed(th.diagonal(dist))

        return Dis ** 2 if squared else Dis

    def off_log_distance(self, X: th.Tensor, Y: th.Tensor = None, squared=False,
                         alpha=0.0, beta_gamma=None) -> th.Tensor:
        """
        Vectorized off-log distance between two batches of SPD correlation matrices.

        Args:
            X: (N, n, n)
            Y: (M, n, n) or None (if None, Y = X)
            alpha, beta, gamma: weights for the quadratic form
        Returns:
            :param beta_gamma:
            :param squared:
            :param Y:
            :param X:
            :param alpha:
        """
        gamma = 1.0
        if beta_gamma is None:
            beta_gamma = th.tensor(1.0)
        beta_gamma = th.clamp(beta_gamma, th.tensor(-3.0 + 1e-6))
        beta = gamma * beta_gamma
        XisY = Y is None

        log_L1 = self.log_map(X)
        L1 = log_L1 - th.diag_embed(th.diagonal(log_L1, dim1=-2, dim2=-1))  # (N, n, n)

        if XisY:
            L2 = L1  # (N, n, n)
        else:
            log_L2 = self.log_map(Y)
            L2 = log_L2 - th.diag_embed(th.diagonal(log_L2, dim1=-2, dim2=-1))  # (M, n, n)

        # Expand L1 (N, 1, n, n) and L2 (1, M, n, n)
        A = L1.unsqueeze(1)  # (N, 1, n, n)
        B = L2.unsqueeze(0)  # (1, M, n, n)
        delta = A - B  # (N, M, n, n)

        # Compute terms of the quadratic form
        tr_X2 = th.einsum('nmij,nmji->nm', delta, delta)  # trace of square: tr(Δ²)
        sum_X2 = (delta ** 2).sum(dim=(-2, -1))  # sum of squares
        sum_X = delta.sum(dim=(-2, -1))  # total sum

        dist_squared = alpha * tr_X2 + beta * sum_X2 + gamma * sum_X ** 2

        if not squared:
            dist_squared = dist_squared.clamp(min=1e-8).sqrt()

        if XisY:
            dist_squared = (dist_squared + dist_squared.T) / 2.0
            dist_squared = dist_squared - th.diag_embed(th.diagonal(dist_squared))

        return dist_squared

    def log_scaled_distance(self, X: th.Tensor, Y: th.Tensor = None, squared=False, alpha=0.0,
                            zeta_delta=None) -> th.Tensor:
        """
        Compute the log-scaled metric distance between two batches of correlation matrices.

        Args:
            C1, C2: (..., n, n) batches of correlation matrices.
            alpha, delta, zeta: Parameters of the quadratic form q*(Y).

        Returns:
            distance: (...,) batch of distances.
        """
        delta = zeta_delta[1]
        zeta = zeta_delta[0]
        # zeta = delta * zeta_delta
        XisY = Y is None

        S1 = self.log_scaled_map(X)
        if XisY:
            S2 = S1
        else:
            S2 = self.log_scaled_map(Y)

        S1 = S1.unsqueeze(1)  # (N, 1, n, n)
        S2 = S2.unsqueeze(0)  # (1, M, n, n)
        delta_S = S2 - S1  # (N, M, n, n)

        tr_Y2 = th.einsum('nmij,nmji->nm', delta_S, delta_S)  # trace of square: tr(Y²)
        tr_diag_Y2 = th.sum(th.diagonal(delta_S, dim1=-2, dim2=-1) ** 2, dim=-1)  # tr(diag(Y)²)
        tr_Y = th.einsum('...nmii->nm', delta_S)  # tr(Y)

        # Quadratic form q*(Y) = α tr(Y²) + δ tr(diag(Y)²) + ζ tr(Y)²
        distance_sq = alpha * tr_Y2 + delta * tr_diag_Y2 + zeta * tr_Y ** 2
        return th.sqrt(distance_sq.clamp(min=1e-8))
