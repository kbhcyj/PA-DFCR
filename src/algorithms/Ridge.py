import numpy as np
from scipy import linalg, sparse



class Ridge_Regression:
    def __init__(self, X, y , alpha ,fit_intercept = False, alpha_scale_list= None, K_scale = None, sigma_1=None, gamma_1=None, mu_1 = None,  beta_1 = None):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.alpha_scale_list= np.array(alpha_scale_list)
        self.k_scale = np.array(K_scale)**2
        self.sigma_1 = sigma_1
        self.gamma_1 = gamma_1
        self.mu_1 = mu_1
        self.beta_1 = beta_1
        self.n_samples, self.n_features = X.shape
        # self.n_samples, self.n_targets = y.shape

    def fit(self):
        coef = self._solve_cholesky()
        return coef


    def _solve_cholesky(self):
        n_features = self.X.shape[1]
        n_targets = self.y.shape[0]

        A = self.safe_sparse_dot(self.X.T, self.X, dense_output=True)
        Xy = self.safe_sparse_dot(self.X.T, self.y, dense_output=True)  + (self.alpha[0] *self.gamma_1/self.sigma_1) * ((self.gamma_1*self.mu_1/self.sigma_1) - self.beta_1)* self.alpha_scale_list #-- 22 without considering sig,gam
        # Xy = self.safe_sparse_dot(self.X.T, self.y, dense_output=True)  + ( (self.alpha[0] * self.mu_1) - (self.alpha[0] * self.sigma_1 * self.beta_1) / self.gamma_1) * self.alpha_scale_list

        # one_alpha = np.array_equal(self.alpha, len(self.alpha) * self.alpha)

        # if one_alpha:
        A.flat[::n_features + 1] += self.alpha[0] * (self.gamma_1/self.sigma_1)**2 * self.alpha_scale_list @ self.alpha_scale_list.T + self.alpha[1]*self.k_scale  # -- 22 without considering sig,gam
        # A.flat[::n_features + 1] += self.alpha[0] * self.alpha_scale_list @ self.alpha_scale_list.T + self.alpha[1] # 22 with considering sig,gam

        #return linalg.solve(A, Xy, overwrite_a=True).T
        eps = 1e-6 * np.eye(A.shape[0])
        coef, *_ = linalg.lstsq(A + eps, Xy, cond=None)
        return coef.T
        # else:
        #     coefs = np.empty([n_targets, n_features], dtype=self.X.dtype)
        #     for coef, target, current_alpha in zip(coefs, Xy.T, self.alpha):
        #         A.flat[::n_features + 1] += current_alpha
        #         coef[:] = linalg.solve(A, target,
        #                                overwrite_a=False).ravel()
        #         A.flat[::n_features + 1] -= current_alpha
        #     return coefs

    def safe_sparse_dot(self, a, b, *, dense_output=False):
        """Dot product that handle the sparse matrix case correctly.
        Parameters
        ----------
        a : {ndarray, sparse matrix}
        b : {ndarray, sparse matrix}
        dense_output : bool, default=False
            When False, ``a`` and ``b`` both being sparse will yield sparse output.
            When True, output will always be a dense array.
        Returns
        -------
        dot_product : {ndarray, sparse matrix}
            Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
        """
        if a.ndim > 2 or b.ndim > 2:
            if sparse.issparse(a):
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif sparse.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = np.dot(a, b)
        else:
            ret = a @ b

        if (sparse.issparse(a) and sparse.issparse(b)
                and dense_output and hasattr(ret, "toarray")):
            return ret.toarray()
        return ret


