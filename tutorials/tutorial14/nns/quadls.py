"""
Quadratic correction via least squares (QuadLS).

Computes the correction as: correction = D_matrix(coeffs) @ operator,
where D_matrix is the quadratic design matrix built from POD coefficients
and the operator is found by solving a least squares problem.
The operator is the same for all parameters (no parameter dependence).
"""
import torch
import time
from nns.base_corr import BaseCorrNet
from pina.model.layers import RBFBlock


class QuadLS(BaseCorrNet):
    """
    Quadratic correction via least squares.

    The correction is: coeff^T @ C @ coeff, where C is a spatially-dependent
    operator (r x N_dof) found by solving a least squares problem. The same
    operator C is used for all parameter values.

    Args:
        pod: A fitted PODBlock instance.
        coeffs: POD-reduced coefficients for training snapshots.
        interp: Interpolation network (default: RBFBlock).
        scaler: Optional scaler for correction terms.
    """

    def __init__(self, pod, coeffs, interp=RBFBlock(), scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)
        self.fictitious_params = torch.nn.Parameter(torch.randn(1))
        self.interp = interp
        self.coeffs = coeffs
        self.rank = pod.rank
        self.operator = None

    def D_matrix(self, coeffs):
        """
        Build the quadratic design matrix from coefficients.

        For each pair (i, j) with j <= i, computes coeffs[:, j] * coeffs[:, i],
        producing the upper-triangular entries of the outer product.

        Args:
            coeffs: POD coefficients (LabelTensor), shape (N, r).

        Returns:
            Design matrix D of shape (N, r*(r+1)/2).
        """
        coefs_i = []
        for i in range(self.rank):
            coef_i = coeffs[:, :i + 1] * coeffs[:, i]
            coefs_i.append(coef_i)
        D = torch.cat([coef.tensor for coef in coefs_i], dim=-1)
        return D

    def fit(self, params, corrections):
        """
        Fit the operator via least squares: D_matrix @ operator = corrections.

        Args:
            params: Input parameters (unused, kept for API consistency).
            corrections: Exact correction terms (LabelTensor).
        """
        R_mat = corrections
        D_mat = self.D_matrix(self.coeffs)
        if torch.linalg.cond(D_mat) >= 1e4:
            driver = 'gelsd'
        else:
            driver = 'gelsy'
        tic = time.time()
        try:
            self.operator = torch.linalg.lstsq(D_mat, R_mat, driver=driver).solution
        except RuntimeError: # on CUDA only gels works
            self.operator = torch.linalg.lstsq(D_mat, R_mat, driver="gels").solution

        toc = time.time()
        print("Time for least squares: ", toc - tic)

    def forward(self, param_test, coeff_test):
        """
        Predict correction terms for given parameters and coefficients.

        Args:
            param_test: Test parameters (unused, kept for API consistency).
            coeff_test: POD coefficients for test parameters.

        Returns:
            Predicted correction terms.
        """
        if self.operator is None:
            raise ValueError("The method needs to be fitted.")
        D_mat = self.D_matrix(coeff_test)
        pred = D_mat @ self.operator
        return pred

    def C(self, input_=None):
        """
        Return the learned spatial correction operator.

        Args:
            input_: Unused, kept for API consistency with QuadNet.

        Returns:
            Operator matrix of shape (r*(r+1)/2, N_dof).
        """
        return self.operator.T
