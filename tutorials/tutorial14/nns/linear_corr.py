"""
Linear correction network via least squares (reference implementation).

Computes the correction as: correction = modes @ coeff_corr,
where coeff_corr are found by solving a least squares problem.
This is the simplest correction strategy but is not used in the
run scripts as it has shown limited accuracy.
"""
import torch
from nns.base_corr import BaseCorrNet
from pina.model.layers import RBFBlock


class LinearCorrNet(BaseCorrNet):
    """
    Linear correction via least squares (not used in run scripts).

    The correction is computed as: correction = modes @ coeff_corr,
    where coeff_corr is found by solving: modes^T @ coeff_corr = corrections.

    Note:
        This class is kept as a reference. It is not included as an option
        in the run scripts due to limited accuracy in practice.
    """

    def __init__(self, pod, interp=RBFBlock(), scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)
        self.fictitious_params = torch.nn.Parameter(torch.randn(1))

    def fit(self, params, corrections):
        """
        Fit the scaler, solve least squares for correction coefficients,
        and fit the interpolation network on the resulting coefficients.

        Args:
            params: Input parameters (LabelTensor).
            corrections: Exact correction terms (LabelTensor).
        """
        if self.scaler is not None:
            corrections = self.scaler.fit_transform(corrections)
        coeff_corr = torch.linalg.lstsq(self.modes, corrections.T).solution.T
        self.params2correction.fit(params, coeff_corr)
        return coeff_corr

    def forward(self, param_test):
        """
        Predict correction terms by interpolating correction coefficients.

        Args:
            param_test: Test parameters (LabelTensor).

        Returns:
            Predicted correction terms.
        """
        coeff_corr_test = self.params2correction.forward(param_test)
        approx_correction = torch.matmul(self.modes, coeff_corr_test.T).T
        if approx_correction.dim() == 1:
            approx_correction = approx_correction.unsqueeze(0)
        if self.scaler is not None:
            approx_correction = self.scaler.inverse_transform(approx_correction)
        return approx_correction
