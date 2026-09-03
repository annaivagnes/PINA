"""
Baseline POD-RBF model and error metrics.

Provides the PODRBF class for baseline (uncorrected) POD+RBF predictions,
and the err() function for computing relative prediction errors.
"""
import torch
from pina.model.layers import PODBlock, RBFBlock


def err(snap, snap_pred):
    """
    Compute mean and median relative L2 errors between snapshots and predictions.

    Args:
        snap: Ground-truth snapshots (LabelTensor).
        snap_pred: Predicted snapshots (LabelTensor).

    Returns:
        Tuple of (mean_error, median_error) as numpy floats.
    """
    import numpy as np

    errs = (
        torch.linalg.norm(snap_pred - snap, dim=-1)
        / torch.linalg.norm(snap, dim=-1)
    )
    errs = errs.tensor.cpu().detach().numpy()
    return float(np.mean(errs)), float(np.median(errs))


class PODRBF(torch.nn.Module):
    """
    Baseline reduced-order model combining POD reduction with RBF interpolation.

    This is the uncorrected ROM that the CorrectedROM aims to improve upon.
    It reduces snapshots via POD, interpolates coefficients via RBF, then
    reconstructs via POD expansion.

    Args:
        pod_rank: Number of POD modes to retain.
        rbf_kernel: RBF kernel type (default: 'thin_plate_spline').
        **rbf_kwargs: Additional keyword arguments passed to the RBFBlock
            (e.g., epsilon, ecc, etc.).
    """

    def __init__(self, pod_rank, rbf_kernel="thin_plate_spline", **rbf_kwargs):
        super().__init__()
        self.pod = PODBlock(pod_rank)
        self.rbf = RBFBlock(kernel=rbf_kernel, **rbf_kwargs)

    def fit(self, params, snaps):
        """
        Fit the POD and RBF on training data.

        Args:
            params: Training parameters (LabelTensor).
            snaps: Training snapshots (LabelTensor).
        """
        self.pod.fit(snaps)
        self.rbf.fit(params, self.pod.reduce(snaps))

    def forward(self, param_test):
        """
        Predict snapshots for test parameters via POD-RBF.

        Args:
            param_test: Test parameters (LabelTensor).

        Returns:
            Predicted snapshots (LabelTensor).
        """
        return self.pod.expand(self.rbf(param_test))
