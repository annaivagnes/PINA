"""
Abstract base class for correction networks in the Corrected ROM.

The correction network learns the residual between high-fidelity snapshots
and their POD-RBF reconstruction. All correction strategies inherit from
this class and implement the `fit` and `forward` methods.
"""
import torch
from abc import ABC, abstractmethod
from pina.model.layers import RBFBlock


class BaseCorrNet(ABC, torch.nn.Module):
    """
    Abstract base class for correction networks.

    A correction network approximates the correction term that accounts
    for the contribution of neglected POD modes. Subclasses must implement
    `fit` (to train/fit the correction) and `forward` (to predict corrections
    for new parameters).

    Args:
        pod: A fitted PODBlock instance providing the reduction/expansion ops.
        interp: An interpolation network (default: RBFBlock) used to
            interpolate correction coefficients to unseen parameters.
        scaler: Optional scaler for normalizing correction terms.
    """

    def __init__(self, pod, interp=RBFBlock(), scaler=None):
        super().__init__()
        self.pod = pod
        self.rank = pod.rank
        self.modes = pod.basis
        self.params2correction = interp
        self.scaler = scaler

    @abstractmethod
    def fit(self, params, corrections):
        """Fit the correction network on training data."""

    @abstractmethod
    def forward(self, *args):
        """Predict correction terms for given parameters."""
