"""
QuadNet: spatially-dependent quadratic correction via neural networks.

Learns a spatially-varying quadratic correction operator C(x,y) using two
FeedForward sub-networks: one operating on POD modes and one on spatial
coordinates. The correction is: coeff^T @ C(x,y) @ coeff.
"""
import torch
import torch.nn as nn
from pina.model import FeedForward


class QuadNet(nn.Module):
    """
    Neural network for spatially-dependent quadratic correction.

    The correction operator C(x,y) is learned as:
        C(x,y) = red(FF_modes(modes) * FF_coords(coords))
    where FF_modes and FF_coords are two separate FeedForward networks,
    and red is a reduction layer.

    The correction is then: coeff^T @ C(x,y) @ coeff, computed via einsum
    over the upper-triangular entries of the coefficient outer product.

    Args:
        modes: POD basis modes (LabelTensor), shape (N_dof, r).
        coordinates: Spatial coordinates (LabelTensor), shape (N_dof, 2).
        scaler: Optional scaler for correction terms (default: None).
    """

    def __init__(self, modes, coordinates, scaler=None):
        super().__init__()
        self.scaler = scaler
        self.modes = modes
        self.coords = coordinates
        r = modes.shape[1]
        self.b = FeedForward(
            input_dimensions=r,
            output_dimensions=r * (r + 1) // 2,
            layers=[20, 20, 20, 20, 20, 20, 20],
            func=nn.Tanh,
        )
        self.t = FeedForward(
            input_dimensions=2,
            output_dimensions=r * (r + 1) // 2,
            layers=[20, 20, 20, 20, 20, 20, 20],
            func=nn.Tanh,
        )
        self.red = FeedForward(
            input_dimensions=r * (r + 1) // 2,
            output_dimensions=r * (r + 1) // 2,
            n_layers=1,
            func=nn.Tanh,
        )

    def forward(self, par, coef):
        """
        Compute the correction for given parameters and coefficients.

        Args:
            par: Input parameters (unused in this variant, kept for API).
            coef: POD coefficients (LabelTensor), shape (N, r).

        Returns:
            Correction terms, shape (N, N_dof).
        """
        z1 = self.b(self.modes)
        z2 = self.t(self.coords)
        c = self.red(z1 * z2)

        r = self.modes.shape[1]
        indices = torch.triu_indices(r, r)
        indices = r * indices[0] + indices[1]
        ai = torch.einsum("ni,nj->nij", coef, coef).flatten(start_dim=1)[:, indices]
        return torch.einsum("ni,Ni->nN", ai, c)

    def C(self, par=None):
        """
        Return the learned spatial correction operator C(x,y).

        Args:
            par: Unused, kept for API consistency with QuadNetMu.

        Returns:
            Operator tensor of shape (N_dof, r*(r+1)/2).
        """
        z1 = self.b(self.modes)
        z2 = self.t(self.coords)
        c = self.red(z1 * z2)
        return c
