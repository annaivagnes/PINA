"""
QuadNetMu: parameter and spatially-dependent quadratic correction.

Extension of QuadNet that also considers the parameter values mu when
computing the correction operator. The operator is:
    C(mu, x, y) = red(FF_modes(modes) * FF_coords(coords) * FF_params(mu))
"""
import torch
import torch.nn as nn
from pina.model import FeedForward


class QuadNetMu(nn.Module):
    """
    Neural network for parameter and spatially-dependent quadratic correction.

    The correction operator C(mu, x, y) is learned as:
        C(mu, x, y) = red(FF_modes(modes) * FF_coords(coords) * FF_params(mu))
    where FF_modes, FF_coords, and FF_params are three separate FeedForward
    networks, and red is a reduction layer.

    The per-parameter and per-point operator is constructed via einsum:
        z = einsum('bi,Ni,Ni->bNi', FF_params(mu), FF_coords(coords), FF_modes(modes))
        c = red(z)

    The correction is then: coeff^T @ C(mu, x, y) @ coeff.

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
        self.bmu = FeedForward(
            input_dimensions=1,
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
            par: Input parameters (LabelTensor), shape (N, 1).
            coef: POD coefficients (LabelTensor), shape (N, r).

        Returns:
            Correction terms, shape (N, N_dof).
        """
        z1 = self.b(self.modes)             # (N_dof, K)
        z2 = self.t(self.coords)            # (N_dof, K)
        z3 = self.bmu(par)                  # (N, K)
        # Per-parameter, per-spatial-point operator: (N, N_dof, K)
        z = torch.einsum('bi,Ni,Ni->bNi', z3, z2, z1)
        c = self.red(z)

        r = self.modes.shape[1]
        indices = torch.triu_indices(r, r)
        indices = r * indices[0] + indices[1]
        ai = torch.einsum("ni,nj->nij", coef, coef).flatten(start_dim=1)[:, indices]
        return torch.einsum("ni,nNi->nN", ai, c)

    def C(self, par):
        """
        Return the learned parameter and spatially-dependent operator C(mu, x, y).

        Args:
            par: Input parameters (LabelTensor), shape (N, 1).

        Returns:
            Operator tensor of shape (N, N_dof, r*(r+1)/2).
        """
        z1 = self.b(self.modes)             # (N_dof, K)
        z2 = self.t(self.coords)            # (N_dof, K)
        z3 = self.bmu(par)                  # (N, K)
        z = torch.einsum('bi,Ni,Ni->bNi', z3, z2, z1)
        c = self.red(z)
        return c
