"""
Corrected ROM solver for parametric PDEs.

Combines POD reduction, RBF interpolation, and a learned correction term
to approximate high-fidelity solutions. The correction network learns the
residual between snapshots and their POD-RBF reconstruction.
"""
import torch
from pina import LabelTensor
from pina.solvers import SupervisedSolver


class CorrectedROM(SupervisedSolver):
    """
    Non-intrusive ROM using POD as reduction and RBF as approximation,
    with an additional correction/closure term that reintroduces the
    contribution of neglected modes.

    The total prediction is:
        u(mu) = POD.expand(RBF(mu)) + CorrectionNetwork(mu, RBF(mu))

    Args:
        problem: A ParametricProblem with a 'correction' condition.
        reduction_network: A fitted PODBlock for dimensionality reduction.
        interpolation_network: A fitted RBFBlock for coefficient interpolation.
        correction_network: A correction network (e.g., QuadLS, QuadNet, QuadNetMu).
        loss: Loss function (default: MSELoss).
        optimizer: Optimizer class (default: Adam).
        optimizer_kwargs: Optimizer keyword arguments.
        scheduler: Learning rate scheduler class.
        scheduler_kwargs: Scheduler keyword arguments.
    """

    def __init__(
        self,
        problem,
        reduction_network,
        interpolation_network,
        correction_network,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        scheduler=torch.optim.lr_scheduler.ConstantLR,
        scheduler_kwargs={"factor": 1, "total_iters": 0},
    ):
        model = torch.nn.ModuleDict(
            {
                "reduction_network": reduction_network,
                "interpolation_network": interpolation_network,
                "correction_network": correction_network,
            }
        )
        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
        )

        if hasattr(correction_network, "fit"):
            correction_network.fit(
                problem.conditions["correction"].input_points,
                problem.conditions["correction"].output_points,
            )
        self.modes = reduction_network.basis

    def forward(self, input_params):
        """
        Compute the ROM solution: POD term + correction term.

        Args:
            input_params: Input parameters (LabelTensor).

        Returns:
            Predicted snapshots (LabelTensor).
        """
        reduction_network = self.neural_net["reduction_network"]
        interpolation_network = self.neural_net["interpolation_network"]
        correction_network = self.neural_net["correction_network"]

        coeff = interpolation_network(input_params)
        pod_term = reduction_network.expand(coeff)
        correction_term = correction_network(input_params, coeff)
        if correction_network.scaler is not None:
            correction_term = correction_network.scaler.inverse_transform(correction_term)

        return pod_term + correction_term

    def _forward_no_interp(self, input_params, snaps):
        """
        Compute the ROM solution using direct projection instead of interpolation.

        Args:
            input_params: Input parameters (LabelTensor).
            snaps: Snapshots to project (LabelTensor).

        Returns:
            Predicted snapshots (LabelTensor).
        """
        reduction_network = self.neural_net["reduction_network"]
        correction_network = self.neural_net["correction_network"]

        coeff = reduction_network.reduce(snaps)
        pod_term = reduction_network.expand(coeff)
        correction_term = correction_network(input_params, coeff)
        if correction_network.scaler is not None:
            correction_term = correction_network.scaler.inverse_transform(correction_term)

        return pod_term + correction_term

    def loss_data(self, input_pts, output_pts):
        """
        Compute the training loss: correction MSE + orthogonality regularizer.

        The loss combines:
            - MSE between predicted and exact correction terms
            - Orthogonality penalty: ||V^T @ C|| to regularize the operator

        Args:
            input_pts: Input parameters.
            output_pts: Exact correction terms (precomputed).

        Returns:
            Combined loss value.
        """
        interpolation_network = self.neural_net["interpolation_network"]
        correction_network = self.neural_net["correction_network"]
        coeff_orig = interpolation_network(input_pts)
        approx_correction = correction_network(input_pts, coeff_orig)
        exact_correction = output_pts
 
        loss_correction = self.loss(approx_correction, exact_correction)

        # Orthogonal components loss: penalize correlation between POD modes
        # and the correction operator. Modes should be oriented as (N_dof, rank).
        V = correction_network.modes
        # ensure V is (N_dof, rank)
        if V.shape[0] < V.shape[1]:
            V = V.T
        Vbar = correction_network.C(input_pts)
        if Vbar.dim() == 3:
            # Vbar: (N, N_dof, K) -> projected to (rank, N, K), normalize by batch
            loss_orthog = torch.norm(torch.einsum("sr,bsk->rbk", V, Vbar)) / Vbar.shape[0]
        else:
            loss_orthog = torch.norm(V.T @ Vbar)

        # Importance of correction over orthonormalisation
        beta = 0.001
        self.log("loss_orthon", float(loss_orthog), prog_bar=True, logger=True)
        self.log("loss_corr", float(loss_correction), prog_bar=True, logger=True)

        return loss_correction + beta * loss_orthog

    @staticmethod
    def compute_exact_correction(pod, snaps):
        """
        Compute the exact correction: residual between snapshots and their
        POD reconstruction.

        This is the ground-truth correction that the correction network
        tries to approximate: exact_correction = snaps - POD.expand(POD.reduce(snaps)).

        Args:
            pod: A fitted PODBlock instance.
            snaps: Snapshots to compute corrections for (LabelTensor).

        Returns:
            Exact correction terms (LabelTensor).
        """
        corr = snaps - pod.expand(pod.reduce(snaps))
        # Rebuild a properly-labeled LabelTensor: PINA's LabelTensor - LabelTensor
        # returns a LabelTensor without _labels, which breaks later .cpu()/indexing.
        if isinstance(snaps, LabelTensor):
            corr = LabelTensor(corr.tensor, snaps.labels)
        return corr

    @property
    def neural_net(self):
        return self._neural_net.torchmodel
