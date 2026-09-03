"""
Unified runner for backward-facing step experiments.

Supports three correction strategies via the --correction flag:
    - quadls: Quadratic correction via least squares (QuadLS)
    - quadnet: Spatially-dependent quadratic correction (QuadNet)
    - quadnetmu: Parameter and spatially-dependent correction (QuadNetMu)

Usage:
    python scripts/run_backstep.py --correction quadls --reddim 3 --train 100
    python scripts/run_backstep.py --correction quadnet --reddim 5 --epochs 5000
    python scripts/run_backstep.py --correction quadnetmu --reddim 7
    python scripts/run_backstep.py --correction quadnet --load <dir> --version 0
"""
import sys
import os

# Add the tutorial14 root to sys.path so local packages (nns, rom, ...)
# are importable when running this script from the scripts/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pytorch_lightning.callbacks import Callback, EarlyStopping

from pina import Trainer, Plotter, LabelTensor
from pina.callbacks import MetricTracker
from pina.loss import LpLoss

from nns.quadls import QuadLS
from nns.quadnet import QuadNet
from nns.quadnet_mu import QuadNetMu
from rom.corrected_rom import CorrectedROM
from rom.pod_rbf import err, PODRBF
from utils.plotting import plot
from problems.setup_backstep import BackstepProblem


def resolve_device(requested):
    """
    Resolve the compute device, falling back to CPU when no GPU is available.

    Args:
        requested: 'cpu' or 'gpu'.

    Returns:
        'cpu' if no CUDA GPU is available, otherwise the requested device.
    """
    if requested == 'gpu' and not torch.cuda.is_available():
        print('WARNING: CUDA not available, falling back to CPU.')
        return 'cpu'
    return requested


class LogCallback(Callback):
    """Logs training time and loss metrics."""

    def on_train_start(self, trainer, pl_module):
        self.tstart = time.time()

    def on_train_end(self, trainer, pl_module):
        self.tend = time.time()
        logging.info(
            f'reddim={pl_module.neural_net["reduction_network"].rank}, '
            f'v_num={pl_module.logger.version}, '
            f'loss_corr={trainer.callback_metrics["loss_corr"].item()}, '
            f'train_time={self.tend - self.tstart}, '
            f'epochs={trainer.current_epoch}'
        )


def create_correction_network(args, backstep):
    """
    Create the correction network based on the --correction argument.

    Args:
        args: Parsed command-line arguments.
        backstep: BackstepProblem instance with fitted POD/RBF.

    Returns:
        Correction network instance.
    """
    match args.correction:
        case "quadls":
            return QuadLS(
                pod=backstep.pod,
                coeffs=backstep.pod.reduce(backstep.snapshots_train),
                interp=backstep.rbf,
                scaler=backstep.scaler,
            )
        case "quadnet":
            return QuadNet(backstep.modes, backstep.coords,
                           scaler=backstep.scaler)
        case "quadnetmu":
            return QuadNetMu(backstep.modes, backstep.coords,
                             scaler=backstep.scaler)
        case _:
            raise ValueError(f"Unknown correction type: {args.correction}")


def get_optimizer_config(args):
    """
    Get optimizer and scheduler configuration based on correction type.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dict with optimizer, optimizer_kwargs, loss, scheduler, scheduler_kwargs.
    """
    match args.correction:
        case "quadls":
            return {
                "optimizer": torch.optim.Adam,
                "optimizer_kwargs": {"lr": 1e-2},
                "loss": torch.nn.MSELoss(),
            }
        case "quadnet":
            return {
                "optimizer": torch.optim.Adam,
                "optimizer_kwargs": {"lr": 1e-2, "weight_decay": 1e-3},
                "loss": LpLoss(relative=True),
                "scheduler": torch.optim.lr_scheduler.MultiStepLR,
                "scheduler_kwargs": {
                    "milestones": [1000, 2000, 3000, 4000],
                    "gamma": 0.9,
                },
            }
        case "quadnetmu":
            return {
                "optimizer": torch.optim.Adam,
                "optimizer_kwargs": {"lr": 1e-2, "weight_decay": 1e-3},
                "loss": LpLoss(relative=True),
                "scheduler": torch.optim.lr_scheduler.MultiStepLR,
                "scheduler_kwargs": {
                    "milestones": [1000, 2000, 3000, 4000],
                    "gamma": 0.9,
                },
            }


def main(arguments=None):
    """
    Main entry point for the backstep experiment.

    Args:
        arguments: Optional list of command-line arguments (for programmatic use).
    """
    parser = argparse.ArgumentParser(description='Corrected ROM: Backward-facing step')
    parser.add_argument('--correction', type=str, required=True,
                        choices=['quadls', 'quadnet', 'quadnetmu'],
                        help='Correction strategy')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--train', type=int, default=10, help='Train set size')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')
    parser.add_argument('--load', type=str, help='Directory to load checkpoint from')
    parser.add_argument('--version', type=int, help='Model version for checkpoint')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--device', type=str, default='gpu',
                        choices=['cpu', 'gpu'], help='Compute device')

    args = parser.parse_args(arguments)
    device = resolve_device(args.device)

    os.makedirs("img", exist_ok=True)
    logging.basicConfig(
        filename='backstep_log.txt', level=logging.INFO, format='%(message)s')

    # Set up data
    backstep = BackstepProblem(
        args.field, args.reddim, subset=None,
        train_size=args.train, device=device)
    problem = backstep.problem

    # Create correction network
    corr_net = create_correction_network(args, backstep)

    if args.train > 10 or args.correction == "quadls":
        num_batches = 4
    else:
        num_batches = 1

    if not args.load:
        # --- Training ---
        opt_config = get_optimizer_config(args)
        rom = CorrectedROM(
            problem=problem,
            reduction_network=backstep.pod,
            interpolation_network=backstep.rbf,
            correction_network=corr_net,
            **opt_config,
        )


        early = EarlyStopping(
            monitor='loss_corr', patience=5000,
            stopping_threshold=1e-2, check_on_train_epoch_end=True)

        trainer = Trainer(
            solver=rom,
            max_epochs=args.epochs,
            accelerator='auto',
            default_root_dir=args.load,
            callbacks=[MetricTracker(), LogCallback(), early],
            batch_size=args.train // num_batches,
        )
        trainer.train()

        # Evaluate
        if device == 'gpu':
            rom.cuda()
        rom.eval()
        predicted_snaps_test = rom(backstep.params_test)
        test_error = err(backstep.snapshots_test, predicted_snaps_test)
        logging.info(
            f'r={args.reddim}, train_size={args.train}, '
            f'correction={args.correction}, test_mean={test_error[0]}\n'
            f'------------------')

    else:
        # --- Load from checkpoint ---
        id_ = args.version

        rom = CorrectedROM.load_from_checkpoint(
            checkpoint_path=os.path.join(
                args.load,
                f'lightning_logs/version_{id_}/checkpoints/'
                f'epoch={args.epochs - 1}-step={args.epochs * num_batches}.ckpt'),
            problem=problem,
            reduction_network=backstep.pod,
            interpolation_network=backstep.rbf,
            correction_network=corr_net,
        )
        rom.eval()

        # Evaluate on train and test
        predicted_snaps_train = rom(backstep.params_train)
        predicted_snaps_test = rom(backstep.params_test)
        train_error = err(backstep.snapshots_train, predicted_snaps_train)
        test_error = err(backstep.snapshots_test, predicted_snaps_test)
        print(f'Train error: {train_error}\nTest error: {test_error}')

        # Plot comparison: truth vs corrected vs baseline POD-RBF
        data = backstep.data
        ind_test = 2
        snap = backstep.snapshots_test[ind_test].tensor.cpu().detach().numpy().reshape(-1)
        pred_snap = predicted_snaps_test[ind_test].tensor.cpu().detach().numpy().reshape(-1)

        pod_rbf = PODRBF(pod_rank=args.reddim, rbf_kernel='thin_plate_spline')
        pod_rbf.fit(backstep.params_train, backstep.snapshots_train)
        pred_pod_rbf = pod_rbf(backstep.params_test).tensor.cpu().detach().numpy()[ind_test].reshape(-1)

        list_fields = [snap, pred_snap, pred_pod_rbf,
                       snap - pred_snap, snap - pred_pod_rbf]
        list_labels = ['Truth', 'Corrected POD-RBF', 'POD',
                       'Error Corrected', 'Error POD']
        plot(data.triang, list_fields, list_labels,
             filename=f'img/{args.correction}_compare')

        # Plot correction: approximated vs exact
        coeff_orig = rom.neural_net["interpolation_network"](backstep.params_test)
        corr_scaler = rom.neural_net["correction_network"].scaler
        corr = corr_net(backstep.params_test, coeff_orig)
        if corr_scaler is not None:
            corr = corr_scaler.inverse_transform(corr)
        corr = corr.tensor.cpu().detach().numpy() if isinstance(corr, LabelTensor) \
            else corr.cpu().detach().numpy()
        corr = corr[ind_test, :].reshape(-1)

        exact_corr = CorrectedROM.compute_exact_correction(backstep.pod, backstep.snapshots_test)
        exact_corr = exact_corr[ind_test].tensor.cpu().detach().numpy().reshape(-1)

        list_fields = [corr, exact_corr, corr - exact_corr]
        list_labels = ['Approximated Correction', 'Exact Correction', 'Error']
        plot(data.triang, list_fields, list_labels,
             filename=f'img/{args.correction}_correction')


if __name__ == "__main__":
    main(sys.argv[1:])
