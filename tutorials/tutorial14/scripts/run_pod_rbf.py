"""
Standalone POD-RBF analysis script.

Evaluates the baseline POD and POD-RBF models on a selected dataset,
printing train/test errors and plotting POD modes.

Usage:
    python scripts/run_pod_rbf.py --dataset cavity --reddim 3
    python scripts/run_pod_rbf.py --dataset backstep --reddim 5 --field mag(v)
"""
import argparse
import os
import sys

# Add the tutorial14 root to sys.path so local packages (nns, rom, ...)
# are importable when running this script from the scripts/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pina.model.layers import PODBlock
from pina import LabelTensor
from smithers.dataset import LidCavity, NavierStokesDataset
from rom.pod_rbf import err, PODRBF
from utils.plotting import plot
os.makedirs("img", exist_ok=True)
torch.manual_seed(42)

def load_dataset(dataset, field):
    """
    Load the requested dataset and extract snapshots and parameters.

    Args:
        dataset: 'cavity' or 'backstep'.
        field: Snapshot field name (e.g., 'mag(v)').

    Returns:
        Tuple of (data, snapshots, params).
    """
    if dataset == 'cavity':
        data = LidCavity()
    elif dataset == 'backstep':
        data = NavierStokesDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'cavity' or 'backstep'.")
    snapshots = data.snapshots[field]
    params = data.params
    return data, snapshots, params

def main():
    parser = argparse.ArgumentParser(description='POD-RBF baseline analysis')
    parser.add_argument('--dataset', type=str, default='cavity',
                        choices=['cavity', 'backstep'],
                        help='Dataset to analyze')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')
    args = parser.parse_args()

    # Load dataset
    data, snapshots, params = load_dataset(args.dataset, args.field)

    # Train/test split
    params_train, params_test, snapshots_train, snapshots_test = train_test_split(
        params, snapshots, test_size=0.20, shuffle=True, random_state=42)

    # Convert to LabelTensors
    params_train = LabelTensor(
        torch.tensor(params_train, dtype=torch.float32), labels=['mu'])
    params_test = LabelTensor(
        torch.tensor(params_test, dtype=torch.float32), labels=['mu'])
    snapshots_train = LabelTensor(
        torch.tensor(snapshots_train, dtype=torch.float32),
        labels=[f's{i}' for i in range(snapshots_train.shape[1])])
    snapshots_test = LabelTensor(
        torch.tensor(snapshots_test, dtype=torch.float32),
        labels=[f's{i}' for i in range(snapshots_test.shape[1])])

    # POD model
    pod = PODBlock(args.reddim)
    pod.fit(snapshots_train)

    # Plot POD modes
    modes = pod.basis.T
    vmin = modes.min()
    vmax = modes.max()
    list_fields = [modes.tensor.cpu().detach().numpy()[:, i].reshape(-1)
                   for i in range(args.reddim)]
    list_labels = [f'Mode {i + 1}' for i in range(args.reddim)]
    plot(data.triang, list_fields, list_labels,
         filename=f'img/pod_modes_{args.dataset}', vmin=vmin, vmax=vmax)

    # Evaluate POD
    predicted_train = pod.expand(pod.reduce(snapshots_train))
    predicted_test = pod.expand(pod.reduce(snapshots_test))
    error_train, error_train_med = err(snapshots_train, predicted_train)
    error_test, error_test_med = err(snapshots_test, predicted_test)
    print(f'POD (rank={args.reddim})')
    print(f'  Train: mean={error_train:.6f}, median={error_train_med:.6f}')
    print(f'  Test:  mean={error_test:.6f}, median={error_test_med:.6f}')

    # POD-RBF model (use an RBF kernel appropriate for the dataset)
    if args.dataset == 'backstep':
        rbf_kwargs = {'rbf_kernel': 'linear'}
    else:
        rbf_kwargs = {'rbf_kernel': 'inverse_multiquadric', 'epsilon': 100.}
    rom_rbf = PODRBF(pod_rank=args.reddim, **rbf_kwargs)
    rom_rbf.fit(params_train, snapshots_train)
    predicted_train_rbf = rom_rbf(params_train)
    predicted_test_rbf = rom_rbf(params_test)
    error_train_rbf, error_train_rbf_med = err(snapshots_train, predicted_train_rbf)
    error_test_rbf, error_test_rbf_med = err(snapshots_test, predicted_test_rbf)
    print(f'POD-RBF (rank={args.reddim})')
    print(f'  Train: mean={error_train_rbf:.6f}, median={error_train_rbf_med:.6f}')
    print(f'  Test:  mean={error_test_rbf:.6f}, median={error_test_rbf_med:.6f}')


if __name__ == "__main__":
    main()
