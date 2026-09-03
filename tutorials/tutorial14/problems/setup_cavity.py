"""
Data pipeline for the lid-driven cavity dataset.

Loads the dataset, performs train/test split, fits POD and RBF,
computes exact corrections, and defines the ParametricProblem.
"""
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from smithers.dataset import LidCavity
from pina.model.layers import PODBlock, RBFBlock
from pina.problem import ParametricProblem
from pina.geometry import CartesianDomain
from pina import Condition, LabelTensor
from rom.corrected_rom import CorrectedROM
from utils.scaler import InfNormScaler
import numpy as np


class CavityProblem:
    """
    Data pipeline for the lid-driven cavity dataset.

    Handles the full workflow: load data, split, fit POD/RBF,
    compute corrections, and define the optimization problem.

    Args:
        field: Snapshot field name (e.g., 'mag(v)').
        reddim: Reduced dimension (number of POD modes).
        subset: If not None, perform importance-based subset selection.
        train_size: Number of training samples.
        test_size: Number of test samples.
        device: 'cpu' or 'gpu'.
        scaler: Scalers the exact corrections to an O(1) scale. Set to None
            to disable correction scaling (default: InfNormScaler()).
    """

    def __init__(self, field, reddim, subset=None, train_size=100, test_size=100,
                 device='cpu', scaler=InfNormScaler()):
        self.field = field
        self.reddim = reddim
        self.train_size = train_size
        self.test_size = test_size
        self.device = device
        self.scaler = scaler
        self._load_data()
        self._train_test_split()
        if self.device == 'gpu':
            self.gpu()
        self._fit_pod()
        if self.device == 'gpu':
            self.gpu()
        self._fit_rbf()
        self._compute_corrections()
        if self.scaler is not None:
            labels = self.exact_correction.labels
            self.exact_correction = self.scaler.fit_transform(self.exact_correction.tensor)
            self.exact_correction = LabelTensor(self.exact_correction, labels)
        if subset is not None:
            self.subset_size = subset
            self._extract_subset()
        self._define_problem()
        if self.device == 'gpu':
            self.gpu()

    def _load_data(self):
        """Load the lid-driven cavity dataset and extract snapshots, coordinates, parameters."""
        self.data = LidCavity()
        self.snapshots = self.data.snapshots[self.field]

        coords = self.data.coordinates
        coords = torch.tensor(coords, dtype=torch.float32)
        self.coords = LabelTensor(coords, ['x', 'y'])

        params = self.data.params
        self.scaler_params = MinMaxScaler()
        self.params = self.scaler_params.fit_transform(params)

        self.Ndof = self.snapshots.shape[1]
        self.Nparams = self.params.shape[1]

    def _train_test_split(self, seed=42):
        """Split data into training and testing sets, convert to LabelTensors."""
        params_train, params_test, snapshots_train, snapshots_test = train_test_split(
            self.params, self.snapshots, test_size=self.test_size,
            train_size=self.train_size, shuffle=True, random_state=seed)

        self.params_train = LabelTensor(
            torch.tensor(params_train, dtype=torch.float32), labels=['mu'])
        self.params_test = LabelTensor(
            torch.tensor(params_test, dtype=torch.float32), labels=['mu'])
        self.snapshots_train = LabelTensor(
            torch.tensor(snapshots_train, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_train.shape[1])])
        self.snapshots_test = LabelTensor(
            torch.tensor(snapshots_test, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_test.shape[1])])

    def _fit_pod(self):
        """Fit the POD on training snapshots."""
        self.pod = PODBlock(self.reddim)
        self.pod.fit(self.snapshots_train)
        self.modes = self.pod.basis.T
        self.modes = LabelTensor(self.modes, [f'{i}' for i in range(self.reddim)])

    def _fit_rbf(self):
        """Fit the RBF on reduced training coefficients."""
        self.rbf = RBFBlock(kernel='inverse_multiquadric', epsilon=100.)
        self.rbf.fit(self.params_train, self.pod.reduce(self.snapshots_train))

    def _compute_corrections(self):
        """Compute the exact correction terms for training snapshots."""
        exact_correction = CorrectedROM.compute_exact_correction(self.pod, self.snapshots_train)
        self.exact_correction = LabelTensor(
            exact_correction, [f's{i}' for i in range(self.Ndof)])

    def _define_problem(self):
        """Define the ParametricProblem with a correction condition."""

        class SnapshotProblem(ParametricProblem):
            input_variables = ['mu']
            output_variables = self.snapshots_train.labels
            parameter_domain = CartesianDomain({'mu': [0, 100]})
            conditions = {
                'correction': Condition(
                    input_points=self.params_train,
                    output_points=self.exact_correction)
            }

        self.problem = SnapshotProblem()

    def _extract_subset(self):
        """Extract a subset of DOFs based on correction magnitude (importance sampling)."""
        N = int(self.subset_size * self.Ndof)
        values_ = self.exact_correction
        avg = torch.mean(torch.abs(values_), dim=0)
        max_vals = torch.max(torch.abs(values_), dim=1).values
        p = torch.exp(-torch.mean(max_vals) / (avg + 1e-6))
        p /= p.sum()

        indices = p.multinomial(N, replacement=False)
        self.indices = indices.tensor
        self.Ndof = N

        self.snapshots_train = self.snapshots_train.tensor[:, indices]
        self.snapshots_train = LabelTensor(
            self.snapshots_train, [f's{i}' for i in range(self.snapshots_train.shape[1])])

        self.snapshots_test = self.snapshots_test.tensor[:, indices]
        self.snapshots_test = LabelTensor(
            self.snapshots_test, [f's{i}' for i in range(self.snapshots_test.shape[1])])

        self.coords = self.coords[indices, :]
        self.modes = self.modes[indices, :]
        self.modes = LabelTensor(self.modes, [f'{i}' for i in range(self.reddim)])

        self.exact_correction = self.exact_correction.tensor[:, indices]
        self.exact_correction = LabelTensor(
            self.exact_correction, [f's{i}' for i in range(self.exact_correction.shape[1])])

    def gpu(self):
        """Move all CUDA-capable attributes to GPU."""
        for arg in dir(self):
            if 'cuda' in dir(getattr(self, arg)):
                setattr(self, arg, getattr(self, arg).cuda())
