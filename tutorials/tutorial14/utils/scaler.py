"""
Scalers for LabelTensors.

Provides two scalers:
    - MinMaxScaler-like `Scaler` for scaling features to a range.
    - `InfNormScaler`, a torch.nn.Module scaler that normalizes correction
      terms by their mean infinity norm, supporting fit/transform/
      inverse_transform and buffering for checkpointing.
"""
import torch
import torch.nn as nn
from pina import LabelTensor


class InfNormScaler(nn.Module):
    """
    Scaler that normalizes data by the mean of its infinity norm.

    Each row is divided by a single scalar `scale` computed as the mean of
    the per-row infinity norms, so corrections are brought to an O(1) scale,
    which generally improves network training.

    Example:
        scaler = InfNormScaler()
        scaled = scaler.fit_transform(exact_correction)   # normalize
        original = scaler.inverse_transform(scaled)       # restore scale
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))

    def fit(self, data):
        """
        Compute the mean infinity norm of the data.

        Args:
            data: Tensor or LabelTensor.
        """
        data = data.tensor if isinstance(data, LabelTensor) else data
        self.scale = torch.linalg.norm(data, ord=float('inf'), dim=-1).mean()

    def transform(self, data):
        """
        Divide data by the fitted scale.

        Args:
            data: Tensor or LabelTensor.

        Returns:
            Scaled tensor (labels preserved if a LabelTensor was given).
        """
        if isinstance(data, LabelTensor):
            return data.tensor / self.scale
        return data / self.scale

    def inverse_transform(self, data):
        """
        Multiply data by the fitted scale to restore the original magnitude.

        Args:
            data: Tensor or LabelTensor.

        Returns:
            Original-scale tensor.
        """
        if isinstance(data, LabelTensor):
            return data.tensor * self.scale
        return data * self.scale

    def fit_transform(self, data):
        """
        Fit and transform in one step.
        """
        self.fit(data)
        return self.transform(data)


class Scaler:
    """
    Min-Max scaler for LabelTensors.

    Scales features to a given range (default [0, 1]) while preserving
    LabelTensor labels.

    Args:
        feature_range: Tuple (min, max) for the desired range.
        axis: Axis along which to compute min/max (default: 0).
    """

    def __init__(self, feature_range=(0., 1.), axis=0):
        self.feature_range = feature_range
        self.axis = axis
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.labels = None

    def fit(self, data):
        """
        Compute min and max values from data.

        Args:
            data: Input LabelTensor.
        """
        self.labels = data.labels
        data_ = self._avoid_nan_in_scaling(data)
        self.data_min_ = torch.min(data_, dim=self.axis).values
        self.data_max_ = torch.max(data_, dim=self.axis).values
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.data_max_ - self.data_min_ + 1e-8)
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, data):
        """
        Scale data using the fitted min and max.

        Args:
            data: Input LabelTensor.

        Returns:
            Scaled LabelTensor.
        """
        data_ = self._avoid_nan_in_scaling(data)
        scaled = data_ * self.scale_ + self.min_
        return LabelTensor(scaled, self.labels) if self.labels is not None else scaled

    def fit_transform(self, data):
        """
        Fit and transform in one step.

        Args:
            data: Input LabelTensor.

        Returns:
            Scaled LabelTensor.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        """
        Reverse the scaling.

        Args:
            data: Scaled LabelTensor.

        Returns:
            Original-scale LabelTensor.
        """
        data_ = self._avoid_nan_in_scaling(data)
        inversed = (data_ - self.min_) / (self.scale_ + 1e-8)
        return LabelTensor(inversed, self.labels) if self.labels is not None else inversed

    def _avoid_nan_in_scaling(self, data):
        """Replace NaN values with zeros."""
        if isinstance(data, LabelTensor):
            return torch.nan_to_num(data.tensor, nan=0.0)
        return torch.nan_to_num(data, nan=0.0)
