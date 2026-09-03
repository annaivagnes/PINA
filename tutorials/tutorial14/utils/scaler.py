"""
Min-Max scaler for LabelTensors.

A custom min-max scaler similar to sklearn's MinMaxScaler,
but designed to work with PINA LabelTensors while preserving labels.
"""
import torch
from pina import LabelTensor


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
