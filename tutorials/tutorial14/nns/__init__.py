"""
Neural network correction classes for the Corrected ROM.

This package provides correction networks that approximate the residual
between high-fidelity snapshots and their POD-RBF reconstruction.

Classes:
    BaseCorrNet: Abstract base class for all correction networks.
    LinearCorrNet: Linear correction via least squares (reference only).
    QuadLS: Quadratic correction via least squares.
    QuadNet: Spatially-dependent quadratic correction via neural networks.
    QuadNetMu: Parameter and spatially-dependent quadratic correction.
"""
