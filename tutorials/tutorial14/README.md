# Tutorial 14: Corrected Reduced Order Model (ROM)

This tutorial implements a **Corrected Reduced Order Model** for parametric
PDEs. The core idea is to build a baseline ROM using **POD** (Proper
Orthogonal Decomposition) plus **RBF** (Radial Basis Function) interpolation,
and then add a learned **correction term** that reintroduces the contribution
of neglected POD modes, improving accuracy at reduced computational cost.

---

## Overview

Given a set of high-fidelity snapshots `u(mu)` parameterized by `mu`, the
ROM architecture is:

```
                    POD (reduction + expansion)
    snapshots  --reduce-->  coefficients  --RBF-->  coefficients(mu)
                                                      |
                                                      v
    prediction(mu) = POD.expand(RBF(mu)) + Correction(mu)
```

The **exact correction** that the ROM tries to approximate is:

```
exact_correction = snaps - POD.expand(POD.reduce(snaps))
```

The correction network is trained to match this exact correction, so the
final prediction recovers the neglected-modes contribution.

### Supported datasets

The datasets are taken from the library `smithers`.

| Problem | Dataset | Variables |
|---|---|---|
| Backward-facing step | `NavierStokesDataset` | `mag(v)` etc. |
| Lid-driven cavity | `LidCavity` | `mag(v)` etc. |

---

## Correction strategies

Three correction strategies are provided, selectable via the `--correction`
flag in the run scripts. They differ in how the correction operator `C` is
built and whether it depends on the parameter `mu`.

| Strategy | Class | Formula | Operator | Parameter dep. |
|---|---|---|---|---|
| `quadls` | `QuadLS` | `coeff^T @ C @ coeff` | solved via least squares | No |
| `quadnet` | `QuadNet` | `coeff^T @ C(x,y) @ coeff` | learned by NNs on modes + coords | No |
| `quadnetmu` | `QuadNetMu` | `coeff^T @ C(mu,x,y) @ coeff` | learned by NNs on modes + coords + mu | Yes |

**QuadLS** solves for the operator `C` directly via `torch.linalg.lstsq`,
so no neural network training is required.

**QuadNet** learns a spatially-varying operator `C(x,y)` using two
FeedForward networks (one on POD modes, one on spatial coordinates), then
projects through a reduction layer. It is parameter-independent, and space-continuous.

**QuadNetMu** extends QuadNet with a third FeedForward network acting on the
parameter `mu`, producing a parameter- and space-dependent operator
`C(mu, x, y)`.

---

## Repository structure

```
tutorial14/
├── environment.yml            # conda environment specification
├── .gitignore
├── README.md
│
├── nns/                       # neural network correction classes
│   ├── base_corr.py           #   abstract base class (BaseCorrNet)
│   ├── quadls.py              #   quadratic correction via least squares (QuadLS)
│   ├── quadnet.py             #   spatially-dependent quadratic NN (QuadNet)
│   ├── quadnet_mu.py          #   parameter + spatially-dependent NN (QuadNetMu)
│   └── linear_corr.py         #   linear correction (reference only, not used)
│
├── rom/                       # ROM solver and baseline
│   ├── corrected_rom.py       #   CorrectedROM solver
│   └── pod_rbf.py             #   PODRBF baseline + err() metric
│
├── problems/                  # dataset-specific data pipelines
│   ├── setup_backstep.py      #   BackstepProblem
│   └── setup_cavity.py        #   CavityProblem
│
├── utils/                     # shared utilities
│   ├── plotting.py            #   plot() for triangular mesh fields
│   └── scaler.py              #   Min-Max scaler for LabelTensors
│
├── scripts/                   # executable entry points
│   ├── run_backstep.py        #   unified backstep runner
│   ├── run_cavity.py          #   unified cavity runner
│   └── run_pod_rbf.py         #   baseline POD-RBF analysis
│
└── notebooks/                 # illustrative Jupyter notebooks
    ├── error_comparison_backstep.ipynb
    ├── quadnet_mu.ipynb
    ├── quadnet_mu_backstep.ipynb
    ├── sensitivity_backstep.ipynb
    └── sensitivity_cavity.ipynb
```

### Package contents

| Module | Class / Function | Description |
|---|---|---|
| `nns.base_corr` | `BaseCorrNet` | Abstract base for correction networks |
| `nns.quadls` | `QuadLS` | Quadratic correction via least squares |
| `nns.quadnet` | `QuadNet` | Spatially-dependent quadratic NN |
| `nns.quadnet_mu` | `QuadNetMu` | Parameter + spatially-dependent quadratic NN |
| `nns.linear_corr` | `LinearCorrNet` | Reference implementation (not used in scripts) |
| `rom.corrected_rom` | `CorrectedROM` | Main ROM solver combining POD + RBF + correction |
| `rom.corrected_rom` | `CorrectedROM.compute_exact_correction` | Static factory for the exact correction |
| `rom.pod_rbf` | `PODRBF` | Baseline un-corrected ROM (POD + RBF) |
| `rom.pod_rbf` | `err` | Mean/median relative L2 error metric |
| `problems.setup_backstep` | `BackstepProblem` | Backstep data pipeline |
| `problems.setup_cavity` | `CavityProblem` | Cavity data pipeline |
| `utils.plotting` | `plot` | Plot fields on a triangular mesh |
| `utils.scaler` | `Scaler` | Min-Max scaler for LabelTensors |

---

## Installation

Create a conda environment with all dependencies:

```bash
conda env create -f environment.yml -n quadrom
conda activate quadrom
```

This environment includes `torch`, `pina`, `smithers`, `scikit-learn`,
`matplotlib`, `scipy`, and `pytorch-lightning`.

---

## Usage

Run the scripts from the **tutorial14 directory** so the local packages
(`nns`, `rom`, `problems`, `utils`) are importable.

### Run a corrected ROM (backward-facing step)

```bash
# Quadratic least squares correction (no training, only LS)
python scripts/run_backstep.py --correction quadls --reddim 3 --train 100

# Neural-network correction (space dependent)
python scripts/run_backstep.py --correction quadnet --reddim 3 --epochs 5000

# Neural-network correction (parameter and space dependent)
python scripts/run_backstep.py --correction quadnetmu --reddim 3 --train 200
```

### Run a corrected ROM (lid-driven cavity)

```bash
python scripts/run_cavity.py --correction quadls --reddim 3 --train 1
python scripts/run_cavity.py --correction quadnet --reddim 3 --epochs 5000
python scripts/run_cavity.py --correction quadnetmu --reddim 3
```

### Load a trained model from a checkpoint

```bash
python scripts/run_backstep.py --correction quadnet --load <checkpoint_dir> --version <version_id> --epochs <epochs>
```

Loading evaluates the model on train/test and produces comparison plots of
the truth, the corrected prediction, and the baseline POD-RBF prediction.

### Baseline POD-RBF analysis

```bash
python scripts/run_pod_rbf.py --reddim 3
python scripts/run_pod_rbf.py --reddim 3 --field mag(v)
```

### Command-line arguments

| Argument | Description |
|---|---|
| `--correction` | Correction strategy: `quadls`, `quadnet`, or `quadnetmu` (required) |
| `--reddim` | Reduced dimension (number of POD modes), default `3` |
| `--train` | Training set size, default `10` |
| `--field` | Snapshot field to reduce, default `mag(v)` |
| `--load` | Directory containing a checkpoint to load for evaluation |
| `--version` | Model version to load (PyTorch-Lightning version index) |
| `--epochs` | Number of training epochs, default `1000` |

---

## Method details

### POD-RBF baseline

The baseline (`PODRBF`) fits a POD basis on training snapshots, projects the
snapshots to a reduced space, and interpolates the reduced coefficients with
an RBF network. The prediction is `POD.expand(RBF(mu))`. This ignores the
truncated modes and therefore carries an intrinsic approximation error.

### Corrected ROM

The `CorrectedROM` solver combines three sub-networks into a `ModuleDict`:

- **reduction_network**: a fitted `PODBlock`
- **interpolation_network**: a fitted `RBFBlock`
- **correction_network**: a `QuadLS`, `QuadNet`, or `QuadNetMu`

The forward pass computes:

```
prediction(mu) = POD.expand(RBF(mu)) + CorrectionNetwork(mu, RBF(mu))
```

The training loss combines the MSE between predicted and exact corrections
with an orthogonality regularizer:

```
loss = MSE(correction_pred, correction_exact) + beta * ||V^T C||            (beta = 0.001)
```

The `CorrectionNetwork` is fitted on the exact correction data before
training via its `fit()` method (for `QuadLS` this solves the least-squares
problem directly).

### Subset selection

Both `BackstepProblem` and `CavityProblem` support extracting a subset of
the degrees of freedom based on correction magnitude (importance sampling).
Pass `subset=<fraction>` to enable this, which samples DOFs with larger
corrections more frequently (used for sparse training).

---

## Notebooks

| Notebook | Description |
|---|---|
| `error_comparison_backstep.ipynb` | Error analysis across correction strategies |
| `quadnet_mu.ipynb` | QuadNetMu on the cavity dataset |
| `quadnet_mu_backstep.ipynb` | QuadNetMu on the backstep dataset |
| `sensitivity_backstep.ipynb` | Sensitivity analysis on the backstep dataset |
| `sensitivity_cavity.ipynb` | Sensitivity analysis on the cavity dataset |

Note: notebooks import the local packages and should be run from the
`tutorial14` directory (or with `tutorial14` on the Python path).
