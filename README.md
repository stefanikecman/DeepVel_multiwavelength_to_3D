# Multi-height Inference of Plasma Flows in the Solar atmosphere using Deep Learning

## About

DeepVel 3D Velocity is a PyTorch-based framework for prediction and analysis of the 3D velocity vector on different optical depths from hyperspectral polarization data.
It includes data preparation steps, model training, evaluation, and visualization scripts.

The DeepVel network is originally published in Ramos, A. A., Requerey, I. S., & Vitas, N. (2017). DeepVel: deep learning for the estimation of horizontal velocities at the solar surface. Astronomy & Astrophysics, 604, A11.
https://www.aanda.org/articles/aa/pdf/2017/08/aa30783-17.pdf

## Features

- Neural network models, based on the DeepVel network, for velocity field prediction with variable spectral, temporal and spatial input extent
- Hybrid input models with multiple physical inputs (Stokes I and Stokes V)
- Data normalization/denormalization codes
- Evaluation metrics: MSE, RMSE, Pearson correlation (vx, vy, vz), slope (vx, vy, vz)- for the velocities, divergence and vorticity
- Visualization: prediction maps, scatter plots

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone <repo_url>
cd DeepVel_3D_velocity
pip install -r requirements.txt
```

### Data Preparation

Methods for data preparation, normalization and other preprocessing steps are located in `prepare_data.py`.

### Training
DeepVel model definition is in `deepvel_torch.py`

Train a model using:

```bash
python deepvel_torch.py
```
Hybrid-input DeepVel model variants are defined in `hybrid_deepvel_torch.py`

Train a model using:

```bash
python hybrid_deepvel_torch.py
```

Multiheight DeepVel model definition is in `multiheight_deepvel_torch.py`

Train a model using:

```bash
python multiheight_deepvel_torch.py
```
Hybrid-input DeepVel model variants are defined in `multiheight_hybrid_deepvel_torch.py`

Train a model using:

```bash
python multiheight_hybrid_deepvel_torch.py
```

### Evaluation and Visualization

Evaluate models and generate plots (first uncomment the chosen evaluation and adjust the paths):

```bash
python test_stokes_deepvel.py
```

or

```bash
python test_hybrid_stokes.py
```

Commands for testing multiheight models are:

```bash
python test_multiheight_stokes_deepvel.py
```

or

```bash
python test_multiheight_hybrid_deepvel.py
```

Methods for output visualization are located in ```metrics_and_plotting.py```

## Data
This project uses simulation data generated with the MURaM  code, courtesy of Matthias Rempel (High Altitude Observatory).
All polarization data (Stokes I and Stokes V), as well as velocity field data used in this project is based on MURaM simulations.
Data is located on the KIS servers and can be obtained upon request.
