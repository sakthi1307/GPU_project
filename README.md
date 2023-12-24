
Presentation Youtube Link : https://youtu.be/_bfZMnXx8Po

This repository contains a CUDA implementation of a differential equation solver using the Runge-Kutta method. The solver is designed to run on GPU for efficient parallel computation.

## Features
- Solves a system of differential equations for multiple particles in parallel on a GPU.
- Utilizes the Runge-Kutta method for numerical integration.
- Shared memory optimization for improved performance.

Requirements
- Python 3.x
- NumPy
- Numba
- CUDA-enabled GPU

## Installation
1. Install Conda 
2. Install numba
```
conda install numba
```
4. Clone the repository:
```
git clone https://github.com/UCR-CSEE217/finalproject-f23-scorpion.git
```

5. Install dependencies( only if minimal version of conda is installed (miniconda)):
```
pip install -r requirements.txt
```
## Usage

```
numba main.py
```
