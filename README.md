# Adaptive MPC for Quadruped Control

This project implements an adaptive Model Predictive Control (MPC) system for quadruped robot control using MuJoCo for simulation.

## Project Structure

```
adaptive-mpc/
├── src/
│   └── adaptive_mpc/
│       ├── __init__.py
│       ├── math_utils.py
│       └── sim_utils.py
├── scripts/
│   └── main.py
├── assets/
│   └── unitree_go1/
│       └── scene.xml
├── pyproject.toml
└── README.md
```

The project is organized as follows:
- `src/adaptive_mpc/`: Core package containing utility functions and shared code
- `scripts/`: Executable scripts that use the package
- `assets/`: Resources needed by the scripts

## Installation

```bash
pip install .
```

## Usage

```bash
adaptive-mpc
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Notes
adaptive needs:
B
P
e_hat
e