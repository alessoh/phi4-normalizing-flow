# README.md
# phi4-normalizing-flow
AI for Quantum Field Theory

# Normalizing Flows for Lattice Field Theory

This repository contains code accompanying Chapter 1 of "Advanced AI Methods 
in Quantum Field Theory." We implement a RealNVP normalizing flow that learns 
to generate field configurations for two-dimensional φ⁴ scalar field theory.

## The Problem

Traditional Monte Carlo methods for simulating quantum field theories suffer 
from **critical slowing down**: near phase transitions, the time required to 
generate statistically independent samples grows dramatically. This is because 
Monte Carlo methods perform a random walk through configuration space, and near 
criticality, correlations extend across the entire system.

## The Solution

Normalizing flows offer a radical alternative. Instead of walking through 
configuration space step by step, we learn a direct mapping from simple 
Gaussian noise to complex field configurations. Each sample is independent 
by construction, eliminating autocorrelation entirely.

The key insight is that we don't need training data—we know the physics 
(the action functional) analytically. We train by minimizing the variational 
free energy, which is equivalent to matching the Boltzmann distribution.

## Results

On a 16×16 lattice with parameters exhibiting spontaneous symmetry breaking:

| Method | Time per independent sample |
|--------|----------------------------|
| Metropolis MCMC | ~1-10 seconds |
| Normalizing Flow | ~0.5 milliseconds |

The flow achieves a **1000-10000x speedup** while producing configurations 
with correct physical observables (magnetization, energy distribution).

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher
- NumPy
- Matplotlib

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/phi4-normalizing-flow.git
cd phi4-normalizing-flow

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

Run the main demonstration:
```bash
python phi4_flow.py
```

This will:
1. Train a normalizing flow for ~5 minutes (faster with GPU)
2. Generate sample field configurations
3. Compare speed with traditional MCMC
4. Save results to `phi4_flow_results.png` and `phi4_flow_model.pt`

## Usage

### Training a Flow
```python
from phi4_flow import Phi4Action, RealNVPFlow, train_flow

# Define the physics
action = Phi4Action(L=16, m_squared=-4.0, lambda_coupling=1.0)

# Create the flow
flow = RealNVPFlow(L=16, n_layers=8, hidden_size=256)

# Train
history = train_flow(flow, action, n_epochs=2000)
```

### Generating Samples
```python
# Generate 1000 independent configurations
samples = flow.sample(1000)  # Shape: (1000, 16, 16)

# Each sample is statistically independent!
# No autocorrelation, no thermalization needed.
```

### Loading a Trained Model
```python
import torch

checkpoint = torch.load('phi4_flow_model.pt')
flow = RealNVPFlow(L=16, n_layers=8, hidden_size=256)
flow.load_state_dict(checkpoint['flow_state_dict'])
```

## Understanding the Code

### The Action Functional

The `Phi4Action` class implements the Euclidean action for φ⁴ theory:
```
S[φ] = Σ_x [ Σ_μ (φ(x) - φ(x+μ))² + m²φ(x)² + λφ(x)⁴ ]
```

For m² < 0, this theory exhibits spontaneous symmetry breaking with 
a second-order phase transition.

### The Flow Architecture

`RealNVPFlow` implements affine coupling layers that transform Gaussian 
noise into field configurations. The key property is that the Jacobian 
determinant can be computed efficiently (O(N) instead of O(N³)).

### The Training Objective

We minimize the variational free energy:
```
L = E_z[ S(f(z)) - log|det(∂f/∂z)| ]
```

This balances two competing objectives:
- **Low action**: Generate configurations with low energy
- **High entropy**: Maintain diversity through the log-determinant term

## Extending the Code

### Different Lattice Sizes
```python
# Larger lattice (slower training, more interesting physics)
action = Phi4Action(L=32, m_squared=-4.0, lambda_coupling=1.0)
flow = RealNVPFlow(L=32, n_layers=12, hidden_size=512)
```

### Different Physics Parameters
```python
# Near the critical point (m² ≈ -4 for λ=1)
action = Phi4Action(L=16, m_squared=-3.8, lambda_coupling=1.0)

# Deep in the broken phase
action = Phi4Action(L=16, m_squared=-6.0, lambda_coupling=1.0)
```

### Using GPU

The code automatically uses GPU if available. To force CPU:
```python
device = torch.device('cpu')
history = train_flow(flow, action, device=device)
```

## Project Structure
```
phi4-normalizing-flow/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── phi4_flow.py       # Main implementation
└── examples/
    └── analyze_samples.py  # Additional analysis tools
```

## Physics Background

### φ⁴ Theory

The φ⁴ theory is the simplest interacting scalar field theory. It serves as:
- A model for the Higgs sector of the Standard Model
- A testbed for non-perturbative methods
- An example of spontaneous symmetry breaking

### Phase Transition

For m² < 0 and appropriate λ, the theory has two degenerate ground states 
at φ = ±√(-m²/2λ). At high temperature (small β), thermal fluctuations 
restore symmetry (⟨φ⟩ = 0). At low temperature, one of the ground states 
is selected spontaneously.

### Critical Slowing Down

Near the critical temperature, fluctuations become correlated over large 
distances. The correlation length ξ diverges as ξ ~ |T - T_c|^(-ν). 
The autocorrelation time τ in MCMC scales as τ ~ ξ^z where z ≈ 2 for 
local update algorithms.

## References

1. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation 
   using Real-NVP. arXiv:1605.08803

2. Albergo, M. S., Kanwar, G., & Shanahan, P. E. (2019). Flow-based 
   generative models for Markov chain Monte Carlo in lattice field theory. 
   Physical Review D, 100(3), 034515.

3. Nicoli, K. A., et al. (2020). Asymptotically unbiased estimation of 
   physical observables with neural samplers. Physical Review E, 101(2), 023304.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```bibtex
@book{alesso2025advanced,
  title={Advanced AI Methods in Quantum Field Theory},
  author={Alesso, H. Peter},
  year={2025},
  publisher={AI HIVE Publications}
}
```