# phi4_flow.py
"""
Normalizing Flow for φ⁴ Scalar Field Theory

This module implements a RealNVP normalizing flow that learns to generate
field configurations distributed according to the Boltzmann distribution
exp(-S[φ]) for two-dimensional φ⁴ theory.

The key insight is that we don't need training data - we know the action S[φ]
analytically, so we can train by minimizing the variational free energy
directly. This is physics-informed generative modeling.

Usage:
    python phi4_flow.py

The script will train a flow and demonstrate generation of field configurations,
comparing the speed to traditional MCMC methods.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time


# =============================================================================
# Physics: The Action Functional
# =============================================================================

class Phi4Action:
    """
    The action functional for two-dimensional φ⁴ scalar field theory.
    
    The Euclidean action is:
        S[φ] = Σ_x [ Σ_μ (φ(x) - φ(x+μ))² + m²φ(x)² + λφ(x)⁴ ]
    
    where the first term is the discretized kinetic energy (favoring smooth
    configurations), the second is the mass term, and the third is the
    quartic self-interaction that makes the theory interacting.
    
    For m² < 0 and appropriate λ, the theory exhibits spontaneous symmetry
    breaking with a phase transition between ordered and disordered phases.
    
    Attributes:
        L: Linear size of the square lattice
        m_squared: Mass squared parameter (can be negative)
        lambda_coupling: Quartic coupling strength
    """
    
    def __init__(self, L: int, m_squared: float = -4.0, lambda_coupling: float = 1.0):
        self.L = L
        self.m_squared = m_squared
        self.lambda_coupling = lambda_coupling
    
    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the action for a batch of field configurations.
        
        Args:
            phi: Field configurations of shape (batch_size, L, L)
            
        Returns:
            Action values of shape (batch_size,)
        """
        # Kinetic term: sum of (φ(x) - φ(x+μ))² over all nearest neighbors
        # We use periodic boundary conditions implemented via torch.roll
        phi_shift_x = torch.roll(phi, shifts=1, dims=1)
        phi_shift_y = torch.roll(phi, shifts=1, dims=2)
        
        kinetic = torch.sum((phi - phi_shift_x)**2 + (phi - phi_shift_y)**2, dim=(1, 2))
        
        # Potential term: m²φ² + λφ⁴ summed over all sites
        potential = torch.sum(
            self.m_squared * phi**2 + self.lambda_coupling * phi**4,
            dim=(1, 2)
        )
        
        return kinetic + potential
    
    def mean_field_magnetization(self) -> float:
        """
        Compute the mean-field prediction for the magnetization.
        
        In mean-field theory, the ordered phase has |⟨φ⟩| = √(-m²/2λ) for m² < 0.
        This provides a reference for validating our samples.
        """
        if self.m_squared >= 0:
            return 0.0
        return np.sqrt(-self.m_squared / (2 * self.lambda_coupling))


# =============================================================================
# Neural Network Components
# =============================================================================

class ScaleTranslateNetwork(nn.Module):
    """
    Neural network that predicts scale and translation parameters.
    
    This network takes the "frozen" half of the lattice as input and produces
    scale (s) and translation (t) factors for transforming the other half.
    The architecture can be simple because the flow's expressiveness comes
    from stacking many coupling layers, not from individual layer complexity.
    
    Attributes:
        input_size: Number of input features (half the lattice)
        hidden_size: Size of hidden layers
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size * 2)  # Output both s and t
        )
        
        # Initialize final layer to near-identity transformation
        # This stabilizes early training
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scale and translation from input.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            s: Scale factors of shape (batch_size, input_size)
            t: Translation factors of shape (batch_size, input_size)
        """
        output = self.network(x)
        s, t = output.chunk(2, dim=1)
        
        # Tanh bounds the scale to prevent numerical instability
        # The factor of 2.0 allows moderate scaling while maintaining stability
        s = 2.0 * torch.tanh(s)
        
        return s, t


class AffineCouplingLayer(nn.Module):
    """
    A single affine coupling layer implementing the RealNVP transformation.
    
    This layer splits the input into two parts using a checkerboard mask,
    leaves one part unchanged, and applies an affine transformation to the
    other part based on the unchanged values.
    
    The transformation is:
        y_A = x_A                           (masked sites unchanged)
        y_B = x_B * exp(s(x_A)) + t(x_A)    (unmasked sites transformed)
    
    The log-determinant of the Jacobian is simply sum(s) over unmasked sites.
    
    Attributes:
        mask: Boolean tensor indicating which sites are frozen (True) vs transformed
        net: Neural network predicting scale and translation
    """
    
    def __init__(self, lattice_size: int, mask: torch.Tensor, hidden_size: int = 256):
        super().__init__()
        
        self.register_buffer('mask', mask)
        self.register_buffer('mask_complement', ~mask)
        
        # Count of masked (input) and unmasked (output) sites
        n_masked = mask.sum().item()
        
        self.net = ScaleTranslateNetwork(n_masked, hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, lattice_size)
            
        Returns:
            y: Transformed tensor of shape (batch_size, lattice_size)
            log_det: Log-determinant of Jacobian of shape (batch_size,)
        """
        # Extract masked (frozen) sites
        x_masked = x[:, self.mask]
        
        # Predict transformation parameters from frozen sites
        s, t = self.net(x_masked)
        
        # Apply affine transformation to unmasked sites
        y = x.clone()
        x_unmasked = x[:, self.mask_complement]
        y_unmasked = x_unmasked * torch.exp(s) + t
        y[:, self.mask_complement] = y_unmasked
        
        # Log-determinant is sum of scale factors
        log_det = s.sum(dim=1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse transformation.
        
        This is needed for computing the latent representation of a given
        field configuration, useful for analysis and validation.
        
        Args:
            y: Transformed tensor of shape (batch_size, lattice_size)
            
        Returns:
            x: Original tensor of shape (batch_size, lattice_size)
        """
        # Extract masked sites (unchanged in forward pass)
        y_masked = y[:, self.mask]
        
        # Predict transformation parameters
        s, t = self.net(y_masked)
        
        # Invert the affine transformation
        x = y.clone()
        y_unmasked = y[:, self.mask_complement]
        x_unmasked = (y_unmasked - t) * torch.exp(-s)
        x[:, self.mask_complement] = x_unmasked
        
        return x


class RealNVPFlow(nn.Module):
    """
    Complete RealNVP normalizing flow for lattice field theory.
    
    This model stacks multiple affine coupling layers with alternating
    checkerboard masks, creating a deep invertible transformation that
    can map simple Gaussian noise to complex field configurations.
    
    The checkerboard pattern ensures that neighboring sites are in opposite
    groups, allowing local correlations to develop through the neural networks
    that connect the two groups.
    
    Attributes:
        L: Linear lattice size
        n_layers: Number of coupling layers
        layers: ModuleList of coupling layers
    """
    
    def __init__(self, L: int, n_layers: int = 8, hidden_size: int = 256):
        super().__init__()
        
        self.L = L
        self.lattice_size = L * L
        self.n_layers = n_layers
        
        # Create checkerboard masks
        mask_even = self._create_checkerboard_mask(L, parity=0)
        mask_odd = self._create_checkerboard_mask(L, parity=1)
        
        # Stack coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = mask_even if i % 2 == 0 else mask_odd
            self.layers.append(AffineCouplingLayer(self.lattice_size, mask, hidden_size))
    
    def _create_checkerboard_mask(self, L: int, parity: int) -> torch.Tensor:
        """Create a checkerboard mask with given parity."""
        mask = torch.zeros(L * L, dtype=torch.bool)
        for i in range(L):
            for j in range(L):
                if (i + j) % 2 == parity:
                    mask[i * L + j] = True
        return mask
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform noise to field configurations.
        
        Args:
            z: Gaussian noise of shape (batch_size, lattice_size)
            
        Returns:
            phi: Field configurations of shape (batch_size, lattice_size)
            log_det: Total log-determinant of shape (batch_size,)
        """
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        
        for layer in self.layers:
            x, log_det = layer(x)
            total_log_det = total_log_det + log_det
        
        return x, total_log_det
    
    def inverse(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Transform field configurations to latent space.
        
        Args:
            phi: Field configurations of shape (batch_size, lattice_size)
            
        Returns:
            z: Latent representations of shape (batch_size, lattice_size)
        """
        x = phi
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
    
    def sample(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate field configuration samples.
        
        Args:
            n_samples: Number of configurations to generate
            device: Device for computation
            
        Returns:
            Field configurations of shape (n_samples, L, L)
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(n_samples, self.lattice_size, device=device)
        phi_flat, _ = self.forward(z)
        
        return phi_flat.view(n_samples, self.L, self.L)
    
    def log_prob(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of configurations under the flow.
        
        This uses the change of variables formula:
        log p(φ) = log p(z) - log|det(∂φ/∂z)|
        
        where z = f⁻¹(φ) is the latent representation.
        
        Args:
            phi: Field configurations of shape (batch_size, L, L)
            
        Returns:
            Log probabilities of shape (batch_size,)
        """
        phi_flat = phi.view(phi.shape[0], -1)
        z = self.inverse(phi_flat)
        
        # Log probability under standard Gaussian prior
        log_pz = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.lattice_size * np.log(2 * np.pi)
        
        # We need the forward log-det, but we computed the inverse
        # For a proper implementation, we'd track this during inverse
        # Here we recompute in forward direction for correctness
        _, log_det = self.forward(z)
        
        return log_pz - log_det


# =============================================================================
# Training
# =============================================================================

def train_flow(
    flow: RealNVPFlow,
    action: Phi4Action,
    n_epochs: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    print_frequency: int = 100
) -> dict:
    """
    Train the normalizing flow to generate Boltzmann-distributed configurations.
    
    The training objective is the variational free energy:
        L = E_z[ S(f(z)) - log|det(∂f/∂z)| ]
    
    This is equivalent to minimizing the KL divergence between the flow's
    distribution and the Boltzmann distribution exp(-S)/Z.
    
    The key insight is that we don't need samples from the target distribution.
    We sample from our flow, compute the action (which we know analytically),
    and train to minimize action while maintaining entropy (via the log-det term).
    
    Args:
        flow: The normalizing flow model
        action: The action functional
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device for computation
        print_frequency: How often to print progress
        
    Returns:
        Dictionary containing training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    history = {'loss': [], 'action': [], 'log_det': [], 'magnetization': []}
    
    print(f"Training on {device}")
    print(f"Lattice size: {action.L}x{action.L}")
    print(f"Parameters: m² = {action.m_squared}, λ = {action.lambda_coupling}")
    print(f"Mean-field magnetization: {action.mean_field_magnetization():.4f}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        # Sample from Gaussian prior
        z = torch.randn(batch_size, flow.lattice_size, device=device)
        
        # Transform to field configurations
        phi_flat, log_det = flow.forward(z)
        phi = phi_flat.view(batch_size, action.L, action.L)
        
        # Compute action (energy) of generated configurations
        S = action(phi)
        
        # Variational free energy loss
        # We want configurations with low action (energy) but high entropy (log_det)
        loss = (S - log_det).mean()
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Record history
        with torch.no_grad():
            magnetization = phi.mean(dim=(1, 2)).abs().mean()
            
            history['loss'].append(loss.item())
            history['action'].append(S.mean().item())
            history['log_det'].append(log_det.mean().item())
            history['magnetization'].append(magnetization.item())
        
        # Print progress
        if (epoch + 1) % print_frequency == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():10.2f} | "
                  f"Action: {S.mean().item():10.2f} | "
                  f"|M|: {magnetization.item():.4f}")
    
    return history


# =============================================================================
# Comparison with MCMC
# =============================================================================

def mcmc_sample(
    action: Phi4Action,
    n_samples: int,
    n_thermalization: int = 1000,
    n_steps_between: int = 100,
    device: torch.device = None
) -> Tuple[torch.Tensor, float]:
    """
    Generate samples using Metropolis-Hastings MCMC.
    
    This provides a baseline for comparison with the normalizing flow.
    We use single-site Metropolis updates, which exhibits critical slowing
    down near phase transitions.
    
    Args:
        action: The action functional
        n_samples: Number of samples to generate
        n_thermalization: Thermalization sweeps before sampling
        n_steps_between: Sweeps between samples for decorrelation
        device: Device for computation
        
    Returns:
        samples: Field configurations of shape (n_samples, L, L)
        time_per_sample: Average time per independent sample
    """
    if device is None:
        device = torch.device('cpu')  # MCMC is typically faster on CPU
    
    L = action.L
    phi = torch.randn(L, L, device=device) * 0.1
    
    def sweep(phi, beta=1.0):
        """One Metropolis sweep over all sites."""
        for i in range(L):
            for j in range(L):
                # Compute local action contribution
                neighbors = (
                    phi[(i+1) % L, j] + phi[(i-1) % L, j] +
                    phi[i, (j+1) % L] + phi[i, (j-1) % L]
                )
                
                old_val = phi[i, j]
                new_val = old_val + torch.randn(1, device=device).item() * 0.5
                
                # Change in action
                delta_S = (
                    4 * (new_val**2 - old_val**2) -
                    2 * neighbors * (new_val - old_val) +
                    action.m_squared * (new_val**2 - old_val**2) +
                    action.lambda_coupling * (new_val**4 - old_val**4)
                )
                
                # Metropolis acceptance
                if delta_S < 0 or torch.rand(1).item() < np.exp(-beta * delta_S):
                    phi[i, j] = new_val
        
        return phi
    
    # Thermalization
    print(f"MCMC: Thermalizing ({n_thermalization} sweeps)...")
    for _ in range(n_thermalization):
        phi = sweep(phi)
    
    # Sampling
    samples = []
    start_time = time.time()
    
    print(f"MCMC: Generating {n_samples} samples...")
    for n in range(n_samples):
        for _ in range(n_steps_between):
            phi = sweep(phi)
        samples.append(phi.clone())
        
        if (n + 1) % 10 == 0:
            print(f"  Sample {n+1}/{n_samples}")
    
    total_time = time.time() - start_time
    time_per_sample = total_time / n_samples
    
    return torch.stack(samples), time_per_sample


def compare_methods(
    flow: RealNVPFlow,
    action: Phi4Action,
    n_samples: int = 100,
    device: torch.device = None
) -> dict:
    """
    Compare normalizing flow and MCMC sampling.
    
    This function generates samples using both methods and compares
    their speed and the quality of physical observables.
    
    Args:
        flow: Trained normalizing flow
        action: Action functional
        n_samples: Number of samples for comparison
        device: Device for flow sampling
        
    Returns:
        Dictionary containing comparison results
    """
    results = {}
    
    # Flow sampling
    print("\n" + "=" * 60)
    print("NORMALIZING FLOW SAMPLING")
    print("=" * 60)
    
    flow.eval()
    with torch.no_grad():
        start_time = time.time()
        flow_samples = flow.sample(n_samples, device)
        flow_time = time.time() - start_time
    
    flow_mag = flow_samples.mean(dim=(1, 2)).abs()
    
    results['flow_time'] = flow_time
    results['flow_time_per_sample'] = flow_time / n_samples
    results['flow_magnetization_mean'] = flow_mag.mean().item()
    results['flow_magnetization_std'] = flow_mag.std().item()
    
    print(f"Time for {n_samples} samples: {flow_time:.4f} seconds")
    print(f"Time per sample: {flow_time/n_samples*1000:.4f} ms")
    print(f"Mean |M|: {results['flow_magnetization_mean']:.4f} ± {results['flow_magnetization_std']:.4f}")
    
    # MCMC sampling (use fewer samples because it's slow)
    print("\n" + "=" * 60)
    print("MCMC SAMPLING (Metropolis-Hastings)")
    print("=" * 60)
    
    n_mcmc = min(n_samples, 20)  # MCMC is slow, use fewer samples
    mcmc_samples, mcmc_time_per = mcmc_sample(
        action, n_mcmc,
        n_thermalization=500,
        n_steps_between=50
    )
    
    mcmc_mag = mcmc_samples.mean(dim=(1, 2)).abs()
    
    results['mcmc_time_per_sample'] = mcmc_time_per
    results['mcmc_magnetization_mean'] = mcmc_mag.mean().item()
    results['mcmc_magnetization_std'] = mcmc_mag.std().item()
    
    print(f"Time per sample: {mcmc_time_per:.4f} seconds")
    print(f"Mean |M|: {results['mcmc_magnetization_mean']:.4f} ± {results['mcmc_magnetization_std']:.4f}")
    
    # Summary
    speedup = mcmc_time_per / results['flow_time_per_sample']
    results['speedup'] = speedup
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Speedup: {speedup:.1f}x faster with normalizing flow")
    print(f"Mean-field prediction for |M|: {action.mean_field_magnetization():.4f}")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    flow: RealNVPFlow,
    action: Phi4Action,
    history: dict,
    output_path: str = "phi4_flow_results.png"
):
    """
    Create visualization of training and samples.
    
    Args:
        flow: Trained flow model
        action: Action functional
        history: Training history dictionary
        output_path: Path for saving figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Variational Free Energy')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Action and log-det
    axes[0, 1].plot(history['action'], label='Action ⟨S⟩')
    axes[0, 1].plot(history['log_det'], label='Log-det ⟨log|J|⟩')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Action vs Entropy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Magnetization during training
    axes[0, 2].plot(history['magnetization'])
    axes[0, 2].axhline(y=action.mean_field_magnetization(), color='r', linestyle='--',
                       label=f'Mean-field: {action.mean_field_magnetization():.3f}')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('|M|')
    axes[0, 2].set_title('Magnetization')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Sample configurations
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(3)
    
    for i in range(3):
        im = axes[1, i].imshow(samples[i].cpu().numpy(), cmap='coolwarm', 
                               vmin=-3, vmax=3)
        mag = samples[i].mean().abs().item()
        axes[1, i].set_title(f'Sample {i+1}, |M| = {mag:.3f}')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function demonstrating the complete workflow."""
    
    print("=" * 60)
    print("Normalizing Flow for φ⁴ Scalar Field Theory")
    print("=" * 60)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Physics parameters
    L = 16  # Lattice size
    m_squared = -4.0  # Negative mass squared -> symmetry breaking
    lambda_coupling = 1.0
    
    # Create action and flow
    action = Phi4Action(L, m_squared, lambda_coupling)
    flow = RealNVPFlow(L, n_layers=8, hidden_size=256)
    
    print(f"\nModel has {sum(p.numel() for p in flow.parameters()):,} parameters")
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = train_flow(
        flow, action,
        n_epochs=2000,
        batch_size=64,
        learning_rate=1e-3,
        device=device,
        print_frequency=200
    )
    
    # Compare with MCMC
    comparison = compare_methods(flow, action, n_samples=100, device=device)
    
    # Visualize
    plot_results(flow, action, history)
    
    # Save model
    torch.save({
        'flow_state_dict': flow.state_dict(),
        'action_params': {'L': L, 'm_squared': m_squared, 'lambda_coupling': lambda_coupling},
        'history': history
    }, 'phi4_flow_model.pt')
    print("\nModel saved to phi4_flow_model.pt")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()