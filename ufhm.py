import jax
import jax.numpy as jnp
import pennylane as qml
import optax
from typing import Callable, Tuple, Optional

# Enforce double precision for quantum gradients
jax.config.update("jax_enable_x64", True)

class TopologicalSuperManifold:
    """
    Manages the continuous relaxation of discrete quantum circuit depths.
    Allows for differentiable architecture search by interpolating costs.
    """
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.depth_array = jnp.arange(1, max_depth + 1)

    def interpolate_costs(self, costs: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolates scalar cost values across discrete circuit depths.
        Memory footprint is O(1) compared to state vector interpolation.
        """
        w_mask = jnp.maximum(0, 1.0 - jnp.abs(self.depth_array - delta))
        w_mask = w_mask / jnp.sum(w_mask)
        return jnp.sum(w_mask * costs)

class UFHMQuantumOptimizer:
    """
    Unified Framework for Hybrid Meta-optimization (UFHM).
    A Differentiable Quantum Architecture Search (DQAS) optimizer for finding
    the optimal depth of parameterized quantum circuits.
    """
    def __init__(self, 
                 num_wires: int, 
                 max_depth: int, 
                 ansatz_block: Optional[Callable] = None,
                 complexity_penalty: float = 0.05):
        self.num_wires = num_wires
        self.max_depth = max_depth
        self.complexity_penalty = complexity_penalty
        self.dev = qml.device('default.qubit', wires=num_wires)
        self.manifold = TopologicalSuperManifold(max_depth)
        
        # Allow users to inject their own quantum circuits, otherwise use MERA
        self.ansatz_block = ansatz_block if ansatz_block else self._default_mera_block
        
        self._build_monolith_qnodes()

    @staticmethod
    def _default_mera_block(weights: jnp.ndarray, wires: list):
        """Default hardware-efficient MERA ansatz block."""
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])

    def _build_monolith_qnodes(self):
        """Programmatically constructs fail-proof static QNodes for JAX tracing."""
        self.circuits = []
        for depth in range(1, self.max_depth + 1):
            def make_qnode(d: int):
                @qml.qnode(self.dev, interface="jax")
                def _circuit(w_slice):
                    for i in range(d):
                        # Apply the user-defined or default ansatz
                        qml.MERA(range(self.num_wires), 2, self.ansatz_block, 2, w_slice[i])
                    return qml.density_matrix(wires=[0, 1])
                return _circuit
            self.circuits.append(make_qnode(depth))

    def total_loss(self, params: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Calculates the purity-based loss with a depth complexity penalty."""
        template_weights, delta = params
        
        # 1. Compute scalar cost (purity) for each depth independently
        costs = jnp.stack([
            jnp.real(jnp.trace(circ(template_weights) @ circ(template_weights)))
            for circ in self.circuits
        ])
        
        # 2. Interpolate the costs mathematically
        interpolated_cost = self.manifold.interpolate_costs(costs, delta)
        
        # 3. Penalize larger topologies (depth)
        penalty = self.complexity_penalty * delta
        return interpolated_cost + penalty

    def fit(self, 
            initial_weights: jnp.ndarray, 
            initial_delta: jnp.ndarray, 
            epochs: int = 50, 
            lr: float = 0.05) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Runs the DQAS optimization loop using Optax Cosine Decay."""
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs)
        optimizer = optax.adam(learning_rate=schedule)
        
        params = (initial_weights, initial_delta)
        opt_state = optimizer.init(params)
        
        @jax.jit
        def update_step(p, o_state):
            loss_val, grads = jax.value_and_grad(self.total_loss)(p)
            updates, new_o_state = optimizer.update(grads, o_state, p)
            new_p = optax.apply_updates(p, updates)
            
            # Clip delta safely within structural bounds
            clipped_delta = jnp.clip(new_p[1], 1.0, float(self.max_depth))
            return (new_p[0], clipped_delta), new_o_state, loss_val

        print(f"\n--- BEGINNING UFHM ARCHITECTURE SEARCH ({epochs} EPOCHS) ---")
        for epoch in range(1, epochs + 1):
            params, opt_state, loss = update_step(params, opt_state)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | Total Loss: {loss:.6f} | Topology (Δ): {params[1].item():.6f}")
        
        print("--- OPTIMIZATION COMPLETE ---\n")
        return params

if __name__ == "__main__":
    print("Initializing UFHM Quantum Framework...")
    MAX_DEPTH = 4
    NUM_WIRES = 4
    
    ufhm = UFHMQuantumOptimizer(num_wires=NUM_WIRES, max_depth=MAX_DEPTH)
    
    key = jax.random.PRNGKey(84)
    n_blocks = qml.MERA.get_n_blocks(wires=range(NUM_WIRES), n_block_wires=2)
    weights = jax.random.uniform(key, shape=(MAX_DEPTH, n_blocks, 2)) * 2 * jnp.pi
    delta = jnp.array(3.8)
    
    final_params = ufhm.fit(weights, delta, epochs=50)
  
