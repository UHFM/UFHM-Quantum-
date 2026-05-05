# UFHM-Quantum: Differentiable Quantum Architecture Search (DQAS) in JAX

A highly scalable, JAX-native framework that automates the discovery of hardware-efficient quantum circuit depths using continuous mathematical relaxation. 

## 🚀 The "Why": Solving the Ansatz Design Problem

In the **Noisy Intermediate-Scale Quantum (NISQ)** era, finding the right circuit depth is a massive bottleneck. Deep circuits suffer from severe physical decoherence (hardware noise), while shallow circuits fail to capture enough entanglement (expressivity). 

Traditionally, engineers solve this using **Grid Search**—building and testing depth 1, then depth 2, then depth 3—which is agonizingly slow. 

**UFHM-Quantum** solves this by turning discrete circuit depth into a continuous, differentiable parameter (`Δ`). By utilizing JAX and `optax`, the framework uses pure gradient descent to automatically prune unnecessary layers and find the mathematically optimal trade-off between entanglement and circuit complexity.

### 🧠 The Memory Breakthrough: Cost-Interpolation
Many differentiable architecture searches fail at scale because they attempt to interpolate the $O(2^N)$ quantum **state vectors** or density matrices. Trying to simulate super-states for anything beyond a few qubits results in immediate Out-Of-Memory (OOM) errors.

UFHM-Quantum uses a highly efficient **Cost-Interpolation** mechanism. Rather than stacking and interpolating states, it calculates the scalar loss (e.g., purity/entanglement) for each depth independently, and mathematically routes the gradients through those *scalars*. 
* **State Interpolation Memory:** $O(4^N)$ (Density Matrices)
* **UFHM Cost-Interpolation Memory:** $O(1)$

This enables the framework to comfortably scale to 12+ qubits on standard hardware while remaining fully differentiable.

---

## 📈 Example Execution & Output

Because JAX compiles the entire topological super-manifold into an XLA graph, optimization takes seconds. Below is a real output trace.

Starting from a heavy bias (`Δ = 3.8`), the optimizer intelligently evaluates the penalty of depth versus the entanglement gain. It recognizes that reaching layer 4 is mathematically superior, pushing `Δ` to `4.0` and smoothing out perfectly via an Optax Cosine Decay schedule.

```text
Initializing UFHM Quantum Framework (Cost Interpolation)...

--- BEGINNING ARCHITECTURE SEARCH (50 EPOCHS) ---
Epoch 01 | Total Loss: 0.612015 | Topology (Δ): 3.800000
Epoch 05 | Total Loss: 0.568065 | Topology (Δ): 3.862090
Epoch 10 | Total Loss: 0.494725 | Topology (Δ): 4.000000
Epoch 15 | Total Loss: 0.456116 | Topology (Δ): 4.000000
Epoch 20 | Total Loss: 0.464514 | Topology (Δ): 4.000000
Epoch 25 | Total Loss: 0.455101 | Topology (Δ): 4.000000
Epoch 30 | Total Loss: 0.452214 | Topology (Δ): 4.000000
Epoch 35 | Total Loss: 0.451449 | Topology (Δ): 4.000000
Epoch 40 | Total Loss: 0.451231 | Topology (Δ): 4.000000
Epoch 45 | Total Loss: 0.450919 | Topology (Δ): 4.000000
Epoch 50 | Total Loss: 0.450801 | Topology (Δ): 4.000000
--- OPTIMIZATION COMPLETE ---

