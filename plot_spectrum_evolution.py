"""
Dimer Model Spectral Flow Visualization
---------------------------------------
This script generates a "spectral flow" diagram for the square lattice dimer
model, showing how energy levels evolve and cross as the Hamiltonian
interpolates between the Rokhsar-Kivelson (RK) point and the kinetic-only limit.

Key Feature: Rescaled Energy Spectrum
To visualize the internal gap dynamics independently of the global bandwidth, 
each eigenvalue E_i(x) is rescaled as:
    E_tilde_i = (E_i - E_min) / (E_max - e_min)

This transformation fixes the ground state at y=0 and the top of the energy 
band at y=1 for all interpolation points, allowing a clear view of level 
crossings and gap evolution.

The plot features:
- Dual Y-axes: RK absolute spectrum (left) and Kinetic spectrum (right).
- Scaled labels: Absolute energies are scaled by a factor of 4 for presentation.
- Publication-ready styling: Thick frames, dark blue level lines, and large tick labels.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time

# Ensure the local repository modules are available
sys.path.append("/Users/tamaghnahazra/code/ObstructedPairs")
from square_lattice import SquareLattice

def build_sparse_components(L):
    """
    Constructs the Adjacency (A) and Diagonal (D) matrix components for an L x L torus.
    
    Args:
        L (int): Linear dimension of the square lattice.
        
    Returns:
        tuple: (A, D) as scipy.sparse.csr_matrix objects.
    """
    lattice = SquareLattice(L)
    edge_lookup = {(s1, s2): idx for idx, (s1, s2) in lattice.edges.items()}
    edge_lookup.update({(s2, s1): idx for idx, (s1, s2) in lattice.edges.items()})
    coord_to_site = {coords: site_idx for site_idx, coords in lattice.sites.items()}
    
    # Precompute bitmasks for each plaquette's horizontal and vertical dimer configurations
    plaquette_masks = []
    for y in range(L):
        for x in range(L):
            p0 = coord_to_site[(x, y)]
            p1 = coord_to_site[((x+1)%L, y)]
            p2 = coord_to_site[((x+1)%L, (y+1)%L)]
            p3 = coord_to_site[(x, (y+1)%L)]
            h_mask = (1 << edge_lookup[(p0, p1)]) | (1 << edge_lookup[(p3, p2)])
            v_mask = (1 << edge_lookup[(p1, p2)]) | (1 << edge_lookup[(p0, p3)])
            plaquette_masks.append((h_mask, v_mask, h_mask | v_mask))

    # Start with a reference dimer configuration (column-packed)
    initial_mask = 0
    for y in range(L):
        for x in range(0, L, 2):
            initial_mask |= (1 << edge_lookup[(coord_to_site[(x, y)], coord_to_site[((x+1)%L, y)])])

    # Explore the reachable state space (winding sector) via BFS
    visited = {initial_mask: 0}
    queue = [initial_mask]
    head = 0
    while head < len(queue):
        curr = queue[head]; head += 1
        for hm, vm, comb in plaquette_masks:
            sub = curr & comb
            if sub == hm or sub == vm:
                nm = curr ^ comb
                if nm not in visited:
                    visited[nm] = len(queue); queue.append(nm)
    
    N = len(queue)
    rows, cols, data = [], [], []
    diag = np.zeros(N)
    for i in range(N):
        curr = queue[i]
        n_fl = 0
        for hm, vm, comb in plaquette_masks:
            sub = curr & comb
            if sub == hm or sub == vm:
                n_fl += 1
                nm = curr ^ comb
                rows.append(i); cols.append(visited[nm]); data.append(-1.0)
        diag[i] = n_fl
            
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    D = csr_matrix((diag, (range(N), range(N))), shape=(N, N))
    return A, D

def main():
    """Calculates and plots the rescaled spectral flow for the L=4 lattice."""
    L = 4
    A, D = build_sparse_components(L)
    N = A.shape[0]
    
    # High-resolution interpolation
    xs = np.linspace(0, 1, 51)
    levels = np.zeros((len(xs), N))
    
    print(f"L={L} Flow calculation...")
    for i, x in enumerate(xs):
        # Hamiltonian: H(x) = A + (1-x)D
        H = (A + (1.0 - x) * D).toarray()
        # Compute the full spectrum for this interpolation point
        evals = np.sort(np.linalg.eigvalsh(H))
        
        # Scaling transformation to fix ground state at 0 and top state at 1
        e_min = evals[0]
        e_max = evals[-1]
        levels[i, :] = (evals - e_min) / (e_max - e_min)
        if i % 10 == 0:
            print(f"  x={x:.2f} done.")

    # Visualization and Styling
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # 0. Set limits with zero padding
    ax1.set_xlim(0, 1)
    
    # 1. Plot all spectral levels in dark blue with transparency for crossing visibility
    for j in range(N):
        ax1.plot(xs, levels[:, j], color='darkblue', alpha=0.4, linewidth=1)
        
    # 2. Plot Aesthetics: Remove decorative text for a clean "figure-only" look
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("")
    
    # 3. Reference Spectra for Y-Axis labeling (scaled by 4 for presentation)
    H_rk = (A + D).toarray()
    e_rk = np.sort(np.linalg.eigvalsh(H_rk))
    
    H_kin = (A).toarray()
    e_kin = np.sort(np.linalg.eigvalsh(H_kin))
    
    def get_tick_details(evals):
        """Identifies key levels (first 6 unique levels + middle + max) for labeling."""
        unique_vals = []
        for v in evals:
            if not unique_vals or not np.isclose(v, unique_vals[-1]):
                unique_vals.append(v)
            if len(unique_vals) >= 6:
                break
        
        mid_val = evals[len(evals)//2]
        max_val = evals[-1]
        
        all_picks = sorted(list(set(unique_vals + [mid_val, max_val])))
        
        e_min_val, e_max_val = evals[0], evals[-1]
        norm_pos = [(v - e_min_val) / (e_max_val - e_min_val) for v in all_picks]
        labels = [f"{4 * v:.2f}" for v in all_picks]
        return norm_pos, labels

    # Configure Left (RK) Y-axis
    l_pos, l_labels = get_tick_details(e_rk)
    ax1.set_yticks(l_pos)
    ax1.set_yticklabels(l_labels)
    
    # Configure Right (Kinetic) Y-axis
    ax2 = ax1.twinx()
    r_pos, r_labels = get_tick_details(e_kin)
    ax2.set_yticks(r_pos)
    ax2.set_yticklabels(r_labels)
    ax2.set_ylim(ax1.get_ylim())

    # Styling for Tick Labels
    ax1.tick_params(axis='y', which='major', labelsize=20, labelcolor='black')
    ax2.tick_params(axis='y', which='major', labelsize=20, labelcolor='black')
    
    # Gridlines marking the first six levels of the RK point
    rk_first_six_pos = l_pos[:6] if len(l_pos) >= 6 else l_pos
    for pos in rk_first_six_pos:
        ax1.axhline(y=pos, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        
    # Thicker frame spines 
    for spine in ax1.spines.values():
        spine.set_linewidth(3.0)
        spine.set_edgecolor('black')
    for spine in ax2.spines.values():
        spine.set_linewidth(3.0)
        spine.set_edgecolor('black')

    # Remove X-axis ticks for cleaner visualization of flow
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', which='both', bottom=False, top=False)

    plt.tight_layout()
    output_svg = "spectrum_scaling_l4.svg"
    plt.savefig(output_svg, format='svg', bbox_inches='tight', pad_inches=0)
    print(f"Spectral flow plot saved successfully to {output_svg}")

if __name__ == "__main__":
    main()
