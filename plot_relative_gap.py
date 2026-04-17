"""
Dimer Model Energy Gap Interpolation Plot
-----------------------------------------
This script computes and visualizes the evolution of the lowest excitation gap 
in the square lattice dimer model as it interpolates between the Adjacency 
matrix Hamiltonian and the Rokhsar-Kivelson (RK) point.

Physics Context:
- Adjacency Matrix (A): Represents the kinetic term (plaquette flips).
- Diagonal Matrix (D): Represents the potential term (counts flippable plaquettes).
- RK Point: H = A + D, where the ground state is a uniform superposition of all states.
- Interpolation: H(x) = A + (1-x)D, where x=0 is RK and x=1 is pure kinetics.

The script uses a bitmask representation for dimer configurations and 
symmetry-agnostic exact diagonalization (via Scipy's Arpack) for small system sizes.
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
    # Map edge tuples to unique bit indices
    edge_lookup = {(s1, s2): idx for idx, (s1, s2) in lattice.edges.items()}
    edge_lookup.update({(s2, s1): idx for idx, (s1, s2) in lattice.edges.items()})
    coord_to_site = {coords: site_idx for site_idx, coords in lattice.sites.items()}
    
    # Precompute bitmasks for each plaquette's horizontal and vertical dimer states
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

    # Initial state: Perfectly column-packed dimers
    initial_mask = 0
    for y in range(L):
        for x in range(0, L, 2):
            initial_mask |= (1 << edge_lookup[(coord_to_site[(x, y)], coord_to_site[((x+1)%L, y)])])

    # Breadth-First Search to find all reachable dimer configurations (single winding sector)
    visited = {initial_mask: 0}
    queue = [initial_mask]
    head = 0
    while head < len(queue):
        curr = queue[head]; head += 1
        for hm, vm, comb in plaquette_masks:
            sub = curr & comb
            if sub == hm or sub == vm:
                nm = curr ^ comb # Flip the plaquette
                if nm not in visited:
                    visited[nm] = len(queue); queue.append(nm)
    
    N = len(queue)
    rows, cols, data = [], [], []
    diag = np.zeros(N)
    
    # Build the matrix elements
    for i in range(N):
        curr = queue[i]
        n_fl = 0
        for hm, vm, comb in plaquette_masks:
            sub = curr & comb
            if sub == hm or sub == vm:
                # Plaquette is flippable
                n_fl += 1
                nm = curr ^ comb
                rows.append(i); cols.append(visited[nm]); data.append(-1.0)
        diag[i] = n_fl # Potential term is the count of flippable plaquettes
            
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    D = csr_matrix((diag, (range(N), range(N))), shape=(N, N))
    return A, D

def get_gap_at_x(A, D, x):
    """
    Computes the energy gap between the ground state and first excitation for H(x).
    
    Args:
        A (sparse matrix): Kinetic term.
        D (sparse matrix): Potential term.
        x (float): Interpolation parameter (0 = RK, 1 = Kinetics).
        
    Returns:
        float: Energy gap Delta E.
    """
    # Hamiltonian mapping: H(x) = A + (1-x)D
    H = A + (1.0 - x) * D
    # Solve for the two lowest eigenvalues (Smallest Algebraic)
    evals = np.sort(eigsh(H, k=2, which='SA', return_eigenvectors=False))
    return evals[1] - evals[0]

def main():
    # Interpolation parameter space
    xs = np.linspace(0, 1, 15)
    
    plt.figure(figsize=(8, 6))
    
    # Analyze small lattice sizes where brute-force is efficient
    for L in [4, 6]:
        print(f"L={L} Building components...")
        A, D = build_sparse_components(L)
        
        gaps = []
        # Calculate RK point gap (x=0) for normalization
        gap0 = get_gap_at_x(A, D, 0)
        
        print(f"L={L} Interpolating...")
        for x in xs:
            if x == 0:
                gaps.append(1.0)
            else:
                gap_x = get_gap_at_x(A, D, x)
                # Store gap relative to the RK gap
                gaps.append(gap_x / gap0)
        
        plt.plot(xs, gaps, 'o-', label=f"L={L}", linewidth=2)

    # Visualization Refinement
    plt.title("")  
    plt.xlabel("") 
    plt.ylabel("")
    
    # Ensure the plot starts from the origin
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, max(gaps)*1.1 if gaps else 1.5)
    
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    output_svg = "relative_gap_interpolation.svg"
    plt.savefig(output_svg, format='svg', bbox_inches='tight', pad_inches=0)
    print(f"Saved plot successfully to {output_svg}")

if __name__ == "__main__":
    main()
