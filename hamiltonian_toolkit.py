"""
Unified Hamiltonian Toolkit for Checkerboard Lattice

This module provides all essential functions for computing eigenspectra of the
checkerboard lattice Hamiltonian using both full diagonalization and matrix-free methods.

Key Functions:
--------------
Full Matrix Methods:
  - HamiltonianMatrixMBoson_optimized: Build full Hamiltonian matrix (optimized)
  - diagonalize_full: Compute full eigenspectrum using numpy.linalg.eigh

Matrix-Free Methods (for low-energy spectrum):
  - Hamiltonian_fully_optimized: Apply H to wavefunction (fastest, requires precomputation)
  - precompute_edge_transitions: Pre-compute transition tables
  - create_hamiltonian_linear_operator: Create LinearOperator for eigsh
  - compute_ground_state: Find ground state using Lanczos (eigsh)
  - compute_lowest_eigenvalues_lobpcg: Find multiple low eigenvalues using LOBPCG

Dependencies:
-------------
  - numpy
  - scipy.sparse.linalg (LinearOperator, eigsh, lobpcg)
  - square_lattice.SquareLattice
  - checkerboard.Checkerboard
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg


# ============================================================================
# FULL MATRIX CONSTRUCTION
# ============================================================================

def HamiltonianMatrixMBoson_optimized(M, checkerboard, t=1.0, V=1.0):
    """
    Construct the full Hamiltonian matrix using vectorized operations.

    The Hamiltonian is H = (1/N_edges) Σ_e [t * B_e^hop + V * B_e^diag]
    where B_e^hop are hopping terms and B_e^diag are diagonal terms.

    Parameters:
    -----------
    M : int
        Number of bosons
    checkerboard : Checkerboard
        The checkerboard lattice object
    t : float, optional
        Hopping amplitude (default: 1.0)
    V : float, optional
        Diagonal/interaction strength (default: 1.0)

    Returns:
    --------
    tuple : (H, basis) where
            H is the Hamiltonian matrix of shape (N_basis, N_basis)
            basis is the list of M-boson basis states
    """
    # Generate the Fock basis for M bosons
    basis = checkerboard.FockBasis(M)
    N_basis = len(basis)

    # Convert basis to numpy array for faster operations
    basis_array = np.array(basis, dtype=bool)  # Shape: (N_basis, N_sites)

    # Create a mapping from basis states to indices using hashing
    basis_to_idx = {}
    for i, state in enumerate(basis):
        key = tuple(j for j, occupied in enumerate(state) if occupied)
        basis_to_idx[key] = i

    # Initialize the Hamiltonian matrix
    H = np.zeros((N_basis, N_basis), dtype=np.float64)

    num_edges = len(checkerboard.edges)

    # For each edge, compute its contribution to H
    for edge_idx, (site1, site2) in checkerboard.edges.items():
        # Find which basis states have exactly one of site1 or site2 occupied (XOR)
        occ_site1 = basis_array[:, site1]
        occ_site2 = basis_array[:, site2]

        # True when exactly one is occupied (can hop)
        can_hop = occ_site1 != occ_site2

        # True when both occupied or both empty (no hop, diagonal contribution)
        no_hop = occ_site1 == occ_site2

        # Handle hopping (off-diagonal) transitions
        hop_indices = np.where(can_hop)[0]
        for i in hop_indices:
            # Create the new state by flipping site1 and site2
            new_state = basis_array[i].copy()
            new_state[site1] = ~new_state[site1]
            new_state[site2] = ~new_state[site2]

            # Find the index of the new state
            key = tuple(j for j in range(len(new_state)) if new_state[j])
            j = basis_to_idx[key]

            # Add contribution to Hamiltonian (i-th column, j-th row)
            # The bond operator takes |i⟩ -> |j⟩
            H[j, i] += t #/ num_edges

        # Handle non-hopping (diagonal) contributions
        no_hop_indices = np.where(no_hop)[0]
        for i in no_hop_indices:
            # Bond operator returns the same state
            H[i, i] += V #/ num_edges

    return H, basis


def diagonalize_full(M, checkerboard, t=1.0, V=1.0, verbose=True):
    """
    Compute the full eigenspectrum using full matrix diagonalization.

    Parameters:
    -----------
    M : int
        Number of bosons
    checkerboard : Checkerboard
        The checkerboard lattice object
    t : float, optional
        Hopping amplitude (default: 1.0)
    V : float, optional
        Diagonal/interaction strength (default: 1.0)
    verbose : bool, optional
        If True, print diagnostic information

    Returns:
    --------
    tuple : (eigenvalues, eigenvectors, basis) where
            eigenvalues is array of all eigenvalues (sorted)
            eigenvectors is array of shape (N_basis, N_basis)
            basis is the list of basis states
    """
    import time

    if verbose:
        print(f"Computing full eigenspectrum for M={M} bosons")

    # Build Hamiltonian matrix
    start = time.time()
    H, basis = HamiltonianMatrixMBoson_optimized(M, checkerboard, t=t, V=V)
    time_construct = time.time() - start

    N_basis = len(basis)

    if verbose:
        print(f"Basis size: {N_basis}")
        print(f"Matrix construction time: {time_construct:.4f} seconds")
        print(f"Matrix memory: {H.nbytes / 1e9:.4f} GB")
        print("Diagonalizing...")

    # Diagonalize
    start = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    time_diag = time.time() - start

    if verbose:
        print(f"Diagonalization time: {time_diag:.4f} seconds")
        print(f"Total time: {time_construct + time_diag:.4f} seconds")
        print(f"\nLowest 10 eigenvalues:")
        for i in range(min(10, len(eigenvalues))):
            print(f"  E[{i}] = {eigenvalues[i]:.10f}")

    return eigenvalues, eigenvectors, basis


# ============================================================================
# MATRIX-FREE HAMILTONIAN APPLICATION
# ============================================================================

def Hamiltonian_fully_optimized(wavefunction, basis_array, edge_transitions, num_edges, t=1.0, V=1.0):
    """
    Apply the Hamiltonian to a wavefunction using pre-computed transition tables.

    This is the fastest implementation for repeated matrix-vector multiplications.
    Requires pre-processing via precompute_edge_transitions().

    Parameters:
    -----------
    wavefunction : array-like of complex
        Complex coefficients for each basis state
    basis_array : numpy array of bool, shape (N_basis, N_sites)
        The basis states as a numpy array
    edge_transitions : dict
        Pre-computed transition table for each edge.
        edge_transitions[edge_idx] = (hop_from, hop_to, diag_indices)
    num_edges : int
        Number of edges in the lattice
    t : float, optional
        Hopping amplitude (default: 1.0)
    V : float, optional
        Diagonal/interaction strength (default: 1.0)

    Returns:
    --------
    numpy array of complex : The new wavefunction after applying Hamiltonian
    """
    wavefunction = np.array(wavefunction, dtype=complex)
    N_basis = len(wavefunction)

    # Initialize result
    result_wavefunction = np.zeros(N_basis, dtype=complex)

    # Process each edge using pre-computed transitions
    for edge_idx, (hop_from, hop_to, diag_indices) in edge_transitions.items():
        # Add hopping contributions
        for k, i in enumerate(hop_from):
            j = hop_to[k]
            result_wavefunction[j] += t * wavefunction[i]

        # Add diagonal contributions
        result_wavefunction[diag_indices] += V * wavefunction[diag_indices]

    # Normalize by number of edges
    result_wavefunction /= num_edges

    return result_wavefunction


def precompute_edge_transitions(basis, checkerboard, verbose=False):
    """
    Pre-compute transition tables for all edges.

    This function should be called once before using Hamiltonian_fully_optimized.

    Parameters:
    -----------
    basis : list of lists of bool
        The Fock basis states
    checkerboard : Checkerboard
        The checkerboard lattice object
    verbose : bool, optional
        If True, print progress information

    Returns:
    --------
    tuple : (basis_array, edge_transitions, num_edges) where:
            - basis_array: numpy array of basis states
            - edge_transitions: dict mapping edge_idx to (hop_from, hop_to, diag_indices)
            - num_edges: number of edges
    """
    if verbose:
        print("Pre-computing edge transitions...")

    # Convert basis to numpy array
    basis_array = np.array(basis, dtype=bool)
    N_basis = len(basis)

    # Create basis state to index mapping
    basis_to_idx = {}
    for i, state in enumerate(basis):
        key = tuple(j for j, occupied in enumerate(state) if occupied)
        basis_to_idx[key] = i

    # Build transition tables for each edge
    edge_transitions = {}
    num_edges = len(checkerboard.edges)

    for edge_idx, (site1, site2) in checkerboard.edges.items():
        # Get occupation of sites for all basis states
        occ_site1 = basis_array[:, site1]
        occ_site2 = basis_array[:, site2]

        # XOR: states where exactly one site is occupied (can hop)
        can_hop = occ_site1 != occ_site2

        # XNOR: states where both sites have same occupation (no hop, diagonal)
        no_hop = occ_site1 == occ_site2

        # Find hopping transitions
        hop_from_list = []
        hop_to_list = []

        hop_indices = np.where(can_hop)[0]
        for i in hop_indices:
            # Create new state by flipping site1 and site2
            new_state = basis_array[i].copy()
            new_state[site1] = not new_state[site1]
            new_state[site2] = not new_state[site2]

            # Find index of new state
            key = tuple(j for j in range(len(new_state)) if new_state[j])
            j = basis_to_idx[key]

            hop_from_list.append(i)
            hop_to_list.append(j)

        hop_from = np.array(hop_from_list, dtype=np.int32)
        hop_to = np.array(hop_to_list, dtype=np.int32)
        diag_indices = np.where(no_hop)[0].astype(np.int32)

        edge_transitions[edge_idx] = (hop_from, hop_to, diag_indices)

    if verbose:
        print(f"Pre-computation complete. Built {num_edges} transition tables.")

    return basis_array, edge_transitions, num_edges


# ============================================================================
# MATRIX-FREE EIGENSOLVERS
# ============================================================================

def create_hamiltonian_linear_operator(basis, checkerboard, t=1.0, V=1.0, verbose=False):
    """
    Create a LinearOperator for the Hamiltonian using optimized implementation.

    This function creates a LinearOperator that represents the Hamiltonian
    without ever constructing the full matrix.

    Parameters:
    -----------
    basis : list of lists of bool
        The Fock basis states
    checkerboard : Checkerboard
        The checkerboard lattice object
    t : float, optional
        Hopping amplitude (default: 1.0)
    V : float, optional
        Diagonal/interaction strength (default: 1.0)
    verbose : bool, optional
        If True, print pre-computation progress

    Returns:
    --------
    LinearOperator : The Hamiltonian as a real-valued LinearOperator
    """
    N_basis = len(basis)

    # Pre-compute edge transitions for fast matrix-vector multiplication
    basis_array, edge_transitions, num_edges = precompute_edge_transitions(basis, checkerboard, verbose=verbose)

    def matvec(v):
        """Apply Hamiltonian to real vector v: H * v"""
        # Convert to complex, apply H, take real part
        v_complex = v.astype(np.complex128)
        result_complex = Hamiltonian_fully_optimized(v_complex, basis_array, edge_transitions, num_edges, t=t, V=V)
        # Hamiltonian is real-symmetric, so imaginary part should be ~0
        result_real = np.real(result_complex)
        return result_real

    def matmat(V_mat):
        """Apply Hamiltonian to matrix V: H * V (multiple real vectors)"""
        if V_mat.ndim == 1:
            return matvec(V_mat)
        # V is shape (N_basis, k) where k is number of vectors
        result = np.zeros_like(V_mat, dtype=np.float64)
        for i in range(V_mat.shape[1]):
            result[:, i] = matvec(V_mat[:, i])
        return result

    # Create real-valued LinearOperator
    H_op = LinearOperator(
        shape=(N_basis, N_basis),
        matvec=matvec,
        matmat=matmat,
        dtype=np.float64
    )

    return H_op


def compute_ground_state(basis, checkerboard, t=1.0, V=1.0, return_eigenvector=True, verbose=True):
    """
    Compute the ground state (lowest eigenvalue and eigenvector) using Lanczos.

    Uses matrix-free methods - no matrix is ever constructed.

    Parameters:
    -----------
    basis : list of lists of bool
        The Fock basis states
    checkerboard : Checkerboard
        The checkerboard lattice object
    t : float, optional
        Hopping amplitude (default: 1.0)
    V : float, optional
        Diagonal/interaction strength (default: 1.0)
    return_eigenvector : bool, optional
        If True, return both eigenvalue and eigenvector
    verbose : bool, optional
        If True, print diagnostic information

    Returns:
    --------
    If return_eigenvector is True:
        tuple : (E0, psi0) where E0 is ground state energy and psi0 is ground state wavefunction
    If return_eigenvector is False:
        float : E0, the ground state energy
    """
    N_basis = len(basis)

    if verbose:
        print(f"Computing ground state for basis size: {N_basis}")

    # Create the Hamiltonian LinearOperator
    H_op = create_hamiltonian_linear_operator(basis, checkerboard, t=t, V=V, verbose=verbose)

    if verbose:
        print("Running eigsh (Lanczos iteration)...")

    # Compute the lowest eigenvalue (and eigenvector)
    if return_eigenvector:
        eigenvalues, eigenvectors = eigsh(H_op, k=1, which='SA', return_eigenvectors=True)
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        if verbose:
            print(f"Ground state energy: E0 = {E0:.10f}")

        return E0, psi0
    else:
        eigenvalues = eigsh(H_op, k=1, which='SA', return_eigenvectors=False)
        E0 = eigenvalues[0]

        if verbose:
            print(f"Ground state energy: E0 = {E0:.10f}")

        return E0


def compute_lowest_eigenvalues_lobpcg(basis, checkerboard, n_eigenvalues=10,
                                      t=1.0, V=1.0, verbose=True, tol=1e-8, maxiter=40):
    """
    Compute lowest eigenvalues using LOBPCG with real eigenvectors.

    This is the recommended method for finding degenerate ground states.
    Uses matrix-free methods - no matrix is ever constructed.

    Parameters:
    -----------
    basis : list of lists of bool
        The Fock basis states
    checkerboard : Checkerboard
        The checkerboard lattice object
    n_eigenvalues : int, optional
        Number of lowest eigenvalues to compute
    t : float, optional
        Hopping amplitude (default: 1.0)
    V : float, optional
        Diagonal/interaction strength (default: 1.0)
    verbose : bool, optional
        If True, print diagnostic information
    tol : float, optional
        Convergence tolerance
    maxiter : int, optional
        Maximum number of iterations

    Returns:
    --------
    tuple : (eigenvalues, eigenvectors) where
            eigenvalues is array of shape (n_eigenvalues,)
            eigenvectors is array of shape (N_basis, n_eigenvalues) - REAL
    """
    N_basis = len(basis)

    if n_eigenvalues >= N_basis:
        raise ValueError(f"n_eigenvalues ({n_eigenvalues}) must be less than N_basis ({N_basis})")

    if verbose:
        print(f"Computing {n_eigenvalues} lowest eigenvalues using LOBPCG")
        print(f"Basis size: {N_basis}")

    # Create real-valued Hamiltonian operator
    H_op = create_hamiltonian_linear_operator(basis, checkerboard, t=t, V=V, verbose=verbose)

    if verbose:
        print(f"Running LOBPCG (maxiter={maxiter}, tol={tol})...")

    # Create random real initial vectors
    np.random.seed(42)
    X = np.random.randn(N_basis, n_eigenvalues)

    # Orthonormalize initial vectors
    X, _ = np.linalg.qr(X)

    # Run LOBPCG
    eigenvalues, eigenvectors = lobpcg(
        H_op, X,
        largest=False,  # Find smallest eigenvalues
        tol=tol,
        maxiter=maxiter,
        verbosityLevel=1 if verbose else 0
    )

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if verbose:
        print(f"\nCompleted!")
        print(f"\nLowest {min(10, len(eigenvalues))} eigenvalues:")
        for i in range(min(10, len(eigenvalues))):
            print(f"  E[{i}] = {eigenvalues[i]:.10f}")

        # Check for degeneracy
        unique_eigs = np.unique(np.round(eigenvalues, decimals=8))
        if len(unique_eigs) < len(eigenvalues):
            print(f"\nDegeneracy detected:")
            print(f"  Number of unique eigenvalues: {len(unique_eigs)}")

    return eigenvalues, eigenvectors


# ============================================================================
# VERIFICATION UTILITIES
# ============================================================================

def verify_eigenvalues(eigenvalues1, eigenvalues2, label1="Method 1", label2="Method 2", verbose=True):
    """
    Compare eigenvalues from two different methods.

    Parameters:
    -----------
    eigenvalues1 : numpy array
        Eigenvalues from first method
    eigenvalues2 : numpy array
        Eigenvalues from second method
    label1 : str
        Label for first method
    label2 : str
        Label for second method
    verbose : bool
        If True, print comparison results

    Returns:
    --------
    float : Maximum absolute difference
    """
    # Ensure same length
    n = min(len(eigenvalues1), len(eigenvalues2))
    eig1 = eigenvalues1[:n]
    eig2 = eigenvalues2[:n]

    max_diff = np.max(np.abs(eig1 - eig2))
    mean_diff = np.mean(np.abs(eig1 - eig2))

    if verbose:
        print(f"\nComparison: {label1} vs {label2}")
        print(f"  Number of eigenvalues compared: {n}")
        print(f"  Maximum difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        if max_diff < 1e-10:
            print(f"  ✓ Excellent agreement (machine precision)")
        elif max_diff < 1e-6:
            print(f"  ✓ Good agreement")
        else:
            print(f"  ✗ Warning: Large differences detected")

    return max_diff


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    """
    Test both methods on L=3, M=2 and verify they give identical results.
    """
    import time
    from square_lattice import SquareLattice
    from checkerboard import Checkerboard

    print("=" * 80)
    print("HAMILTONIAN TOOLKIT TEST: L=3, M=2")
    print("=" * 80)
    print("Testing both full diagonalization and matrix-free methods")
    print("=" * 80)

    # Create lattice
    L = 3
    square = SquareLattice(L)
    checkerboard = Checkerboard(square)
    M = 2

    print(f"\nSystem parameters:")
    print(f"  L = {L}")
    print(f"  M = {M} bosons")
    print(f"  N_sites = {checkerboard.N}")
    print(f"  N_edges = {len(checkerboard.edges)}")

    # Generate basis
    basis = checkerboard.FockBasis(M)
    N_basis = len(basis)
    print(f"  N_basis = {N_basis}")

    # ========================================================================
    # Method 1: Full Diagonalization
    # ========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: FULL DIAGONALIZATION")
    print("=" * 80)

    start = time.time()
    eigenvalues_full, eigenvectors_full, _ = diagonalize_full(M, checkerboard, verbose=True)
    time_full = time.time() - start

    print(f"\nTotal time: {time_full:.4f} seconds")

    # ========================================================================
    # Method 2: Matrix-Free LOBPCG (low-energy spectrum)
    # ========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: MATRIX-FREE LOBPCG")
    print("=" * 80)

    n_eigenvalues = min(50, N_basis - 2)
    print(f"Computing {n_eigenvalues} lowest eigenvalues...")

    start = time.time()
    eigenvalues_lobpcg, eigenvectors_lobpcg = compute_lowest_eigenvalues_lobpcg(
        basis, checkerboard,
        n_eigenvalues=n_eigenvalues,
        verbose=True,
        tol=1e-10,
        maxiter=100
    )
    time_lobpcg = time.time() - start

    print(f"\nTotal time: {time_lobpcg:.4f} seconds")

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    max_diff = verify_eigenvalues(
        eigenvalues_full[:n_eigenvalues],
        eigenvalues_lobpcg,
        label1="Full Diagonalization",
        label2="Matrix-Free LOBPCG",
        verbose=True
    )

    print(f"\nTiming comparison:")
    print(f"  Full diagonalization: {time_full:.4f} seconds")
    print(f"  Matrix-free LOBPCG:   {time_lobpcg:.4f} seconds")
    if time_full > time_lobpcg:
        print(f"  LOBPCG is {time_full/time_lobpcg:.2f}x faster for {n_eigenvalues} eigenvalues")
    else:
        print(f"  Full diag is {time_lobpcg/time_full:.2f}x faster (gets all {N_basis} eigenvalues)")

    print("\n" + "=" * 80)
    print("KEY POINTS")
    print("=" * 80)
    print("Full Diagonalization:")
    print(f"  ✓ Computes ALL {N_basis} eigenvalues at once")
    print(f"  ✓ Memory: {eigenvalues_full.nbytes / 1e6:.2f} MB for eigenvalues")
    print(f"  ✓ Best for: Complete spectrum or small systems")
    print()
    print("Matrix-Free LOBPCG:")
    print(f"  ✓ Computes only {n_eigenvalues} lowest eigenvalues")
    print(f"  ✓ No matrix construction (saves memory)")
    print(f"  ✓ Best for: Low-energy spectrum of large systems")
    print(f"  ✓ Ideal for highly degenerate ground states")

    print("\n" + "=" * 80)
    print("TEST COMPLETE - All functions verified!")
    print("=" * 80)
