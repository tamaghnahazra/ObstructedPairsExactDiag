"""
Square lattice implementation with sites and edges.
This script creates a dictionary representation of a square lattice.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class SquareLattice:
    """A square lattice with sites and edges."""

    def __init__(self, L):
        """
        Initialize a square lattice.

        Parameters:
        -----------
        L : int
            Linear size of the lattice (L x L)
        """
        self.L = L
        self.sites = {}
        self.edges = {}
        self._build_lattice()

    def _build_lattice(self):
        """Build the sites and edges dictionaries with periodic boundary conditions."""
        # Create sites dictionary: maps site index to (x, y) coordinates
        site_idx = 0
        coord_to_idx = {}  # Helper to map coordinates to indices

        for y in range(self.L):
            for x in range(self.L):
                self.sites[site_idx] = (x, y)
                coord_to_idx[(x, y)] = site_idx
                site_idx += 1

        # Create edges dictionary: maps edge index to (site1, site2) pairs
        # With periodic boundary conditions, each site has exactly 4 edges
        edge_idx = 0

        for y in range(self.L):
            for x in range(self.L):
                current_site = coord_to_idx[(x, y)]

                # Horizontal edge (to the right, wraps around)
                right_x = (x + 1) % self.L
                right_site = coord_to_idx[(right_x, y)]
                self.edges[edge_idx] = (current_site, right_site)
                edge_idx += 1

                # Vertical edge (upward, wraps around)
                up_y = (y + 1) % self.L
                up_site = coord_to_idx[(x, up_y)]
                self.edges[edge_idx] = (current_site, up_site)
                edge_idx += 1

        # Build adjacency matrix
        self.adjacency_matrix = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self):
        """
        Build the adjacency matrix for the lattice.

        Returns
        -------
        np.ndarray
            A 2D numpy array of shape (N, N) where N is the number of sites.
            adjacency_matrix[i, j] = True if sites i and j are connected by an edge,
            False otherwise.
        """
        N = len(self.sites)
        adj_matrix = np.zeros((N, N), dtype=bool)

        # Fill in the adjacency matrix from edges
        for site1, site2 in self.edges.values():
            adj_matrix[site1, site2] = True
            adj_matrix[site2, site1] = True

        return adj_matrix

    def get_neighbors(self, site_idx):
        """
        Get all neighboring sites of a given site using the adjacency matrix.

        Parameters:
        -----------
        site_idx : int
            Index of the site

        Returns:
        --------
        list : List of neighboring site indices
        """
        if not (0 <= site_idx < self.N):
            raise IndexError(f"site_idx out of bounds: {site_idx}")

        neighbors = list(np.nonzero(self.adjacency_matrix[site_idx])[0])
        return neighbors

    def visualize(self):
        """Create a visualization of the lattice."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw edges
        for edge_idx, (site1, site2) in self.edges.items():
            x1, y1 = self.sites[site1]
            x2, y2 = self.sites[site2]

            # Check if this is a periodic boundary edge
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            is_periodic = (dx > 1) or (dy > 1)

            if is_periodic:
                # Draw periodic edge with wraparound indicators
                # Draw two arrow segments pointing toward the boundary
                if dx > 1:  # Horizontal wraparound
                    # Draw from left site to left boundary
                    x_left = min(x1, x2)
                    y_left = y1 if x1 < x2 else y2
                    ax.annotate('', xy=(-0.4, y_left), xytext=(x_left, y_left),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))

                    # Draw from right site to right boundary
                    x_right = max(x1, x2)
                    y_right = y1 if x1 > x2 else y2
                    ax.annotate('', xy=(self.L - 0.6, y_right), xytext=(x_right, y_right),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))

                if dy > 1:  # Vertical wraparound
                    # Draw from bottom site to bottom boundary
                    x_bottom = x1 if y1 < y2 else x2
                    y_bottom = min(y1, y2)
                    ax.annotate('', xy=(x_bottom, -0.4), xytext=(x_bottom, y_bottom),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))

                    # Draw from top site to top boundary
                    x_top = x1 if y1 > y2 else x2
                    y_top = max(y1, y2)
                    ax.annotate('', xy=(x_top, self.L - 0.6), xytext=(x_top, y_top),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))
            else:
                # Regular edge
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.6)

        # Draw sites
        for site_idx, (x, y) in self.sites.items():
            ax.plot(x, y, 'ro', markersize=8)
            ax.text(x, y, str(site_idx), fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlim(-0.5, self.L - 0.5)
        ax.set_ylim(-0.5, self.L - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{self.L}x{self.L} Square Lattice (Periodic BC)\n{len(self.sites)} sites, {len(self.edges)} edges')

        plt.tight_layout()
        plt.savefig('/Users/tamaghnahazra/code/ObstructedPairs/lattice_visualization.png', dpi=150)
        plt.close()

        return fig, ax

    def print_info(self):
        """Print information about the lattice."""
        print(f"Square Lattice: {self.L}x{self.L} (Periodic Boundary Conditions)")
        print(f"Number of sites: {len(self.sites)}")
        print(f"Number of edges: {len(self.edges)}")
        print(f"Edges / Sites ratio: {len(self.edges) / len(self.sites):.1f}")
        print(f"Verification: Edges = 2 × Sites? {len(self.edges) == 2 * len(self.sites)}")
        print(f"\nFirst 10 sites:")
        for i in range(min(10, len(self.sites))):
            print(f"  Site {i}: {self.sites[i]}")
        print(f"\nFirst 10 edges:")
        for i in range(min(10, len(self.edges))):
            s1, s2 = self.edges[i]
            print(f"  Edge {i}: {s1} -- {s2} (coords: {self.sites[s1]} -- {self.sites[s2]})")
        print(f"\nLast 5 edges (showing periodic connections):")
        for i in range(max(0, len(self.edges) - 5), len(self.edges)):
            s1, s2 = self.edges[i]
            print(f"  Edge {i}: {s1} -- {s2} (coords: {self.sites[s1]} -- {self.sites[s2]})")


if __name__ == "__main__":
    # Create a 4x4 square lattice
    L = 4
    lattice = SquareLattice(L)

    # Print lattice information
    lattice.print_info()

    # Example: Get neighbors of site 5
    print(f"\nNeighbors of site 5: {lattice.get_neighbors(5)}")

    # Visualize the lattice
    print("\nGenerating visualization...")
    lattice.visualize()
    print("Visualization saved as 'lattice_visualization.png'")
