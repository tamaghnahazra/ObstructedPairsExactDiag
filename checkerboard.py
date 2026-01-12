"""
Checkerboard lattice implementation as the line graph of a square lattice.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


class Checkerboard:
    """
    Checkerboard lattice created as the line graph of a SquareLattice.

    Sites correspond to edges of the parent SquareLattice.
    Edges connect sites whose parent edges share a common vertex.
    """

    def __init__(self, parent_lattice):
        """
        Initialize a Checkerboard lattice from a parent SquareLattice.

        Parameters:
        -----------
        parent_lattice : SquareLattice
            The parent square lattice whose line graph we're computing
        """
        self.parent_lattice = parent_lattice
        self.sites = {}
        self.edges = {}
        self.N = len(parent_lattice.edges)  # Number of sites = number of parent edges
        self._build_lattice()

    def _build_lattice(self):
        """Build the checkerboard lattice as the line graph of the parent."""
        # Each edge of the parent becomes a site in the checkerboard
        # Site coordinates are the midpoints of parent edges
        for edge_idx, (s1, s2) in self.parent_lattice.edges.items():
            x1, y1 = self.parent_lattice.sites[s1]
            x2, y2 = self.parent_lattice.sites[s2]

            # Check if this is a periodic boundary edge
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # For periodic edges, place midpoint outside the lattice bounds
            if dx > 1:  # Horizontal wraparound
                # Place midpoint to the right of the rightmost site
                mid_x = max(x1, x2) + 0.5
                mid_y = y1  # y coordinate is the same for both sites
            elif dy > 1:  # Vertical wraparound
                # Place midpoint above the topmost site
                mid_x = x1  # x coordinate is the same for both sites
                mid_y = max(y1, y2) + 0.5
            else:  # Regular edge
                # Calculate normal midpoint
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

            self.sites[edge_idx] = (mid_x, mid_y)

        # Build edges: connect sites if their parent edges share a vertex
        new_edge_idx = 0
        edge_list = list(self.parent_lattice.edges.items())

        for i in range(len(edge_list)):
            edge_i_idx, (s1_i, s2_i) = edge_list[i]
            vertices_i = {s1_i, s2_i}

            for j in range(i + 1, len(edge_list)):
                edge_j_idx, (s1_j, s2_j) = edge_list[j]
                vertices_j = {s1_j, s2_j}

                # Check if edges share a common vertex
                if len(vertices_i & vertices_j) > 0:
                    self.edges[new_edge_idx] = (edge_i_idx, edge_j_idx)
                    new_edge_idx += 1

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

        neighbors = np.nonzero(self.adjacency_matrix[site_idx])[0].tolist()
        return neighbors

    def FockBasis(self, M):
        """
        Generate all basis states for M hardcore bosons on N sites.

        Parameters:
        -----------
        M : int
            Number of hardcore bosons (particles)

        Returns:
        --------
        list of lists : Each element is a list of N booleans representing a basis state.
                        True indicates an occupied site, False indicates empty.
        """
        if not (0 <= M <= self.N):
            raise ValueError(f"M must be between 0 and {self.N}")

        # Generate all combinations of M sites from N total sites
        basis_states = []
        for occupied_sites in combinations(range(self.N), M):
            # Create a boolean array for this basis state
            state = [False] * self.N
            for site_idx in occupied_sites:
                state[site_idx] = True
            basis_states.append(state)

        return basis_states

    def visualize_basis_state(self, basis_state, filename='basis_state_visualization.png', title=None):
        """
        Visualize a basis state on the checkerboard lattice.

        Parameters:
        -----------
        basis_state : list of bool
            A list of N booleans indicating which sites are occupied
        filename : str
            Output filename for the visualization
        title : str, optional
            Custom title for the plot
        """
        if len(basis_state) != self.N:
            raise ValueError(f"basis_state must have length {self.N}")

        fig, ax = plt.subplots(figsize=(10, 10))

        L = self.parent_lattice.L

        # Helper function to get replicated positions for periodic boundary sites
        def get_replicas(x, y):
            """Return main position and replicas for a site."""
            positions = [(x, y)]

            # If site is outside on the right (x = L - 0.5), also draw on left (x = -0.5)
            if x >= L - 0.5:
                positions.append((x - L, y))

            # If site is outside on the top (y = L - 0.5), also draw on bottom (y = -0.5)
            if y >= L - 0.5:
                positions.append((x, y - L))

            # If site is at a corner
            if x >= L - 0.5 and y >= L - 0.5:
                positions.append((x - L, y - L))

            return positions

        # Draw edges using closest replicas
        for edge_idx, (site1, site2) in self.edges.items():
            x1, y1 = self.sites[site1]
            x2, y2 = self.sites[site2]

            # Get all possible positions for both sites
            pos1_list = get_replicas(x1, y1)
            pos2_list = get_replicas(x2, y2)

            # Find the pair with minimum distance
            min_dist = float('inf')
            best_pair = None
            for p1 in pos1_list:
                for p2 in pos2_list:
                    dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (p1, p2)

            # Draw the edge using the closest pair
            if best_pair:
                ax.plot([best_pair[0][0], best_pair[1][0]],
                       [best_pair[0][1], best_pair[1][1]],
                       'lightgray', linewidth=1, alpha=0.3)

        # Draw sites with different colors based on occupation
        for site_idx, (x, y) in self.sites.items():
            is_occupied = basis_state[site_idx]
            color = 'red' if is_occupied else 'lightblue'
            marker_size = 12 if is_occupied else 6

            for rx, ry in get_replicas(x, y):
                ax.plot(rx, ry, 'o', color=color, markersize=marker_size)
                # Only label occupied sites or show site numbers
                if is_occupied:
                    ax.text(rx, ry, str(site_idx), fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2))
                else:
                    ax.text(rx, ry, str(site_idx), fontsize=5, ha='center', va='center',
                           color='gray', alpha=0.6)

        # Set axis limits to include replicas
        ax.set_xlim(-1, L + 0.5)
        ax.set_ylim(-1, L + 0.5)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Create title
        num_occupied = sum(basis_state)
        if title is None:
            occupied_sites = [i for i, occupied in enumerate(basis_state) if occupied]
            title = f'Fock Basis State: {num_occupied} bosons on {self.N} sites\nOccupied sites: {occupied_sites}'

        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        return fig, ax

    def visualize(self, filename='checkerboard_visualization.png'):
        """
        Create a visualization of the checkerboard lattice.

        Parameters:
        -----------
        filename : str
            Output filename for the visualization
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        L = self.parent_lattice.L

        # Helper function to get replicated positions for periodic boundary sites
        def get_replicas(x, y):
            """Return main position and replicas for a site."""
            positions = [(x, y)]

            # If site is outside on the right (x = L - 0.5), also draw on left (x = -0.5)
            if x >= L - 0.5:
                positions.append((x - L, y))

            # If site is outside on the top (y = L - 0.5), also draw on bottom (y = -0.5)
            if y >= L - 0.5:
                positions.append((x, y - L))

            # If site is at a corner
            if x >= L - 0.5 and y >= L - 0.5:
                positions.append((x - L, y - L))

            return positions

        # Draw edges using closest replicas
        for edge_idx, (site1, site2) in self.edges.items():
            x1, y1 = self.sites[site1]
            x2, y2 = self.sites[site2]

            # Get all possible positions for both sites
            pos1_list = get_replicas(x1, y1)
            pos2_list = get_replicas(x2, y2)

            # Find the pair with minimum distance
            min_dist = float('inf')
            best_pair = None
            for p1 in pos1_list:
                for p2 in pos2_list:
                    dist = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (p1, p2)

            # Draw the edge using the closest pair
            if best_pair:
                ax.plot([best_pair[0][0], best_pair[1][0]],
                       [best_pair[0][1], best_pair[1][1]],
                       'g-', linewidth=1, alpha=0.5)

        # Draw sites (with replicas for boundary sites)
        for site_idx, (x, y) in self.sites.items():
            for rx, ry in get_replicas(x, y):
                ax.plot(rx, ry, 'bo', markersize=6)
                ax.text(rx, ry, str(site_idx), fontsize=6, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Optionally overlay the parent lattice in light gray for reference (with replicas)
        for edge_idx, (s1, s2) in self.parent_lattice.edges.items():
            x1, y1 = self.parent_lattice.sites[s1]
            x2, y2 = self.parent_lattice.sites[s2]

            # Check if this is a boundary edge
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx > 1:  # Horizontal wraparound - draw on both sides
                ax.plot([x1, x1], [y1, y1], 'o', color='lightgray', markersize=4, alpha=0.5)
                ax.plot([x2, x2], [y2, y2], 'o', color='lightgray', markersize=4, alpha=0.5)
                # Draw edge segments
                if x1 < x2:  # x1 is on left (0), x2 is on right (L-1)
                    ax.plot([x1, -0.5], [y1, y1], 'gray', linewidth=0.5, alpha=0.3)
                    ax.plot([x2, L - 0.5], [y2, y2], 'gray', linewidth=0.5, alpha=0.3)
                else:  # x2 is on left, x1 is on right
                    ax.plot([x2, -0.5], [y2, y2], 'gray', linewidth=0.5, alpha=0.3)
                    ax.plot([x1, L - 0.5], [y1, y1], 'gray', linewidth=0.5, alpha=0.3)
            elif dy > 1:  # Vertical wraparound
                ax.plot([x1, x1], [y1, y1], 'o', color='lightgray', markersize=4, alpha=0.5)
                ax.plot([x2, x2], [y2, y2], 'o', color='lightgray', markersize=4, alpha=0.5)
                # Draw edge segments
                if y1 < y2:  # y1 is on bottom, y2 is on top
                    ax.plot([x1, x1], [y1, -0.5], 'gray', linewidth=0.5, alpha=0.3)
                    ax.plot([x2, x2], [y2, L - 0.5], 'gray', linewidth=0.5, alpha=0.3)
                else:  # y2 is on bottom, y1 is on top
                    ax.plot([x2, x2], [y2, -0.5], 'gray', linewidth=0.5, alpha=0.3)
                    ax.plot([x1, x1], [y1, L - 0.5], 'gray', linewidth=0.5, alpha=0.3)
            else:  # Regular edge
                ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.3)

        # Draw parent lattice sites with replicas on boundaries
        for site_idx, (x, y) in self.parent_lattice.sites.items():
            ax.plot(x, y, 'o', color='lightgray', markersize=4, alpha=0.5)
            # Replicate boundary sites
            if x == 0:
                ax.plot(L, y, 'o', color='lightgray', markersize=4, alpha=0.5)
            if y == 0:
                ax.plot(x, L, 'o', color='lightgray', markersize=4, alpha=0.5)
            if x == 0 and y == 0:
                ax.plot(L, L, 'o', color='lightgray', markersize=4, alpha=0.5)

        # Set axis limits to include replicas
        ax.set_xlim(-1, L + 0.5)
        ax.set_ylim(-1, L + 0.5)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Checkerboard Lattice (Line graph of {self.parent_lattice.L}x{self.parent_lattice.L} Square Lattice)\n{len(self.sites)} sites, {len(self.edges)} edges')

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        return fig, ax

    def print_info(self):
        """Print information about the lattice."""
        print(f"Checkerboard Lattice (Line graph of {self.parent_lattice.L}x{self.parent_lattice.L} Square Lattice)")
        print(f"Number of sites: {len(self.sites)}")
        print(f"Number of edges: {len(self.edges)}")
        if len(self.sites) > 0:
            print(f"Edges / Sites ratio: {len(self.edges) / len(self.sites):.2f}")
        print(f"\nFirst 10 sites (midpoints of parent edges):")
        for i in range(min(10, len(self.sites))):
            parent_edge = self.parent_lattice.edges[i]
            print(f"  Site {i}: {self.sites[i]} (from parent edge {parent_edge})")
        print(f"\nFirst 10 edges:")
        for i in range(min(10, len(self.edges))):
            s1, s2 = self.edges[i]
            print(f"  Edge {i}: {s1} -- {s2} (coords: {self.sites[s1]} -- {self.sites[s2]})")
        if len(self.edges) > 10:
            print(f"\nLast 5 edges:")
            for i in range(max(0, len(self.edges) - 5), len(self.edges)):
                s1, s2 = self.edges[i]
                print(f"  Edge {i}: {s1} -- {s2} (coords: {self.sites[s1]} -- {self.sites[s2]})")


if __name__ == "__main__":
    from square_lattice import SquareLattice

    # Create a 4x4 square lattice
    L = 4
    square = SquareLattice(L)

    print("Parent Square Lattice:")
    print("=" * 60)
    square.print_info()

    # Create checkerboard lattice
    print("\n\nCheckerboard Lattice:")
    print("=" * 60)
    checkerboard = Checkerboard(square)
    checkerboard.print_info()

    # Example: Get neighbors of site 5 in checkerboard
    print(f"\nNeighbors of checkerboard site 5: {checkerboard.get_neighbors(5)}")

    # Visualize both lattices
    print("\nGenerating visualizations...")
    square.visualize()
    print("Square lattice saved as 'lattice_visualization.png'")

    checkerboard.visualize()
    print("Checkerboard lattice saved as 'checkerboard_visualization.png'")
