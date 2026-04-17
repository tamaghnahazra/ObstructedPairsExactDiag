import sys
import subprocess
from square_lattice import SquareLattice
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
# Force interactive backend after SquareLattice imports and sets Agg
plt.switch_backend('macosx') 
import numpy as np

class DimerApp:
    def __init__(self, L=4, render_mode="3D"):
        self.lattice = SquareLattice(L)
        self.render_mode = render_mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title(f"Dimer App - Square Lattice L={L}")
        
        self.dimers = set()
        self.covered_sites = set()
        
        self.dragging_dimer = None
        
        # We store lines over plotted dimers to update efficiently
        self.dimer_plot_elements = []
        self.hover_plot_elements = []
        self.plaquette_plot_elements = []

        self.edge_segments = {}
        self._build_edge_segments()
        
        self.plaquettes = []
        self._build_plaquettes()
        
        self.site_edge_map = {}
        self._build_site_edge_map()
        
        self.draw_lattice()
        
        self.eigvals = None
        self.eigvecs = None
        self.graph_N_nodes = 0
        self.graph_pos_x = None
        self.graph_pos_y = None
        self.graph_pos_z = None
        self.scatter_nodes = None
        self.current_eigen_idx = 0
        self.double_occ = False
        self.idx_to_state = {}
        
        # Initialize secondary window upfront to prevent Cocoa segfaults
        self.graph_fig = plt.figure(figsize=(20, 5))
        self.graph_fig.canvas.manager.set_window_title("Analysis Window (Press 'c' in main map to compute)")
        
        if self.render_mode == "3D":
             self.ax_graph = self.graph_fig.add_subplot(141, projection='3d')
             self.annot = self.ax_graph.text2D(0.05, 0.95, "", transform=self.ax_graph.transAxes, verticalalignment='top',
                                 bbox=dict(boxstyle="round", fc="w"))
        else:
             self.ax_graph = self.graph_fig.add_subplot(141)
             self.annot = self.ax_graph.text(0.05, 0.95, "", transform=self.ax_graph.transAxes, verticalalignment='top',
                                 bbox=dict(boxstyle="round", fc="w"))
             
        self.ax_mat = self.graph_fig.add_subplot(142)
        self.ax_eig = self.graph_fig.add_subplot(143)
        self.ax_state = self.graph_fig.add_subplot(144)
        
        self.ax_graph.set_title("Covering Graph (Empty)")
        self.ax_mat.set_title("Adjacency Matrix (Empty)")
        self.ax_eig.set_title("Eigenvalues (Empty)")
        self.ax_state.set_title("State Viewer (Empty)")
        self.ax_state.axis('off')
        self.graph_fig.subplots_adjust(bottom=0.15)
        
        self.annot.set_visible(False)
        
        self.graph_fig.canvas.mpl_connect('button_press_event', self.on_graph_click)
        self.graph_fig.canvas.mpl_connect('motion_notify_event', self.on_graph_hover)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.graph_fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()

    def _build_edge_segments(self):
        """Precomputes the visual segments for each edge to enable interaction targeting."""
        for edge_idx, (site1, site2) in self.lattice.edges.items():
            x1, y1 = self.lattice.sites[site1]
            x2, y2 = self.lattice.sites[site2]
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            is_periodic = (dx > 1) or (dy > 1)
            segments = []
            
            if is_periodic:
                if dx > 1:
                    x_left = min(x1, x2)
                    y_left = y1 if x1 < x2 else y2
                    segments.append(((x_left, y_left), (-0.4, y_left)))
                    
                    x_right = max(x1, x2)
                    y_right = y1 if x1 > x2 else y2
                    segments.append(((x_right, y_right), (self.lattice.L - 0.6, y_right)))
                if dy > 1:
                    x_bottom = x1 if y1 < y2 else x2
                    y_bottom = min(y1, y2)
                    segments.append(((x_bottom, y_bottom), (x_bottom, -0.4)))
                    
                    x_top = x1 if y1 > y2 else x2
                    y_top = max(y1, y2)
                    segments.append(((x_top, y_top), (x_top, self.lattice.L - 0.6)))
            else:
                segments.append(((x1, y1), (x2, y2)))
                
            self.edge_segments[edge_idx] = segments

    def _build_plaquettes(self):
        """Construct plaquettes defined by their 4 edges for L x L square lattice."""
        self.edge_lookup = {}
        for edge_idx, (s1, s2) in self.lattice.edges.items():
            self.edge_lookup[(s1, s2)] = edge_idx
            self.edge_lookup[(s2, s1)] = edge_idx

        coord_to_site = {coords: site_idx for site_idx, coords in self.lattice.sites.items()}
        
        L = self.lattice.L
        for y in range(L):
            for x in range(L):
                p0 = coord_to_site[(x, y)]
                p1 = coord_to_site[((x+1)%L, y)]
                p2 = coord_to_site[((x+1)%L, (y+1)%L)]
                p3 = coord_to_site[(x, (y+1)%L)]
                
                e_bottom = self.edge_lookup[(p0, p1)]
                e_right = self.edge_lookup[(p1, p2)]
                e_top = self.edge_lookup[(p3, p2)]
                e_left = self.edge_lookup[(p0, p3)]
                
                self.plaquettes.append({
                    'coords': (x, y),
                    'edges': (e_bottom, e_right, e_top, e_left),
                })

    def _build_site_edge_map(self):
        """Precompute the edges connected to each site for fast neighbor lookups."""
        self.site_edge_map = {s: [] for s in self.lattice.sites}
        for e_idx, (s1, s2) in self.lattice.edges.items():
            self.site_edge_map[s1].append(e_idx)
            self.site_edge_map[s2].append(e_idx)

    def get_defect_counts(self, state_dimers):
        degrees = {site: 0 for site in self.lattice.sites}
        for d in state_dimers:
            s1, s2 = self.lattice.edges[d]
            degrees[s1] += 1
            degrees[s2] += 1
        holes = [site for site, deg in degrees.items() if deg == 0]
        doublons = [site for site, deg in degrees.items() if deg > 1]
        return holes, doublons

    def get_single_dimer_moves(self, state_dimers):
        holes, _ = self.get_defect_counts(state_dimers)
        moves = []
        
        for h in holes:
            # For each neighbor edge of the hole
            for e_idx in self.site_edge_map[h]:
                s1, s2 = self.lattice.edges[e_idx]
                n = s2 if s1 == h else s1
                
                # Check if this neighbor 'n' is connected to a dimer
                for d in state_dimers:
                    ds1, ds2 = self.lattice.edges[d]
                    if ds1 == n or ds2 == n:
                        # Swing this dimer to the hole
                        new_state = set(state_dimers)
                        new_state.remove(d)
                        new_state.add(e_idx)
                        moves.append(new_state)
                        break # Standard dimer overlap prevention implies max 1 incident natively
                        
        unique_moves = []
        seen = set()
        for m in moves:
            fs = frozenset(m)
            if fs not in seen:
                seen.add(fs)
                unique_moves.append(m)
        return unique_moves

    def get_extra_dimer_moves(self, state_dimers):
        _, doublons = self.get_defect_counts(state_dimers)
        moves = []
        
        for d_site in doublons:
            # find dimers incident to the doublon
            incident_dimers = []
            for e_idx in self.site_edge_map[d_site]:
                if e_idx in state_dimers:
                    incident_dimers.append(e_idx)
                    
            for d in incident_dimers:
                # This dimer connects d_site to some neighbor n
                s1, s2 = self.lattice.edges[d]
                n = s2 if s1 == d_site else s1
                
                # Swing this dimer 'd' around 'n' to another neighbor 'm'
                # Meaning new edge e_new connects n and m
                for e_new in self.site_edge_map[n]:
                    if e_new == d: continue
                    es1, es2 = self.lattice.edges[e_new]
                    m = es2 if es1 == n else es1
                    
                    if e_new not in state_dimers:
                        # Swing: remove d, add e_new
                        new_state = set(state_dimers)
                        new_state.remove(d)
                        new_state.add(e_new)
                        moves.append(new_state)

        unique_moves = []
        seen = set()
        for m in moves:
            fs = frozenset(m)
            if fs not in seen:
                seen.add(fs)
                unique_moves.append(m)
        return unique_moves

    def get_flippable_plaquettes_of_state(self, state_dimers):
        """Pure function variant of get_flippable_plaquettes for exploring states."""
        flippable = []
        for pl in self.plaquettes:
            e_bottom, e_right, e_top, e_left = pl['edges']
            if e_bottom in state_dimers and e_top in state_dimers and e_left not in state_dimers and e_right not in state_dimers:
                flippable.append((pl, 'horizontal'))
            elif e_left in state_dimers and e_right in state_dimers and e_bottom not in state_dimers and e_top not in state_dimers:
                flippable.append((pl, 'vertical'))
        return flippable

    def get_flippable_plaquettes(self):
        """Returns a list of (plaquette, type) that have parallel dimers."""
        return self.get_flippable_plaquettes_of_state(self.dimers)

    def spring_layout(self, adj_mat, iters=100, dim=3):
        """A force-directed layout implementation leveraging numpy."""
        N = len(adj_mat)
        if N == 0:
            return tuple(np.array([]) for _ in range(dim))
        if N == 1:
            return tuple(np.array([0.0]) for _ in range(dim))
            
        pos = np.zeros((N, dim))
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pos[:, 0] = np.cos(angles)
        pos[:, 1] = np.sin(angles)
        if dim == 3:
            # Add some z-variation
            pos[:, 2] = np.sin(3*angles) * 0.5
            
        # Add a tiny bit of noise to prevent perfectly unstable locked symmetries
        pos += np.random.RandomState(42).rand(N, dim) * 0.05
        k = 1.0 / (N**(1/dim))
        t = 1.0
        dt = t / (iters + 1)
        
        for _ in range(iters):
            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.linalg.norm(delta, axis=-1)
            dist[dist < 1e-5] = 1e-5
            
            rep_mag = k**2 / dist
            np.fill_diagonal(rep_mag, 0)
            disp = np.sum((delta / dist[:, :, np.newaxis]) * rep_mag[:, :, np.newaxis], axis=1)
            
            rows, cols = np.where(adj_mat > 0)
            for i, j in zip(rows, cols):
                if i < j:
                    d_vec = pos[i] - pos[j]
                    d = np.linalg.norm(d_vec)
                    if d > 1e-5:
                        attr_mag = d**2 / k
                        force = (d_vec / d) * attr_mag
                        disp[i] -= force
                        disp[j] += force
                        
            disp_norm = np.linalg.norm(disp, axis=1)
            disp_norm[disp_norm < 1e-5] = 1e-5
            
            move = (disp / disp_norm[:, np.newaxis]) * np.minimum(disp_norm, t)[:, np.newaxis]
            pos += move
            t -= dt
            
        pos -= np.mean(pos, axis=0)
        max_val = np.max(np.abs(pos))
        if max_val > 0:
            pos /= max_val
            
        if dim == 3:
            return pos[:, 0], pos[:, 1], pos[:, 2]
        else:
            return pos[:, 0], pos[:, 1]

    def compute_covering_graph(self):
        """BFS exploration of flippable spaces starting from current state, plotted sequentially."""
        start_state = frozenset(self.dimers)
        visited = set()
        queue = [start_state]
        
        state_to_idx = {start_state: 0}
        idx_to_state = {0: start_state}
        adjacency_edges = set()
        
        print("Computing accessibility graph from current state...")
        
        while queue:
            current_state = queue.pop(0)
            u = state_to_idx[current_state]
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            flips = self.get_flippable_plaquettes_of_state(current_state)
            
            for pl, p_type in flips:
                e_bottom, e_right, e_top, e_left = pl['edges']
                new_state = set(current_state)
                if p_type == 'horizontal':
                    new_state.remove(e_bottom)
                    new_state.remove(e_top)
                    new_state.add(e_left)
                    new_state.add(e_right)
                else:
                    new_state.remove(e_left)
                    new_state.remove(e_right)
                    new_state.add(e_bottom)
                    new_state.add(e_top)
                    
                new_state_fs = frozenset(new_state)
                if new_state_fs not in state_to_idx:
                    new_idx = len(state_to_idx)
                    state_to_idx[new_state_fs] = new_idx
                    idx_to_state[new_idx] = new_state_fs
                    queue.append(new_state_fs)
                
                v = state_to_idx[new_state_fs]
                adjacency_edges.add(frozenset([u, v]))
                
        self.graph_edges = adjacency_edges
        self.double_occ = False
        self._render_state_graph(state_to_idx, idx_to_state, adjacency_edges, "Covering Graph")

    def compute_extra_dimer_graph(self):
        """BFS exploration of extra dimer swings starting from current state."""
        start_state = frozenset(self.dimers)
        
        holes, doublons = self.get_defect_counts(start_state)
        if len(doublons) == 0:
            print("Cannot explore extra dimer graph: no doublons present!")
            self.ax_graph.set_title("Cannot execute 'e' (no doublons)")
            self.graph_fig.canvas.draw_idle()
            return
            
        visited = set()
        queue = [start_state]
        
        state_to_idx = {start_state: 0}
        idx_to_state = {0: start_state}
        adjacency_edges = set()
        
        print("Computing extra-dimer defect accessibility graph from current state...")
        
        while queue:
            current_state = queue.pop(0)
            u = state_to_idx[current_state]
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            new_states = self.get_extra_dimer_moves(current_state)
            
            for ns in new_states:
                new_state_fs = frozenset(ns)
                if new_state_fs not in state_to_idx:
                    new_idx = len(state_to_idx)
                    state_to_idx[new_state_fs] = new_idx
                    idx_to_state[new_idx] = new_state_fs
                    queue.append(new_state_fs)
                
                v = state_to_idx[new_state_fs]
                adjacency_edges.add(frozenset([u, v]))
                
        self.graph_edges = adjacency_edges
        self.double_occ = True
        self._render_state_graph(state_to_idx, idx_to_state, adjacency_edges, "Extra Dimer Graph")

    def compute_defect_graph(self):
        """BFS exploration of single-dimer swings starting from current state."""
        start_state = frozenset(self.dimers)
        
        holes, _ = self.get_defect_counts(start_state)
        if len(holes) == 0:
            print("Cannot explore defect graph: the current covering is close packed (0 holes).")
            self.ax_graph.set_title("Cannot execute 'd' (close packed)")
            self.graph_fig.canvas.draw_idle()
            return

        visited = set()
        queue = [start_state]
        
        state_to_idx = {start_state: 0}
        idx_to_state = {0: start_state}
        adjacency_edges = set()
        
        print("Computing single-dimer defect accessibility graph from current state...")
        
        while queue:
            current_state = queue.pop(0)
            u = state_to_idx[current_state]
            
            if current_state in visited:
                continue
            visited.add(current_state)
            
            new_states = self.get_single_dimer_moves(current_state)
            
            for ns in new_states:
                new_state_fs = frozenset(ns)
                if new_state_fs not in state_to_idx:
                    new_idx = len(state_to_idx)
                    state_to_idx[new_state_fs] = new_idx
                    idx_to_state[new_idx] = new_state_fs
                    queue.append(new_state_fs)
                
                v = state_to_idx[new_state_fs]
                adjacency_edges.add(frozenset([u, v]))
                
        self.graph_edges = adjacency_edges
        self.double_occ = False
        self._render_state_graph(state_to_idx, idx_to_state, adjacency_edges, "Defect Graph")

    def _render_state_graph(self, state_to_idx, idx_to_state, adjacency_edges, title_prefix):
        self.idx_to_state = idx_to_state
        N_nodes = len(state_to_idx)
        self.graph_N_nodes = N_nodes
        print(f"Graph constructed with {N_nodes} reachable configurations.")
        
        adj_mat = np.zeros((N_nodes, N_nodes))
        for edge_fs in adjacency_edges:
            nodes = list(edge_fs)
            u = nodes[0]
            v = nodes[1] if len(nodes) > 1 else nodes[0]
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
            
        self.ax_graph.clear()
        self.ax_mat.clear()
        self.ax_eig.clear()
        
        if hasattr(self, 'graph_cbar') and self.graph_cbar is not None:
            try:
                self.graph_cbar.remove()
            except Exception:
                pass
            self.graph_cbar = None
        
        # Add annotation back to ax_graph since we cleared it
        if self.render_mode == "3D":
            self.annot = self.ax_graph.text2D(0.05, 0.95, "", transform=self.ax_graph.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle="round", fc="w"))
        else:
            self.annot = self.ax_graph.text(0.05, 0.95, "", transform=self.ax_graph.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle="round", fc="w"))
        self.annot.set_visible(False)
        
        self.ax_graph.set_title(f"{title_prefix} ({N_nodes} reachable states)")
        if N_nodes > 0:
            if self.render_mode == "3D":
                x, y, z = self.spring_layout(adj_mat, dim=3)
                self.graph_pos_x = x
                self.graph_pos_y = y
                self.graph_pos_z = z
                
                for edge_fs in adjacency_edges:
                    nodes = list(edge_fs)
                    u = nodes[0]
                    v = nodes[1] if len(nodes) > 1 else nodes[0]
                    self.ax_graph.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], 'gray', alpha=0.5, zorder=1)
                    
                self.scatter_nodes = self.ax_graph.scatter(x, y, z, s=300, c=np.zeros(N_nodes), cmap='RdYlBu', zorder=2, edgecolors='white', linewidths=0.5)
                
                for i in range(N_nodes):
                    self.ax_graph.text(x[i], y[i], z[i], str(i), fontsize=8, color='black', ha='center', va='center')
                    
                self.ax_graph.set_axis_off()
            else:
                x, y = self.spring_layout(adj_mat, dim=2)
                self.graph_pos_x = x
                self.graph_pos_y = y
                
                for edge_fs in adjacency_edges:
                    nodes = list(edge_fs)
                    u = nodes[0]
                    v = nodes[1] if len(nodes) > 1 else nodes[0]
                    self.ax_graph.plot([x[u], x[v]], [y[u], y[v]], 'gray', alpha=0.5, zorder=1)
                    
                self.scatter_nodes = self.ax_graph.scatter(x, y, s=300, c=np.zeros(N_nodes), cmap='RdYlBu', zorder=2, edgecolors='white', linewidths=0.5)
                
                for i in range(N_nodes):
                    self.ax_graph.text(x[i], y[i], str(i), fontsize=8, color='black', ha='center', va='center')
                    
                self.ax_graph.axis('off')
                
            self.graph_cbar = self.graph_fig.colorbar(self.scatter_nodes, ax=self.ax_graph, fraction=0.046, pad=0.04)
            self.graph_cbar.set_label('Eigenvector Weight')
        
        self.ax_mat.set_title("Adjacency Matrix")
        if N_nodes > 0:
            matrix_img = self.ax_mat.imshow(adj_mat, cmap='Blues', interpolation='nearest')
            
            eigvals, eigvecs = np.linalg.eigh(adj_mat)
            self.eigvals = eigvals
            self.eigvecs = eigvecs
            self.current_eigen_idx = np.argmin(eigvals)
            
            self.draw_eigenvalues()
            self.update_eigenvector_colors()
            
            if 0 in self.idx_to_state:
                self.draw_state_in_axes(self.idx_to_state[0], self.ax_state)
                self.ax_state.set_title(f"State 0")
            
        self.graph_fig.canvas.draw_idle()

    def draw_eigenvalues(self):
        self.ax_eig.clear()
        self.ax_eig.set_title("Eigenvalues")
        if self.eigvals is None: return
        
        indices = np.arange(len(self.eigvals))
        colors = ['red' if i == self.current_eigen_idx else 'blue' for i in indices]
        
        self.ax_eig.scatter(indices, self.eigvals, c=colors, zorder=2, picker=True)
        self.ax_eig.plot(indices, self.eigvals, 'blue', alpha=0.3, zorder=1)
        self.ax_eig.set_xlabel('Index (Click to select)')
        self.ax_eig.set_ylabel('Eigenvalue')
        
        min_val = self.eigvals[self.current_eigen_idx]
        self.graph_fig.text(0.5, 0.05, f"Selected Eigenvalue: {min_val:.5f} (Index: {self.current_eigen_idx})", ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.9))

    def update_eigenvector_colors(self):
        if self.eigvecs is None or self.scatter_nodes is None:
            return
            
        vec = self.eigvecs[:, self.current_eigen_idx]
        self.scatter_nodes.set_array(vec)
        self.scatter_nodes.set_cmap('RdYlBu')
        vmax = np.max(np.abs(vec))
        if vmax == 0: vmax = 1.0 # prevent division by zero in color scaling
        self.scatter_nodes.set_clim(-vmax, vmax)
        
        if hasattr(self, 'graph_cbar') and self.graph_cbar is not None:
            self.graph_cbar.update_normal(self.scatter_nodes)
            
        self.graph_fig.canvas.draw_idle()

    def draw_state_in_axes(self, state_fs, ax):
        ax.clear()
        
        # draw grid
        for edge_idx, segments in self.edge_segments.items():
            for ((x1, y1), (x2, y2)) in segments:
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.4)
                
        # draw sites
        for site_idx, (x, y) in self.lattice.sites.items():
            ax.plot(x, y, 'ko', markersize=4)
            
        # draw dimers
        for d in state_fs:
            segments = self.edge_segments[d]
            for ((x1, y1), (x2, y2)) in segments:
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=4, alpha=0.8, solid_capstyle='round')

        ax.set_xlim(-0.5, self.lattice.L - 0.5)
        ax.set_ylim(-0.5, self.lattice.L - 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def on_graph_click(self, event):
        if event.inaxes == self.ax_eig and self.eigvals is not None:
            if event.xdata is None: return
            idx = int(round(event.xdata))
            if 0 <= idx < len(self.eigvals):
                # Clear previous text
                [t.remove() for t in self.graph_fig.texts]
                self.current_eigen_idx = idx
                self.draw_eigenvalues()
                self.update_eigenvector_colors()
        elif event.inaxes == self.ax_graph and self.scatter_nodes is not None:
            cont, ind = self.scatter_nodes.contains(event)
            if cont:
                idx = ind["ind"][0]
                if idx in self.idx_to_state:
                    state_fs = self.idx_to_state[idx]
                    self.draw_state_in_axes(state_fs, self.ax_state)
                    self.ax_state.set_title(f"State {idx}")
                    self.graph_fig.canvas.draw_idle()

    def on_graph_hover(self, event):
        if event.inaxes == self.ax_graph and self.scatter_nodes is not None and self.eigvecs is not None:
            cont, ind = self.scatter_nodes.contains(event)
            if cont:
                idx = ind["ind"][0]
                weight = self.eigvecs[idx, self.current_eigen_idx]
                
                self.annot.set_text(f"{weight:.4f}")
                self.annot.set_visible(True)
                self.graph_fig.canvas.draw_idle()
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.graph_fig.canvas.draw_idle()

    def point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        """Calculates distance between point and line segment."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.hypot(px - x1, py - y1)
            
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return np.hypot(px - closest_x, py - closest_y)

    def get_closest_edge(self, x, y):
        """Returns the index of the closest edge to coordinates (x,y) under a threshold distance."""
        if x is None or y is None:
            return None, float('inf')
            
        min_dist = float('inf')
        closest_edge = None
        
        for edge_idx, segments in self.edge_segments.items():
            for ((x1, y1), (x2, y2)) in segments:
                d = self.point_to_segment_dist(x, y, x1, y1, x2, y2)
                if d < min_dist:
                    min_dist = d
                    closest_edge = edge_idx
                    
        return closest_edge, min_dist

    def update_covered_sites(self):
        """Recompute covered sites based on current list of dimers."""
        self.covered_sites.clear()
        for d in self.dimers:
            s1, s2 = self.lattice.edges[d]
            self.covered_sites.add(s1)
            self.covered_sites.add(s2)

    def is_valid_dimer(self, edge_idx):
        """Check if we can place a dimer here without violating constraints."""
        if edge_idx in self.dimers:
            return True # it already is one
        s1, s2 = self.lattice.edges[edge_idx]
        return (not s1 in self.covered_sites) and (not s2 in self.covered_sites)

    def get_event_button(self, event):
        # Handle the event.button across potential different matplotlib versions/backends
        return event.button
        
    def flip_plaquette_if_clicked(self, px, py):
        """Identify if click falls inside a flippable plaquette to trigger a flip."""
        if px is None or py is None:
            return False
            
        L = self.lattice.L
        target_x = int(np.floor(px % L))
        target_y = int(np.floor(py % L))
        
        for pl, p_type in self.get_flippable_plaquettes():
            if pl['coords'] == (target_x, target_y):
                e_bottom, e_right, e_top, e_left = pl['edges']
                if p_type == 'horizontal':
                    self.dimers.remove(e_bottom)
                    self.dimers.remove(e_top)
                    self.dimers.add(e_left)
                    self.dimers.add(e_right)
                else: # vertical
                    self.dimers.remove(e_left)
                    self.dimers.remove(e_right)
                    self.dimers.add(e_bottom)
                    self.dimers.add(e_top)
                
                self.update_covered_sites()
                self.redraw_dimers()
                return True
        return False

    def on_key_press(self, event):
        if event.key == 'c':
            self.compute_covering_graph()
        elif event.key == 'd':
            self.compute_defect_graph()
        elif event.key == 'e':
            self.compute_extra_dimer_graph()
        elif event.key == 'g':
            self.save_graph_pdf()
        elif event.key == 'p':
            self.save_pdf()
        elif event.key == 'm':
            self.save_matrix_pdf()
        elif event.key == 'a':
            self.save_matrix_data()
        elif event.key == 'v':
            self.save_eigenvalues_pdf()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
            
        # Double click to flip plaquette
        if event.dblclick:
            if self.flip_plaquette_if_clicked(event.xdata, event.ydata):
                # Success, abort further click handling.
                # NOTE: the first click of the double-click sequence might have picked up a dimer.
                # If a dimer was mistakenly picked up from inside the plaquette area...
                if self.dragging_dimer is not None:
                    # Cancel the drag by putting it back before the UI thinks it's being dragged
                    self.dimers.add(self.dragging_dimer)
                    self.dragging_dimer = None
                    self.update_covered_sites()
                    self.redraw_dimers()
                return
            
        edge_idx, d = self.get_closest_edge(event.xdata, event.ydata)
        if d > 0.25 or edge_idx is None:
            return # too far away
            
        btn = self.get_event_button(event)
        
        # Left click (add or pick up dimer)
        if btn == 1:
            if edge_idx in self.dimers:
                # Pick up the dimer
                self.dragging_dimer = edge_idx
                self.dimers.remove(edge_idx)
                self.update_covered_sites()
                self.redraw_dimers()
            else:
                # Add generic dimer if valid or bypass validity if double_occ is set
                if self.is_valid_dimer(edge_idx) or self.double_occ or event.key == 'shift':
                    self.dimers.add(edge_idx)
                    self.update_covered_sites()
                    self.redraw_dimers()
                
        # Right click (remove dimer)
        elif btn == 3:
            if edge_idx in self.dimers:
                self.dimers.remove(edge_idx)
                self.update_covered_sites()
                self.redraw_dimers()
                
    def on_motion(self, event):
        if self.dragging_dimer is not None:
            if event.inaxes != self.ax:
                 # Clean hover elements if mouse drags out of axis
                 for el in self.hover_plot_elements:
                     el.remove()
                 self.hover_plot_elements.clear()
                 self.fig.canvas.draw_idle()
                 return
                 
            edge_idx, d = self.get_closest_edge(event.xdata, event.ydata)
            for el in self.hover_plot_elements:
                el.remove()
            self.hover_plot_elements.clear()
            
            if d <= 0.3 and edge_idx is not None:
                # preview dragged dimer
                color = 'green' if (self.is_valid_dimer(edge_idx) or self.double_occ) else 'red'
                
                segments = self.edge_segments[edge_idx]
                for ((x1, y1), (x2, y2)) in segments:
                    line, = self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=6, alpha=0.5, solid_capstyle='round')
                    self.hover_plot_elements.append(line)
                    
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.dragging_dimer is not None:
            # Clear preview elements
            for el in self.hover_plot_elements:
                el.remove()
            self.hover_plot_elements.clear()
            
            dropped = False
            
            if event.inaxes == self.ax:
                edge_idx, d = self.get_closest_edge(event.xdata, event.ydata)
                if d <= 0.3 and edge_idx is not None:
                    if self.is_valid_dimer(edge_idx):
                        self.dimers.add(edge_idx)
                        dropped = True
            
            if not dropped:
                # return to original place
                self.dimers.add(self.dragging_dimer)
                
            self.dragging_dimer = None
            self.update_covered_sites()
            self.redraw_dimers()

    def _draw_link_ellipse(self, ax, p1, p2, fill, edge):
        mid = (np.array(p1) + np.array(p2)) / 2
        dx, dy = np.array(p2) - np.array(p1)
        angle = np.degrees(np.arctan2(dy, dx))
        # Aspect ratio 5: width 1.0, height 0.2
        ell = patches.Ellipse(mid, 1.0, 0.2, angle=angle, facecolor=fill, edgecolor=edge, linewidth=2.0, zorder=4)
        ax.add_patch(ell)

    def _draw_zigzag(self, ax, start, end, color):
        dx, dy = np.array(end) - np.array(start)
        length = np.hypot(dx, dy)
        num_points = int(length * 20)
        t = np.linspace(0, 1, num_points)
        
        # Main line path
        x = start[0] + t * dx
        y = start[1] + t * dy
        
        # Perpendicular vector
        nx = -dy / length
        ny = dx / length
        
        # Zigzag oscillation (amplitude 0.05)
        oscillation = 0.05 * np.sin(t * length * 10 * np.pi)
        
        x += nx * oscillation
        y += ny * oscillation
        ax.plot(x, y, color=color, linewidth=0.8, zorder=10)

    def _get_save_path(self, title, initial_file, extension='.pdf'):
        """Helper to get save path using macOS native dialog (doesn't require tkinter)."""
        applescript = f'POSIX path of (choose file name with prompt "{title}" default name "{initial_file}")'
        try:
            # Use osascript to show a native macOS file dialog
            proc = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)
            if proc.returncode == 0:
                path = proc.stdout.strip()
                if not path.lower().endswith(extension.lower()):
                    path += extension
                return path
        except Exception:
            pass
        return None

    def save_pdf(self):
        """Export the current dimer covering to a PDF with professional vector styling."""
        L = self.lattice.L
        # Use Figure directly to avoid interactive backend crashes on macOS
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        # 1. Background grid (subtle but visible)
        for i in range(L):
            ax.plot([-0.5, L-0.5], [i, i], 'k-', linewidth=0.8, alpha=0.15, zorder=1)
            ax.plot([i, i], [-0.5, L-0.5], 'k-', linewidth=0.8, alpha=0.15, zorder=1)
            
        # 2. Draw dimers as blue ovals
        # Colors: fill a9c0f6ff, line 0011ffff
        fill_color = '#a9c0f6' # a9c0f6ff
        line_color = '#0011ff' # 0011ffff
        
        for d_idx in self.dimers:
            s1, s2 = self.lattice.edges[d_idx]
            p1 = np.array(self.lattice.sites[s1])
            p2 = np.array(self.lattice.sites[s2])
            
            dp = p2 - p1
            # Check for x-wrap
            if abs(dp[0]) > 1:
                if p1[0] < p2[0]: 
                    self._draw_link_ellipse(ax, p1, p1 + [-1, 0], fill_color, line_color)
                    self._draw_link_ellipse(ax, p2, p2 + [1, 0], fill_color, line_color)
                else: 
                    self._draw_link_ellipse(ax, p2, p2 + [-1, 0], fill_color, line_color)
                    self._draw_link_ellipse(ax, p1, p1 + [1, 0], fill_color, line_color)
            elif abs(dp[1]) > 1:
                if p1[1] < p2[1]: 
                    self._draw_link_ellipse(ax, p1, p1 + [0, -1], fill_color, line_color)
                    self._draw_link_ellipse(ax, p2, p2 + [0, 1], fill_color, line_color)
                else: 
                    self._draw_link_ellipse(ax, p2, p2 + [0, -1], fill_color, line_color)
                    self._draw_link_ellipse(ax, p1, p1 + [0, 1], fill_color, line_color)
            else:
                self._draw_link_ellipse(ax, p1, p2, fill_color, line_color)

        # 3. Draw zigzag lines at x=0.5 and y=0.5
        self._draw_zigzag(ax, [0.5, -0.5], [0.5, L-0.5], 'lightcoral')
        self._draw_zigzag(ax, [-0.5, 0.5], [L-0.5, 0.5], 'lightcoral')
        
        # 4. Final adjustments
        ax.set_xlim(-0.5, L-0.5)
        ax.set_ylim(-0.5, L-0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Open file dialog to choose path
        file_path = self._get_save_path("Export Dimer Covering as PDF", "dimer_covering.pdf")
        
        if not file_path:
            print("Export cancelled or failed. Using default 'dimer_covering.pdf' if path was not chosen.")
            return

        fig.savefig(file_path, format='pdf', bbox_inches='tight')
        print(f"Exported dimer covering to {file_path}")

    def save_graph_pdf(self):
        """Export the current state graph to a PDF with professional vector styling."""
        if self.graph_pos_x is None:
            print("Cannot export: state graph has not been computed. Press 'c', 'd', or 'e' first.")
            return

        # Revert to raw Figure (non-interactive) to prevent macOS Cocoa segfaults
        fig = Figure(figsize=(10, 8))
        if self.render_mode == "3D":
             ax = fig.add_subplot(111, projection='3d')
        else:
             ax = fig.add_subplot(111)

        x, y = self.graph_pos_x, self.graph_pos_y
        z = self.graph_pos_z if self.render_mode == "3D" else None
        
        # 1. Draw edges
        for edge_fs in self.graph_edges:
            nodes = list(edge_fs)
            u = nodes[0]
            v = nodes[1] if len(nodes) > 1 else nodes[0]
            if self.render_mode == "3D":
                ax.plot([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], 'gray', alpha=0.3, zorder=1)
            else:
                ax.plot([x[u], x[v]], [y[u], y[v]], 'gray', alpha=0.3, zorder=1)

        # 2. Draw nodes
        weights = self.scatter_nodes.get_array() if self.scatter_nodes else np.zeros(len(x))
        clim = self.scatter_nodes.get_clim() if self.scatter_nodes else (-1, 1)
        
        if self.render_mode == "3D":
            scatter = ax.scatter(x, y, z, s=500, c=weights, cmap='RdYlBu', zorder=2, edgecolors='white', linewidths=0.5)
        else:
            scatter = ax.scatter(x, y, s=500, c=weights, cmap='RdYlBu', zorder=2, edgecolors='white', linewidths=0.5)
        
        scatter.set_clim(clim)

        # 3. Draw state indices (conditionally: only if nodes <= 100)
        if len(x) <= 100:
            pe = [withStroke(linewidth=1.5, foreground='black')]
            for i in range(len(x)):
                if self.render_mode == "3D":
                    ax.text(x[i], y[i], z[i], str(i), fontsize=10, color='white', fontweight='bold', ha='center', va='center', zorder=10, path_effects=pe)
                else:
                    ax.text(x[i], y[i], str(i), fontsize=10, color='white', fontweight='bold', ha='center', va='center', zorder=10, path_effects=pe)

        # 4. Colorbar (explicitly set labels and ticks to 3x larger fonts)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Eigenvector Weight', color='black', fontsize=36)
        cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=30)

        # 5. Clean up
        ax.set_axis_off()
        ax.set_title("")
        
        # Preserve current viewing angle and zoom for 3D
        if self.render_mode == "3D":
            # Sync limits first as view_init sometimes depends on them for internal scaling
            ax.set_xlim3d(self.ax_graph.get_xlim3d())
            ax.set_ylim3d(self.ax_graph.get_ylim3d())
            ax.set_zlim3d(self.ax_graph.get_zlim3d())
            
            e, a = self.ax_graph.elev, self.ax_graph.azim
            r = getattr(self.ax_graph, 'roll', 0)
            ax.view_init(elev=e, azim=a, roll=r)
            
            if hasattr(self.ax_graph, 'dist'):
                ax.dist = self.ax_graph.dist
        
        # Open file dialog to choose path
        file_path = self._get_save_path("Export State Graph as PDF", "state_graph.pdf")
        
        if not file_path:
            print("Export cancelled or failed. Using default 'state_graph.pdf' if path was not chosen.")
            return

        # For non-interactive figures, savefig handles the rendering.
        fig.savefig(file_path, format='pdf', bbox_inches='tight')
        print(f"Exported state graph to {file_path}")

    def save_matrix_pdf(self):
        """Export the Adjacency Matrix of the last computed graph to a PDF."""
        # Calculate matrix if not done
        if not hasattr(self, 'graph_edges') or not self.graph_edges:
            print("Cannot export: compute a graph first (press 'c', 'd', or 'e').")
            return
            
        n = len(self.idx_to_state)
        H_mat = np.zeros((n, n))
        for edge_fs in self.graph_edges:
            nodes = list(edge_fs)
            u = nodes[0]
            v = nodes[1] if len(nodes) > 1 else nodes[0]
            H_mat[u, v] = 1
            H_mat[v, u] = 1

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(H_mat, cmap='Blues', interpolation='nearest')
        ax.set_axis_off()
        ax.set_title("")
        
        file_path = self._get_save_path("Export Adjacency Matrix as PDF", "adjacency_matrix.pdf")
        if not file_path: return
        
        fig.savefig(file_path, format='pdf', bbox_inches='tight')
        print(f"Exported matrix to {file_path}")

    def save_matrix_data(self):
        """Save the Adjacency Matrix of the last computed graph to a text file."""
        if not hasattr(self, 'graph_edges') or not self.graph_edges:
            print("Cannot save: compute a graph first (press 'c', 'd', or 'e').")
            return
            
        n = len(self.idx_to_state)
        H_mat = np.zeros((n, n), dtype=int)
        for edge_fs in self.graph_edges:
            nodes = list(edge_fs)
            u = nodes[0]
            v = nodes[1] if len(nodes) > 1 else nodes[0]
            H_mat[u, v] = 1
            H_mat[v, u] = 1

        file_path = self._get_save_path("Export Adjacency Matrix as Text File", "adjacency_matrix.txt", extension='.txt')
        if not file_path: return
        
        try:
            np.savetxt(file_path, H_mat, fmt='%d')
            print(f"Saved adjacency matrix data to {file_path}")
        except Exception as e:
            print(f"Error saving matrix data: {e}")

    def save_eigenvalues_pdf(self):
        """Export the Eigenvalue Spectrum of the last computed graph to a PDF."""
        if self.eigvals is None:
            print("No eigenvalues to export. Compute a graph first.")
            return
            
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        indices = np.arange(len(self.eigvals))
        ax.scatter(indices, self.eigvals, c='blue', s=20, alpha=0.6, label='Spectrum')
        
        if hasattr(self, 'current_eigen_idx') and self.current_eigen_idx is not None:
            ax.scatter([self.current_eigen_idx], [self.eigvals[self.current_eigen_idx]], 
                       c='red', s=100, edgecolors='black', zorder=5, label='Selected')
        
        ax.set_xlabel('Index', fontsize=36)
        ax.set_ylabel('Eigenvalue', fontsize=36)
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_title("")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        file_path = self._get_save_path("Export Eigenvalues as PDF", "eigenvalues_spectrum.pdf")
        if not file_path: return
        
        fig.savefig(file_path, format='pdf', bbox_inches='tight')
        print(f"Exported eigenvalues to {file_path}")

    def redraw_dimers(self):
        for el in self.dimer_plot_elements:
            el.remove()
        self.dimer_plot_elements.clear()
        
        for el in self.plaquette_plot_elements:
            el.remove()
        self.plaquette_plot_elements.clear()

        L = self.lattice.L
        
        # Draw highlighted flippable plaquettes
        for pl, p_type in self.get_flippable_plaquettes():
            x, y = pl['coords']
            for ix in [-1, 0, 1]:
                for iy in [-1, 0, 1]:
                    sx = x + ix * L
                    sy = y + iy * L
                    if sx > L or sx + 1 < -1 or sy > L or sy + 1 < -1:
                        continue
                    rect = patches.Rectangle((sx, sy), 1, 1, linewidth=0, facecolor='yellow', alpha=0.4, zorder=2)
                    self.ax.add_patch(rect)
                    self.plaquette_plot_elements.append(rect)
        
        for d_idx in self.dimers:
            segments = self.edge_segments[d_idx]
            for ((x1, y1), (x2, y2)) in segments:
                # Dimers drawn as thick green lines
                line, = self.ax.plot([x1, x2], [y1, y2], color='green', linewidth=5, solid_capstyle='round', zorder=4)
                self.dimer_plot_elements.append(line)
                
        self.fig.canvas.draw_idle()

    def draw_lattice(self):
        self.ax.clear()

        # Draw lattice edges
        for edge_idx, (site1, site2) in self.lattice.edges.items():
            x1, y1 = self.lattice.sites[site1]
            x2, y2 = self.lattice.sites[site2]
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            is_periodic = (dx > 1) or (dy > 1)
            
            if is_periodic:
                if dx > 1:
                    x_left = min(x1, x2)
                    y_left = y1 if x1 < x2 else y2
                    self.ax.annotate('', xy=(-0.4, y_left), xytext=(x_left, y_left),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.3))

                    x_right = max(x1, x2)
                    y_right = y1 if x1 > x2 else y2
                    self.ax.annotate('', xy=(self.lattice.L - 0.6, y_right), xytext=(x_right, y_right),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.3))

                if dy > 1:
                    x_bottom = x1 if y1 < y2 else x2
                    y_bottom = min(y1, y2)
                    self.ax.annotate('', xy=(x_bottom, -0.4), xytext=(x_bottom, y_bottom),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.3))

                    x_top = x1 if y1 > y2 else x2
                    y_top = max(y1, y2)
                    self.ax.annotate('', xy=(x_top, self.lattice.L - 0.6), xytext=(x_top, y_top),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.3))
            else:
                self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.4)

        # Draw lattice sites
        for site_idx, (x, y) in self.lattice.sites.items():
            self.ax.plot(x, y, 'ko', markersize=6)
            self.ax.text(x, y, str(site_idx), fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=5)

        self.ax.set_xlim(-0.5, self.lattice.L - 0.5)
        self.ax.set_ylim(-0.5, self.lattice.L - 0.5)
        self.ax.set_xticks(range(self.lattice.L))
        self.ax.set_yticks(range(self.lattice.L))
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2)
        
        # Adjust layout to make room for a side box
        self.fig.subplots_adjust(right=0.72, left=0.08, top=0.9, bottom=0.1)
        
        # Controls text box on the right
        controls_text = (
            "$\mathbf{Controls:}$"
            "\n\n"
            "$\mathbf{Lattice\ Interaction}$\n"
            "• Left Click: Add/Pick Up\n"
            "• Shift+Click: Extra Dimer\n"
            "• Right Click: Remove\n"
            "• Double Click: Flip\n"
            "\n"
            "$\mathbf{Graph\ Analysis}$\n"
            "• 'c': Covering Graph\n"
            "• 'd': Defect Graph\n"
            "• 'e': Extra Dimer Graph\n"
            "\n"
            "$\mathbf{PDF\ Export}$\n"
            "• 'p': Coverage Map\n"
            "• 'g': State Graph\n"
            "• 'm': Adjacency Matrix\n"
            "• 'a': Save Matrix Data\n"
            "• 'v': Eigenvalues"
        )
        
        # Remove old title if necessary and set a clean one
        self.ax.set_title(f"{self.lattice.L}x{self.lattice.L} Square Lattice", fontsize=14, fontweight='bold', pad=20)
        
        # Add a static text box adjacent to the figure
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        self.fig.text(0.74, 0.5, controls_text, fontsize=10, 
                     verticalalignment='center', bbox=props)
        
        plt.tight_layout(rect=[0, 0, 0.72, 1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Dimer App")
    parser.add_argument("L_arg", nargs='?', type=int, default=4, help="Backward compatibility placeholder for L positional argument")
    parser.add_argument("-L", type=int, default=4, help="Lattice size (L x L). Overrides positional argument if provided.")
    parser.add_argument("-r", "--render", choices=["2D", "3D"], default="3D", help="Render mode for the state graph (2D or 3D)")
    args = parser.parse_args()
    
    # Simple logic to preserve old usage e.g `python dimer_app.py 6`
    L_val = args.L_arg if args.L == 4 and args.L_arg != 4 else args.L
    
    app = DimerApp(L=L_val, render_mode=args.render)
