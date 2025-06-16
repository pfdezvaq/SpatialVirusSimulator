import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import label, binary_dilation
from collections import Counter
import seaborn as sns
import time


class SpatialVirusSimulator:
    """
    A spatial simulation of virus replication dynamics in cell culture.

    This simulator models the spatial distribution and temporal dynamics of wild-type (WT)
    and amplicon virions in a cell culture plate, including:
    - Spatial distribution of cells and virions
    - Cell infection dynamics with multiple infection states
    - Virion production, diffusion, and degradation
    - Microscopic view capabilities for detailed analysis

    The simulation uses a 2D grid to represent a cell culture plate and tracks:
    - Wild-type virions that can replicate autonomously
    - Amplicon virions that require helper functions for replication
    - Cell states: healthy, infected (WT/amplicon/coinfected), dead
    - Temporal dynamics over specified time periods
    """

    def __init__(self, params):
        """
        Initialize the simulator with specific parameters.

        Args:
            params (dict): Simulation parameters containing:
                - 'b' (int): Virions produced per cell (burst size)
                - 'num_cells' (int): Number of cells in the simulation
                - 'wt_initial' (int): Initial wild-type virions
                - 'a_initial' (int): Initial amplicon virions
                - 'sigma' (float): Relative production parameter (WT fraction in coinfection)
                - 'gamma' (float): Amplicon inhibition factor due to WT excess
                - 'max_time' (int): Maximum simulation time (hours)
                - 'tau' (int): Virus replication time (hours)
                - 'degradation_rate' (float): Virion degradation rate per hour
                - 'diffusion_rate' (float): Virion diffusion rate
                - 'plate_diameter' (float): Plate diameter in mm
                - 'cell_diameter' (float): Cell diameter in μm
                - 'grid_size' (int): Grid size for the plate representation
                - 'local_retention_rate' (float): Fraction of virions remaining at origin
                - 'extracelular_diffusion_rate' (float): Fraction dispersed globally
                - 'cell_to_cell_diffusion_rate' (float): Fraction transferred to neighbors
        """
        self.params = params
        self.temporal_snapshots = {}  # Store temporal snapshots for microscopic view
        self.initialize_simulation()

    def calculate_lambda(self, virions_count, cells_in_position):
        """
        Calculate lambda parameter for Poisson distribution based on MOI.

        Multiplicity of Infection (MOI) determines the probability of infection
        using a Poisson distribution: P(infection) = 1 - exp(-MOI)

        Args:
            virions_count (float): Number of virions at the position
            cells_in_position (int): Number of cells at the position

        Returns:
            float: Lambda value for Poisson distribution (0 to 1)
        """
        if cells_in_position <= 0 or virions_count < 0:
            return 0

        moi = virions_count / cells_in_position
        lambda_value = 1 - np.exp(-moi)

        if np.isnan(lambda_value) or lambda_value < 0:
            return 0

        return min(lambda_value, 1)

    def initialize_simulation(self):
        """
        Initialize the spatial matrices for the simulation.

        Creates 2D grids representing:
        - Cell positions and states
        - Wild-type and amplicon virion distributions
        - Infection states and timing
        """
        # Create spatial matrix to represent the plate
        self.grid_size = self.params['grid_size']
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Matrix for wild-type virions
        self.wt_virions = np.zeros((self.grid_size, self.grid_size))

        # Matrix for amplicon virions
        self.a_virions = np.zeros((self.grid_size, self.grid_size))

        # Cell state in each position: 0: no cells, 1+: number of cells
        self.cells_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Cell infection states: 2D matrix (one cell per position max)
        # -1: dead cell, 0: healthy, 1: WT, 2: Amplicon, 3: Coinfected,
        # 4: Multiple WT, 5: Multiple coinfected
        self.cell_states = None

        # Infection time for each cell
        self.infection_times = None

        # Distribute cells on the plate
        self.distribute_cells()

        # Distribute initial virions
        self.distribute_initial_virions()

        # Initialize history tracking for plotting
        self.wt_history = [np.sum(self.wt_virions)]
        self.a_history = [np.sum(self.a_virions)]
        self.total_virions_history = [self.wt_history[0] + self.a_history[0]]
        self.infected_history = [0]  # Initially no infected cells
        self.save_temporal_snapshot(0)

    def distribute_cells(self):
        """
        Distribute cells randomly on the plate simulating realistic confluence.

        Uses circular plate geometry with maximum one cell per position to avoid overlap.
        Cell distribution considers:
        - Plate diameter and cell diameter for realistic cell density
        - Circular boundary conditions
        - Random positioning within available space
        """
        # Calculate area of P35 plate (typically 35mm diameter)
        plate_diameter_mm = self.params['plate_diameter']  # in mm
        plate_area_mm2 = np.pi * (plate_diameter_mm / 2) ** 2

        # Convert cell diameter to mm for consistent calculations
        cell_diameter_um = self.params['cell_diameter']  # in μm
        cell_diameter_mm = cell_diameter_um / 1000

        # Calculate cell area
        cell_area_mm2 = np.pi * (cell_diameter_mm / 2) ** 2

        # Calculate plate radius in grid units
        plate_radius_grid = self.grid_size // 2

        # Calculate plate center
        center_x, center_y = plate_radius_grid, plate_radius_grid

        # Calculate maximum available positions within the plate
        max_positions = 0
        available_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= plate_radius_grid ** 2:
                    max_positions += 1
                    available_positions.append((x, y))

        # Check if all requested cells can be placed
        num_cells = self.params['num_cells']
        if num_cells > max_positions:
            print(f"Warning: Requested {num_cells} cells but only {max_positions} positions available.")
            print(f"Placing {max_positions} cells (100% confluence).")
            num_cells = max_positions

        # Create matrices for cell states and infection (2D since max 1 cell per position)
        self.cell_states = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.infection_times = np.full((self.grid_size, self.grid_size), np.inf)

        # Randomly select positions for cell placement
        selected_positions = np.random.choice(len(available_positions), size=num_cells, replace=False)

        cells_placed = 0
        for pos_idx in selected_positions:
            x, y = available_positions[pos_idx]
            self.cells_grid[x, y] = 1  # One cell per position
            cells_placed += 1

        print(f"Cells distributed: {cells_placed} out of {max_positions} available positions "
              f"({cells_placed / max_positions * 100:.1f}% confluence)")

    def distribute_initial_virions(self):
        """
        Distribute initial virions randomly on the plate.

        Uses random placement within circular plate boundaries for both
        wild-type and amplicon virions according to initial conditions.
        """
        # Calculate plate radius in grid units
        plate_radius_grid = self.grid_size // 2
        center_x, center_y = plate_radius_grid, plate_radius_grid

        # Distribute WT virions
        wt_placed = 0
        while wt_placed < self.params['wt_initial']:
            # Generate random position within the plate using polar coordinates
            r = np.sqrt(np.random.random()) * plate_radius_grid
            theta = np.random.random() * 2 * np.pi

            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))

            # Check boundaries
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Verify position is within the circular plate
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= plate_radius_grid ** 2:
                    self.wt_virions[x, y] += 1
                    wt_placed += 1

        # Distribute amplicon virions
        a_placed = 0
        while a_placed < self.params['a_initial']:
            # Generate random position within the plate
            r = np.sqrt(np.random.random()) * plate_radius_grid
            theta = np.random.random() * 2 * np.pi

            x = int(center_x + r * np.cos(theta))
            y = int(center_y + r * np.sin(theta))

            # Check boundaries
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Verify position is within the circular plate
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= plate_radius_grid ** 2:
                    self.a_virions[x, y] += 1
                    a_placed += 1

    def diffuse_virions(self):
        """
        Simulate multiple types of virion distribution:

        1. Local retention: virions remain at original position
        2. Extracellular diffusion: global dispersal through medium
        3. Cell-to-cell transfer: direct transfer between neighboring cells

        The sum of all coefficients must equal 1.0 for mass conservation.
        """
        # Distribution parameters
        local_retention_rate = self.params.get('local_retention_rate', 0.7)
        extracelular_diffusion_rate = self.params.get('extracelular_diffusion_rate', 0.1)
        cell_to_cell_diffusion_rate = self.params.get('cell_to_cell_diffusion_rate', 0.2)

        # Validate that rates sum to 1.0
        total_rate = local_retention_rate + extracelular_diffusion_rate + cell_to_cell_diffusion_rate
        if not np.isclose(total_rate, 1.0, atol=1e-5):
            raise ValueError(f"Distribution rates must sum to 1.0. Currently sum to {total_rate}")

        # Create temporary matrices to store new states
        new_wt = np.copy(self.wt_virions)
        new_a = np.copy(self.a_virions)

        # Calculate diffusion (excluding boundaries to avoid edge effects)
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                for virion_matrix, new_matrix in [(self.wt_virions, new_wt), (self.a_virions, new_a)]:
                    total_virions = virion_matrix[i, j]

                    # 1. Virions that remain locally (no movement)
                    local_virions = total_virions * local_retention_rate

                    # 2. Extracellular diffusion (global dispersal)
                    extracelular_diffusion = total_virions * extracelular_diffusion_rate
                    global_diffusion_per_direction = extracelular_diffusion / 4

                    # 3. Cell-to-cell diffusion (direct transfer)
                    cell_to_cell_diffusion = total_virions * cell_to_cell_diffusion_rate
                    cell_diffusion_per_neighbor = cell_to_cell_diffusion / 4

                    # Apply extracellular diffusion
                    new_matrix[i, j] = local_virions  # Reset to local value
                    new_matrix[i + 1, j] += global_diffusion_per_direction
                    new_matrix[i - 1, j] += global_diffusion_per_direction
                    new_matrix[i, j + 1] += global_diffusion_per_direction
                    new_matrix[i, j - 1] += global_diffusion_per_direction

                    # Direct cell-to-cell transfer
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        neighbor_i, neighbor_j = i + di, j + dj

                        # If there are cells in the neighboring position
                        if (0 <= neighbor_i < self.grid_size and
                                0 <= neighbor_j < self.grid_size and
                                self.cells_grid[neighbor_i, neighbor_j] > 0):
                            new_matrix[neighbor_i, neighbor_j] += cell_diffusion_per_neighbor

        # Update matrices with new values
        self.wt_virions = np.maximum(0, new_wt)  # Ensure no negative values
        self.a_virions = np.maximum(0, new_a)

    def simulate_infection_step(self, t):
        """
        Simulate one time step of infection including:
        - Virion production from cells completing their cycle
        - Virion diffusion
        - New cell infections
        - Virion degradation

        Args:
            t (int): Current time step

        Returns:
            int: Number of newly infected cells in this step
        """
        newly_infected = 0
        tau = self.params['tau']
        b = self.params['b']
        sigma = self.params['sigma']

        # STEP 1: Virion production by infected cells after tau hours
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cells_grid[i, j] > 0:  # If there's a cell at this position
                    if self.infection_times[i, j] + tau == t:  # Time to produce virions
                        if self.cell_states[i, j] == 1:  # WT infection
                            self.wt_virions[i, j] += b
                        elif self.cell_states[i, j] == 2:  # Amplicon infection
                            self.a_virions[i, j] += 0  # Amplicons alone don't produce
                        elif self.cell_states[i, j] == 3:  # Coinfected
                            # Modification: WT inhibition by amplicon presence
                            gamma = self.params.get('gamma', 0.0)
                            prob_wt = sigma * np.exp(-gamma)
                            prob_amp = 1 - prob_wt
                            burst = np.random.multinomial(b, [prob_wt, prob_amp])
                            self.wt_virions[i, j] += burst[0]
                            self.a_virions[i, j] += burst[1]
                        elif self.cell_states[i, j] == 4:  # Multiple WT
                            self.wt_virions[i, j] += b
                        elif self.cell_states[i, j] == 5:  # Multiple coinfected
                            # Complex production model for multiple infections
                            wt_produced = b * (2.98476 * sigma - 4.65143 * sigma ** 2 + 2.66667 * sigma ** 3)
                            a_produced = b - wt_produced
                            self.wt_virions[i, j] += max(0, wt_produced)
                            self.a_virions[i, j] += max(0, a_produced)

                        # Cell dies after producing virions
                        self.cell_states[i, j] = -1

        # STEP 2: Virion diffusion
        self.diffuse_virions()

        # STEP 3: Attempt to infect healthy cells with available virions
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.cells_grid[i, j] > 0 and self.cell_states[i, j] == 0:  # Healthy cell present
                    # Calculate infection probabilities
                    lambda_w = self.calculate_lambda(self.wt_virions[i, j], 1)  # One cell per position
                    lambda_a = self.calculate_lambda(self.a_virions[i, j], 1)

                    wt_infections = np.random.poisson(lambda_w)
                    a_infections = np.random.poisson(lambda_a)

                    # Determine new cell state
                    if wt_infections > 1 and a_infections > 1:
                        self.cell_states[i, j] = 5  # Multiple coinfected
                    elif wt_infections > 1:
                        self.cell_states[i, j] = 4  # Multiple WT
                    elif a_infections > 1:
                        self.cell_states[i, j] = 2  # Amplicon
                    elif wt_infections > 0 and a_infections > 0:
                        self.cell_states[i, j] = 3  # Coinfected
                    elif wt_infections > 0:
                        self.cell_states[i, j] = 1  # WT
                    elif a_infections > 0:
                        self.cell_states[i, j] = 2  # Amplicon

                    # If cell became infected
                    if self.cell_states[i, j] > 0:
                        self.infection_times[i, j] = t  # Record infection time
                        newly_infected += 1

                        # Remove virions that caused infection
                        self.wt_virions[i, j] = max(0, self.wt_virions[i, j] - wt_infections)
                        self.a_virions[i, j] = max(0, self.a_virions[i, j] - a_infections)

        # STEP 4: Apply degradation to free virions
        degradation_rate = self.params['degradation_rate']
        self.wt_virions = np.maximum(0, self.wt_virions * (1 - degradation_rate))
        self.a_virions = np.maximum(0, self.a_virions * (1 - degradation_rate))

        # Count total infected cells
        infected_count = np.sum(self.cell_states > 0)

        # Update history
        self.wt_history.append(np.sum(self.wt_virions))
        self.a_history.append(np.sum(self.a_virions))
        self.total_virions_history.append(self.wt_history[-1] + self.a_history[-1])
        self.infected_history.append(infected_count)

        # Save temporal snapshot every 5 hours or as needed
        if t % 5 == 0 or t <= 10:  # More frequent at the beginning
            self.save_temporal_snapshot(t)

        return newly_infected

    def save_temporal_snapshot(self, time_point):
        """
        Save a snapshot of the current state at a specific time point.

        Args:
            time_point (int): Time point to save snapshot for
        """
        self.temporal_snapshots[time_point] = {
            'wt_virions': self.wt_virions.copy(),
            'a_virions': self.a_virions.copy(),
            'cells_grid': self.cells_grid.copy(),
            'cell_states': self.cell_states.copy() if self.cell_states is not None else None
        }

    def get_microscopic_view(self, time_point, center_x=None, center_y=None, view_size=50):
        """
        Get a microscopic view of a specific area at a given time point.

        Args:
            time_point (int): Time point for the snapshot
            center_x (int, optional): X coordinate of view center (default: plate center)
            center_y (int, optional): Y coordinate of view center (default: plate center)
            view_size (int): Size of the area to visualize (view_size x view_size)

        Returns:
            dict: Dictionary containing cropped matrices for microscopic view
        """
        if time_point not in self.temporal_snapshots:
            print(f"Warning: No snapshot available for time {time_point}")
            return None

        snapshot = self.temporal_snapshots[time_point]

        # Use plate center if no center specified
        if center_x is None or center_y is None:
            center_x = self.grid_size // 2
            center_y = self.grid_size // 2

        # Calculate microscopic view boundaries
        half_size = view_size // 2
        x_start = max(0, center_x - half_size)
        x_end = min(self.grid_size, center_x + half_size)
        y_start = max(0, center_y - half_size)
        y_end = min(self.grid_size, center_y + half_size)

        # Extract submatrices
        microscopic_view = {
            'wt_virions': snapshot['wt_virions'][x_start:x_end, y_start:y_end],
            'a_virions': snapshot['a_virions'][x_start:x_end, y_start:y_end],
            'cells_grid': snapshot['cells_grid'][x_start:x_end, y_start:y_end],
            'cell_states': snapshot['cell_states'][x_start:x_end, y_start:y_end] if snapshot[
                                                                                        'cell_states'] is not None else None,
            'coordinates': (x_start, x_end, y_start, y_end),
            'center': (center_x, center_y),
            'time_point': time_point
        }

        return microscopic_view

    def ask_microscopic_parameters(self):
        """
        Prompt user for microscopic view parameters.

        Returns:
            tuple: (micro_time, center_x, center_y, view_size)
        """
        print(f"\nAvailable temporal snapshots: {sorted(self.temporal_snapshots.keys())}")

        # Time for microscopic view
        while True:
            try:
                micro_time = int(input("At what time point do you want the microscopic view? (hours): "))
                if micro_time in self.temporal_snapshots:
                    break
                else:
                    print(f"Time {micro_time} not available. Options: {sorted(self.temporal_snapshots.keys())}")
            except ValueError:
                print("Please enter a valid integer.")

        # View size
        while True:
            try:
                view_size = int(input("Microscopic view size? (e.g., 50 for 50x50): "))
                if 10 <= view_size <= self.grid_size:
                    break
                else:
                    print(f"Size must be between 10 and {self.grid_size}")
            except ValueError:
                print("Please enter a valid integer.")

        # Center position
        print(f"Full plate: {self.grid_size}x{self.grid_size}")
        print("Press Enter to use plate center, or specify coordinates:")

        center_input = input("Center X (or Enter for automatic center): ").strip()
        if center_input:
            try:
                center_x = int(center_input)
                center_y = int(input("Center Y: "))
                # Validate coordinates are within bounds
                if not (0 <= center_x < self.grid_size and 0 <= center_y < self.grid_size):
                    print("Coordinates out of range, using automatic center.")
                    center_x = center_y = None
            except ValueError:
                print("Invalid coordinates, using automatic center.")
                center_x = center_y = None
        else:
            center_x = center_y = None

        return micro_time, center_x, center_y, view_size

    def run_simulation(self):
        """
        Execute the complete simulation.

        Returns:
            tuple: (wt_history, a_history, total_virions_history, infected_history)
        """
        max_time = self.params['max_time']

        print(f"Starting spatial simulation for {max_time} hours...")
        start_time = time.time()

        for t in range(1, max_time + 1):
            newly_infected = self.simulate_infection_step(t)

            if t % 10 == 0 or t == max_time:
                elapsed = time.time() - start_time
                wt_count = self.wt_history[-1]
                a_count = self.a_history[-1]
                infected = self.infected_history[-1]

                print(f"Hour {t}: WT={wt_count:.1f}, A={a_count:.1f}, Infected={infected} " +
                      f"(New: {newly_infected}) - Time: {elapsed:.1f}s")

        return (self.wt_history, self.a_history, self.total_virions_history, self.infected_history)

    def plot_spatial_state(self, title=None, save_path=None, snapshot_time=None,
                           microscopic_view=True, micro_time=None, center_x=None,
                           center_y=None, view_size=50):
        """
        Generate a visualization of the spatial simulation state.

        Args:
            title (str, optional): Title for the plot
            save_path (str, optional): Path to save the plot
            snapshot_time (int, optional): Time point for snapshot
            microscopic_view (bool): Whether to include microscopic view
            micro_time (int, optional): Time for microscopic view
            center_x (int, optional): X center for microscopic view
            center_y (int, optional): Y center for microscopic view
            view_size (int): Size of microscopic view
        """
        # Request microscopic parameters if microscopic view is required but parameters not provided
        if microscopic_view and micro_time is None:
            micro_time, center_x, center_y, view_size = self.ask_microscopic_parameters()

        # Get microscopic view data
        if microscopic_view:
            micro_view = self.get_microscopic_view(micro_time, center_x, center_y, view_size)
            if micro_view is None:
                print("Could not obtain microscopic view, showing complete view.")
                microscopic_view = False

        # Create figure with grid layout for subplots
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1], figure=fig)

        # Plot 1: Cell distribution (complete view)
        ax1 = fig.add_subplot(gs[0, 0])
        cell_map = ax1.imshow(self.cells_grid.T, cmap='Blues', interpolation='nearest')
        ax1.set_title('Cells distribution (Complete plate)')
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Position Y')
        plt.colorbar(cell_map, ax=ax1, label='Presence of cells')

        # Mark microscopic area if applicable
        if microscopic_view:
            x_start, x_end, y_start, y_end = micro_view['coordinates']
            # Draw red rectangle to indicate the microscopic view area
            rect = plt.Rectangle((y_start, x_start), y_end - y_start, x_end - x_start,
                                 fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)

        if microscopic_view:
            # Plots 2, 3, 4: Microscopic views

            # Plot 2: Microscopic view of WT virions (scatter plot)
            ax2 = fig.add_subplot(gs[0, 1])
            wt_positions = np.where(micro_view['wt_virions'] > 0)
            if len(wt_positions[0]) > 0:
                # Scale size for better visibility
                sizes = micro_view['wt_virions'][wt_positions] * 10
                ax2.scatter(wt_positions[1], wt_positions[0], c=micro_view['wt_virions'][wt_positions],
                            s=sizes, cmap='Greens', alpha=0.7, edgecolors='darkgreen', linewidth=0.5)
                plt.colorbar(ax2.collections[0], ax=ax2, label='WT virions')

            # Show cells as background with transparency
            cell_bg = ax2.imshow(micro_view['cells_grid'].T, cmap='Blues', alpha=0.3, interpolation='nearest')
            ax2.set_title(f'Microscopic WT concentration view (t={micro_time}h)')
            ax2.set_xlabel('Position X (microscopic)')
            ax2.set_ylabel('Position Y (microscopic)')
            ax2.set_xlim(-0.5, micro_view['wt_virions'].shape[1] - 0.5)
            ax2.set_ylim(-0.5, micro_view['wt_virions'].shape[0] - 0.5)

            # Plot 3: Microscopic view of amplicon virions (scatter plot)
            ax3 = fig.add_subplot(gs[0, 2])
            a_positions = np.where(micro_view['a_virions'] > 0)
            if len(a_positions[0]) > 0:
                # Scale size for better visibility
                sizes = micro_view['a_virions'][a_positions] * 10
                ax3.scatter(a_positions[1], a_positions[0], c=micro_view['a_virions'][a_positions],
                            s=sizes, cmap='Reds', alpha=0.7, edgecolors='darkred', linewidth=0.5)
                plt.colorbar(ax3.collections[0], ax=ax3, label='Amplicon virions')

            # Show cells as background with transparency
            cell_bg = ax3.imshow(micro_view['cells_grid'].T, cmap='Blues', alpha=0.3, interpolation='nearest')
            ax3.set_title(f'Microscopic amplicon concentration view (t={micro_time}h)')
            ax3.set_xlabel('Position X (microscopic)')
            ax3.set_ylabel('Position Y (microscopic)')
            ax3.set_xlim(-0.5, micro_view['a_virions'].shape[1] - 0.5)
            ax3.set_ylim(-0.5, micro_view['a_virions'].shape[0] - 0.5)

            # Plot 4: Microscopic view of infection status
            ax4 = fig.add_subplot(gs[1, 0])

            # Create infection status map for microscopic view (now 2D)
            if micro_view['cell_states'] is not None:
                micro_infection_map = micro_view['cell_states'].copy()
            else:
                micro_infection_map = np.zeros((view_size, view_size))

            # Create custom colormap for infection states
            colors = ['white', 'green', 'red', 'purple', 'darkgreen', 'darkviolet']
            infection_cmap = LinearSegmentedColormap.from_list('infection_cmap', colors, N=6)

            inf_micro_map = ax4.imshow(micro_infection_map.T, cmap=infection_cmap,
                                       interpolation='nearest', vmin=0, vmax=5)
            ax4.set_title(f'Microscopic infection status (t={micro_time}h)')
            ax4.set_xlabel('Position X (microscopic)')
            ax4.set_ylabel('Position Y (microscopic)')

            # Create legend for infection states
            handles = [
                mpatches.Patch(color='white', label='Not infected'),
                mpatches.Patch(color='green', label='WT'),
                mpatches.Patch(color='red', label='Amplicon'),
                mpatches.Patch(color='purple', label='Coinfected'),
                mpatches.Patch(color='darkgreen', label='Multiple WT'),
                mpatches.Patch(color='darkviolet', label='Multiple coinfected')
            ]
            ax4.legend(handles=handles, loc='upper right', fontsize=8)

        else:
            # Original version for plots 2, 3, 4 (complete view)

            # Plot 2: WT virions distribution (complete view)
            ax2 = fig.add_subplot(gs[0, 1])
            wt_map = ax2.imshow(self.wt_virions.T, cmap='Greens', interpolation='nearest')
            ax2.set_title('WT Virions')
            ax2.set_xlabel('Position X')
            ax2.set_ylabel('Position Y')
            plt.colorbar(wt_map, ax=ax2, label='Number of WT virions')

            # Plot 3: Amplicon virions distribution (complete view)
            ax3 = fig.add_subplot(gs[0, 2])
            a_map = ax3.imshow(self.a_virions.T, cmap='Reds', interpolation='nearest')
            ax3.set_title('Amplicon virions')
            ax3.set_xlabel('Position X')
            ax3.set_ylabel('Position Y')
            plt.colorbar(a_map, ax=ax3, label='Number of amplicon virions')

            # Plot 4: Infection status map (complete view)
            ax4 = fig.add_subplot(gs[1, 0])
            # Now the infection map is directly 2D
            infection_map = self.cell_states.copy()

            # Create custom colormap for infection states
            colors = ['white', 'green', 'red', 'purple', 'darkgreen', 'darkviolet']
            infection_cmap = LinearSegmentedColormap.from_list('infection_cmap', colors, N=6)
            inf_map = ax4.imshow(infection_map.T, cmap=infection_cmap, interpolation='nearest', vmin=0, vmax=5)
            ax4.set_title('Infection Status')
            ax4.set_xlabel('Position X')
            ax4.set_ylabel('Position Y')

            # Create legend for infection states
            handles = [
                mpatches.Patch(color='white', label='Not infected'),
                mpatches.Patch(color='green', label='WT'),
                mpatches.Patch(color='red', label='Amplicon'),
                mpatches.Patch(color='purple', label='Coinfected'),
                mpatches.Patch(color='darkgreen', label='Multiple WT'),
                mpatches.Patch(color='darkviolet', label='Multiple coinfected')
            ]
            ax4.legend(handles=handles, loc='upper right')

        # Plots 5 and 6: Temporal dynamics (unchanged)

        # Plot 5: Virion temporal dynamics
        ax5 = fig.add_subplot(gs[1, 1])
        time_axis = range(len(self.wt_history))
        ax5.plot(time_axis, self.wt_history, color='green', label='WT')
        ax5.plot(time_axis, self.a_history, color='red', label='Amplicon')
        ax5.plot(time_axis, self.total_virions_history, color='purple', linestyle='--', label='Total')
        ax5.set_title('Virion temporal dynamics')
        ax5.set_xlabel('Time (h)')
        ax5.set_ylabel('Virion number')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, len(self.wt_history) - 1)

        # Mark microscopic view time point if applicable
        if microscopic_view:
            ax5.axvline(x=micro_time, color='red', linestyle='--', alpha=0.7,
                        label=f'Micro view (t={micro_time}h)')
            ax5.legend()

        # Plot 6: Infected cells over time
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time_axis, self.infected_history, color='blue')
        ax6.set_title('Infected cells')
        ax6.set_xlabel('Time (h)')
        ax6.set_ylabel('Number of cells')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, len(self.infected_history) - 1)

        # Mark microscopic view time point if applicable
        if microscopic_view:
            ax6.axvline(x=micro_time, color='red', linestyle='--', alpha=0.7)

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Add main title if provided
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        # Display the plot
        plt.show()

    def analyse_clustering(self, time_point=None, center_x=None, center_y=None, view_size=None):
        """
        Method to add to SpatialVirusSimulator.
        Analyzes clustering of WT and coinfected cells.

        Parameters:
        -----------
        time_point : int, optional
            Time point for analysis. If None, defaults to 15.
        center_x : int, optional
            X coordinate of the center for microscopic view
        center_y : int, optional
            Y coordinate of the center for microscopic view
        view_size : int, optional
            Size of the microscopic view window

        Returns:
        --------
        dict
            Dictionary containing clustering analysis results
        """
        if time_point is None:
            time_point = 15  # Fixed to 15 hours

        return analyse_clustering_for_snapshot(self, time_point, center_x, center_y, view_size)

    def plot_clustering(self, time_point=None, save_path=None):
        """
        Method to add to SpatialVirusSimulator.
        Generates visualization of clustering analysis.

        Parameters:
        -----------
        time_point : int, optional
            Time point for clustering analysis
        save_path : str, optional
            Path to save the clustering plot

        Returns:
        --------
        matplotlib.figure.Figure
            The generated clustering analysis plot
        """
        clustering_results = self.analyse_clustering(time_point)
        return plot_clustering_analysis(clustering_results, save_path)


def ask_user_criterion():
    """
    Prompts the user to select which type of viral production to maximise
    across generations of infection.

    Returns:
        str: Criterion for optimisation ('wt', 'amplicon', or 'total')
    """
    print("Which type of viral production would you like to maximise?")
    print("1. Total production (WT + amplicon)")
    print("2. WT only")
    print("3. Amplicons only")
    option = input("Please select an option (1/2/3): ").strip()

    if option == '2':
        return 'wt'
    elif option == '3':
        return 'amplicon'
    else:
        return 'total'


def simulate_multiple_generations(base_params, num_generations):
    """
    Simulates multiple generations of viral infection in spatial context.

    This function models the evolutionary dynamics of viral populations
    over successive infection cycles, maintaining constant MOI whilst
    allowing population composition to evolve based on fitness.

    Args:
        base_params (dict): Base simulation parameters
        num_generations (int): Number of generations to simulate

    Returns:
        tuple: Complete simulation histories and metrics for all generations
    """
    criterion = ask_user_criterion()

    # Initialise storage for all generation data
    all_wt_histories = []
    all_a_histories = []
    all_total_histories = []
    all_infected_histories = []

    max_production_times = []
    wt_counts = []
    a_counts = []

    # Establish initial conditions for first generation
    wt_initial_first_gen = base_params['wt_initial']
    a_initial_first_gen = base_params['a_initial']

    # Calculate and maintain constant MOI across all generations
    initial_total_virions = wt_initial_first_gen + a_initial_first_gen
    constant_moi = initial_total_virions / base_params['num_cells']
    print(f"Constant MOI for all generations: {constant_moi:.3f}")

    # Record initial population sizes
    wt_counts.append(wt_initial_first_gen)
    a_counts.append(a_initial_first_gen)

    current_params = base_params.copy()

    # Execute simulation for each generation
    for gen in range(num_generations):
        print(f"\nSimulating generation {gen + 1}...")

        simulator = SpatialVirusSimulator(current_params)
        wt_history, a_history, total_history, infected_history = simulator.run_simulation()

        # Store complete temporal dynamics
        all_wt_histories.append(wt_history)
        all_a_histories.append(a_history)
        all_total_histories.append(total_history)
        all_infected_histories.append(infected_history)

        # Determine optimal harvest time based on selected criterion
        if criterion == 'wt':
            target_array = wt_history
        elif criterion == 'amplicon':
            target_array = a_history
        else:
            target_array = total_history

        # Find time point of maximum production (excluding t=0)
        max_time_idx = np.argmax(target_array[1:]) + 1
        max_production_times.append(max_time_idx)

        # Extract viral populations at optimal harvest time
        max_wt = wt_history[max_time_idx]
        max_a = a_history[max_time_idx]
        max_total = max_wt + max_a

        wt_counts.append(max_wt)
        a_counts.append(max_a)

        # Prepare inoculum for next generation (if not final generation)
        if gen < num_generations - 1:
            # Calculate population proportions at harvest
            wt_proportion = max_wt / max_total if max_total > 0 else 0
            a_proportion = max_a / max_total if max_total > 0 else 0

            # Determine inoculum composition to maintain constant MOI
            target_total = constant_moi * base_params['num_cells']
            new_wt_initial = target_total * wt_proportion
            new_a_initial = target_total * a_proportion

            print(f"  Adjusting virions based on maximum {criterion} at t={max_time_idx}h")
            print(f"  WT: {max_wt:.1f}, A: {max_a:.1f}, Total: {max_total:.1f}")
            print(f"  Next generation inoculum: WT={new_wt_initial:.1f}, A={new_a_initial:.1f}")

            # Update parameters for next generation
            current_params['wt_initial'] = new_wt_initial
            current_params['a_initial'] = new_a_initial

        # Generate spatial visualisation at optimal harvest time
        simulator.plot_spatial_state(
            title=f"Spatial state at t={max_time_idx}h (maximum {criterion})",
            save_path=f"generation_{gen + 1}_t{max_time_idx}_{criterion}.png",
            snapshot_time=max_time_idx
        )

    # Generate comprehensive analysis plots
    plot_log_comparison(all_wt_histories, all_a_histories, max_production_times,
                        wt_counts, a_counts, criterion)

    return (all_wt_histories, all_a_histories, all_total_histories, all_infected_histories,
            max_production_times, wt_counts, a_counts)


def plot_individual_generations(all_wt, all_a, all_total, all_infected, max_times,
                                num_generations, criterion):
    """
    Creates individual plots for each generation showing complete temporal dynamics.

    This visualisation displays the full time course of each infection cycle,
    highlighting the optimal harvest time for the selected criterion.

    Args:
        all_wt (list): WT viral population histories for all generations
        all_a (list): Amplicon viral population histories for all generations
        all_total (list): Total viral population histories for all generations
        all_infected (list): Infected cell count histories for all generations
        max_times (list): Optimal harvest times for each generation
        num_generations (int): Total number of generations simulated
        criterion (str): Optimisation criterion used
    """
    # Determine subplot layout
    rows = int(np.ceil(num_generations / 2))
    cols = min(2, num_generations)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Handle different subplot configurations
    if num_generations == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each generation
    for gen in range(num_generations):
        row = gen // cols
        col = gen % cols
        ax = axes[row, col]

        # Plot complete temporal dynamics
        ax.plot(range(len(all_wt[gen])), all_wt[gen],
                label='WT', color='green', linewidth=2)
        ax.plot(range(len(all_a[gen])), all_a[gen],
                label='Amplicon', color='red', linewidth=2)
        ax.plot(range(len(all_total[gen])), all_total[gen],
                label='Total virions', color='purple', linestyle='--', linewidth=2)
        ax.plot(range(len(all_infected[gen])), all_infected[gen],
                label='Infected cells', color='blue', linestyle=':', linewidth=2)

        # Mark optimal harvest time
        t_max = max_times[gen]
        ax.axvline(x=t_max, color='black', linestyle='--', alpha=0.7,
                   label=f'Optimal harvest (t={t_max}h)')

        ax.set_title(f'Generation {gen + 1} (maximum at t={t_max}h)', fontsize=12)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Count (virions/cells)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # Remove unused subplots
    for gen in range(num_generations, rows * cols):
        row = gen // cols
        col = gen % cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(f'individual_infections_complete_{criterion}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_log_comparison(all_wt, all_a, max_times, wt_counts, a_counts, criterion):
    """
    Generates logarithmic comparison plots showing relative viral production
    across generations.

    This analysis reveals the evolutionary trajectory of each viral type,
    normalised to their initial abundances to highlight relative fitness changes.

    Args:
        all_wt (list): WT viral histories
        all_a (list): Amplicon viral histories
        max_times (list): Optimal harvest times
        wt_counts (list): WT counts at each generation
        a_counts (list): Amplicon counts at each generation
        criterion (str): Optimisation criterion
    """
    num_generations = len(wt_counts) - 1  # Exclude initial population

    # Calculate relative production compared to initial populations
    wt_relative = [wt_counts[i + 1] / wt_counts[0] if wt_counts[0] > 0 else 0
                   for i in range(num_generations)]
    a_relative = [a_counts[i + 1] / a_counts[0] if a_counts[0] > 0 else 0
                  for i in range(num_generations)]

    generations = np.arange(1, num_generations + 1)

    # Create figure with dual subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # WT relative production subplot
    ax1.plot(generations, wt_relative, marker='o', color='green',
             linewidth=3, markersize=8, label='WT relative production')
    ax1.set_title('Wild-type Virions: Relative Maximum Production', fontsize=14, weight='bold')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Maximum Production / Initial Population')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add reference line at initial level
    ax1.axhline(y=1, color='darkgreen', linestyle='--', alpha=0.6,
                label='Initial level')
    ax1.legend()

    # Amplicon relative production subplot
    ax2.plot(generations, a_relative, marker='s', color='red',
             linewidth=3, markersize=8, label='Amplicon relative production')
    ax2.set_title('Amplicon Virions: Relative Maximum Production', fontsize=14, weight='bold')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Maximum Production / Initial Population')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add reference line at initial level
    ax2.axhline(y=1, color='darkred', linestyle='--', alpha=0.6,
                label='Initial level')
    ax2.legend()

    # Synchronise y-axis scales for comparison
    if any(wt_relative) and any(a_relative):
        min_val = min(min(v for v in wt_relative if v > 0),
                      min(v for v in a_relative if v > 0))
        max_val = max(max(wt_relative), max(a_relative))

        # Add margins for better visualisation
        min_val = min_val * 0.3
        max_val = max_val * 3

        ax1.set_ylim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

    # Overall figure title
    fig.suptitle(f'Evolutionary Dynamics: Relative Production Trends (optimised for {criterion})',
                 fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig(f'log_relative_maxima_comparison_{criterion}.png', dpi=300, bbox_inches='tight')
    plt.show()


def find_clusters(cell_states, include_states=[1, 3, 4, 5]):
    """
    Identifies clusters of connected cells in specified infection states.

    Uses 8-connectivity (including diagonals) to define spatial clusters
    of infected cells that would be experimentally observable.

    Args:
        cell_states (ndarray): 2D array of cellular infection states
        include_states (list): States to include in clustering analysis
                              (1: WT, 3: Coinfected, 4: Multiple WT, 5: Multiple coinfected)

    Returns:
        tuple: (labelled_array, cluster_sizes, num_clusters)
    """
    # Create binary mask for cells of interest
    mask = np.isin(cell_states, include_states)

    # Define 8-connectivity structure (includes diagonal neighbours)
    connectivity_structure = np.array([[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]])

    # Identify connected components
    labelled_array, num_clusters = label(mask, structure=connectivity_structure)

    # Calculate cluster size distribution
    if num_clusters > 0:
        cluster_sizes = [np.sum(labelled_array == i) for i in range(1, num_clusters + 1)]
    else:
        cluster_sizes = []

    return labelled_array, cluster_sizes, num_clusters


def analyse_clustering_for_snapshot(simulator, time_point, center_x=None, center_y=None, view_size=None):
    """
    Performs comprehensive clustering analysis for a specific temporal snapshot.

    This function quantifies the spatial organisation of infected cells,
    providing metrics relevant to experimental plaque assay interpretation.

    Args:
        simulator: SpatialVirusSimulator instance
        time_point (int): Time point for analysis
        center_x, center_y (int, optional): Centre coordinates for regional analysis
        view_size (int, optional): Size of analysis region (None for full plate)

    Returns:
        dict: Comprehensive clustering analysis results
    """
    if time_point not in simulator.temporal_snapshots:
        print(f"Warning: No snapshot available for time point {time_point}")
        return None

    # Extract relevant data from snapshot
    if view_size is not None and center_x is not None and center_y is not None:
        # Regional analysis
        snapshot_data = simulator.get_microscopic_view(time_point, center_x, center_y, view_size)
        cell_states = snapshot_data['cell_states']
        cells_grid = snapshot_data['cells_grid']
        region_info = f"Region {view_size}×{view_size} centred at ({center_x},{center_y})"
    else:
        # Full plate analysis
        snapshot = simulator.temporal_snapshots[time_point]
        cell_states = snapshot['cell_states']
        cells_grid = snapshot['cells_grid']
        region_info = "Complete plate"

    if cell_states is None:
        return None

    # Identify spatial clusters of experimentally visible infections
    labelled_clusters, cluster_sizes, num_clusters = find_clusters(cell_states,
                                                                   include_states=[1, 3, 4, 5])

    # Calculate clustering statistics
    total_visible_cells = np.sum(np.isin(cell_states, [1, 3, 4, 5]))
    cells_in_clusters = sum(cluster_sizes) if cluster_sizes else 0
    isolated_cells = total_visible_cells - cells_in_clusters

    # Compute cluster size statistics
    cluster_statistics = {}
    if cluster_sizes:
        cluster_statistics = {
            'mean_size': np.mean(cluster_sizes),
            'median_size': np.median(cluster_sizes),
            'max_size': np.max(cluster_sizes),
            'min_size': np.min(cluster_sizes),
            'std_size': np.std(cluster_sizes)
        }

    # Compile comprehensive results
    results = {
        'time_point': time_point,
        'region_info': region_info,
        'cell_states': cell_states,
        'cells_grid': cells_grid,
        'labelled_clusters': labelled_clusters,
        'cluster_sizes': cluster_sizes,
        'num_clusters': num_clusters,
        'total_visible_cells': total_visible_cells,
        'cells_in_clusters': cells_in_clusters,
        'isolated_cells': isolated_cells,
        'cluster_statistics': cluster_statistics
    }

    return results


def plot_clustering_analysis(clustering_results, save_path=None):
    """
    Creates comprehensive visualisation of spatial clustering analysis.

    Generates a four-panel figure showing infection states, identified clusters,
    size distributions, and summary statistics.

    Args:
        clustering_results (dict): Results from analyse_clustering_for_snapshot
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if clustering_results is None:
        print("No clustering data available for visualisation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Original cellular infection states
    ax1 = axes[0, 0]
    cell_states = clustering_results['cell_states']

    # Define colour scheme for infection states
    state_colours = ['white', 'green', 'red', 'purple', 'darkgreen', 'darkviolet', 'black']
    infection_colormap = LinearSegmentedColormap.from_list('infection_states',
                                                           state_colours, N=7)

    im1 = ax1.imshow(cell_states.T, cmap=infection_colormap,
                     interpolation='nearest', vmin=-1, vmax=5)
    ax1.set_title(f'Cellular Infection States (t={clustering_results["time_point"]}h)',
                  fontsize=12, weight='bold')
    ax1.set_xlabel('Position X')
    ax1.set_ylabel('Position Y')

    # Create comprehensive legend
    legend_elements = [
        mpatches.Patch(color='white', label='Healthy'),
        mpatches.Patch(color='green', label='WT infected'),
        mpatches.Patch(color='red', label='Amplicon infected'),
        mpatches.Patch(color='purple', label='Coinfected'),
        mpatches.Patch(color='darkgreen', label='Multiple WT'),
        mpatches.Patch(color='darkviolet', label='Multiple coinfected'),
        mpatches.Patch(color='black', label='Dead')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Panel 2: Identified spatial clusters
    ax2 = axes[0, 1]
    labelled_clusters = clustering_results['labelled_clusters']

    # Display only experimentally visible clusters
    visible_mask = np.isin(cell_states, [1, 3, 4, 5])
    display_clusters = labelled_clusters * visible_mask

    # Use distinctive colour scheme for cluster identification
    cluster_colormap = plt.cm.tab20
    im2 = ax2.imshow(display_clusters.T, cmap=cluster_colormap, interpolation='nearest')
    ax2.set_title(f'Identified Clusters (n={clustering_results["num_clusters"]})',
                  fontsize=12, weight='bold')
    ax2.set_xlabel('Position X')
    ax2.set_ylabel('Position Y')

    # Panel 3: Cluster size distribution
    ax3 = axes[1, 0]
    cluster_sizes = clustering_results['cluster_sizes']

    if cluster_sizes:
        # Create histogram of cluster sizes
        bins = range(1, max(cluster_sizes) + 2)
        ax3.hist(cluster_sizes, bins=bins, alpha=0.7, color='skyblue',
                 edgecolor='navy', linewidth=1.2)
        ax3.set_xlabel('Cluster Size (number of cells)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Cluster Size Distribution', fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)

        # Add statistical summary
        stats = clustering_results['cluster_statistics']
        stats_text = f'Mean: {stats["mean_size"]:.1f}\n'
        stats_text += f'Median: {stats["median_size"]:.1f}\n'
        stats_text += f'Maximum: {stats["max_size"]}\n'
        stats_text += f'Std Dev: {stats["std_size"]:.1f}'

        ax3.text(0.7, 0.7, stats_text, transform=ax3.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
                 fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No clusters identified', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Cluster Size Distribution', fontsize=12, weight='bold')

    # Panel 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Prepare summary data
    summary_data = [
        ['Total visible cells', clustering_results['total_visible_cells']],
        ['Cells in clusters', clustering_results['cells_in_clusters']],
        ['Isolated cells', clustering_results['isolated_cells']],
        ['Number of clusters', clustering_results['num_clusters']],
        ['Analysis region', clustering_results['region_info']]
    ]

    if cluster_sizes:
        summary_data.extend([
            ['Largest cluster', clustering_results['cluster_statistics']['max_size']],
            ['Average cluster size', f"{clustering_results['cluster_statistics']['mean_size']:.2f}"]
        ])

    # Create formatted table
    table = ax4.table(cellText=summary_data,
                      colLabels=['Metric', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # Style the table
    table[(0, 0)].set_facecolor('#4CAF50')
    table[(0, 1)].set_facecolor('#4CAF50')

    ax4.set_title('Clustering Analysis Summary', pad=20, fontsize=12, weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
    return fig


def compare_clustering_across_mois(base_params, moi_values, time_point=None, num_simulations=3):
    """
    Comparative analysis of spatial clustering patterns across different MOI values.

    This function investigates how multiplicity of infection affects the spatial
    organisation of viral plaques, providing insights into transmission dynamics
    and experimental design considerations.

    Args:
        base_params (dict): Base simulation parameters
        moi_values (list): MOI values to compare
        time_point (int, optional): Analysis time point (defaults to maximum production)
        num_simulations (int): Number of replicate simulations per MOI

    Returns:
        dict: Comprehensive comparative analysis results
    """
    print(f"Comparative clustering analysis for MOI values: {moi_values}")
    print(f"Replicate simulations per MOI: {num_simulations}")

    all_results = {}

    for moi in moi_values:
        print(f"\n--- Analysing MOI = {moi} ---")

        # Calculate required initial viral load for target MOI
        total_virions_required = moi * base_params['num_cells']

        moi_results = {
            'moi': moi,
            'cluster_sizes_all_simulations': [],
            'num_clusters_all_simulations': [],
            'clustering_analyses': []
        }

        # Perform replicate simulations
        for simulation in range(num_simulations):
            print(f"  Simulation {simulation + 1}/{num_simulations}")

            # Configure parameters for this simulation
            current_params = base_params.copy()
            current_params['wt_initial'] = total_virions_required
            current_params['a_initial'] = 0  # Pure WT infection for MOI comparison

            # Execute simulation
            simulator = SpatialVirusSimulator(current_params)
            wt_history, a_history, total_history, infected_history = simulator.run_simulation()

            # Set fixed analysis time point
            analysis_time = 15  # Fixed at 15 hours for consistency

            # Perform clustering analysis
            clustering_result = analyse_clustering_for_snapshot(simulator, analysis_time)

            if clustering_result:
                moi_results['cluster_sizes_all_simulations'].extend(clustering_result['cluster_sizes'])
                moi_results['num_clusters_all_simulations'].append(clustering_result['num_clusters'])
                moi_results['clustering_analyses'].append(clustering_result)

        all_results[moi] = moi_results

    # Generate comparative visualisation
    plot_moi_clustering_comparison(all_results)

    return all_results


def plot_moi_clustering_comparison(all_results, save_path=None):
    """
    Creates comprehensive comparative visualisation of clustering patterns across MOI values.

    Generates multi-panel figure showing distribution patterns, average metrics,
    and detailed statistical comparisons.

    Args:
        all_results (dict): Results from compare_clustering_across_mois
        save_path (str, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The generated comparative figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mois = sorted(all_results.keys())
    colours = plt.cm.viridis(np.linspace(0, 1, len(mois)))

    # Panel 1: Cluster size distribution by MOI (box plot)
    ax1 = axes[0, 0]

    cluster_data = []
    moi_labels = []

    for moi in mois:
        cluster_sizes = all_results[moi]['cluster_sizes_all_simulations']
        if cluster_sizes:
            cluster_data.extend(cluster_sizes)
            moi_labels.extend([f'MOI {moi}'] * len(cluster_sizes))

    if cluster_data:
        # Create structured data for visualisation
        import pandas as pd
        df = pd.DataFrame({'MOI': moi_labels, 'Cluster Size': cluster_data})

        sns.boxplot(data=df, x='MOI', hue='MOI', y='Cluster Size', ax=ax1,
                    palette='viridis', legend=False)
        ax1.set_title('Cluster Size Distributions by MOI', fontsize=14, weight='bold')
        ax1.set_xlabel('Multiplicity of Infection (MOI)')
        ax1.set_ylabel('Cluster Size (number of cells)')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No clusters identified across MOI range',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    # Panel 2: Average number of clusters vs MOI
    ax2 = axes[0, 1]

    average_cluster_counts = []
    cluster_count_std = []

    for moi in mois:
        cluster_counts = all_results[moi]['num_clusters_all_simulations']
        if cluster_counts:
            average_cluster_counts.append(np.mean(cluster_counts))
            cluster_count_std.append(np.std(cluster_counts))
        else:
            average_cluster_counts.append(0)
            cluster_count_std.append(0)

    ax2.errorbar(mois, average_cluster_counts, yerr=cluster_count_std,
                 marker='o', linestyle='-', linewidth=2.5, markersize=8,
                 capsize=5, capthick=2)
    ax2.set_xlabel('Multiplicity of Infection (MOI)')
    ax2.set_ylabel('Average Number of Clusters')
    ax2.set_title('Cluster Frequency vs MOI', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Average cluster size vs MOI
    ax3 = axes[1, 0]

    average_cluster_sizes = []
    cluster_size_std = []

    for moi in mois:
        cluster_sizes = all_results[moi]['cluster_sizes_all_simulations']
        if cluster_sizes:
            average_cluster_sizes.append(np.mean(cluster_sizes))
            cluster_size_std.append(np.std(cluster_sizes))
        else:
            average_cluster_sizes.append(0)
            cluster_size_std.append(0)

    ax3.errorbar(mois, average_cluster_sizes, yerr=cluster_size_std,
                 marker='s', linestyle='-', linewidth=2.5, markersize=8,
                 color='red', capsize=5, capthick=2)
    ax3.set_xlabel('Multiplicity of Infection (MOI)')
    ax3.set_ylabel('Average Cluster Size (cells)')
    ax3.set_title('Mean Cluster Size vs MOI', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Detailed distribution comparison (violin plots)
    ax4 = axes[1, 1]

    all_cluster_data = []
    all_moi_labels = []

    for moi in mois:
        cluster_sizes = all_results[moi]['cluster_sizes_all_simulations']
        all_cluster_data.extend(cluster_sizes)
        all_moi_labels.extend([moi] * len(cluster_sizes))

    if all_cluster_data:
        df_violin = pd.DataFrame({'MOI': all_moi_labels, 'Cluster Size': all_cluster_data})
        sns.violinplot(data=df_violin, x='MOI', y='Cluster Size', ax=ax4)
        ax4.set_title('Cluster Size Distributions by MOI')
        ax4.set_xlabel('MOI')
        ax4.set_ylabel('Cluster Size')
    else:
        ax4.text(0.5, 0.5, 'No data for violin plot',
                 ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

    return fig