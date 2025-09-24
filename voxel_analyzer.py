"""
Voxel-Based Analysis Module
Comprehensive voxel-based analysis of binding sites with field calculations
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import gaussian_filter, binary_dilation, maximum_filter, minimum_filter, binary_erosion, label
from scipy.spatial.distance import cdist
import warnings

from config import LigandForgeConfig
from data_structures import (
    VoxelAnalysisResult, PropertyGrid, PocketVoxel, AtomicEnvironment,
    create_empty_property_grid, validate_position_array
)
from pdb_parser import PDBParser


class VoxelBasedAnalyzer:
    """Comprehensive voxel-based analysis of binding sites with field calculations"""
    
    def __init__(self, config: LigandForgeConfig):
        self.config = config
        self.pdb_parser = PDBParser()
        # Cache frequently used values
        self._coulomb_const = self.config.coulomb_constant
        self._dielectric_const = self.config.dielectric_constant
        self._probe_radius = self.config.probe_radius
    
    def analyze_binding_site_voxels(self, pdb_text: str, center: np.ndarray, 
                                  radius: float = 10.0) -> VoxelAnalysisResult:
        """Perform comprehensive voxel-based binding site analysis"""
        
        center = validate_position_array(center, "center")
        
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        # Parse structure and create atomic environments
        structure = self.pdb_parser.parse_pdb_structure(pdb_text)
        atomic_environments = self.pdb_parser.create_atomic_environments(
            structure, center, radius + 5.0
        )
        
        if not atomic_environments:
            raise ValueError("No atoms found in the specified region")
        
        # Pre-filter atoms with significant contributions to reduce computation
        filtered_environments = self._filter_significant_atoms(atomic_environments, center, radius)
        
        # Set up voxel grid
        grid_spacing = self.config.get_grid_spacing_for_resolution()
        grid_radius = radius + self.config.grid_padding
        grid_origin = center - grid_radius
        n_points = int(2 * grid_radius / grid_spacing) + 1
        grid_dimensions = (n_points, n_points, n_points)
        
        # Create coordinate arrays for vectorized calculations
        coords_1d = self._generate_grid_coordinates(grid_origin, grid_spacing, n_points)
        
        # Compute all field components with progress tracking
        print(f"Computing fields on {len(coords_1d):,} voxels with {len(filtered_environments)} atoms...")
        
        # Batch process fields for memory efficiency
        batch_size = min(100000, len(coords_1d))  # Process in chunks to manage memory
        
        electrostatic_grid = self._compute_field_batched(
            coords_1d, filtered_environments, batch_size, 
            self._compute_electrostatic_contribution
        ).reshape(grid_dimensions)
        
        hydrophobicity_grid = self._compute_field_batched(
            coords_1d, filtered_environments, batch_size,
            self._compute_hydrophobicity_contribution
        ).reshape(grid_dimensions)
        
        steric_grid = self._compute_field_batched(
            coords_1d, filtered_environments, batch_size,
            self._compute_steric_contribution
        ).reshape(grid_dimensions)
        
        accessibility_grid = self._compute_field_batched(
            coords_1d, filtered_environments, batch_size,
            self._compute_accessibility_contribution
        ).reshape(grid_dimensions)
        
        shape_grid = self._compute_shape_field_optimized(
            coords_1d, filtered_environments, center, radius
        ).reshape(grid_dimensions)
        
        # Apply smoothing to reduce noise
        if self.config.field_smoothing_sigma > 0:
            sigma = self.config.field_smoothing_sigma
            electrostatic_grid = gaussian_filter(electrostatic_grid, sigma=sigma)
            hydrophobicity_grid = gaussian_filter(hydrophobicity_grid, sigma=sigma)
        
        # Compute field gradients for interaction analysis
        gradient_magnitude = self._compute_field_gradients(electrostatic_grid, hydrophobicity_grid)
        
        # Create property grid
        property_grid = PropertyGrid(
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            grid_dimensions=grid_dimensions,
            electrostatic_grid=electrostatic_grid,
            hydrophobicity_grid=hydrophobicity_grid,
            steric_grid=steric_grid,
            accessibility_grid=accessibility_grid,
            shape_grid=shape_grid,
            gradient_magnitude=gradient_magnitude
        )
        
        # Extract voxel-based hotspots
        hotspot_voxels = self._extract_hotspot_voxels(property_grid, center, radius)
        
        # Calculate comprehensive metrics efficiently
        metrics = self._calculate_all_metrics(property_grid, center, grid_spacing)
        
        return VoxelAnalysisResult(
            property_grid=property_grid,
            hotspot_voxels=hotspot_voxels,
            cavity_volume=metrics['cavity_volume'],
            surface_area=metrics['surface_area'],
            electrostatic_variance=metrics['electrostatic_variance'],
            hydrophobic_moment=metrics['hydrophobic_moment'],
            field_complexity_score=metrics['field_complexity_score'],
            shape_descriptors=metrics['shape_descriptors'],
            pocket_depth=metrics['shape_descriptors'].get('depth', 0.0),
            pocket_width=metrics['shape_descriptors'].get('width', 0.0),
            convexity=metrics['shape_descriptors'].get('convexity', 0.0)
        )
    
    def _filter_significant_atoms(self, atomic_environments: List[AtomicEnvironment], 
                                center: np.ndarray, radius: float) -> List[AtomicEnvironment]:
        """Pre-filter atoms that contribute significantly to reduce computation"""
        filtered = []
        cutoff_distance = radius + 10.0  # Extended cutoff for field effects
        
        for env in atomic_environments:
            distance_to_center = np.linalg.norm(env.position - center)
            
            # Keep atoms that might contribute to fields
            if distance_to_center <= cutoff_distance:
                # Additional filtering based on properties
                has_charge = abs(env.partial_charge) > 0.01
                has_hydrophobicity = (env.is_hydrophobic or 
                                    env.element in ['O', 'N', 'C'] or 
                                    env.is_aromatic)
                
                if has_charge or has_hydrophobicity:
                    filtered.append(env)
        
        print(f"Filtered {len(atomic_environments)} atoms to {len(filtered)} significant contributors")
        return filtered
    
    def _generate_grid_coordinates(self, origin: np.ndarray, spacing: float, 
                                 n_points: int) -> np.ndarray:
        """Generate flattened grid coordinates for vectorized calculations"""
        # Use more memory-efficient coordinate generation
        coords = np.mgrid[0:n_points, 0:n_points, 0:n_points] * spacing
        coords = coords.reshape(3, -1).T + origin
        return coords
    
    def _compute_field_batched(self, grid_coords: np.ndarray, 
                             atomic_environments: List[AtomicEnvironment],
                             batch_size: int, contribution_func) -> np.ndarray:
        """Compute fields in batches to manage memory usage"""
        result = np.zeros(len(grid_coords))
        
        for i in range(0, len(grid_coords), batch_size):
            end_idx = min(i + batch_size, len(grid_coords))
            batch_coords = grid_coords[i:end_idx]
            
            batch_result = np.zeros(len(batch_coords))
            for env in atomic_environments:
                batch_result += contribution_func(batch_coords, env)
            
            result[i:end_idx] = batch_result
        
        return result
    
    def _compute_electrostatic_contribution(self, coords: np.ndarray, 
                                          env: AtomicEnvironment) -> np.ndarray:
        """Compute electrostatic contribution from single atom"""
        if abs(env.partial_charge) < 0.01:
            return np.zeros(len(coords))
        
        # Vectorized distance calculation
        distances = np.linalg.norm(coords - env.position, axis=1)
        
        # Apply electrostatic cutoff efficiently
        if self.config.electrostatic_cutoff > 0:
            valid_mask = distances <= self.config.electrostatic_cutoff
            if not np.any(valid_mask):
                return np.zeros(len(coords))
        else:
            valid_mask = np.ones(len(coords), dtype=bool)
        
        # Avoid singularities and compute potential
        distances = np.maximum(distances, 0.1)
        dielectric = self._dielectric_const * distances
        potential = np.zeros(len(coords))
        
        coulomb_potential = (self._coulomb_const * env.partial_charge) / (dielectric * distances)
        potential[valid_mask] = coulomb_potential[valid_mask]
        
        return potential
    
    def _compute_hydrophobicity_contribution(self, coords: np.ndarray, 
                                           env: AtomicEnvironment) -> np.ndarray:
        """Compute hydrophobicity contribution from single atom"""
        # Get residue hydrophobicity from scale
        residue_hydrophobicity = self.config.residue_hydrophobicity.get(env.residue_name, 0.0)
        
        # Atomic contribution to hydrophobicity (optimized logic)
        atom_hydrophobicity = self._get_atomic_hydrophobicity(env)
        
        # Combined hydrophobicity score
        total_hydrophobicity = 0.6 * residue_hydrophobicity + 0.4 * atom_hydrophobicity
        
        if abs(total_hydrophobicity) < 0.01:
            return np.zeros(len(coords))
        
        # Vectorized distance calculation and decay
        distances = np.linalg.norm(coords - env.position, axis=1)
        sigma = env.van_der_waals_radius + self.config.hydrophobicity_decay
        decay_factor = np.exp(-distances / sigma)
        
        return total_hydrophobicity * decay_factor
    
    def _get_atomic_hydrophobicity(self, env: AtomicEnvironment) -> float:
        """Get atomic hydrophobicity efficiently"""
        if env.is_hydrophobic:
            return 1.0
        elif env.element in ['O', 'N'] and (env.is_hbd or env.is_hba):
            return -0.5
        elif env.element == 'C' and not env.is_aromatic:
            return 0.2
        elif env.is_aromatic:
            return 0.4
        else:
            return 0.0
    
    def _compute_steric_contribution(self, coords: np.ndarray, 
                                   env: AtomicEnvironment) -> np.ndarray:
        """Compute steric contribution from single atom"""
        distances = np.linalg.norm(coords - env.position, axis=1)
        return (distances <= env.van_der_waals_radius).astype(float)
    
    def _compute_accessibility_contribution(self, coords: np.ndarray, 
                                          env: AtomicEnvironment) -> np.ndarray:
        """Compute accessibility contribution from single atom"""
        distances = np.linalg.norm(coords - env.position, axis=1)
        effective_radius = env.van_der_waals_radius + self._probe_radius
        
        # More efficient accessibility calculation
        accessibility_factor = np.where(
            distances <= effective_radius,
            0.0,
            np.minimum(1.0, (distances - effective_radius) / 2.0)
        )
        return accessibility_factor
    
    def _compute_shape_field_optimized(self, grid_coords: np.ndarray, 
                                     atomic_environments: List[AtomicEnvironment], 
                                     center: np.ndarray, radius: float) -> np.ndarray:
        """Define binding cavity shape with optimized computation"""
        # Start with spherical cavity
        distances_to_center = np.linalg.norm(grid_coords - center, axis=1)
        shape = (distances_to_center <= radius).astype(float)
        
        # Vectorized protein volume subtraction
        for env in atomic_environments:
            distances = np.linalg.norm(grid_coords - env.position, axis=1)
            exclusion_radius = env.van_der_waals_radius + 0.5
            shape = np.where(distances <= exclusion_radius, 0.0, shape)
        
        return shape
    
    def _compute_field_gradients(self, electrostatic_grid: np.ndarray, 
                               hydrophobicity_grid: np.ndarray) -> np.ndarray:
        """Compute magnitude of field gradients efficiently"""
        # Calculate gradients using numpy gradient
        grad_elec = np.gradient(electrostatic_grid)
        grad_hydro = np.gradient(hydrophobicity_grid)
        
        # More efficient gradient magnitude calculation
        grad_mag_elec = np.sqrt(np.sum([g**2 for g in grad_elec], axis=0))
        grad_mag_hydro = np.sqrt(np.sum([g**2 for g in grad_hydro], axis=0))
        
        return grad_mag_elec + grad_mag_hydro
    
    def _extract_hotspot_voxels(self, property_grid: PropertyGrid, 
                              center: np.ndarray, radius: float) -> List[PocketVoxel]:
        """Extract hotspot voxels from computed fields with improved efficiency"""
        hotspot_voxels = []
        accessibility_threshold = 0.3
        
        # Pre-compute accessibility mask to avoid repeated checks
        accessible_mask = property_grid.accessibility_grid > accessibility_threshold
        
        # Find significant electrostatic extrema
        elec_extrema = self._find_combined_extrema(
            property_grid.electrostatic_grid, threshold=2.0
        )
        
        # Find hydrophobic hotspots
        hydro_maxima = self._find_local_extrema_3d(
            property_grid.hydrophobicity_grid, threshold=0.5
        )
        
        # Combine all potential hotspots
        all_extrema = elec_extrema + hydro_maxima
        
        # Process hotspots efficiently
        for idx in all_extrema:
            if not accessible_mask[idx]:
                continue
                
            world_pos = property_grid.get_world_coordinates(*idx)
            if np.linalg.norm(world_pos - center) > radius:
                continue
            
            voxel = PocketVoxel(
                position=world_pos,
                is_occupied=False,
                electrostatic_potential=property_grid.electrostatic_grid[idx],
                hydrophobicity=property_grid.hydrophobicity_grid[idx],
                accessibility=property_grid.accessibility_grid[idx],
                distance_to_surface=self._distance_to_surface(idx, property_grid.shape_grid)
            )
            hotspot_voxels.append(voxel)
        
        return hotspot_voxels
    
    def _find_combined_extrema(self, grid: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int, int]]:
        """Find both maxima and minima in a single pass"""
        # Find maxima
        maxima = self._find_local_extrema_3d(grid, threshold)
        # Find minima
        minima = self._find_local_extrema_3d(-grid, threshold)
        return maxima + minima
    
    def _find_local_extrema_3d(self, grid: np.ndarray, threshold: float = 0.1, 
                             min_distance: int = 2) -> List[Tuple[int, int, int]]:
        """Find local extrema in 3D grid with minimum distance constraint"""
        # Use maximum filter to find local maxima
        local_max = maximum_filter(grid, size=min_distance*2+1) == grid
        
        # Apply threshold
        above_threshold = np.abs(grid) > threshold
        
        # Combine conditions
        extrema_mask = local_max & above_threshold
        
        # Get coordinates
        extrema_coords = np.where(extrema_mask)
        return list(zip(extrema_coords[0], extrema_coords[1], extrema_coords[2]))
    
    def _distance_to_surface(self, voxel_idx: Tuple[int, int, int], 
                           shape_grid: np.ndarray) -> float:
        """Calculate distance from voxel to cavity surface efficiently"""
        i, j, k = voxel_idx
        
        # Check immediate neighbors for boundary
        neighbors = [
            (i+di, j+dj, k+dk) 
            for di in [-1, 0, 1] for dj in [-1, 0, 1] for dk in [-1, 0, 1]
            if not (di == 0 and dj == 0 and dk == 0)
        ]
        
        min_distance = float('inf')
        for ni, nj, nk in neighbors:
            if (0 <= ni < shape_grid.shape[0] and 
                0 <= nj < shape_grid.shape[1] and 
                0 <= nk < shape_grid.shape[2]):
                if shape_grid[ni, nj, nk] < 0.5:  # Outside cavity
                    distance = np.sqrt((ni-i)**2 + (nj-j)**2 + (nk-k)**2)
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _calculate_all_metrics(self, property_grid: PropertyGrid, center: np.ndarray, 
                             grid_spacing: float) -> Dict[str, any]:
        """Calculate all metrics in a single pass for efficiency"""
        shape_grid = property_grid.shape_grid
        cavity_mask = shape_grid > 0.5
        
        # Early return if no cavity
        if not np.any(cavity_mask):
            empty_descriptors = {'depth': 0.0, 'width': 0.0, 'convexity': 0.0, 
                               'aspect_ratio': 0.0, 'sphericity': 0.0}
            return {
                'cavity_volume': 0.0,
                'surface_area': 0.0,
                'electrostatic_variance': 0.0,
                'hydrophobic_moment': 0.0,
                'field_complexity_score': 0.0,
                'shape_descriptors': empty_descriptors
            }
        
        # Calculate volume
        voxel_volume = grid_spacing ** 3
        cavity_volume = float(np.sum(cavity_mask) * voxel_volume)
        
        # Calculate surface area
        eroded = binary_erosion(cavity_mask)
        surface = cavity_mask & ~eroded
        voxel_face_area = grid_spacing ** 2
        surface_area = float(np.sum(surface) * 6 * voxel_face_area)
        
        # Electrostatic variance
        elec_values = property_grid.electrostatic_grid[cavity_mask]
        electrostatic_variance = float(np.var(elec_values))
        
        # Hydrophobic moment
        hydrophobic_moment = self._calculate_hydrophobic_moment_optimized(
            property_grid, cavity_mask, center, grid_spacing
        )
        
        # Field complexity
        field_complexity_score = self._calculate_field_complexity_optimized(
            property_grid, cavity_mask
        )
        
        # Shape descriptors
        shape_descriptors = self._calculate_shape_descriptors_optimized(
            cavity_mask, grid_spacing
        )
        
        return {
            'cavity_volume': cavity_volume,
            'surface_area': surface_area,
            'electrostatic_variance': electrostatic_variance,
            'hydrophobic_moment': hydrophobic_moment,
            'field_complexity_score': field_complexity_score,
            'shape_descriptors': shape_descriptors
        }
    
    def _calculate_hydrophobic_moment_optimized(self, property_grid: PropertyGrid, 
                                              cavity_mask: np.ndarray, center: np.ndarray,
                                              grid_spacing: float) -> float:
        """Optimized hydrophobic moment calculation"""
        hydro_values = property_grid.hydrophobicity_grid[cavity_mask]
        positive_hydro = hydro_values > 0
        
        if not np.any(positive_hydro):
            return 0.0
        
        # Get cavity indices and convert to world coordinates
        cavity_indices = np.where(cavity_mask)
        world_coords = (np.array(cavity_indices).T * grid_spacing + 
                       property_grid.grid_origin)
        
        # Filter for positive hydrophobicity
        positive_coords = world_coords[positive_hydro]
        positive_values = hydro_values[positive_hydro]
        
        # Calculate moment vector
        displacements = positive_coords - center
        weighted_displacement = np.sum(positive_values[:, np.newaxis] * displacements, axis=0)
        total_hydrophobicity = np.sum(positive_values)
        
        if total_hydrophobicity > 0:
            moment_vector = weighted_displacement / total_hydrophobicity
            return float(np.linalg.norm(moment_vector))
        
        return 0.0
    
    def _calculate_field_complexity_optimized(self, property_grid: PropertyGrid, 
                                            cavity_mask: np.ndarray) -> float:
        """Optimized field complexity calculation"""
        elec_values = property_grid.electrostatic_grid[cavity_mask]
        hydro_values = property_grid.hydrophobicity_grid[cavity_mask]
        
        elec_variance = np.var(elec_values)
        hydro_variance = np.var(hydro_values)
        
        # Gradient-based complexity
        grad_complexity = (np.mean(property_grid.gradient_magnitude[cavity_mask]) 
                          if hasattr(property_grid, 'gradient_magnitude') else 0.0)
        
        # Combine metrics with bounds checking
        complexity_score = (
            0.4 * min(1.0, elec_variance / 100.0) +
            0.4 * min(1.0, hydro_variance / 4.0) +
            0.2 * min(1.0, grad_complexity / 10.0)
        )
        
        return float(complexity_score)
    
    def _calculate_shape_descriptors_optimized(self, cavity_mask: np.ndarray, 
                                             grid_spacing: float) -> Dict[str, float]:
        """Optimized shape descriptors calculation"""
        cavity_coords = np.where(cavity_mask)
        cavity_positions = np.array(cavity_coords).T * grid_spacing
        
        if len(cavity_positions) == 0:
            return {'depth': 0.0, 'width': 0.0, 'convexity': 0.0, 
                   'aspect_ratio': 0.0, 'sphericity': 0.0}
        
        # Bounding box dimensions
        min_coords = np.min(cavity_positions, axis=0)
        max_coords = np.max(cavity_positions, axis=0)
        dimensions = max_coords - min_coords
        
        depth = float(np.max(dimensions))
        width = float(np.mean(dimensions))
        
        # Convexity calculation
        convexity = self._calculate_convexity_safe(cavity_positions)
        
        # Sphericity
        volume = len(cavity_positions) * (grid_spacing ** 3)
        sphericity = self._calculate_sphericity_optimized(cavity_mask, grid_spacing, volume)
        
        return {
            'depth': depth,
            'width': width,
            'convexity': convexity,
            'aspect_ratio': float(depth / width if width > 0 else 0.0),
            'sphericity': sphericity
        }
    
    def _calculate_convexity_safe(self, cavity_positions: np.ndarray) -> float:
        """Calculate convexity with error handling"""
        try:
            from scipy.spatial import ConvexHull
            if len(cavity_positions) > 3:
                hull = ConvexHull(cavity_positions)
                convex_volume = hull.volume
                actual_volume = len(cavity_positions)
                return float(actual_volume / convex_volume if convex_volume > 0 else 0.0)
        except Exception:
            pass
        return 0.0
    
    def _calculate_sphericity_optimized(self, cavity_mask: np.ndarray, 
                                      grid_spacing: float, volume: float) -> float:
        """Optimized sphericity calculation"""
        if volume == 0:
            return 0.0
        
        # Surface area calculation
        eroded = binary_erosion(cavity_mask)
        surface = cavity_mask & ~eroded
        surface_area = np.sum(surface) * 6 * (grid_spacing ** 2)
        
        if surface_area == 0:
            return 0.0
        
        # Sphericity formula
        sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
        return float(min(sphericity, 1.0))
    
    def analyze_pocket_druggability(self, voxel_result: VoxelAnalysisResult) -> Dict[str, float]:
        """Analyze pocket druggability based on voxel analysis"""
        
        # Volume-based score with improved scaling
        volume_score = min(1.0, voxel_result.cavity_volume / 500.0)
        
        # Complexity-based score
        complexity_score = voxel_result.field_complexity_score
        
        # Shape-based score with better weighting
        convexity = voxel_result.shape_descriptors.get('convexity', 0.0)
        sphericity = voxel_result.shape_descriptors.get('sphericity', 0.0)
        aspect_ratio = voxel_result.shape_descriptors.get('aspect_ratio', 0.0)
        
        # Penalize extreme aspect ratios
        aspect_penalty = 1.0 / (1.0 + aspect_ratio / 3.0) if aspect_ratio > 0 else 0.0
        shape_score = 0.4 * convexity + 0.4 * sphericity + 0.2 * aspect_penalty
        
        # Hotspot-based score with improved scaling
        hotspot_score = min(1.0, len(voxel_result.hotspot_voxels) / 10.0)
        
        # Combined druggability score with refined weights
        druggability = (
            0.35 * volume_score +
            0.25 * complexity_score +
            0.25 * shape_score +
            0.15 * hotspot_score
        )
        
        return {
            'overall_druggability': float(druggability),
            'volume_score': float(volume_score),
            'complexity_score': float(complexity_score),
            'shape_score': float(shape_score),
            'hotspot_score': float(hotspot_score),
            'aspect_penalty': float(aspect_penalty)
        }
    
    def export_grid_data(self, property_grid: PropertyGrid, filepath: str, 
                        format: str = 'dx') -> None:
        """Export grid data to file with improved error handling"""
        try:
            if format.lower() == 'dx':
                self._export_dx_format(property_grid, filepath)
            elif format.lower() == 'cube':
                self._export_cube_format(property_grid, filepath)
            else:
                raise ValueError(f"Unsupported export format: {format}. Supported: 'dx', 'cube'")
        except Exception as e:
            raise RuntimeError(f"Failed to export grid data: {str(e)}")
    
    def _export_dx_format(self, property_grid: PropertyGrid, filepath: str) -> None:
        """Export grid in DX format for visualization with improved formatting"""
        try:
            with open(filepath, 'w') as f:
                # Write header
                f.write("# Data from LigandForge voxel analysis\n")
                dims = property_grid.grid_dimensions
                f.write(f"object 1 class gridpositions counts {dims[0]} {dims[1]} {dims[2]}\n")
                
                origin = property_grid.grid_origin
                f.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
                
                spacing = property_grid.grid_spacing
                f.write(f"delta {spacing:.6f} 0.000000 0.000000\n")
                f.write(f"delta 0.000000 {spacing:.6f} 0.000000\n")
                f.write(f"delta 0.000000 0.000000 {spacing:.6f}\n")
                f.write(f"object 2 class gridconnections counts {dims[0]} {dims[1]} {dims[2]}\n")
                f.write(f"object 3 class array type double rank 0 items "
                       f"{np.prod(dims)} data follows\n")
                
                # Write data efficiently
                data = property_grid.electrostatic_grid.flatten()
                for i in range(0, len(data), 3):
                    batch = data[i:i+3]
                    f.write(" ".join(f"{value:.6f}" for value in batch))
                    f.write("\n")
                
                f.write("object \"electrostatic potential\" class field\n")
        except IOError as e:
            raise RuntimeError(f"Failed to write DX file: {str(e)}")
    
    def _export_cube_format(self, property_grid: PropertyGrid, filepath: str) -> None:
        """Export grid in Gaussian cube format with improved formatting"""
        try:
            with open(filepath, 'w') as f:
                # Write header
                f.write("LigandForge electrostatic potential\n")
                f.write("Generated by voxel analysis\n")
                
                # Number of atoms (0 for grid-only file) and origin
                origin = property_grid.grid_origin
                f.write(f"    0 {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
                
                # Grid dimensions and spacing
                dims = property_grid.grid_dimensions
                spacing = property_grid.grid_spacing
                for i in range(3):
                    spacing_vec = [0.0, 0.0, 0.0]
                    spacing_vec[i] = spacing
                    f.write(f"{dims[i]:5d} {spacing_vec[0]:.6f} {spacing_vec[1]:.6f} {spacing_vec[2]:.6f}\n")
                
                # Write data efficiently
                data = property_grid.electrostatic_grid
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        for k in range(0, data.shape[2], 6):  # 6 values per line
                            batch = data[i, j, k:k+6]
                            f.write(" ".join(f"{value:13.5e}" for value in batch))
                            f.write("\n")
        except IOError as e:
            raise RuntimeError(f"Failed to write cube file: {str(e)}")


# Additional utility functions for performance monitoring and validation

def profile_voxel_analysis(analyzer: VoxelBasedAnalyzer, pdb_text: str, 
                          center: np.ndarray, radius: float = 10.0) -> Dict[str, float]:
    """Profile the performance of voxel analysis"""
    import time
    
    timings = {}
    
    start_time = time.time()
    result = analyzer.analyze_binding_site_voxels(pdb_text, center, radius)
    timings['total_time'] = time.time() - start_time
    
    timings['grid_points'] = np.prod(result.property_grid.grid_dimensions)
    timings['hotspots_found'] = len(result.hotspot_voxels)
    timings['cavity_volume'] = result.cavity_volume
    
    return timings


def validate_voxel_results(result: VoxelAnalysisResult) -> Dict[str, bool]:
    """Validate the results of voxel analysis"""
    validation = {}
    
    # Check for reasonable values
    validation['positive_volume'] = result.cavity_volume > 0
    validation['positive_surface_area'] = result.surface_area > 0
    validation['finite_electrostatic_variance'] = np.isfinite(result.electrostatic_variance)
    validation['finite_hydrophobic_moment'] = np.isfinite(result.hydrophobic_moment)
    validation['valid_complexity_score'] = 0 <= result.field_complexity_score <= 1
    
    # Check grid consistency
    grid = result.property_grid
    validation['consistent_grid_dimensions'] = all(
        grid.electrostatic_grid.shape == grid.hydrophobicity_grid.shape == 
        grid.steric_grid.shape == grid.accessibility_grid.shape == 
        grid.shape_grid.shape
    )
    
    # Check for NaN values
    validation['no_nan_electrostatic'] = not np.any(np.isnan(grid.electrostatic_grid))
    validation['no_nan_hydrophobicity'] = not np.any(np.isnan(grid.hydrophobicity_grid))
    
    # Check hotspots are within reasonable bounds
    if result.hotspot_voxels:
        positions = np.array([v.position for v in result.hotspot_voxels])
        center_estimate = np.mean(positions, axis=0)
        max_distance = np.max(np.linalg.norm(positions - center_estimate, axis=1))
        validation['hotspots_reasonable_distance'] = max_distance < 50.0  # Reasonable cutoff
    else:
        validation['hotspots_reasonable_distance'] = True
    
    return validation


def optimize_grid_resolution(analyzer: VoxelBasedAnalyzer, pdb_text: str, 
                            center: np.ndarray, radius: float = 10.0,
                            target_time: float = 60.0) -> float:
    """Find optimal grid resolution for target computation time"""
    import time
    
    # Test different resolutions
    test_resolutions = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    timings = []
    
    original_spacing = analyzer.config.get_grid_spacing_for_resolution()
    
    for resolution in test_resolutions:
        # Temporarily modify config
        analyzer.config.grid_spacing = resolution
        
        start_time = time.time()
        try:
            result = analyzer.analyze_binding_site_voxels(pdb_text, center, radius)
            elapsed = time.time() - start_time
            timings.append((resolution, elapsed))
            
            if elapsed > target_time * 2:  # Stop if getting too slow
                break
                
        except Exception:
            continue
    
    # Restore original spacing
    analyzer.config.grid_spacing = original_spacing
    
    # Find resolution closest to target time
    if timings:
        best_resolution = min(timings, key=lambda x: abs(x[1] - target_time))[0]
        return best_resolution
    
    return 1.0  # Default fallback