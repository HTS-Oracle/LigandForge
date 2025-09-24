
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from enum import Enum


# ============================================================================
# Enumerations
# ============================================================================

class InteractionType(Enum):
    """Types of molecular interactions"""
    HYDROGEN_BOND_DONOR = "hbd"
    HYDROGEN_BOND_ACCEPTOR = "hba"
    HYDROPHOBIC = "hydrophobic"
    AROMATIC = "aromatic"
    ELECTROSTATIC = "electrostatic"
    VAN_DER_WAALS = "vdw"
    HALOGEN_BOND = "halogen"
    CATION_PI = "cation_pi"
    METAL_COORDINATION = "metal"


class OptimizationMethod(Enum):
    """Available optimization methods"""
    REINFORCEMENT_LEARNING = "rl"
    GENETIC_ALGORITHM = "ga"
    HYBRID = "hybrid"
    RANDOM_SEARCH = "random"
    SIMULATED_ANNEALING = "sa"


class ScaffoldType(Enum):
    """Types of molecular scaffolds"""
    CORE = "core"
    LINKER = "linker"
    SUBSTITUENT = "substituent"
    BIOISOSTERE = "bioisostere"
    FRAGMENT = "fragment"


# ============================================================================
# Basic Molecular Data Structures
# ============================================================================

@dataclass
class AtomicEnvironment:
    """Detailed atomic environment information for field calculations"""
    atom_id: int
    position: np.ndarray
    element: str
    residue_name: str
    residue_id: int
    chain: str
    partial_charge: float
    is_hbd: bool = False
    is_hba: bool = False
    is_hydrophobic: bool = False
    is_aromatic: bool = False
    van_der_waals_radius: float = 1.7
    accessibility: float = 0.0
    hybridization: str = "sp3"
    formal_charge: int = 0
    b_factor: float = 0.0
    occupancy: float = 1.0
    
    def __post_init__(self):
        """Validate atomic environment data"""
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("Position must be a 3D numpy array")
        if self.van_der_waals_radius <= 0:
            raise ValueError("Van der Waals radius must be positive")


@dataclass
class PocketVoxel:
    """Individual voxel in pocket grid with comprehensive properties"""
    position: np.ndarray
    is_occupied: bool = False
    electrostatic_potential: float = 0.0
    hydrophobicity: float = 0.0
    accessibility: float = 0.0
    nearest_residue: Optional[str] = None
    distance_to_surface: float = 0.0
    steric_clash: bool = False
    field_gradient: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_density: float = 0.0
    shape_index: float = 0.0
    curvature: float = 0.0
    
    def __post_init__(self):
        """Validate voxel data"""
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("Position must be a 3D numpy array")
        if not isinstance(self.field_gradient, np.ndarray) or self.field_gradient.shape != (3,):
            self.field_gradient = np.zeros(3)


@dataclass
class PropertyGrid:
    """3D property grid for comprehensive pocket analysis"""
    grid_origin: np.ndarray
    grid_spacing: float
    grid_dimensions: Tuple[int, int, int]
    electrostatic_grid: np.ndarray
    hydrophobicity_grid: np.ndarray
    steric_grid: np.ndarray
    accessibility_grid: np.ndarray
    shape_grid: np.ndarray
    gradient_magnitude: np.ndarray = field(default_factory=lambda: np.array([]))
    interaction_energy_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_factor_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    conservation_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Validate grid data consistency"""
        if not isinstance(self.grid_origin, np.ndarray) or self.grid_origin.shape != (3,):
            raise ValueError("Grid origin must be a 3D numpy array")
        if self.grid_spacing <= 0:
            raise ValueError("Grid spacing must be positive")
        
        # Check that all required grids have the same dimensions
        required_grids = [
            self.electrostatic_grid, self.hydrophobicity_grid,
            self.steric_grid, self.accessibility_grid, self.shape_grid
        ]
        for i, grid in enumerate(required_grids):
            if grid.shape != self.grid_dimensions:
                raise ValueError(f"Grid {i} dimensions don't match grid_dimensions")
    
    def get_world_coordinates(self, i: int, j: int, k: int) -> np.ndarray:
        """Convert grid indices to world coordinates"""
        return self.grid_origin + np.array([i, j, k]) * self.grid_spacing
    
    def get_grid_indices(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices"""
        indices = np.round((world_pos - self.grid_origin) / self.grid_spacing).astype(int)
        return tuple(indices)


# ============================================================================
# Interaction and Binding Site Data Structures
# ============================================================================

@dataclass
class InteractionHotspot:
    """Protein interaction site with field-based properties"""
    position: np.ndarray
    interaction_type: str
    strength: float
    residue_name: str
    residue_id: int = 0
    chain: str = "A"
    flexibility: float = 0.5
    conservation: float = 1.0
    field_gradient: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accessibility_score: float = 1.0
    complementarity_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pharmacophore_features: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate hotspot data"""
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("Position must be a 3D numpy array")
        if not 0 <= self.strength <= 1:
            raise ValueError("Strength must be between 0 and 1")
        if not 0 <= self.flexibility <= 1:
            raise ValueError("Flexibility must be between 0 and 1")
        if not 0 <= self.conservation <= 1:
            raise ValueError("Conservation must be between 0 and 1")


@dataclass
class WaterSite:
    """Crystallographic water analysis with voxel context"""
    position: np.ndarray
    b_factor: float
    hydrogen_bonds: List[str]
    replaceability_score: float
    conservation_score: float
    local_field_strength: float = 0.0
    displacement_energy: float = 0.0
    coordination_number: int = 0
    residence_time: float = 0.0
    entropy_contribution: float = 0.0
    
    def __post_init__(self):
        """Validate water site data"""
        if not isinstance(self.position, np.ndarray) or self.position.shape != (3,):
            raise ValueError("Position must be a 3D numpy array")
        if not 0 <= self.replaceability_score <= 1:
            raise ValueError("Replaceability score must be between 0 and 1")
        if not 0 <= self.conservation_score <= 1:
            raise ValueError("Conservation score must be between 0 and 1")


@dataclass
class PocketAnalysis:
    """Comprehensive binding site analysis with voxel-based fields"""
    center: np.ndarray
    volume: float
    surface_area: float
    druggability_score: float
    hotspots: List[InteractionHotspot]
    water_sites: List[WaterSite]
    flexibility_map: Dict[str, float]
    electrostatic_potential: np.ndarray
    hydrophobic_patches: List[np.ndarray]
    voxel_analysis: Optional['VoxelAnalysisResult'] = None
    field_complexity: float = 0.0
    interaction_complementarity: float = 0.0
    cavity_shape_descriptor: Dict[str, float] = field(default_factory=dict)
    pharmacophore_points: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate pocket analysis data"""
        if not isinstance(self.center, np.ndarray) or self.center.shape != (3,):
            raise ValueError("Center must be a 3D numpy array")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")
        if self.surface_area < 0:
            raise ValueError("Surface area must be non-negative")
        if not 0 <= self.druggability_score <= 1:
            raise ValueError("Druggability score must be between 0 and 1")
    
    def get_interaction_summary(self) -> Dict[str, int]:
        """Get summary of interaction types"""
        summary = {}
        for hotspot in self.hotspots:
            interaction_type = hotspot.interaction_type
            summary[interaction_type] = summary.get(interaction_type, 0) + 1
        return summary


# ============================================================================
# Voxel Analysis Results
# ============================================================================

@dataclass
class VoxelAnalysisResult:
    """Results from voxel-based binding site analysis"""
    property_grid: PropertyGrid
    hotspot_voxels: List[PocketVoxel]
    cavity_volume: float
    surface_area: float
    electrostatic_variance: float
    hydrophobic_moment: float
    field_complexity_score: float
    shape_descriptors: Dict[str, float] = field(default_factory=dict)
    pocket_depth: float = 0.0
    pocket_width: float = 0.0
    convexity: float = 0.0
    
    def __post_init__(self):
        """Validate voxel analysis results"""
        if self.cavity_volume < 0:
            raise ValueError("Cavity volume must be non-negative")
        if self.surface_area < 0:
            raise ValueError("Surface area must be non-negative")
    
    def get_hotspot_summary(self) -> Dict[str, Any]:
        """Get summary statistics of hotspot voxels"""
        if not self.hotspot_voxels:
            return {}
        
        electrostatic_values = [v.electrostatic_potential for v in self.hotspot_voxels]
        hydrophobicity_values = [v.hydrophobicity for v in self.hotspot_voxels]
        
        return {
            'count': len(self.hotspot_voxels),
            'electrostatic_mean': np.mean(electrostatic_values),
            'electrostatic_std': np.std(electrostatic_values),
            'hydrophobicity_mean': np.mean(hydrophobicity_values),
            'hydrophobicity_std': np.std(hydrophobicity_values),
            'accessibility_mean': np.mean([v.accessibility for v in self.hotspot_voxels])
        }


# ============================================================================
# Molecular Generation and Scoring
# ============================================================================

@dataclass
class MolecularScore:
    """Comprehensive scoring results for a molecule"""
    total_score: float
    pharmacophore_score: float
    synthetic_score: float
    drug_likeness_score: float
    novelty_score: float
    selectivity_score: float
    water_displacement_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    property_values: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate scoring data"""
        scores = [
            self.total_score, self.pharmacophore_score, self.synthetic_score,
            self.drug_likeness_score, self.novelty_score, self.selectivity_score,
            self.water_displacement_score
        ]
        for score in scores:
            if not 0 <= score <= 1:
                raise ValueError("All scores must be between 0 and 1")


@dataclass
class GenerationResult:
    """Results from molecular generation process"""
    molecules: List[Any]  # List of RDKit Mol objects
    scores: List[MolecularScore]
    generation_statistics: Dict[str, Any]
    diversity_statistics: Dict[str, Any]
    optimization_history: List[Dict] = field(default_factory=list)
    successful_generations: int = 0
    failed_generations: int = 0
    total_time: float = 0.0
    
    def __post_init__(self):
        """Validate generation results"""
        if len(self.molecules) != len(self.scores):
            raise ValueError("Number of molecules must match number of scores")
    
    def get_top_molecules(self, n: int = 10) -> List[Tuple[Any, MolecularScore]]:
        """Get top N molecules by total score"""
        paired = list(zip(self.molecules, self.scores))
        sorted_pairs = sorted(paired, key=lambda x: x[1].total_score, reverse=True)
        return sorted_pairs[:n]


# ============================================================================
# Optimization State and History
# ============================================================================

@dataclass
class OptimizationState:
    """Current state of optimization process"""
    iteration: int
    current_population: List[Any]
    current_scores: List[float]
    best_molecule: Optional[Any]
    best_score: float
    population_diversity: float
    convergence_metric: float
    stagnation_counter: int = 0
    
    def is_converged(self, threshold: float = 0.001) -> bool:
        """Check if optimization has converged"""
        return self.convergence_metric < threshold
    
    def is_stagnated(self, max_stagnation: int = 10) -> bool:
        """Check if optimization is stagnated"""
        return self.stagnation_counter >= max_stagnation


@dataclass
class OptimizationHistory:
    """Complete history of optimization process"""
    method: str
    states: List[OptimizationState]
    parameter_history: List[Dict] = field(default_factory=list)
    performance_metrics: List[Dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def get_convergence_plot_data(self) -> Tuple[List[int], List[float]]:
        """Get data for plotting convergence"""
        iterations = [state.iteration for state in self.states]
        best_scores = [state.best_score for state in self.states]
        return iterations, best_scores
    
    def get_diversity_plot_data(self) -> Tuple[List[int], List[float]]:
        """Get data for plotting population diversity"""
        iterations = [state.iteration for state in self.states]
        diversities = [state.population_diversity for state in self.states]
        return iterations, diversities


# ============================================================================
# Pipeline Configuration and Results
# ============================================================================

@dataclass
class PipelineConfiguration:
    """Complete configuration for LigandForge pipeline"""
    config: 'LigandForgeConfig'
    target_interactions: List[str]
    optimization_method: str
    pdb_file: Optional[str] = None
    binding_site_center: Optional[np.ndarray] = None
    binding_site_radius: float = 10.0
    reference_molecules: List[Any] = field(default_factory=list)
    custom_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResults:
    """Complete results from LigandForge pipeline"""
    configuration: PipelineConfiguration
    pocket_analysis: PocketAnalysis
    voxel_analysis: Optional[VoxelAnalysisResult]
    generation_results: GenerationResult
    optimization_history: OptimizationHistory
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save_summary(self, filepath: str):
        """Save a summary of results to file"""
        import json
        
        summary = {
            'total_molecules': len(self.generation_results.molecules),
            'best_score': max(score.total_score for score in self.generation_results.scores),
            'average_score': np.mean([score.total_score for score in self.generation_results.scores]),
            'pocket_druggability': self.pocket_analysis.druggability_score,
            'optimization_method': self.configuration.optimization_method,
            'execution_time': self.generation_results.total_time,
            'successful_generations': self.generation_results.successful_generations,
            'failed_generations': self.generation_results.failed_generations
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# Utility Functions
# ============================================================================

def create_empty_property_grid(origin: np.ndarray, spacing: float, dimensions: Tuple[int, int, int]) -> PropertyGrid:
    """Create an empty property grid with specified parameters"""
    return PropertyGrid(
        grid_origin=origin,
        grid_spacing=spacing,
        grid_dimensions=dimensions,
        electrostatic_grid=np.zeros(dimensions),
        hydrophobicity_grid=np.zeros(dimensions),
        steric_grid=np.zeros(dimensions),
        accessibility_grid=np.ones(dimensions),
        shape_grid=np.zeros(dimensions)
    )


def validate_position_array(position: Union[List, Tuple, np.ndarray], name: str = "position") -> np.ndarray:
    """Validate and convert position to numpy array"""
    if not isinstance(position, np.ndarray):
        position = np.array(position)
    
    if position.shape != (3,):
        raise ValueError(f"{name} must be a 3D array")
    
    if not np.all(np.isfinite(position)):
        raise ValueError(f"{name} must contain finite values")
    
    return position


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two positions"""
    return float(np.linalg.norm(pos1 - pos2))


def calculate_center_of_mass(positions: List[np.ndarray], masses: Optional[List[float]] = None) -> np.ndarray:
    """Calculate center of mass for a collection of positions"""
    positions_array = np.array(positions)
    
    if masses is None:
        return np.mean(positions_array, axis=0)
    else:
        masses_array = np.array(masses)
        total_mass = np.sum(masses_array)
        if total_mass == 0:
            raise ValueError("Total mass cannot be zero")
        return np.sum(positions_array * masses_array[:, np.newaxis], axis=0) / total_mass
