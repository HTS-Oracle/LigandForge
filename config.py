"""
Configuration Management Module
Centralized configuration for LigandForge pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class LigandForgeConfig:
    """Centralized configuration management for LigandForge pipeline"""
    
    # ============================================================================
    # Fragment Generation Parameters
    # ============================================================================
    max_heavy_atoms: int = 35
    min_heavy_atoms: int = 15
    max_rings: int = 6
    max_rotatable_bonds: int = 8
    max_molecular_weight: float = 500.0
    min_molecular_weight: float = 150.0
    
    # ============================================================================
    # Voxel Grid Parameters
    # ============================================================================
    voxel_spacing: float = 0.5  # Angstroms
    grid_padding: float = 3.0   # Angstroms around binding site
    probe_radius: float = 1.4   # Angstroms, water probe radius
    grid_resolution: str = "medium"  # "low", "medium", "high"
    
    # ============================================================================
    # Field Calculation Parameters
    # ============================================================================
    dielectric_constant: float = 4.0
    coulomb_constant: float = 332.0  # kcal/(mol*e*A)
    hydrophobicity_decay: float = 1.5
    field_smoothing_sigma: float = 1.0
    electrostatic_cutoff: float = 12.0  # Angstroms
    
    # ============================================================================
    # Diversity Parameters
    # ============================================================================
    diversity_threshold: float = 0.7  # Tanimoto similarity threshold
    max_similar_molecules: int = 3
    cluster_count: int = 10
    scaffold_diversity_weight: float = 0.6
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    
    # ============================================================================
    # Scoring Weights (must sum to 1.0)
    # ============================================================================
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'pharmacophore': 0.25,
        'synthetic': 0.20,
        'drug_likeness': 0.20,
        'novelty': 0.15,
        'selectivity': 0.10,
        'water_displacement': 0.10
    })
    
    # ============================================================================
    # Reinforcement Learning Parameters
    # ============================================================================
    rl_algorithm: str = 'PPO'  # 'PPO', 'DQN', 'A3C'
    learning_rate: float = 0.001
    exploration_factor: float = 0.15
    policy_update_frequency: int = 10
    rl_batch_size: int = 32
    rl_memory_size: int = 10000
    
    # ============================================================================
    # Genetic Algorithm Parameters
    # ============================================================================
    ga_population_size: int = 150
    ga_generations: int = 30
    ga_crossover_rate: float = 0.7
    ga_mutation_rate: float = 0.3
    ga_elitism: int = 10
    ga_tournament_size: int = 3
    ga_diversity_pressure: float = 0.1
    
    # ============================================================================
    # Physical Constants and Radii
    # ============================================================================
    vdw_radii: Dict[str, float] = field(default_factory=lambda: {
        "H": 1.20, "He": 1.40,
        "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
        "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
        "K": 2.75, "Ca": 2.31, "Sc": 2.11, "Ti": 2.00, "V": 1.95, "Cr": 1.90, "Mn": 1.61, "Fe": 2.00,
        "Co": 1.92, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39, "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90,
        "Br": 1.85, "Kr": 2.02, "Rb": 3.03, "Sr": 2.49, "Y": 2.32, "Zr": 2.23, "Nb": 2.18, "Mo": 2.17,
        "Tc": 2.16, "Ru": 2.13, "Rh": 2.10, "Pd": 2.10, "Ag": 1.72, "Cd": 1.58, "In": 1.93, "Sn": 2.17,
        "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16
    })
    
    # ============================================================================
    # Hydrophobicity Scales (Eisenberg scale)
    # ============================================================================
    residue_hydrophobicity: Dict[str, float] = field(default_factory=lambda: {
        'ALA': 0.62, 'ARG': -2.53, 'ASN': -0.78, 'ASP': -0.90, 'CYS': 0.29,
        'GLN': -0.85, 'GLU': -0.74, 'GLY': 0.48, 'HIS': -0.40, 'ILE': 1.38,
        'LEU': 1.06, 'LYS': -1.50, 'MET': 0.64, 'PHE': 1.19, 'PRO': 0.12,
        'SER': -0.18, 'THR': -0.05, 'TRP': 0.81, 'TYR': 0.26, 'VAL': 1.08
    })
    
    # ============================================================================
    # Chemical Property Thresholds
    # ============================================================================
    lipinski_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'mw': (0.0, 500.0),
        'logp': (-3.0, 5.0),
        'hbd': (0, 5),
        'hba': (0, 10),
        'tpsa': (0.0, 140.0),
        'rotatable_bonds': (0, 10)
    })
    
    # ============================================================================
    # Synthetic Feasibility Parameters
    # ============================================================================
    synthetic_complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        'ring_complexity': 0.3,
        'stereocenter_penalty': 0.2,
        'heteroatom_penalty': 0.1,
        'bond_complexity': 0.2,
        'functional_group_penalty': 0.2
    })
    
    # ============================================================================
    # Output and Logging Parameters
    # ============================================================================
    output_format: str = "sdf"  # "sdf", "mol2", "pdb", "smiles"
    max_output_molecules: int = 100
    save_intermediate_results: bool = True
    verbose_logging: bool = True
    random_seed: Optional[int] = 42
    
    # ============================================================================
    # Performance Parameters
    # ============================================================================
    max_parallel_workers: int = 4
    memory_limit_gb: float = 8.0
    timeout_seconds: int = 3600  # 1 hour
    batch_processing_size: int = 50
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._normalize_weights()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate molecular weight range
        if self.max_molecular_weight <= self.min_molecular_weight:
            raise ValueError("max_molecular_weight must be greater than min_molecular_weight")
        
        # Validate heavy atom range
        if self.max_heavy_atoms <= self.min_heavy_atoms:
            raise ValueError("max_heavy_atoms must be greater than min_heavy_atoms")
        
        # Validate diversity threshold
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be between 0.0 and 1.0")
        
        # Validate GA rates
        if not 0.0 <= self.ga_crossover_rate <= 1.0:
            raise ValueError("ga_crossover_rate must be between 0.0 and 1.0")
        if not 0.0 <= self.ga_mutation_rate <= 1.0:
            raise ValueError("ga_mutation_rate must be between 0.0 and 1.0")
        
        # Validate exploration factor
        if not 0.0 <= self.exploration_factor <= 1.0:
            raise ValueError("exploration_factor must be between 0.0 and 1.0")
        
        # Validate voxel spacing
        if self.voxel_spacing <= 0:
            raise ValueError("voxel_spacing must be positive")
        
        # Validate grid resolution
        if self.grid_resolution not in ["low", "medium", "high"]:
            raise ValueError("grid_resolution must be 'low', 'medium', or 'high'")
    
    def _normalize_weights(self):
        """Normalize scoring weights to sum to 1.0"""
        total_weight = sum(self.reward_weights.values())
        if total_weight > 0:
            self.reward_weights = {k: v / total_weight for k, v in self.reward_weights.items()}
    
    def get_grid_spacing_for_resolution(self) -> float:
        """Get voxel spacing based on grid resolution setting"""
        resolution_map = {
            "low": self.voxel_spacing * 2.0,
            "medium": self.voxel_spacing,
            "high": self.voxel_spacing * 0.5
        }
        return resolution_map[self.grid_resolution]
    
    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes"""
        return int(self.memory_limit_gb * 1024 * 1024 * 1024)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights and normalize"""
        self.reward_weights.update(new_weights)
        self._normalize_weights()
    
    def get_optimization_config(self, method: str) -> Dict:
        """Get configuration for specific optimization method"""
        if method.lower() == "ga":
            return {
                'population_size': self.ga_population_size,
                'generations': self.ga_generations,
                'crossover_rate': self.ga_crossover_rate,
                'mutation_rate': self.ga_mutation_rate,
                'elitism': self.ga_elitism,
                'tournament_size': self.ga_tournament_size,
                'diversity_pressure': self.ga_diversity_pressure
            }
        elif method.lower() == "rl":
            return {
                'algorithm': self.rl_algorithm,
                'learning_rate': self.learning_rate,
                'exploration_factor': self.exploration_factor,
                'policy_update_frequency': self.policy_update_frequency,
                'batch_size': self.rl_batch_size,
                'memory_size': self.rl_memory_size
            }
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def copy(self):
        """Create a deep copy of the configuration"""
        import copy
        return copy.deepcopy(self)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        import dataclasses
        
        config_dict = dataclasses.asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"LigandForgeConfig(diversity_threshold={self.diversity_threshold}, " \
               f"max_heavy_atoms={self.max_heavy_atoms}, " \
               f"voxel_spacing={self.voxel_spacing})"
    
    def __repr__(self) -> str:
        """Detailed representation of configuration"""
        return f"LigandForgeConfig({self.__dict__})"


# ============================================================================
# Configuration Presets
# ============================================================================

class ConfigPresets:
    """Predefined configuration presets for different use cases"""
    
    @staticmethod
    def fast_screening() -> LigandForgeConfig:
        """Configuration optimized for fast screening of many molecules"""
        config = LigandForgeConfig()
        config.voxel_spacing = 1.0
        config.grid_resolution = "low"
        config.max_heavy_atoms = 25
        config.ga_generations = 15
        config.ga_population_size = 100
        config.diversity_threshold = 0.6
        return config
    
    @staticmethod
    def high_quality() -> LigandForgeConfig:
        """Configuration optimized for high-quality results"""
        config = LigandForgeConfig()
        config.voxel_spacing = 0.3
        config.grid_resolution = "high"
        config.max_heavy_atoms = 45
        config.ga_generations = 50
        config.ga_population_size = 200
        config.diversity_threshold = 0.8
        config.field_smoothing_sigma = 0.5
        return config
    
    @staticmethod
    def fragment_based() -> LigandForgeConfig:
        """Configuration optimized for fragment-based drug design"""
        config = LigandForgeConfig()
        config.max_heavy_atoms = 20
        config.min_heavy_atoms = 8
        config.max_rings = 3
        config.reward_weights = {
            'pharmacophore': 0.35,
            'synthetic': 0.25,
            'drug_likeness': 0.15,
            'novelty': 0.15,
            'selectivity': 0.05,
            'water_displacement': 0.05
        }
        return config
    
    @staticmethod
    def lead_optimization() -> LigandForgeConfig:
        """Configuration optimized for lead optimization"""
        config = LigandForgeConfig()
        config.diversity_threshold = 0.5
        config.max_similar_molecules = 5
        config.reward_weights = {
            'pharmacophore': 0.20,
            'synthetic': 0.30,
            'drug_likeness': 0.25,
            'novelty': 0.10,
            'selectivity': 0.10,
            'water_displacement': 0.05
        }
        return config
    
    @staticmethod
    def kinase_focused() -> LigandForgeConfig:
        """Configuration optimized for kinase inhibitors"""
        config = LigandForgeConfig()
        config.max_heavy_atoms = 40
        config.max_rings = 5
        config.lipinski_thresholds = {
            'mw': (200.0, 600.0),
            'logp': (1.0, 6.0),
            'hbd': (0, 3),
            'hba': (2, 8),
            'tpsa': (40.0, 120.0),
            'rotatable_bonds': (0, 8)
        }
        return config
    
    @staticmethod
    def gpcr_focused() -> LigandForgeConfig:
        """Configuration optimized for GPCR ligands"""
        config = LigandForgeConfig()
        config.max_heavy_atoms = 35
        config.max_rings = 4
        config.lipinski_thresholds = {
            'mw': (150.0, 450.0),
            'logp': (0.0, 5.0),
            'hbd': (0, 4),
            'hba': (1, 8),
            'tpsa': (20.0, 100.0),
            'rotatable_bonds': (0, 10)
        }
        return config