# LigandForge 2.0 - Modular Pipeline Structure

## Core Module Organization

### 1. **config.py** - Configuration Management
```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class LigandForgeConfig:
    """Centralized configuration management"""
    # Fragment generation parameters
    max_heavy_atoms: int = 35
    min_heavy_atoms: int = 15
    max_rings: int = 6
    max_rotatable_bonds: int = 8
    
    # Voxel grid parameters
    voxel_spacing: float = 0.5
    grid_padding: float = 3.0
    probe_radius: float = 1.4
    
    # Field calculation parameters
    dielectric_constant: float = 4.0
    coulomb_constant: float = 332.0
    hydrophobicity_decay: float = 1.5
    
    # Diversity parameters
    diversity_threshold: float = 0.7
    max_similar_molecules: int = 3
    cluster_count: int = 10
    
    # Scoring weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'pharmacophore': 0.25,
        'synthetic': 0.20,
        'drug_likeness': 0.20,
        'novelty': 0.15,
        'selectivity': 0.10,
        'water_displacement': 0.10
    })
    
    # Physical constraints and scales
    vdw_radii: Dict[str, float] = field(default_factory=lambda: {
        "H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47,
        "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98
    })
```

### 2. **data_structures.py** - Core Data Classes
```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

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

@dataclass
class InteractionHotspot:
    """Protein interaction site with field-based properties"""
    position: np.ndarray
    interaction_type: str
    strength: float
    residue_name: str
    flexibility: float = 0.5
    conservation: float = 1.0
    field_gradient: np.ndarray = field(default_factory=lambda: np.zeros(3))

# ... other data structures
```

### 3. **pdb_parser.py** - Structure Parsing
```python
from typing import Dict, List
import numpy as np

class PDBParser:
    """PDB structure parsing and validation"""
    
    def parse_pdb_structure(self, pdb_text: str) -> Dict:
        """Parse PDB structure efficiently"""
        # Implementation here
        pass
    
    def extract_binding_site_atoms(self, structure: Dict, center: np.ndarray, radius: float) -> List:
        """Extract atoms within binding site radius"""
        # Implementation here
        pass
    
    def validate_structure(self, structure: Dict) -> bool:
        """Validate parsed PDB structure"""
        # Implementation here
        pass
```

### 4. **voxel_analyzer.py** - Voxel-Based Analysis
```python
from .config import LigandForgeConfig
from .data_structures import VoxelAnalysisResult, PropertyGrid

class VoxelBasedAnalyzer:
    """Comprehensive voxel-based analysis of binding sites"""
    
    def __init__(self, config: LigandForgeConfig):
        self.config = config
    
    def analyze_binding_site_voxels(self, pdb_text: str, center: np.ndarray, radius: float = 10.0) -> VoxelAnalysisResult:
        """Perform comprehensive voxel-based binding site analysis"""
        # Implementation here
        pass
    
    def _compute_electrostatic_field_vectorized(self, grid_coords: np.ndarray, atomic_environments: List) -> np.ndarray:
        """Compute electrostatic potential field"""
        # Implementation here
        pass
    
    def _compute_hydrophobicity_field_vectorized(self, grid_coords: np.ndarray, atomic_environments: List) -> np.ndarray:
        """Compute hydrophobicity field"""
        # Implementation here
        pass
```

### 5. **pocket_analyzer.py** - Pocket Analysis
```python
from .config import LigandForgeConfig
from .data_structures import PocketAnalysis

class LigandForgeEnhancedAnalyzer:
    """Enhanced pocket analyzer with comprehensive binding site analysis"""
    
    def __init__(self, config: LigandForgeConfig):
        self.config = config
    
    def analyze_binding_site_comprehensive(self, pdb_text: str, center: np.ndarray, radius: float = 10.0) -> Dict:
        """Comprehensive binding site analysis"""
        # Implementation here
        pass
    
    def _calculate_druggability_score(self, hotspots: List, water_sites: List, volume: float, surface_area: float) -> float:
        """Calculate enhanced druggability score"""
        # Implementation here
        pass
```

### 6. **molecular_assembly.py** - Fragment Assembly
```python
from rdkit import Chem
from .config import LigandForgeConfig
from fragment_library7 import FragmentLibrary

class StructureGuidedAssembly:
    """Structure-guided molecular assembly"""
    
    def __init__(self, fragment_library: FragmentLibrary, pocket_analysis, config: LigandForgeConfig):
        self.fragment_lib = fragment_library
        self.pocket = pocket_analysis
        self.config = config
        self.reaction_templates = self._load_reaction_templates()
    
    def generate_structure_guided(self, target_interactions: List[str], n_molecules: int = 100) -> List[Chem.Mol]:
        """Generate molecules targeting specific interactions"""
        # Implementation here
        pass
    
    def _hotspot_guided_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Assemble molecule to target specific hotspots"""
        # Implementation here
        pass
```

### 7. **diversity_manager.py** - Diversity and Deduplication
```python
from typing import List, Dict
from rdkit import Chem
from collections import defaultdict

class EnhancedDiversityManager:
    """Manage molecular diversity and prevent duplicate generation"""
    
    def __init__(self, config):
        self.config = config
        self.generated_fingerprints = []
        self.molecular_scaffolds = set()
        self.generated_smiles = set()
        self.generated_inchi_keys = set()
        self.generated_hashes = set()
        self.scaffold_counts = defaultdict(int)
    
    def is_duplicate(self, mol: Chem.Mol) -> bool:
        """Comprehensive duplicate checking"""
        # Implementation here
        pass
    
    def is_diverse(self, mol: Chem.Mol, skip_duplicate_check: bool = False) -> bool:
        """Check if molecule is sufficiently diverse"""
        # Implementation here
        pass
```

### 8. **scoring.py** - Multi-Objective Scoring
```python
from .config import LigandForgeConfig
from .diversity_manager import EnhancedDiversityManager

class MultiObjectiveScorer:
    """Multi-objective scoring for generated molecules"""
    
    def __init__(self, config: LigandForgeConfig, pocket_analysis, diversity_manager: EnhancedDiversityManager):
        self.config = config
        self.pocket = pocket_analysis
        self.diversity = diversity_manager
    
    def calculate_comprehensive_score(self, mol: Chem.Mol, generation_round: int = 0) -> Dict[str, float]:
        """Calculate comprehensive multi-objective score"""
        # Implementation here
        pass
    
    def _pharmacophore_score(self, mol: Chem.Mol) -> float:
        """Score based on pharmacophore fit"""
        # Implementation here
        pass
```

### 9. **optimization/** - Optimization Algorithms
#### 9a. **optimization/rl_optimizer.py** - Reinforcement Learning
```python
from ..config import LigandForgeConfig
from ..scoring import MultiObjectiveScorer
from ..molecular_assembly import StructureGuidedAssembly

class RLOptimizer:
    """Reinforcement learning-based molecular optimization"""
    
    def __init__(self, config: LigandForgeConfig, scorer: MultiObjectiveScorer, assembly: StructureGuidedAssembly):
        self.config = config
        self.scorer = scorer
        self.assembly = assembly
    
    def optimize_generation(self, initial_molecules: List[Chem.Mol], n_iterations: int = 20) -> List[Chem.Mol]:
        """Optimize molecule generation using RL"""
        # Implementation here
        pass
```

#### 9b. **optimization/genetic_algorithm.py** - Genetic Algorithm
```python
from ..config import LigandForgeConfig
from ..scoring import MultiObjectiveScorer

class GeneticAlgorithm:
    """Genetic Algorithm for molecular optimization"""
    
    def __init__(self, config: LigandForgeConfig, scorer: MultiObjectiveScorer, assembly, diversity):
        self.config = config
        self.scorer = scorer
        self.assembly = assembly
        self.diversity = diversity
    
    def run(self, target_interactions: List[str], population_size: int = 100, generations: int = 30) -> List[Chem.Mol]:
        """Execute GA and return top molecules"""
        # Implementation here
        pass
```

### 10. **visualization.py** - Plotting and Visualization
```python
import pandas as pd
import plotly.graph_objects as go

def create_score_distribution_plot(df: pd.DataFrame):
    """Create score distribution plot"""
    # Implementation here
    pass

def create_property_space_plot(df: pd.DataFrame):
    """Create property space visualization"""
    # Implementation here
    pass

def create_sdf_from_molecules(molecules: List, results_data: List[Dict]) -> str:
    """Create SDF format string from molecules"""
    # Implementation here
    pass
```

### 11. **pipeline.py** - Main Pipeline Orchestrator
```python
from .config import LigandForgeConfig
from .pdb_parser import PDBParser
from .voxel_analyzer import VoxelBasedAnalyzer
from .pocket_analyzer import LigandForgeEnhancedAnalyzer
from .molecular_assembly import StructureGuidedAssembly
from .diversity_manager import EnhancedDiversityManager
from .scoring import MultiObjectiveScorer
from .optimization.rl_optimizer import RLOptimizer
from .optimization.genetic_algorithm import GeneticAlgorithm

class LigandForgePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: LigandForgeConfig = None):
        self.config = config or LigandForgeConfig()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        self.pdb_parser = PDBParser()
        self.voxel_analyzer = VoxelBasedAnalyzer(self.config)
        self.pocket_analyzer = LigandForgeEnhancedAnalyzer(self.config)
        self.diversity_manager = EnhancedDiversityManager(self.config)
    
    def run_full_pipeline(self, pdb_text: str, center: np.ndarray, target_interactions: List[str], 
                         optimization_method: str = "hybrid", **kwargs) -> Dict:
        """Run complete ligand design pipeline"""
        # 1. Parse structure
        structure = self.pdb_parser.parse_pdb_structure(pdb_text)
        
        # 2. Analyze binding site
        pocket_analysis = self.pocket_analyzer.analyze_binding_site_comprehensive(
            pdb_text, center, kwargs.get('radius', 10.0)
        )
        
        # 3. Initialize assembly and scoring
        fragment_lib = FragmentLibrary(self.config)
        assembly = StructureGuidedAssembly(fragment_lib, pocket_analysis['compatible_format'], self.config)
        scorer = MultiObjectiveScorer(self.config, pocket_analysis['compatible_format'], self.diversity_manager)
        
        # 4. Generate initial molecules
        seed_molecules = assembly.generate_structure_guided(target_interactions, kwargs.get('n_initial', 200))
        
        # 5. Optimize
        if optimization_method == "rl":
            optimizer = RLOptimizer(self.config, scorer, assembly)
            final_molecules = optimizer.optimize_generation(seed_molecules, kwargs.get('rl_iterations', 20))
        elif optimization_method == "ga":
            optimizer = GeneticAlgorithm(self.config, scorer, assembly, self.diversity_manager)
            final_molecules = optimizer.run(target_interactions, kwargs.get('population_size', 150), 
                                          kwargs.get('generations', 30))
        elif optimization_method == "hybrid":
            # RL then GA
            rl_optimizer = RLOptimizer(self.config, scorer, assembly)
            rl_results = rl_optimizer.optimize_generation(seed_molecules, kwargs.get('rl_iterations', 15))
            
            ga_optimizer = GeneticAlgorithm(self.config, scorer, assembly, self.diversity_manager)
            final_molecules = ga_optimizer.run(target_interactions, kwargs.get('population_size', 150),
                                             kwargs.get('generations', 15), seed_population=rl_results)
        
        return {
            'molecules': final_molecules,
            'pocket_analysis': pocket_analysis,
            'statistics': self.diversity_manager.get_statistics()
        }
```

### 12. **app.py** - Streamlit Application
```python
import streamlit as st
from pipeline import LigandForgePipeline
from config import LigandForgeConfig
from visualization import create_score_distribution_plot, create_property_space_plot
from fragment_library7 import FragmentLibrary

def create_streamlit_app():
    """Main Streamlit application with modular backend"""
    
    st.title("🧬 LigandForge 2.0 - AI-Driven Structure-Based Drug Design")
    
    # Initialize pipeline
    config = LigandForgeConfig()
    pipeline = LigandForgePipeline(config)
    
    # UI components here...
    
    if st.button("🚀 Run Optimization"):
        with st.spinner("Running optimization..."):
            results = pipeline.run_full_pipeline(
                pdb_text=pdb_text,
                center=center,
                target_interactions=chosen_interactions,
                optimization_method=opt_method.lower().replace(" ", "_"),
                population_size=int(ga_population),
                generations=int(ga_generations),
                n_initial=int(n_init)
            )
            
            # Process and display results...

if __name__ == "__main__":
    create_streamlit_app()

```
ligandforge/
├── __init__.py
├── config.py
├── data_structures.py
├── pdb_parser.py
├── voxel_analyzer.py
├── pocket_analyzer.py
├── molecular_assembly.py
├── diversity_manager.py
├── scoring.py
├── pipeline.py
├── visualization.py
├── app.py
├── optimization/
│   ├── __init__.py
│   ├── rl_optimizer.py
│   └── genetic_algorithm.py
├── utils/
│   ├── __init__.py
│   ├── chemistry_utils.py
│   └── math_utils.py
└── tests/
    ├── test_pdb_parser.py
    ├── test_voxel_analyzer.py
    ├── test_molecular_assembly.py
    └── ...
```


