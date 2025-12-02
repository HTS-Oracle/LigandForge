"""
Retrosynthetic Analysis Module
Analyzes and visualizes synthetic routes for generated molecules
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, Fragments, rdMolDescriptors

logger = logging.getLogger(__name__)


@dataclass
class SyntheticStep:
    """Represents a single synthetic step"""
    step_number: int
    reaction_type: str
    reactant_1: str  # SMILES
    reactant_2: Optional[str]  # SMILES
    product: str  # SMILES
    reagents: List[str]
    conditions: str
    yield_estimate: float
    difficulty: float  # 0-1 scale


@dataclass
class RetrosyntheticRoute:
    """Complete retrosynthetic route for a molecule"""
    target_smiles: str
    total_steps: int
    steps: List[SyntheticStep]
    starting_materials: List[str]
    total_yield_estimate: float
    overall_difficulty: float
    synthetic_complexity_score: float
    route_feasibility: float
    key_intermediates: List[str]
    critical_reactions: List[str]


class RetrosyntheticAnalyzer:
    """Analyzes molecules to determine synthetic routes"""
    
    def __init__(self):
        self.reaction_library = self._build_reaction_library()
        self.functional_group_transforms = self._build_fg_transforms()
        self.common_building_blocks = self._load_common_building_blocks()
        
    def _build_reaction_library(self) -> Dict[str, Dict]:
        """Build library of common reactions with templates"""
        return {
            'amide_coupling': {
                'name': 'Amide Coupling',
                'forward_smarts': '[C:1](=[O:2])[OH:3].[N:4][H:5]>>[C:1](=[O:2])[N:4]',
                'backward_smarts': '[C:1](=[O:2])[N:4]>>[C:1](=[O:2])[OH:3].[N:4][H:5]',
                'reagents': ['EDC/HOBt', 'HATU', 'T3P'],
                'conditions': 'RT, DMF or DCM',
                'typical_yield': 0.75,
                'difficulty': 0.2
            },
            'esterification': {
                'name': 'Esterification',
                'forward_smarts': '[C:1](=[O:2])[OH:3].[O:4][H:5]>>[C:1](=[O:2])[O:4]',
                'backward_smarts': '[C:1](=[O:2])[O:4]>>[C:1](=[O:2])[OH:3].[O:4][H:5]',
                'reagents': ['DCC/DMAP', 'EDC'],
                'conditions': 'RT, DCM',
                'typical_yield': 0.80,
                'difficulty': 0.15
            },
            'reductive_amination': {
                'name': 'Reductive Amination',
                'forward_smarts': '[C:1]=[O:2].[N:3][H:4]>>[C:1][N:3]',
                'backward_smarts': '[C:1][N:3]>>[C:1]=[O:2].[N:3][H:4]',
                'reagents': ['NaBH(OAc)3', 'NaCNBH3'],
                'conditions': 'RT, MeOH or AcOH',
                'typical_yield': 0.70,
                'difficulty': 0.25
            },
            'suzuki_coupling': {
                'name': 'Suzuki Coupling',
                'forward_smarts': '[c:1][Br,I:2].[c:3][B:4]([OH:5])[OH:6]>>[c:1][c:3]',
                'backward_smarts': '[c:1][c:3]>>[c:1][Br,I].[c:3][B]([OH])[OH]',
                'reagents': ['Pd(PPh3)4', 'K2CO3'],
                'conditions': '80°C, DME/H2O',
                'typical_yield': 0.65,
                'difficulty': 0.35
            },
            'n_alkylation': {
                'name': 'N-Alkylation',
                'forward_smarts': '[N:1][H:2].[C:3][Br,Cl,I:4]>>[N:1][C:3]',
                'backward_smarts': '[N:1][C:3]>>[N:1][H].[C:3][Br,Cl,I]',
                'reagents': ['K2CO3', 'NaH', 'Cs2CO3'],
                'conditions': 'RT-80°C, DMF or acetone',
                'typical_yield': 0.75,
                'difficulty': 0.20
            },
            'buchwald_hartwig': {
                'name': 'Buchwald-Hartwig Amination',
                'forward_smarts': '[c:1][Br,I:2].[N:3][H:4]>>[c:1][N:3]',
                'backward_smarts': '[c:1][N:3]>>[c:1][Br,I].[N:3][H]',
                'reagents': ['Pd2(dba)3', 'BINAP', 'NaOtBu'],
                'conditions': '80-100°C, toluene',
                'typical_yield': 0.60,
                'difficulty': 0.40
            },
            'mitsunobu': {
                'name': 'Mitsunobu Reaction',
                'forward_smarts': '[C:1][OH:2].[O:3][H:4]>>[C:1][O:3]',
                'backward_smarts': '[C:1][O:3]>>[C:1][OH].[O:3][H]',
                'reagents': ['DIAD', 'PPh3'],
                'conditions': '0°C-RT, THF',
                'typical_yield': 0.65,
                'difficulty': 0.35
            },
            'sonogashira': {
                'name': 'Sonogashira Coupling',
                'forward_smarts': '[c:1][Br,I:2].[C:3]#[C:4][H:5]>>[c:1][C:3]#[C:4]',
                'backward_smarts': '[c:1][C:3]#[C:4]>>[c:1][Br,I].[C:3]#[C:4][H]',
                'reagents': ['Pd(PPh3)2Cl2', 'CuI', 'Et3N'],
                'conditions': '50-80°C, DMF',
                'typical_yield': 0.70,
                'difficulty': 0.30
            },
            'wittig': {
                'name': 'Wittig Olefination',
                'forward_smarts': '[C:1]=[O:2].[P:3]=[C:4]>>[C:1]=[C:4]',
                'backward_smarts': '[C:1]=[C:4]>>[C:1]=[O:2].[P:3]=[C:4]',
                'reagents': ['Ph3P=CHR'],
                'conditions': 'RT, THF or DCM',
                'typical_yield': 0.65,
                'difficulty': 0.30
            },
            'click_chemistry': {
                'name': 'Click Chemistry (CuAAC)',
                'forward_smarts': '[C:1]#[C:2].[N:3]=[N:4]=[N:5]>>[c:1]1[n:3][n:4][n:5][c:2]1',
                'backward_smarts': '[c:1]1[n:3][n:4][n:5][c:2]1>>[C:1]#[C:2].[N:3]=[N:4]=[N:5]',
                'reagents': ['CuSO4', 'Na-ascorbate'],
                'conditions': 'RT, H2O/tBuOH',
                'typical_yield': 0.85,
                'difficulty': 0.15
            }
        }
    
    def _build_fg_transforms(self) -> Dict[str, List[str]]:
        """Build functional group transformation library"""
        return {
            'alcohol_to_halide': ['SOCl2', 'PBr3', 'Appel reaction'],
            'halide_to_amine': ['NaN3 then reduction', 'Gabriel synthesis'],
            'ketone_to_alcohol': ['NaBH4', 'LiAlH4'],
            'alcohol_to_ketone': ['PCC', 'Swern', 'DMP'],
            'nitrile_to_amine': ['LiAlH4', 'Raney Ni/H2'],
            'ester_to_acid': ['LiOH', 'NaOH'],
            'acid_to_ester': ['MeOH/H+', 'alcohol/DCC'],
            'nitro_to_amine': ['H2/Pd', 'Fe/AcOH', 'SnCl2'],
            'halide_to_nitrile': ['NaCN', 'KCN'],
            'alkene_epoxidation': ['mCPBA', 'Sharpless'],
            'aromatic_nitration': ['HNO3/H2SO4'],
            'aromatic_sulfonation': ['H2SO4', 'ClSO3H'],
            'friedel_crafts_acylation': ['RCOCl/AlCl3'],
            'friedel_crafts_alkylation': ['RCl/AlCl3'],
            'halogenation': ['Br2/FeBr3', 'Cl2/AlCl3']
        }
    
    def _load_common_building_blocks(self) -> Set[str]:
        """Load common commercially available building blocks"""
        return {
            'c1ccccc1',  # benzene
            'c1ccncc1',  # pyridine
            'c1ccc2ccccc2c1',  # naphthalene
            'C1CCCCC1',  # cyclohexane
            'c1ccc(N)cc1',  # aniline
            'c1ccc(O)cc1',  # phenol
            'c1ccc(C(=O)O)cc1',  # benzoic acid
            'CC(=O)O',  # acetic acid
            'CCO',  # ethanol
            'CO',  # methanol
            'C1CCNCC1',  # piperidine
            'C1CCOCC1',  # tetrahydropyran
            'c1c[nH]cc1',  # pyrrole
            'c1cncc1',  # pyrimidine
            'C(=O)O',  # formic acid
            'C#N',  # acetonitrile
            # Add more as needed
        }
    
    def analyze_molecule(self, mol: Chem.Mol, 
                        generation_metadata: Optional[Dict] = None) -> RetrosyntheticRoute:
        """
        Analyze a molecule and generate its retrosynthetic route
        
        Args:
            mol: RDKit molecule object
            generation_metadata: Optional metadata about how the molecule was generated
        
        Returns:
            RetrosyntheticRoute object with complete synthetic route
        """
        if mol is None:
            return self._create_failed_route("Invalid molecule")
        
        try:
            target_smiles = Chem.MolToSmiles(mol)
            
            # If we have generation metadata, use it to reconstruct the route
            if generation_metadata and 'assembly_history' in generation_metadata:
                route = self._reconstruct_from_assembly(mol, generation_metadata)
            else:
                # Perform retrosynthetic analysis
                route = self._perform_retrosynthesis(mol)
            
            return route
            
        except Exception as e:
            logger.error(f"Error analyzing molecule: {e}")
            return self._create_failed_route(str(e))
    
    def _reconstruct_from_assembly(self, mol: Chem.Mol, 
                                   metadata: Dict) -> RetrosyntheticRoute:
        """Reconstruct synthetic route from assembly metadata"""
        try:
            assembly_history = metadata.get('assembly_history', [])
            strategy = metadata.get('strategy', 'unknown')
            
            steps = []
            current_step = 1
            
            # Process assembly history to create synthetic steps
            for entry in assembly_history:
                operation = entry.get('operation', '')
                
                if operation == 'attach_fragment':
                    step = self._create_step_from_attachment(
                        entry, current_step
                    )
                    if step:
                        steps.append(step)
                        current_step += 1
                
                elif operation == 'link_cores':
                    step = self._create_step_from_linking(
                        entry, current_step
                    )
                    if step:
                        steps.append(step)
                        current_step += 1
                
                elif operation == 'add_functional_group':
                    step = self._create_step_from_fg_addition(
                        entry, current_step
                    )
                    if step:
                        steps.append(step)
                        current_step += 1
            
            # Calculate overall metrics
            starting_materials = self._identify_starting_materials(steps)
            total_yield = self._calculate_overall_yield(steps)
            overall_difficulty = self._calculate_overall_difficulty(steps)
            
            return RetrosyntheticRoute(
                target_smiles=Chem.MolToSmiles(mol),
                total_steps=len(steps),
                steps=steps,
                starting_materials=starting_materials,
                total_yield_estimate=total_yield,
                overall_difficulty=overall_difficulty,
                synthetic_complexity_score=self._calculate_complexity(mol),
                route_feasibility=self._assess_route_feasibility(steps),
                key_intermediates=self._identify_key_intermediates(steps),
                critical_reactions=self._identify_critical_reactions(steps)
            )
            
        except Exception as e:
            logger.error(f"Error reconstructing from assembly: {e}")
            return self._perform_retrosynthesis(mol)
    
    def _perform_retrosynthesis(self, mol: Chem.Mol) -> RetrosyntheticRoute:
        """Perform retrosynthetic analysis on a molecule"""
        try:
            target_smiles = Chem.MolToSmiles(mol)
            steps = []
            current_mol = mol
            step_number = 1
            
            # Iteratively break down the molecule
            max_steps = 10
            while step_number <= max_steps:
                # Try to find a disconnection
                disconnection = self._find_best_disconnection(current_mol)
                
                if disconnection is None:
                    # Can't disconnect further - reached starting materials
                    break
                
                step = self._create_step_from_disconnection(
                    disconnection, step_number
                )
                steps.append(step)
                
                # Continue with one of the fragments
                if disconnection['fragment_1_mol'] is not None:
                    current_mol = disconnection['fragment_1_mol']
                else:
                    break
                
                step_number += 1
            
            # Reverse steps for forward synthesis
            steps.reverse()
            for i, step in enumerate(steps, 1):
                step.step_number = i
            
            starting_materials = self._identify_starting_materials(steps)
            total_yield = self._calculate_overall_yield(steps)
            overall_difficulty = self._calculate_overall_difficulty(steps)
            
            return RetrosyntheticRoute(
                target_smiles=target_smiles,
                total_steps=len(steps),
                steps=steps,
                starting_materials=starting_materials,
                total_yield_estimate=total_yield,
                overall_difficulty=overall_difficulty,
                synthetic_complexity_score=self._calculate_complexity(mol),
                route_feasibility=self._assess_route_feasibility(steps),
                key_intermediates=self._identify_key_intermediates(steps),
                critical_reactions=self._identify_critical_reactions(steps)
            )
            
        except Exception as e:
            logger.error(f"Error in retrosynthesis: {e}")
            return self._create_failed_route(str(e))
    
    def _find_best_disconnection(self, mol: Chem.Mol) -> Optional[Dict]:
        """Find the best strategic bond to disconnect"""
        try:
            # Look for functional groups that suggest synthetic disconnections
            for reaction_name, reaction_data in self.reaction_library.items():
                try:
                    # Try backward transformation
                    rxn = AllChem.ReactionFromSmarts(reaction_data['backward_smarts'])
                    products = rxn.RunReactants((mol,))
                    
                    if products:
                        # Found a valid disconnection
                        reactants = products[0]
                        return {
                            'reaction_type': reaction_name,
                            'reaction_data': reaction_data,
                            'fragment_1_mol': reactants[0] if len(reactants) > 0 else None,
                            'fragment_2_mol': reactants[1] if len(reactants) > 1 else None,
                            'fragment_1_smiles': Chem.MolToSmiles(reactants[0]) if len(reactants) > 0 else None,
                            'fragment_2_smiles': Chem.MolToSmiles(reactants[1]) if len(reactants) > 1 else None,
                        }
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error finding disconnection: {e}")
            return None
    
    def _create_step_from_disconnection(self, disconnection: Dict, 
                                       step_number: int) -> SyntheticStep:
        """Create a synthetic step from a disconnection"""
        reaction_data = disconnection['reaction_data']
        
        return SyntheticStep(
            step_number=step_number,
            reaction_type=reaction_data['name'],
            reactant_1=disconnection['fragment_1_smiles'],
            reactant_2=disconnection.get('fragment_2_smiles'),
            product="",  # Will be filled in later
            reagents=reaction_data['reagents'],
            conditions=reaction_data['conditions'],
            yield_estimate=reaction_data['typical_yield'],
            difficulty=reaction_data['difficulty']
        )
    
    def _create_step_from_attachment(self, entry: Dict, 
                                    step_number: int) -> Optional[SyntheticStep]:
        """Create synthetic step from fragment attachment"""
        try:
            fragment_smiles = entry.get('fragment_smiles', '')
            base_smiles = entry.get('base_smiles', '')
            
            # Infer reaction type from fragments
            reaction_type = self._infer_reaction_type(fragment_smiles)
            reaction_data = self.reaction_library.get(reaction_type, {})
            
            return SyntheticStep(
                step_number=step_number,
                reaction_type=reaction_data.get('name', 'Fragment Coupling'),
                reactant_1=base_smiles,
                reactant_2=fragment_smiles,
                product=entry.get('product_smiles', ''),
                reagents=reaction_data.get('reagents', ['Coupling reagents']),
                conditions=reaction_data.get('conditions', 'Standard conditions'),
                yield_estimate=reaction_data.get('typical_yield', 0.70),
                difficulty=reaction_data.get('difficulty', 0.25)
            )
        except Exception:
            return None
    
    def _create_step_from_linking(self, entry: Dict, 
                                 step_number: int) -> Optional[SyntheticStep]:
        """Create synthetic step from core linking"""
        try:
            return SyntheticStep(
                step_number=step_number,
                reaction_type='Core Linking',
                reactant_1=entry.get('core1_smiles', ''),
                reactant_2=entry.get('core2_smiles', ''),
                product=entry.get('product_smiles', ''),
                reagents=['Linker', 'Coupling reagents'],
                conditions='Standard coupling conditions',
                yield_estimate=0.65,
                difficulty=0.30
            )
        except Exception:
            return None
    
    def _create_step_from_fg_addition(self, entry: Dict, 
                                     step_number: int) -> Optional[SyntheticStep]:
        """Create synthetic step from functional group addition"""
        try:
            fg_type = entry.get('fg_type', 'functional_group')
            transform = self.fg_transforms.get(fg_type, ['Standard reagents'])
            
            return SyntheticStep(
                step_number=step_number,
                reaction_type=f'{fg_type.replace("_", " ").title()} Addition',
                reactant_1=entry.get('base_smiles', ''),
                reactant_2=None,
                product=entry.get('product_smiles', ''),
                reagents=transform,
                conditions='Standard conditions',
                yield_estimate=0.75,
                difficulty=0.20
            )
        except Exception:
            return None
    
    def _infer_reaction_type(self, fragment_smiles: str) -> str:
        """Infer likely reaction type from fragment structure"""
        try:
            mol = Chem.MolFromSmiles(fragment_smiles)
            if mol is None:
                return 'n_alkylation'
            
            # Check for functional groups
            if mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3;H2,H1]')):
                return 'amide_coupling'
            elif mol.HasSubstructMatch(Chem.MolFromSmarts('[OX2H]')):
                return 'esterification'
            elif mol.HasSubstructMatch(Chem.MolFromSmarts('c[Br,I]')):
                return 'suzuki_coupling'
            elif mol.HasSubstructMatch(Chem.MolFromSmarts('[C]=[O]')):
                return 'reductive_amination'
            else:
                return 'n_alkylation'
                
        except Exception:
            return 'n_alkylation'
    
    def _identify_starting_materials(self, steps: List[SyntheticStep]) -> List[str]:
        """Identify starting materials from synthetic steps"""
        if not steps:
            return []
        
        starting_materials = set()
        
        # First step reactants are starting materials
        if steps:
            first_step = steps[0]
            if first_step.reactant_1:
                starting_materials.add(first_step.reactant_1)
            if first_step.reactant_2:
                starting_materials.add(first_step.reactant_2)
        
        # Remove any that appear as products in later steps
        products = {step.product for step in steps if step.product}
        starting_materials = starting_materials - products
        
        return list(starting_materials)
    
    def _calculate_overall_yield(self, steps: List[SyntheticStep]) -> float:
        """Calculate overall yield from individual step yields"""
        if not steps:
            return 0.0
        
        overall_yield = 1.0
        for step in steps:
            overall_yield *= step.yield_estimate
        
        return overall_yield
    
    def _calculate_overall_difficulty(self, steps: List[SyntheticStep]) -> float:
        """Calculate overall synthetic difficulty"""
        if not steps:
            return 0.0
        
        # Use maximum difficulty (bottleneck step) and average
        difficulties = [step.difficulty for step in steps]
        max_difficulty = max(difficulties)
        avg_difficulty = np.mean(difficulties)
        
        # Weight toward the hardest step
        return 0.7 * max_difficulty + 0.3 * avg_difficulty
    
    def _calculate_complexity(self, mol: Chem.Mol) -> float:
        """Calculate synthetic complexity score"""
        try:
            score = 0.0
            
            # Ring complexity
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            score += num_rings * 0.1
            
            # Stereocenter complexity
            chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            score += chiral_centers * 0.15
            
            # Functional group diversity
            num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            score += num_heteroatoms * 0.05
            
            # Molecular weight factor
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            score += (mw / 500) * 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _assess_route_feasibility(self, steps: List[SyntheticStep]) -> float:
        """Assess overall route feasibility"""
        if not steps:
            return 0.0
        
        factors = []
        
        # Number of steps (fewer is better)
        step_score = max(0.0, 1.0 - len(steps) * 0.1)
        factors.append(step_score)
        
        # Average yield
        avg_yield = self._calculate_overall_yield(steps)
        factors.append(avg_yield)
        
        # Difficulty
        avg_difficulty = self._calculate_overall_difficulty(steps)
        difficulty_score = 1.0 - avg_difficulty
        factors.append(difficulty_score)
        
        return np.mean(factors)
    
    def _identify_key_intermediates(self, steps: List[SyntheticStep]) -> List[str]:
        """Identify key intermediates in the route"""
        if len(steps) <= 2:
            return []
        
        # Intermediates are products of steps (except the last step)
        intermediates = []
        for step in steps[:-1]:
            if step.product:
                intermediates.append(step.product)
        
        return intermediates
    
    def _identify_critical_reactions(self, steps: List[SyntheticStep]) -> List[str]:
        """Identify critical/challenging reactions"""
        critical = []
        
        for step in steps:
            # Reactions with difficulty > 0.3 or yield < 0.65 are critical
            if step.difficulty > 0.3 or step.yield_estimate < 0.65:
                critical.append(f"Step {step.step_number}: {step.reaction_type}")
        
        return critical
    
    def _create_failed_route(self, error_msg: str) -> RetrosyntheticRoute:
        """Create a failed route with error information"""
        return RetrosyntheticRoute(
            target_smiles="",
            total_steps=0,
            steps=[],
            starting_materials=[],
            total_yield_estimate=0.0,
            overall_difficulty=1.0,
            synthetic_complexity_score=1.0,
            route_feasibility=0.0,
            key_intermediates=[],
            critical_reactions=[f"Analysis failed: {error_msg}"]
        )
    
    def generate_route_summary(self, route: RetrosyntheticRoute) -> str:
        """Generate human-readable summary of the route"""
        if not route.steps:
            return "No synthetic route available"
        
        summary_lines = [
            f"Retrosynthetic Analysis",
            f"=" * 60,
            f"Target: {route.target_smiles}",
            f"Total Steps: {route.total_steps}",
            f"Estimated Overall Yield: {route.total_yield_estimate*100:.1f}%",
            f"Overall Difficulty: {route.overall_difficulty:.2f}/1.0",
            f"Route Feasibility: {route.route_feasibility:.2f}/1.0",
            f"",
            f"Starting Materials:",
        ]
        
        for i, sm in enumerate(route.starting_materials, 1):
            summary_lines.append(f"  {i}. {sm}")
        
        summary_lines.extend([
            f"",
            f"Synthetic Steps:",
        ])
        
        for step in route.steps:
            summary_lines.extend([
                f"",
                f"Step {step.step_number}: {step.reaction_type}",
                f"  Reactant 1: {step.reactant_1}",
            ])
            if step.reactant_2:
                summary_lines.append(f"  Reactant 2: {step.reactant_2}")
            summary_lines.extend([
                f"  Reagents: {', '.join(step.reagents)}",
                f"  Conditions: {step.conditions}",
                f"  Estimated Yield: {step.yield_estimate*100:.0f}%",
                f"  Difficulty: {step.difficulty:.2f}/1.0",
            ])
        
        if route.critical_reactions:
            summary_lines.extend([
                f"",
                f"Critical Steps:",
            ])
            for critical in route.critical_reactions:
                summary_lines.append(f"  - {critical}")
        
        return "\n".join(summary_lines)
    
    def visualize_route_graphviz(self, route: RetrosyntheticRoute) -> Optional[str]:
        """Generate Graphviz DOT representation of the route"""
        if not route.steps:
            return None
        
        try:
            dot_lines = [
                'digraph RetrosyntheticRoute {',
                '  rankdir=LR;',
                '  node [shape=box, style=rounded];',
                '  edge [fontsize=10];',
                ''
            ]
            
            # Add starting materials
            for i, sm in enumerate(route.starting_materials):
                sm_id = f"SM{i}"
                label = self._truncate_smiles(sm, 20)
                dot_lines.append(f'  {sm_id} [label="{label}\\nStarting Material", fillcolor=lightblue, style="rounded,filled"];')
            
            # Add steps
            for step in route.steps:
                step_id = f"Step{step.step_number}"
                product_id = f"P{step.step_number}"
                
                # Step node
                label = f"{step.reaction_type}\\nYield: {step.yield_estimate*100:.0f}%"
                color = self._get_difficulty_color(step.difficulty)
                dot_lines.append(f'  {step_id} [label="{label}", fillcolor={color}, style="rounded,filled"];')
                
                # Product node (except for last step which is the target)
                if step.step_number < route.total_steps:
                    prod_label = self._truncate_smiles(step.product, 20)
                    dot_lines.append(f'  {product_id} [label="{prod_label}\\nIntermediate", fillcolor=lightyellow, style="rounded,filled"];')
                else:
                    dot_lines.append(f'  {product_id} [label="Target\\nMolecule", fillcolor=lightgreen, style="rounded,filled"];')
                
                # Connect reactants to step
                if step.step_number == 1:
                    # Connect starting materials
                    for i in range(len(route.starting_materials)):
                        dot_lines.append(f'  SM{i} -> {step_id};')
                else:
                    # Connect previous product
                    prev_product_id = f"P{step.step_number-1}"
                    dot_lines.append(f'  {prev_product_id} -> {step_id};')
                
                # Connect step to product
                dot_lines.append(f'  {step_id} -> {product_id};')
            
            dot_lines.append('}')
            
            return '\n'.join(dot_lines)
            
        except Exception as e:
            logger.error(f"Error generating Graphviz: {e}")
            return None
    
    def _truncate_smiles(self, smiles: str, max_len: int = 20) -> str:
        """Truncate SMILES for display"""
        if len(smiles) <= max_len:
            return smiles
        return smiles[:max_len] + "..."
    
    def _get_difficulty_color(self, difficulty: float) -> str:
        """Get color based on difficulty"""
        if difficulty < 0.25:
            return "lightgreen"
        elif difficulty < 0.35:
            return "yellow"
        else:
            return "orange"
