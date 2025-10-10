import numpy as np
import random
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import warnings

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdMolDescriptors
from rdkit.Chem import Crippen, Lipinski, QED
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import LigandForgeConfig
from data_structures import PocketAnalysis, InteractionHotspot
from fragment_library7 import FragmentLibrary, FragmentInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttachmentPoint:
    """Represents an attachment point in a molecule"""
    atom_idx: int
    map_num: int  # The :1, :2, etc. from [*:1]
    neighbor_idx: int  # The atom it's attached to


@dataclass
class ChemicalEnvironment:
    """Describes the chemical environment around an attachment point"""
    atom_type: str
    hybridization: str  # sp, sp2, sp3
    is_aromatic: bool
    is_in_ring: bool
    ring_size: int
    formal_charge: int
    num_neighbors: int
    available_valence: int
    neighbor_types: List[str]
    functional_groups: List[str]


@dataclass
class MoleculeConstraints:
    """Molecular property constraints with validation"""
    min_heavy_atoms: int = 10
    max_heavy_atoms: int = 50
    min_molecular_weight: float = 150.0
    max_molecular_weight: float = 700.0
    max_rotatable_bonds: int = 15
    max_rings: int = 8
    max_logp: float = 7.0
    min_logp: float = -4.0
    max_tpsa: float = 250.0
    large_fragment_threshold: float = 0.7


@dataclass
class GrowthSite:
    """Represents a potential growth site on a molecule"""
    atom_idx: int
    atom_type: str
    hybridization: str
    available_valence: int
    is_aromatic: bool
    is_in_ring: bool
    distance_to_target: float
    environment_score: float
    position_3d: Optional[np.ndarray] = None


class AttachmentPointManager:
    """Manages chemistry-aware attachment point parsing and matching"""
    
    @staticmethod
    def find_attachment_points(mol: Chem.Mol) -> List[AttachmentPoint]:
        """Find all attachment points ([*:N]) in a molecule"""
        attachment_points = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    # Find the neighbor (the real atom it's attached to)
                    neighbors = list(atom.GetNeighbors())
                    if neighbors:
                        attachment_points.append(AttachmentPoint(
                            atom_idx=atom.GetIdx(),
                            map_num=map_num,
                            neighbor_idx=neighbors[0].GetIdx()
                        ))
        return attachment_points
    
    @staticmethod
    def analyze_chemical_environment(mol: Chem.Mol, atom_idx: int) -> ChemicalEnvironment:
        """Analyze the chemical environment around an attachment point"""
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Determine hybridization
        hybridization_map = {
            Chem.HybridizationType.SP: 'sp',
            Chem.HybridizationType.SP2: 'sp2',
            Chem.HybridizationType.SP3: 'sp3',
            Chem.HybridizationType.SP3D: 'sp3d',
            Chem.HybridizationType.SP3D2: 'sp3d2'
        }
        hybrid = hybridization_map.get(atom.GetHybridization(), 'unknown')
        
        # Get ring information
        ring_info = mol.GetRingInfo()
        is_in_ring = atom.IsInRing()
        ring_size = 0
        if is_in_ring:
            for ring in ring_info.AtomRings():
                if atom_idx in ring:
                    ring_size = len(ring)
                    break
        
        # Get neighbor information
        neighbors = atom.GetNeighbors()
        neighbor_types = [n.GetSymbol() for n in neighbors if n.GetAtomicNum() != 0]
        
        # Calculate available valence
        max_valence = AttachmentPointManager._get_max_valence(atom.GetAtomicNum())
        current_valence = atom.GetTotalValence()
        available_valence = max_valence - current_valence
        
        # Detect functional groups
        functional_groups = AttachmentPointManager._detect_functional_groups(mol, atom_idx)
        
        return ChemicalEnvironment(
            atom_type=atom.GetSymbol(),
            hybridization=hybrid,
            is_aromatic=atom.GetIsAromatic(),
            is_in_ring=is_in_ring,
            ring_size=ring_size,
            formal_charge=atom.GetFormalCharge(),
            num_neighbors=len(neighbors),
            available_valence=available_valence,
            neighbor_types=neighbor_types,
            functional_groups=functional_groups
        )
    
    @staticmethod
    def _detect_functional_groups(mol: Chem.Mol, atom_idx: int) -> List[str]:
        """Detect functional groups near an atom"""
        groups = []
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Common functional group SMARTS patterns
        fg_patterns = {
            'carbonyl': '[CX3](=O)',
            'carboxyl': '[CX3](=O)[OX2H1]',
            'ester': '[CX3](=O)[OX2]',
            'amide': '[CX3](=O)[NX3]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'hydroxyl': '[OX2H]',
            'ether': '[OD2]([#6])[#6]',
            'aromatic': 'c',
            'alkene': '[CX3]=[CX3]',
            'alkyne': '[CX2]#[CX2]',
            'nitrile': '[CX2]#[NX1]',
            'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
            'halogen': '[F,Cl,Br,I]',
            'sulfone': '[SX4](=O)(=O)',
            'phosphate': '[PX4](=O)'
        }
        
        for fg_name, smarts in fg_patterns.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt:
                    matches = mol.GetSubstructMatches(patt)
                    for match in matches:
                        if atom_idx in match or any(atom_idx in [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors()] for m in match):
                            groups.append(fg_name)
                            break
            except:
                continue
        
        return groups
    
    @staticmethod
    def _get_max_valence(atomic_num: int) -> int:
        """Get maximum valence for an atom"""
        valence_map = {
            1: 1,   # H
            6: 4,   # C
            7: 3,   # N
            8: 2,   # O
            9: 1,   # F
            15: 5,  # P
            16: 6,  # S
            17: 1,  # Cl
            35: 1,  # Br
            53: 1   # I
        }
        return valence_map.get(atomic_num, 4)
    
    @staticmethod
    def calculate_compatibility_score(env1: ChemicalEnvironment, env2: ChemicalEnvironment) -> float:
        """Calculate chemistry-aware compatibility score between two attachment points"""
        score = 0.0
        
        # 1. Hybridization compatibility (most important)
        if env1.hybridization == env2.hybridization:
            score += 40.0
        elif {env1.hybridization, env2.hybridization} == {'sp2', 'sp3'}:
            score += 25.0  # Acceptable but not ideal
        elif 'unknown' in {env1.hybridization, env2.hybridization}:
            score += 15.0
        else:
            score += 5.0  # Poor match
        
        # 2. Aromaticity matching
        if env1.is_aromatic == env2.is_aromatic:
            score += 20.0
        else:
            score += 5.0  # Can still connect aromatic to non-aromatic
        
        # 3. Valence availability check (critical)
        if env1.available_valence > 0 and env2.available_valence > 0:
            score += 30.0
        else:
            return 0.0  # Cannot connect if no valence available
        
        # 4. Atom type compatibility
        compatible_pairs = [
            {'C', 'C'}, {'C', 'N'}, {'C', 'O'}, {'C', 'S'},
            {'N', 'C'}, {'N', 'N'}, {'O', 'C'}, {'S', 'C'}
        ]
        if {env1.atom_type, env2.atom_type} in compatible_pairs:
            score += 15.0
        else:
            score += 5.0
        
        # 5. Ring system considerations
        if env1.is_in_ring and env2.is_in_ring:
            # Prefer similar ring sizes
            if abs(env1.ring_size - env2.ring_size) <= 1:
                score += 10.0
            else:
                score += 3.0
        elif not env1.is_in_ring and not env2.is_in_ring:
            score += 8.0
        else:
            score += 5.0
        
        # 6. Charge compatibility
        if env1.formal_charge == 0 and env2.formal_charge == 0:
            score += 10.0
        elif env1.formal_charge * env2.formal_charge < 0:
            score += 5.0  # Opposite charges
        else:
            score += 2.0
        
        # 7. Functional group synergy
        complementary_groups = [
            ('carbonyl', 'amine'), ('carboxyl', 'amine'),
            ('carbonyl', 'hydroxyl'), ('aromatic', 'aromatic')
        ]
        for fg1 in env1.functional_groups:
            for fg2 in env2.functional_groups:
                if (fg1, fg2) in complementary_groups or (fg2, fg1) in complementary_groups:
                    score += 8.0
                    break
        
        # 8. Neighbor diversity penalty (avoid overcrowding)
        total_neighbors = env1.num_neighbors + env2.num_neighbors
        if total_neighbors > 6:
            score -= 10.0  # Penalty for crowded environment
        
        return score
    
    @staticmethod
    def determine_bond_type(mol1: Chem.Mol, mol2: Chem.Mol,
                           ap1: AttachmentPoint, ap2: AttachmentPoint) -> Chem.BondType:
        """Determine appropriate bond type based on hybridization"""
        try:
            atom1 = mol1.GetAtomWithIdx(ap1.neighbor_idx)
            atom2 = mol2.GetAtomWithIdx(ap2.neighbor_idx)
            
            hyb1 = atom1.GetHybridization()
            hyb2 = atom2.GetHybridization()
            
            # sp2-sp2 or aromatic
            if (hyb1 == Chem.HybridizationType.SP2 and 
                hyb2 == Chem.HybridizationType.SP2):
                if atom1.GetIsAromatic() and atom2.GetIsAromatic():
                    return Chem.BondType.AROMATIC
                # Could be double bond, but safer to use single for general case
                return Chem.BondType.SINGLE
            
            # sp-sp (triple bond)
            if (hyb1 == Chem.HybridizationType.SP and 
                hyb2 == Chem.HybridizationType.SP):
                return Chem.BondType.TRIPLE
            
            # sp2-sp or sp-sp2 (double bond potential)
            if ({hyb1, hyb2} == {Chem.HybridizationType.SP, Chem.HybridizationType.SP2}):
                return Chem.BondType.DOUBLE
            
            # Default to single bond
            return Chem.BondType.SINGLE
            
        except Exception as e:
            logger.debug(f"Error determining bond type: {e}")
            return Chem.BondType.SINGLE
    
    @staticmethod
    def remove_attachment_point(mol: Chem.Mol, attachment_point: AttachmentPoint) -> Chem.Mol:
        """Remove an attachment point dummy atom, returning modified molecule"""
        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAtom(attachment_point.atom_idx)
        return rw_mol.GetMol()
    
    @staticmethod
    def connect_at_attachment_points(mol1: Chem.Mol, mol2: Chem.Mol,
                                     ap1: AttachmentPoint, ap2: AttachmentPoint,
                                     bond_type: Optional[Chem.BondType] = None) -> Optional[Chem.Mol]:
        """Connect two molecules at specified attachment points with smart bond typing"""
        try:
            # Combine molecules
            combined = Chem.CombineMols(mol1, mol2)
            rw_mol = Chem.RWMol(combined)
            
            # Get the indices after combination
            mol1_size = mol1.GetNumAtoms()
            neighbor1_idx = ap1.neighbor_idx
            neighbor2_idx = mol1_size + ap2.neighbor_idx
            dummy1_idx = ap1.atom_idx
            dummy2_idx = mol1_size + ap2.atom_idx
            
            # Determine bond type if not specified
            if bond_type is None:
                bond_type = AttachmentPointManager.determine_bond_type(mol1, mol2, ap1, ap2)
            
            # Add bond between the real atoms
            rw_mol.AddBond(neighbor1_idx, neighbor2_idx, bond_type)
            
            # Remove dummy atoms (in reverse order to maintain indices)
            for idx in sorted([dummy1_idx, dummy2_idx], reverse=True):
                rw_mol.RemoveAtom(idx)
            
            result = rw_mol.GetMol()
            Chem.SanitizeMol(result)
            return result
            
        except Exception as e:
            logger.debug(f"Connection at attachment points failed: {e}")
            return None
    
    @staticmethod
    def match_attachment_points(mol1: Chem.Mol, mol2: Chem.Mol,
                               prefer_matching: bool = True) -> Optional[Tuple[AttachmentPoint, AttachmentPoint]]:
        """
        Chemistry-aware attachment point matching with compatibility scoring
        """
        ap1_list = AttachmentPointManager.find_attachment_points(mol1)
        ap2_list = AttachmentPointManager.find_attachment_points(mol2)
        
        if not ap1_list or not ap2_list:
            return None
        
        # Build compatibility matrix
        compatibility_scores = []
        
        for ap1 in ap1_list:
            env1 = AttachmentPointManager.analyze_chemical_environment(mol1, ap1.neighbor_idx)
            
            for ap2 in ap2_list:
                env2 = AttachmentPointManager.analyze_chemical_environment(mol2, ap2.neighbor_idx)
                
                # Calculate base compatibility score
                compatibility = AttachmentPointManager.calculate_compatibility_score(env1, env2)
                
                # Bonus for matching map numbers if prefer_matching is True
                if prefer_matching and ap1.map_num == ap2.map_num:
                    compatibility += 50.0
                
                compatibility_scores.append({
                    'ap1': ap1,
                    'ap2': ap2,
                    'score': compatibility,
                    'env1': env1,
                    'env2': env2
                })
        
        # Filter out incompatible matches (score <= 0)
        valid_matches = [m for m in compatibility_scores if m['score'] > 0]
        
        if not valid_matches:
            logger.debug("No chemically compatible attachment points found")
            return None
        
        # Sort by compatibility score
        valid_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Select best match
        best_match = valid_matches[0]
        
        logger.debug(f"Selected attachment match with compatibility score: {best_match['score']:.1f} "
                    f"({best_match['env1'].atom_type}-{best_match['env1'].hybridization} + "
                    f"{best_match['env2'].atom_type}-{best_match['env2'].hybridization})")
        
        return (best_match['ap1'], best_match['ap2'])


class ValidationEngine:
    """Handles molecular validation and property calculation"""
    
    def __init__(self, constraints: MoleculeConstraints, fragment_library: FragmentLibrary):
        self.constraints = constraints
        self.fragment_lib = fragment_library
    
    @lru_cache(maxsize=1000)
    def calculate_properties_cached(self, smiles: str) -> Optional[Dict]:
        """Calculate and cache molecular properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            return {
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'mw': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'qed': QED.qed(mol)
            }
        except Exception as e:
            logger.debug(f"Property calculation error: {e}")
            return None
    
    def validate_with_reason(self, mol: Chem.Mol, props: Dict) -> Tuple[bool, Optional[str]]:
        """Validate and return reason for failure"""
        try:
            if not (self.constraints.min_heavy_atoms <= props['heavy_atoms'] <= self.constraints.max_heavy_atoms):
                return False, f"Heavy atoms {props['heavy_atoms']} outside range [{self.constraints.min_heavy_atoms}, {self.constraints.max_heavy_atoms}]"
            
            if not (self.constraints.min_molecular_weight <= props['mw'] <= self.constraints.max_molecular_weight):
                return False, f"MW {props['mw']:.1f} outside range [{self.constraints.min_molecular_weight}, {self.constraints.max_molecular_weight}]"
            
            if props['rotatable_bonds'] > self.constraints.max_rotatable_bonds:
                return False, f"Too many rotatable bonds: {props['rotatable_bonds']}"
            
            if props['rings'] > self.constraints.max_rings:
                return False, f"Too many rings: {props['rings']}"
            
            if not (self.constraints.min_logp <= props['logp'] <= self.constraints.max_logp):
                return False, f"LogP {props['logp']:.2f} outside range [{self.constraints.min_logp}, {self.constraints.max_logp}]"
            
            if props['tpsa'] > self.constraints.max_tpsa:
                return False, f"TPSA too high: {props['tpsa']:.1f}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation exception: {str(e)}"


class PocketAwareFunctionalGroupManager:
    """Manages functional group addition based on pocket geometry and requirements"""
    
    def __init__(self, pocket: PocketAnalysis, fg_library: Dict, pharmacophore_map: Dict, scorer=None):
        self.pocket = pocket
        self.fg_library = fg_library
        self.pharmacophore_map = pharmacophore_map
        self.scorer = scorer
        self.addition_stats = defaultdict(int)
        self.fg_patterns = self._build_fg_patterns()
    
    def _build_fg_patterns(self) -> Dict[str, List[str]]:
        """Build SMARTS patterns for detecting functional groups"""
        return {
            'hbond_donor': ['[NH2]', '[NH]', '[OH]', '[SH]'],
            'hbond_acceptor': ['[O;H0]', '[N;H0]', 'C(=O)', 'C#N'],
            'hydrophobic': ['[CH3]', '[CH2]', 'c', '[Cl,Br,I,F]'],
            'aromatic': ['c1ccccc1', 'c1ccncc1', 'c1cncnc1'],
            'positive': ['[NH3+]', '[NH2+]', '[NH+]', '[N+]'],
            'negative': ['[O-]', 'C(=O)[O-]', 'S(=O)(=O)[O-]'],
            'polar': ['[OH]', '[NH]', 'C(=O)'],
            'basic': ['[NH2]', '[NH]', 'N'],
            'acidic': ['C(=O)[OH]', 'S(=O)(=O)[OH]'],
        }
    
    def analyze_unsatisfied_requirements(self, mol: Chem.Mol, 
                                         target_interactions: List[str]) -> List[Dict]:
        """Identify which pocket requirements are not yet satisfied by the molecule"""
        unsatisfied = []
        
        if not self.pocket.pharmacophore_points:
            return unsatisfied
        
        for point in self.pocket.pharmacophore_points:
            point_type = point.get('type', '')
            if point_type not in target_interactions:
                continue
            
            point_position = np.array(point.get('position', [0, 0, 0]))
            
            if not self._has_fg_near_position(mol, point_position, point_type):
                unsatisfied.append({
                    'type': point_type,
                    'position': point_position,
                    'priority': point.get('importance', point.get('strength', 1.0))
                })
        
        unsatisfied.sort(key=lambda x: x['priority'], reverse=True)
        return unsatisfied
    
    def _has_fg_near_position(self, mol: Chem.Mol, position: np.ndarray, 
                              fg_type: str, threshold: float = 5.0) -> bool:
        """Check if molecule has appropriate functional group near a position"""
        if not mol.GetNumConformers():
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                return False
        
        try:
            conformer = mol.GetConformer()
        except:
            return False
        
        fg_patterns = self.fg_patterns.get(fg_type, [])
        
        for pattern_smarts in fg_patterns:
            try:
                patt = Chem.MolFromSmarts(pattern_smarts)
                if patt is None:
                    continue
                
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    positions = [conformer.GetAtomPosition(idx) for idx in match]
                    center = np.mean([[p.x, p.y, p.z] for p in positions], axis=0)
                    
                    distance = np.linalg.norm(center - position)
                    if distance < threshold:
                        return True
            except Exception as e:
                logger.debug(f"Pattern matching error for {pattern_smarts}: {e}")
                continue
        
        return False
    
    def add_targeted_functional_groups(self, mol: Chem.Mol, 
                                      target_interactions: List[str],
                                      max_additions: int = 5) -> Chem.Mol:
        """Add functional groups targeting specific pocket requirements with scoring feedback"""
        if mol is None:
            return None
        
        initial_score = None
        if self.scorer:
            try:
                from data_structures import MolecularScore
                initial_score = self.scorer.calculate_comprehensive_score(mol)
                logger.debug(f"Initial score before FG addition: {initial_score.total_score:.3f}")
            except Exception as e:
                logger.debug(f"Could not calculate initial score: {e}")
        
        unsatisfied = self.analyze_unsatisfied_requirements(mol, target_interactions)
        
        if not unsatisfied:
            logger.debug("All pharmacophore requirements satisfied")
            return mol
        
        logger.info(f"Found {len(unsatisfied)} unsatisfied pocket requirements")
        
        additions_made = 0
        best_mol = mol
        best_score = initial_score.total_score if initial_score else 0.0
        
        for requirement in unsatisfied[:max_additions]:
            if additions_made >= max_additions:
                break
            
            fg_type_options = self.pharmacophore_map.get(requirement['type'], [])
            if not fg_type_options:
                continue
            
            for fg_type in fg_type_options:
                if fg_type not in self.fg_library:
                    continue
                
                target_pos = requirement['position']
                growth_sites = self._find_growth_sites_near_position(
                    mol, target_pos, max_sites=3
                )
                
                if not growth_sites:
                    continue
                
                new_mol = self._add_fg_at_sites(
                    mol, fg_type, growth_sites, requirement['type']
                )
                
                if new_mol is not None and new_mol != mol:
                    if self.scorer:
                        try:
                            new_score_obj = self.scorer.calculate_comprehensive_score(new_mol)
                            new_score = new_score_obj.total_score
                            
                            if new_score > best_score:
                                best_mol = new_mol
                                best_score = new_score
                                mol = new_mol
                                additions_made += 1
                                logger.debug(f"Added {fg_type} for {requirement['type']}: score improved to {new_score:.3f}")
                                break
                            else:
                                logger.debug(f"FG addition decreased score ({new_score:.3f} vs {best_score:.3f}), reverting")
                        except Exception as e:
                            logger.debug(f"Could not score new molecule: {e}")
                            mol = new_mol
                            additions_made += 1
                            break
                    else:
                        mol = new_mol
                        additions_made += 1
                        logger.debug(f"Added {fg_type} for {requirement['type']} requirement")
                        break
        
        logger.info(f"Added {additions_made} targeted functional groups")
        return best_mol
    
    def _find_growth_sites_near_position(self, mol: Chem.Mol, 
                                        target_position: np.ndarray,
                                        max_sites: int = 3) -> List[int]:
        """Find growth sites closest to target position"""
        if not mol.GetNumConformers():
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                return []
        
        try:
            conformer = mol.GetConformer()
        except:
            return []
        
        growth_sites = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [6, 7, 8]:
                current_valence = atom.GetTotalValence()
                max_valence = AttachmentPointManager._get_max_valence(atom.GetAtomicNum())
                
                if current_valence < max_valence:
                    try:
                        pos = conformer.GetAtomPosition(atom.GetIdx())
                        atom_pos = np.array([pos.x, pos.y, pos.z])
                        distance = np.linalg.norm(atom_pos - target_position)
                        
                        growth_sites.append({
                            'idx': atom.GetIdx(),
                            'distance': distance,
                            'atom_type': atom.GetSymbol()
                        })
                    except Exception as e:
                        logger.debug(f"Error getting atom position: {e}")
                        continue
        
        growth_sites.sort(key=lambda x: x['distance'])
        return [site['idx'] for site in growth_sites[:max_sites]]
    
    def _add_fg_at_sites(self, mol: Chem.Mol, fg_type: str, 
                        sites: List[int], interaction_type: str) -> Optional[Chem.Mol]:
        """Attempt to add functional group at specified sites"""
        available_groups = self.fg_library.get(fg_type, [])
        if not available_groups:
            return mol
        
        for site_idx in sites:
            for fg_info in available_groups:
                fg_smiles = fg_info['smiles']
                fg_mol = Chem.MolFromSmiles(fg_smiles)
                
                if fg_mol is None:
                    continue
                
                try:
                    combined = Chem.CombineMols(mol, fg_mol)
                    rw_mol = Chem.RWMol(combined)
                    
                    mol_size = mol.GetNumAtoms()
                    rw_mol.AddBond(site_idx, mol_size, Chem.BondType.SINGLE)
                    
                    result = rw_mol.GetMol()
                    Chem.SanitizeMol(result)
                    
                    self.addition_stats[fg_info['name']] += 1
                    return result
                    
                except Exception as e:
                    logger.debug(f"Failed to add {fg_info['name']} at site {site_idx}: {e}")
                    continue
        
        return mol


class MolecularElaborationEngine:
    """Post-assembly elaboration engine for strategic molecule growth"""
    
    def __init__(self, fragment_library: FragmentLibrary, pocket_analysis: PocketAnalysis,
                 constraints: MoleculeConstraints, validator: ValidationEngine, scorer=None):
        self.fragment_lib = fragment_library
        self.pocket = pocket_analysis
        self.constraints = constraints
        self.validator = validator
        self.scorer = scorer
        self.ap_manager = AttachmentPointManager()
        
        self.elaboration_stats = {
            'molecules_elaborated': 0,
            'elaboration_attempts': 0,
            'successful_additions': 0,
            'score_improvements': 0,
            'property_violations': 0,
            'avg_additions_per_mol': [],
            'avg_score_delta': []
        }
    
    def _identify_growth_sites(self, mol: Chem.Mol, target_interactions: List[str]) -> List[GrowthSite]:
        """Identify strategic growth sites on the molecule"""
        growth_sites = []
        
        # Ensure 3D conformation exists
        has_conformer = mol.GetNumConformers() > 0
        if not has_conformer:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                has_conformer = True
            except:
                has_conformer = False
        
        # Get pocket hotspot positions for guidance
        hotspot_positions = []
        hotspot_types = []
        if self.pocket.hotspots:
            for hotspot in self.pocket.hotspots:
                if hotspot.interaction_type in target_interactions:
                    hotspot_positions.append(hotspot.position)
                    hotspot_types.append(hotspot.interaction_type)
        
        # Analyze each atom for growth potential
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            
            # Only consider C, N, O atoms for growth
            if atomic_num not in [6, 7, 8]:
                continue
            
            # Check available valence
            current_valence = atom.GetTotalValence()
            max_valence = AttachmentPointManager._get_max_valence(atomic_num)
            available_valence = max_valence - current_valence
            
            if available_valence <= 0:
                continue
            
            # Get chemical environment
            env = self.ap_manager.analyze_chemical_environment(mol, atom_idx)
            
            # Calculate environment score (prefer less crowded, accessible sites)
            environment_score = 100.0
            environment_score -= env.num_neighbors * 15  # Penalty for crowding
            if env.is_in_ring:
                environment_score -= 20  # Prefer growing outside rings
            if env.is_aromatic:
                environment_score += 10  # Aromatic sites are good for substitution
            environment_score += available_valence * 20  # Reward high valence availability
            
            # Calculate distance to nearest hotspot
            distance_to_target = float('inf')
            position_3d = None
            
            if has_conformer and hotspot_positions:
                try:
                    conformer = mol.GetConformer()
                    pos = conformer.GetAtomPosition(atom_idx)
                    atom_pos = np.array([pos.x, pos.y, pos.z])
                    position_3d = atom_pos
                    
                    for hotspot_pos in hotspot_positions:
                        dist = np.linalg.norm(atom_pos - hotspot_pos)
                        if dist < distance_to_target:
                            distance_to_target = dist
                except:
                    pass
            
            growth_sites.append(GrowthSite(
                atom_idx=atom_idx,
                atom_type=atom.GetSymbol(),
                hybridization=env.hybridization,
                available_valence=available_valence,
                is_aromatic=env.is_aromatic,
                is_in_ring=env.is_in_ring,
                distance_to_target=distance_to_target,
                environment_score=environment_score,
                position_3d=position_3d
            ))
        
        # Sort by composite score: environment quality + proximity to targets
        for site in growth_sites:
            if site.distance_to_target < float('inf'):
                # Closer to hotspot = higher priority
                proximity_score = max(0, 100 - site.distance_to_target * 10)
            else:
                proximity_score = 0
            site.environment_score = site.environment_score * 0.6 + proximity_score * 0.4
        
        growth_sites.sort(key=lambda x: x.environment_score, reverse=True)
        
        return growth_sites
    
    def _select_fragment_for_site(self, site: GrowthSite, target_interactions: List[str]) -> Optional[FragmentInfo]:
        """Select appropriate fragment for a growth site"""
        candidates = []
        
        # Get fragments matching target interactions
        for interaction in target_interactions:
            frags = self.fragment_lib.get_fragments_for_interaction(interaction)
            candidates.extend(frags)
        
        if not candidates:
            # Fall back to substituents
            candidates = self.fragment_lib.fragments.get('substituents', [])
        
        if not candidates:
            return None
        
        # Filter by size - prefer small to medium fragments for elaboration
        candidates = [f for f in candidates if f.mw < 200]
        if not candidates:
            return None
        
        # Prefer fragments with compatible chemistry
        scored_candidates = []
        for frag in candidates:
            score = 50.0
            
            # Prefer fragments that complement the site
            if site.is_aromatic and 'aromatic' in frag.interaction_types:
                score += 20
            if not site.is_aromatic and 'aliphatic' in frag.interaction_types:
                score += 15
            
            # Size penalty - prefer smaller additions
            score -= (frag.mw / 20.0)
            
            # Diversity bonus
            if frag.ring_count > 0 and not site.is_in_ring:
                score += 10
            
            scored_candidates.append((frag, score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top candidates with some randomness
        top_n = min(5, len(scored_candidates))
        return random.choice([c[0] for c in scored_candidates[:top_n]])
    
    def _try_fragment_addition(self, mol: Chem.Mol, site: GrowthSite, 
                               fragment: FragmentInfo) -> Optional[Chem.Mol]:
        """Attempt to add fragment at growth site"""
        try:
            frag_mol = Chem.MolFromSmiles(fragment.smiles)
            if frag_mol is None:
                return None
            
            # Use chemistry-aware attachment if fragment has attachment points
            frag_aps = self.ap_manager.find_attachment_points(frag_mol)
            
            if frag_aps:
                # Try attachment point method
                # Add temporary attachment point to molecule at growth site
                rw_mol = Chem.RWMol(mol)
                dummy_idx = rw_mol.AddAtom(Chem.Atom(0))  # Add dummy atom
                rw_mol.GetAtomWithIdx(dummy_idx).SetAtomMapNum(1)
                rw_mol.AddBond(site.atom_idx, dummy_idx, Chem.BondType.SINGLE)
                
                temp_mol = rw_mol.GetMol()
                
                # Try to connect
                result = self.ap_manager._attach_fragments_by_points(temp_mol, frag_mol)
                if result:
                    return result
            
            # Fall back to direct attachment
            combined = Chem.CombineMols(mol, frag_mol)
            rw_mol = Chem.RWMol(combined)
            
            mol_size = mol.GetNumAtoms()
            # Connect first atom of fragment to growth site
            rw_mol.AddBond(site.atom_idx, mol_size, Chem.BondType.SINGLE)
            
            result = rw_mol.GetMol()
            Chem.SanitizeMol(result)
            
            return result
            
        except Exception as e:
            logger.debug(f"Fragment addition failed: {e}")
            return None
    
    def _check_druglikeness(self, mol: Chem.Mol) -> Tuple[bool, Dict]:
        """Quick druglikeness check"""
        try:
            props = {
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'mw': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'qed': QED.qed(mol)
            }
            
            is_valid, reason = self.validator.validate_with_reason(mol, props)
            return is_valid, props
            
        except Exception as e:
            logger.debug(f"Property calculation failed: {e}")
            return False, {}
    
    def elaborate_molecule(self, mol: Chem.Mol, target_interactions: List[str],
                          max_additions: int = 5, min_score_improvement: float = 0.01) -> Chem.Mol:
        """
        Elaborate a molecule through strategic growth while maintaining drug-likeness
        
        Args:
            mol: Initial molecule
            target_interactions: Target interaction types for the pocket
            max_additions: Maximum number of fragments to add
            min_score_improvement: Minimum score improvement to accept an addition
            
        Returns:
            Elaborated molecule (or original if no improvements)
        """
        if mol is None:
            return None
        
        self.elaboration_stats['molecules_elaborated'] += 1
        
        # Get initial score
        initial_score = 0.0
        if self.scorer:
            try:
                score_obj = self.scorer.calculate_comprehensive_score(mol)
                initial_score = score_obj.total_score
                logger.debug(f"Initial molecule score: {initial_score:.3f}")
            except:
                pass
        
        best_mol = mol
        best_score = initial_score
        additions_made = 0
        
        for iteration in range(max_additions):
            self.elaboration_stats['elaboration_attempts'] += 1
            
            # Identify growth sites
            growth_sites = self._identify_growth_sites(best_mol, target_interactions)
            
            if not growth_sites:
                logger.debug("No suitable growth sites found")
                break
            
            # Try top growth sites
            success = False
            for site in growth_sites[:3]:  # Try top 3 sites
                # Select fragment
                fragment = self._select_fragment_for_site(site, target_interactions)
                if fragment is None:
                    continue
                
                # Attempt addition
                new_mol = self._try_fragment_addition(best_mol, site, fragment)
                if new_mol is None:
                    continue
                
                # Sanitize
                try:
                    Chem.SanitizeMol(new_mol)
                except:
                    continue
                
                # Check druglikeness
                is_druglike, props = self._check_druglikeness(new_mol)
                if not is_druglike:
                    self.elaboration_stats['property_violations'] += 1
                    logger.debug(f"Addition violated constraints: {props.get('mw', 0):.1f} Da")
                    continue
                
                # Score new molecule
                new_score = best_score  # Default if scorer unavailable
                if self.scorer:
                    try:
                        new_score_obj = self.scorer.calculate_comprehensive_score(new_mol)
                        new_score = new_score_obj.total_score
                    except:
                        new_score = best_score
                
                # Accept if improvement
                score_delta = new_score - best_score
                if score_delta >= min_score_improvement:
                    best_mol = new_mol
                    best_score = new_score
                    additions_made += 1
                    success = True
                    self.elaboration_stats['successful_additions'] += 1
                    self.elaboration_stats['score_improvements'] += 1
                    
                    logger.debug(f"Elaboration {iteration+1}: Added {fragment.smiles[:20]}... "
                               f"(score: {best_score:.3f}, Δ={score_delta:.3f}, "
                               f"MW={props['mw']:.1f}, atoms={props['heavy_atoms']})")
                    break
                else:
                    logger.debug(f"Addition rejected: score change {score_delta:.3f} < threshold")
            
            if not success:
                logger.debug(f"No beneficial addition found in iteration {iteration+1}")
                break
        
        # Record statistics
        self.elaboration_stats['avg_additions_per_mol'].append(additions_made)
        if self.scorer:
            score_delta = best_score - initial_score
            self.elaboration_stats['avg_score_delta'].append(score_delta)
        
        if additions_made > 0:
            logger.info(f"Elaboration complete: {additions_made} additions, "
                       f"score {initial_score:.3f} → {best_score:.3f}")
        else:
            logger.debug("No elaborations accepted")
        
        return best_mol
    
    def get_statistics(self) -> Dict:
        """Get elaboration statistics"""
        stats = self.elaboration_stats.copy()
        
        if stats['avg_additions_per_mol']:
            stats['mean_additions'] = float(np.mean(stats['avg_additions_per_mol']))
        else:
            stats['mean_additions'] = 0.0
        
        if stats['avg_score_delta']:
            stats['mean_score_improvement'] = float(np.mean(stats['avg_score_delta']))
        else:
            stats['mean_score_improvement'] = 0.0
        
        if stats['elaboration_attempts'] > 0:
            stats['success_rate'] = stats['successful_additions'] / stats['elaboration_attempts']
        else:
            stats['success_rate'] = 0.0
        
        return stats


class StructureGuidedAssembly:
    """Structure-guided molecular assembly with chemistry-aware attachment point handling and 3D guidance"""

    def __init__(self, fragment_library: FragmentLibrary, pocket_analysis: PocketAnalysis,
                 config: LigandForgeConfig, random_seed: Optional[int] = None, scorer=None):
        self.fragment_lib = fragment_library
        self.pocket = pocket_analysis
        self.config = config
        self.scorer = scorer
        self.ap_manager = AttachmentPointManager()
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed}")
        
        self.constraints = MoleculeConstraints(
            min_heavy_atoms=getattr(config, 'min_heavy_atoms', 10),
            max_heavy_atoms=max(50, getattr(config, 'max_heavy_atoms', 30)),
            min_molecular_weight=getattr(config, 'min_molecular_weight', 150),
            max_molecular_weight=max(700, getattr(config, 'max_molecular_weight', 500)),
            max_rotatable_bonds=max(15, getattr(config, 'max_rotatable_bonds', 10)),
            max_rings=max(8, getattr(config, 'max_rings', 5))
        )
        self.validator = ValidationEngine(self.constraints, fragment_library)
        
        self.reaction_templates = self._load_reaction_templates()
        self.pharmacophore_fragment_map = self._build_pharmacophore_fragment_map()
        self.functional_group_library = self._build_functional_group_library()
        self.pharmacophore_functional_group_map = self._build_pharmacophore_functional_group_map()
        
        self.pocket_fg_manager = PocketAwareFunctionalGroupManager(
            pocket_analysis,
            self.functional_group_library,
            self.pharmacophore_functional_group_map,
            scorer=scorer
        )
        
        # Initialize elaboration engine
        self.elaboration_engine = MolecularElaborationEngine(
            fragment_library,
            pocket_analysis,
            self.constraints,
            self.validator,
            scorer=scorer
        )
        
        self._cache_pocket_data()
        
        self.generation_stats = {
            'total_attempts': 0,
            'successful_assemblies': 0,
            'validation_failures': 0,
            'strategy_counts': defaultdict(int),
            'functional_group_additions': defaultdict(int),
            'failure_reasons': defaultdict(int),
            'pocket_satisfaction_scores': [],
            'molecule_sizes': [],
            'attachment_point_usage': defaultdict(int),
            'scorer_improvements': 0,
            'hotspot_matches': defaultdict(int),
            'chemistry_aware_matches': 0,
            'compatibility_scores': [],
            '3d_guided_assemblies': 0,
            '3d_optimization_successes': 0,
            'elaborations_performed': 0,
            'elaboration_improvements': 0
        }
    
    def _cache_pocket_data(self):
        """Cache important pocket data for quick access during assembly"""
        self.pocket_hotspot_positions = np.array([h.position for h in self.pocket.hotspots]) if self.pocket.hotspots else np.array([])
        self.pocket_hotspot_types = [h.interaction_type for h in self.pocket.hotspots] if self.pocket.hotspots else []
        self.pocket_hotspot_strengths = [h.strength for h in self.pocket.hotspots] if self.pocket.hotspots else []
        
        self.high_priority_hotspots = [
            h for h in self.pocket.hotspots 
            if h.strength > 0.7
        ] if self.pocket.hotspots else []
        
        self.displaceable_waters = [
            w for w in self.pocket.water_sites 
            if w.replaceability_score > 0.6
        ] if self.pocket.water_sites else []
        
        logger.info(f"Cached {len(self.high_priority_hotspots)} high-priority hotspots and {len(self.displaceable_waters)} displaceable waters")

    def _build_functional_group_library(self) -> Dict[str, List[Dict[str, str]]]:
        """Build comprehensive functional group library"""
        return {
            'halogen': [
                {'name': 'fluorine', 'smiles': 'F', 'attachment': '[*]F'},
                {'name': 'chlorine', 'smiles': 'Cl', 'attachment': '[*]Cl'},
                {'name': 'bromine', 'smiles': 'Br', 'attachment': '[*]Br'},
                {'name': 'iodine', 'smiles': 'I', 'attachment': '[*]I'},
                {'name': 'trifluoromethyl', 'smiles': 'C(F)(F)F', 'attachment': '[*]C(F)(F)F'},
            ],
            'ether': [
                {'name': 'methoxy', 'smiles': 'OC', 'attachment': '[*]OC'},
                {'name': 'ethoxy', 'smiles': 'OCC', 'attachment': '[*]OCC'},
                {'name': 'propoxy', 'smiles': 'OCCC', 'attachment': '[*]OCCC'},
                {'name': 'phenoxy', 'smiles': 'Oc1ccccc1', 'attachment': '[*]Oc1ccccc1'},
                {'name': 'benzyloxy', 'smiles': 'OCc1ccccc1', 'attachment': '[*]OCc1ccccc1'},
            ],
            'amide': [
                {'name': 'acetamide', 'smiles': 'NC(=O)C', 'attachment': '[*]NC(=O)C'},
                {'name': 'benzamide', 'smiles': 'NC(=O)c1ccccc1', 'attachment': '[*]NC(=O)c1ccccc1'},
                {'name': 'formamide', 'smiles': 'NC=O', 'attachment': '[*]NC=O'},
                {'name': 'carboxamide', 'smiles': 'C(=O)N', 'attachment': '[*]C(=O)N'},
                {'name': 'sulfonamide', 'smiles': 'NS(=O)(=O)C', 'attachment': '[*]NS(=O)(=O)C'},
            ],
            'amine': [
                {'name': 'methyl_amine', 'smiles': 'NC', 'attachment': '[*]NC'},
                {'name': 'dimethyl_amine', 'smiles': 'N(C)C', 'attachment': '[*]N(C)C'},
                {'name': 'tertiary_amine', 'smiles': 'N(C)(C)C', 'attachment': '[*]N(C)(C)C'},
                {'name': 'aniline', 'smiles': 'Nc1ccccc1', 'attachment': '[*]Nc1ccccc1'},
                {'name': 'piperidine', 'smiles': 'N1CCCCC1', 'attachment': '[*]N1CCCCC1'},
                {'name': 'morpholine', 'smiles': 'N1CCOCC1', 'attachment': '[*]N1CCOCC1'},
            ],
            'carboxylic_acid': [
                {'name': 'carboxyl', 'smiles': 'C(=O)O', 'attachment': '[*]C(=O)O'},
                {'name': 'acetic_acid', 'smiles': 'CC(=O)O', 'attachment': '[*]CC(=O)O'},
                {'name': 'benzoic_acid', 'smiles': 'c1ccc(cc1)C(=O)O', 'attachment': '[*]c1ccc(cc1)C(=O)O'},
            ],
            'ester': [
                {'name': 'methyl_ester', 'smiles': 'C(=O)OC', 'attachment': '[*]C(=O)OC'},
                {'name': 'ethyl_ester', 'smiles': 'C(=O)OCC', 'attachment': '[*]C(=O)OCC'},
                {'name': 'phenyl_ester', 'smiles': 'C(=O)Oc1ccccc1', 'attachment': '[*]C(=O)Oc1ccccc1'},
                {'name': 'acetate', 'smiles': 'OC(=O)C', 'attachment': '[*]OC(=O)C'},
            ],
            'alkyl': [
                {'name': 'methyl', 'smiles': 'C', 'attachment': '[*]C'},
                {'name': 'ethyl', 'smiles': 'CC', 'attachment': '[*]CC'},
                {'name': 'propyl', 'smiles': 'CCC', 'attachment': '[*]CCC'},
                {'name': 'isopropyl', 'smiles': 'C(C)C', 'attachment': '[*]C(C)C'},
                {'name': 'butyl', 'smiles': 'CCCC', 'attachment': '[*]CCCC'},
                {'name': 'tert_butyl', 'smiles': 'C(C)(C)C', 'attachment': '[*]C(C)(C)C'},
            ],
            'hydroxyl': [
                {'name': 'hydroxyl', 'smiles': 'O', 'attachment': '[*]O'},
                {'name': 'phenol', 'smiles': 'Oc1ccccc1', 'attachment': '[*]Oc1ccccc1'},
            ],
            'thiol': [
                {'name': 'thiol', 'smiles': 'S', 'attachment': '[*]S'},
                {'name': 'methylthio', 'smiles': 'SC', 'attachment': '[*]SC'},
            ],
            'ketone': [
                {'name': 'acetyl', 'smiles': 'C(=O)C', 'attachment': '[*]C(=O)C'},
                {'name': 'benzoyl', 'smiles': 'C(=O)c1ccccc1', 'attachment': '[*]C(=O)c1ccccc1'},
            ],
            'aldehyde': [
                {'name': 'formyl', 'smiles': 'C=O', 'attachment': '[*]C=O'},
            ],
            'nitro': [
                {'name': 'nitro', 'smiles': '[N+](=O)[O-]', 'attachment': '[*][N+](=O)[O-]'},
            ],
            'nitrile': [
                {'name': 'cyano', 'smiles': 'C#N', 'attachment': '[*]C#N'},
            ],
            'sulfone': [
                {'name': 'methylsulfone', 'smiles': 'S(=O)(=O)C', 'attachment': '[*]S(=O)(=O)C'},
            ],
            'aromatic': [
                {'name': 'phenyl', 'smiles': 'c1ccccc1', 'attachment': '[*]c1ccccc1'},
                {'name': 'pyridyl', 'smiles': 'c1ccncc1', 'attachment': '[*]c1ccncc1'},
                {'name': 'pyrimidyl', 'smiles': 'c1cncnc1', 'attachment': '[*]c1cncnc1'},
                {'name': 'benzyl', 'smiles': 'Cc1ccccc1', 'attachment': '[*]Cc1ccccc1'},
            ]
        }

    def _build_pharmacophore_functional_group_map(self) -> Dict[str, List[str]]:
        """Map pharmacophore types to suitable functional groups"""
        return {
            'hydrophobic': ['alkyl', 'aromatic', 'halogen'],
            'aromatic': ['aromatic', 'ester', 'amide'],
            'hbond_donor': ['amine', 'amide', 'hydroxyl', 'thiol', 'carboxylic_acid'],
            'hbond_acceptor': ['ether', 'ester', 'amide', 'ketone', 'aldehyde', 'nitrile'],
            'hbd': ['amine', 'amide', 'hydroxyl', 'thiol', 'carboxylic_acid'],
            'hba': ['ether', 'ester', 'amide', 'ketone', 'aldehyde', 'nitrile'],
            'positive': ['amine'],
            'negative': ['carboxylic_acid', 'sulfone'],
            'electrostatic': ['amine', 'carboxylic_acid'],
            'polar': ['hydroxyl', 'ether', 'amide', 'ester', 'amine'],
            'metal_coordination': ['carboxylic_acid', 'amine', 'hydroxyl'],
            'metal': ['carboxylic_acid', 'amine', 'hydroxyl'],
            'pi_stacking': ['aromatic', 'nitrile'],
            'halogen_bond': ['halogen', 'nitro'],
            'basic': ['amine'],
            'acidic': ['carboxylic_acid'],
            'charged': ['amine', 'carboxylic_acid'],
            'donor': ['amine', 'amide', 'hydroxyl', 'thiol'],
            'acceptor': ['ether', 'ester', 'ketone', 'nitrile'],
            'aliphatic': ['alkyl', 'ether'],
            'hydrogen_bond': ['amine', 'amide', 'hydroxyl', 'carboxylic_acid', 'ether']
        }

    def _sanitize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Sanitize molecule and remove any remaining dummy atoms"""
        if mol is None:
            return None
            
        try:
            rw_mol = Chem.RWMol(mol)
            atoms_to_remove = []
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atoms_to_remove.append(atom.GetIdx())
            
            for atom_idx in sorted(atoms_to_remove, reverse=True):
                rw_mol.RemoveAtom(atom_idx)
            
            mol = rw_mol.GetMol()
            
            if mol.GetNumAtoms() == 0:
                return None
            
            Chem.SanitizeMol(mol)
            
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                fragments = smiles.split('.')
                largest_fragment = max(fragments, key=len)
                mol = Chem.MolFromSmiles(largest_fragment)
                if mol is not None:
                    Chem.SanitizeMol(mol)
            
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if any(char in smiles for char in ['A', '*', '[*]']):
                    return None
                    
            return mol
            
        except Exception as e:
            logger.debug(f"Error sanitizing molecule: {e}")
            return None

    # ========== 3D-GUIDED ASSEMBLY METHODS ==========
    
    def _attach_fragment_at_position(self, mol: Chem.Mol, fragment_smiles: str, 
                                    target_position: np.ndarray,
                                    distance_threshold: float = 5.0) -> Optional[Chem.Mol]:
        """Attach fragment and optimize to place it near target 3D position"""
        try:
            fragment_mol = Chem.MolFromSmiles(fragment_smiles)
            if fragment_mol is None:
                return None
            
            # Attach fragment using chemistry-aware method
            new_mol = self._attach_fragments_by_points(mol, fragment_mol)
            if new_mol is None:
                return None
            
            # Ensure the molecule has a 3D conformer
            if not new_mol.GetNumConformers():
                try:
                    # Try multiple embedding attempts
                    for seed in range(3):
                        result = AllChem.EmbedMolecule(new_mol, randomSeed=42 + seed, useRandomCoords=True)
                        if result == 0:
                            break
                    
                    if result != 0:
                        # Embedding failed, return without 3D optimization
                        logger.debug("3D embedding failed, returning 2D structure")
                        return new_mol
                    
                    # Optimize geometry
                    try:
                        AllChem.MMFFOptimizeMolecule(new_mol, maxIters=200)
                        self.generation_stats['3d_optimization_successes'] += 1
                    except:
                        try:
                            AllChem.UFFOptimizeMolecule(new_mol, maxIters=200)
                            self.generation_stats['3d_optimization_successes'] += 1
                        except:
                            logger.debug("Geometry optimization failed")
                            
                except Exception as e:
                    logger.debug(f"3D conformer generation error: {e}")
                    return new_mol
            
            # Check if newly added atoms are near target position
            conformer = new_mol.GetConformer()
            original_num_atoms = mol.GetNumAtoms()
            
            # Get positions of new atoms (those added from fragment)
            new_atom_positions = []
            for atom_idx in range(original_num_atoms, new_mol.GetNumAtoms()):
                try:
                    pos = conformer.GetAtomPosition(atom_idx)
                    new_atom_positions.append(np.array([pos.x, pos.y, pos.z]))
                except:
                    continue
            
            if new_atom_positions:
                # Calculate center of new atoms
                new_center = np.mean(new_atom_positions, axis=0)
                distance_to_target = np.linalg.norm(new_center - target_position)
                
                logger.debug(f"3D-guided: New fragment center is {distance_to_target:.2f}Ã… from target")
                
                if distance_to_target <= distance_threshold:
                    self.generation_stats['3d_guided_assemblies'] += 1
                    logger.debug("3D-guided assembly successful - fragment near target position")
                else:
                    logger.debug(f"Fragment not optimally placed (>{distance_threshold}Ã… from target)")
            
            return new_mol
            
        except Exception as e:
            logger.debug(f"3D-guided attachment error: {e}")
            # Fall back to standard attachment
            return self._attach_fragments_by_points(mol, Chem.MolFromSmiles(fragment_smiles))
    
    def _grow_toward_position(self, mol: Chem.Mol, target_position: np.ndarray,
                             interaction_type: str, max_attempts: int = 5) -> Optional[Chem.Mol]:
        """Grow molecule toward a specific 3D position in the pocket"""
        suitable_fragments = self.fragment_lib.get_fragments_for_interaction(interaction_type)
        if not suitable_fragments:
            return mol
        
        best_mol = mol
        best_distance = float('inf')
        
        for attempt in range(min(max_attempts, len(suitable_fragments))):
            fragment = random.choice(suitable_fragments)
            
            # Try 3D-guided attachment
            test_mol = self._attach_fragment_at_position(
                mol, fragment.smiles, target_position, distance_threshold=6.0
            )
            
            if test_mol is not None and test_mol != mol:
                # Check if this placement is better
                if test_mol.GetNumConformers():
                    try:
                        conformer = test_mol.GetConformer()
                        original_size = mol.GetNumAtoms()
                        
                        # Get center of new atoms
                        new_positions = []
                        for idx in range(original_size, test_mol.GetNumAtoms()):
                            pos = conformer.GetAtomPosition(idx)
                            new_positions.append(np.array([pos.x, pos.y, pos.z]))
                        
                        if new_positions:
                            center = np.mean(new_positions, axis=0)
                            distance = np.linalg.norm(center - target_position)
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_mol = test_mol
                                logger.debug(f"Better placement found: {distance:.2f}Ã… from target")
                    except:
                        pass
        
        return best_mol if best_mol != mol else mol

    # ========== ATTACHMENT METHODS ==========
    
    def _attach_fragments_by_points(self, mol1: Chem.Mol, mol2: Chem.Mol,
                                    prefer_matching: bool = True) -> Optional[Chem.Mol]:
        """Attach two fragments using chemistry-aware matching"""
        match = self.ap_manager.match_attachment_points(mol1, mol2, prefer_matching)
        
        if match is None:
            logger.debug("No attachment points found for connection")
            return None
        
        ap1, ap2 = match
        self.generation_stats['attachment_point_usage'][f'{ap1.map_num}-{ap2.map_num}'] += 1
        self.generation_stats['chemistry_aware_matches'] += 1
        
        result = self.ap_manager.connect_at_attachment_points(mol1, mol2, ap1, ap2)
        return result
    
    def _link_fragments(self, core1_smiles: str, core2_smiles: str,
                       linker_smiles: str) -> Optional[Chem.Mol]:
        """Link two cores with a linker using chemistry-aware attachment points"""
        try:
            mol1 = Chem.MolFromSmiles(core1_smiles)
            mol2 = Chem.MolFromSmiles(core2_smiles)
            linker_mol = Chem.MolFromSmiles(linker_smiles)
            
            if None in [mol1, mol2, linker_mol]:
                return None
            
            linker_aps = self.ap_manager.find_attachment_points(linker_mol)
            if len(linker_aps) < 2:
                logger.debug(f"Linker has insufficient attachment points: {len(linker_aps)}")
                return None
            
            intermediate = self._attach_fragments_by_points(mol1, linker_mol, prefer_matching=False)
            if intermediate is None:
                return None
            
            final = self._attach_fragments_by_points(intermediate, mol2, prefer_matching=False)
            return final
            
        except Exception as e:
            logger.debug(f"Linking failed: {e}")
            return None
    
    def _attach_fragment(self, mol: Chem.Mol, fragment_smiles: str,
                        target_map_num: Optional[int] = None) -> Optional[Chem.Mol]:
        """Attach a fragment to a molecule using chemistry-aware matching"""
        try:
            fragment_mol = Chem.MolFromSmiles(fragment_smiles)
            if fragment_mol is None:
                return None
            
            mol_aps = self.ap_manager.find_attachment_points(mol)
            frag_aps = self.ap_manager.find_attachment_points(fragment_mol)
            
            if not mol_aps or not frag_aps:
                logger.debug("Missing attachment points for fragment attachment")
                return None
            
            if target_map_num is not None:
                target_ap = [ap for ap in mol_aps if ap.map_num == target_map_num]
                if target_ap:
                    mol_ap = target_ap[0]
                else:
                    mol_ap = mol_aps[0]
            else:
                mol_ap = mol_aps[0]
            
            frag_ap = frag_aps[0]
            
            result = self.ap_manager.connect_at_attachment_points(mol, fragment_mol, mol_ap, frag_ap)
            return result
            
        except Exception as e:
            logger.debug(f"Fragment attachment failed: {e}")
            return None
    
    def _grow_at_attachment_point(self, mol: Chem.Mol, fragment: FragmentInfo,
                                  map_num: Optional[int] = None) -> Optional[Chem.Mol]:
        """Grow molecule by adding fragment at attachment point"""
        return self._attach_fragment(mol, fragment.smiles, map_num)
    
    def _score_and_select_fragment(self, current_mol: Chem.Mol, 
                                  fragment_candidates: List[FragmentInfo],
                                  target_interaction: str) -> Optional[FragmentInfo]:
        """Score fragment candidates and select the best one"""
        if not self.scorer or not fragment_candidates:
            return random.choice(fragment_candidates) if fragment_candidates else None
        
        best_fragment = None
        best_improvement = -float('inf')
        current_score = 0.0
        
        try:
            current_score_obj = self.scorer.calculate_comprehensive_score(current_mol)
            current_score = current_score_obj.total_score
        except:
            pass
        
        test_candidates = random.sample(fragment_candidates, min(5, len(fragment_candidates)))
        
        for fragment in test_candidates:
            try:
                test_mol = self._attach_fragments_by_points(current_mol, Chem.MolFromSmiles(fragment.smiles))
                if test_mol is None:
                    continue
                
                test_mol = self._sanitize_molecule(test_mol)
                if test_mol is None:
                    continue
                
                test_score_obj = self.scorer.calculate_comprehensive_score(test_mol)
                improvement = test_score_obj.total_score - current_score
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_fragment = fragment
                    
            except Exception as e:
                logger.debug(f"Error scoring fragment: {e}")
                continue
        
        if best_improvement > 0:
            self.generation_stats['scorer_improvements'] += 1
            logger.debug(f"Selected fragment with score improvement: +{best_improvement:.3f}")
        
        return best_fragment if best_fragment else (random.choice(fragment_candidates) if fragment_candidates else None)

    # ========== ASSEMBLY STRATEGIES ==========
    
    def _load_reaction_templates(self) -> Dict[str, rdChemReactions.ChemicalReaction]:
        """Load reaction templates"""
        reactions: Dict[str, rdChemReactions.ChemicalReaction] = {}
        try:
            reactions["amide"] = rdChemReactions.ReactionFromSmarts(
                "[C:1](=[O:2])[OH:3].[N:4][H:5]>>[C:1](=[O:2])[N:4].[OH2:3]"
            )
            reactions["esterification"] = rdChemReactions.ReactionFromSmarts(
                "[C:1](=[O:2])[OH:3].[O:4][H:5]>>[C:1](=[O:2])[O:4].[OH2:3]"
            )
            reactions["ether"] = rdChemReactions.ReactionFromSmarts(
                "[C:1][OH:2].[C:3][Br,Cl,I:4]>>[C:1][O:2][C:3].[Br,Cl,I:4][H]"
            )
            reactions["n_alkylation"] = rdChemReactions.ReactionFromSmarts(
                "[N:1][H:2].[C:3][Br,Cl,I:4]>>[N:1][C:3].[Br,Cl,I:4][H:2]"
            )
            reactions["reductive_amination"] = rdChemReactions.ReactionFromSmarts(
                "[C:1]=[O:2].[N:3][H:4]>>[C:1][N:3].[O:2][H:4]"
            )
            logger.info(f"Loaded {len(reactions)} reaction templates")
        except Exception as e:
            logger.error(f"Error loading reaction templates: {e}")
        return reactions

    def _build_pharmacophore_fragment_map(self) -> Dict[str, List[str]]:
        """Build mapping between pharmacophore types and fragment interaction types"""
        return {
            'hydrophobic': ['aromatic', 'hydrophobic', 'aliphatic'],
            'aromatic': ['aromatic', 'pi_stacking', 'hydrophobic'],
            'hbond_donor': ['donor', 'hydrogen_bond', 'polar'],
            'hbond_acceptor': ['acceptor', 'hydrogen_bond', 'polar'],
            'hbd': ['donor', 'hydrogen_bond', 'polar'],
            'hba': ['acceptor', 'hydrogen_bond', 'polar'],
            'positive': ['basic', 'cation', 'charged'],
            'negative': ['acidic', 'anion', 'charged'],
            'electrostatic': ['charged', 'polar'],
            'polar': ['polar', 'hydrogen_bond', 'donor', 'acceptor'],
            'metal_coordination': ['metal_binding', 'coordination', 'polar'],
            'metal': ['metal_binding', 'coordination', 'polar'],
            'pi_stacking': ['aromatic', 'pi_stacking'],
            'halogen_bond': ['halogen', 'polar']
        }

    def _choose_assembly_strategy(self, target_interactions: List[str]) -> str:
        """Strategy selection with pocket awareness"""
        num_hotspots = len(self.pocket.hotspots)
        pocket_volume = self.pocket.volume
        interaction_diversity = len(set(h.interaction_type for h in self.pocket.hotspots)) if self.pocket.hotspots else 0
        num_pharmacophore = len(self.pocket.pharmacophore_points)
        
        druggability = self.pocket.druggability_score

        candidates: List[str] = []
        
        candidates += ['mandatory_linking_assembly'] * 8
        candidates += ['multi_core_assembly'] * 6
        candidates += ['scaffold_decoration'] * 5
        candidates += ['fragment_growing'] * 4
        
        if num_pharmacophore >= 3:
            weight = 3 if druggability > 0.7 else 2
            candidates += ['pharmacophore_guided'] * weight
            
        if num_hotspots >= 3 and interaction_diversity >= 2:
            candidates += ['linking'] * 2
            candidates += ['hotspot_guided']
            
        if pocket_volume > 600:
            candidates += ['conjugation_extension']
        
        if len(self.high_priority_hotspots) >= 3:
            candidates += ['hotspot_guided'] * 2
        
        if random.random() < 0.1:
            candidates += ['sp3_enrichment', 'ring_closure']
        
        return random.choice(candidates) if candidates else 'mandatory_linking_assembly'

    def _mandatory_linking_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Assembly that MUST use linkers between multiple cores"""
        cores = self.fragment_lib.fragments['cores']
        linkers = self.fragment_lib.fragments.get('linkers', [])
        
        if len(cores) < 2 or not linkers:
            return None
        
        num_cores = min(4, max(3, len(cores)))
        selected_cores = random.sample(cores, num_cores)
        
        mol = Chem.MolFromSmiles(selected_cores[0].smiles)
        if mol is None:
            return None
        
        logger.debug(f"Mandatory linking: assembling {num_cores} cores with linkers")
        
        for i, core_info in enumerate(selected_cores[1:]):
            linker = random.choice(linkers)
            linked = self._link_fragments(
                Chem.MolToSmiles(mol),
                core_info.smiles,
                linker.smiles
            )
            
            if linked is not None:
                mol = linked
                logger.debug(f"Successfully linked core {i+2}")
            else:
                linked = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(core_info.smiles))
                if linked is not None:
                    mol = linked
        
        substituents = self.fragment_lib.fragments.get('substituents', [])
        if substituents:
            for _ in range(random.randint(3, 6)):
                sub = random.choice(substituents)
                decorated = self._attach_fragment(mol, sub.smiles)
                if decorated is not None:
                    mol = decorated
        
        return self._sanitize_molecule(mol)

    def _multi_core_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Assemble multiple cores together"""
        cores = self.fragment_lib.fragments['cores']
        linkers = self.fragment_lib.fragments.get('linkers', [])
        
        if len(cores) < 2:
            return None
            
        num_cores = min(3, len(cores))
        selected_cores = random.sample(cores, num_cores)
        selected_cores = sorted(selected_cores, key=lambda x: getattr(x, 'mw', 0), reverse=True)
        
        mol = Chem.MolFromSmiles(selected_cores[0].smiles)
        if mol is None:
            return None
        
        for core_info in selected_cores[1:]:
            if linkers and random.random() < 0.7:
                linker = random.choice(linkers)
                linked = self._link_fragments(
                    Chem.MolToSmiles(mol),
                    core_info.smiles,
                    linker.smiles
                )
                if linked is not None:
                    mol = linked
            else:
                attached = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(core_info.smiles))
                if attached is not None:
                    mol = attached
        
        return self._sanitize_molecule(mol)

    def _fragment_growing_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Fragment growing with chemistry-aware attachment"""
        suitable_cores: List[FragmentInfo] = []
        for interaction in target_interactions:
            cores = self.fragment_lib.get_fragments_for_interaction(interaction)
            suitable_cores.extend([c for c in cores if getattr(c, 'scaffold_type', '') == 'core'])
        
        if not suitable_cores:
            suitable_cores = self.fragment_lib.fragments['cores']
        if not suitable_cores:
            return None

        core = random.choice(self._select_larger_fragments(suitable_cores) or suitable_cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        for iteration in range(random.randint(6, 12)):
            interaction_needed = target_interactions[iteration % len(target_interactions)]
            suitable_fragments = self.fragment_lib.get_fragments_for_interaction(interaction_needed)
            
            if suitable_fragments:
                larger_fragments = [f for f in suitable_fragments if f.mw > 100] or suitable_fragments
                
                fragment = self._score_and_select_fragment(mol, larger_fragments, interaction_needed)
                if fragment is None:
                    fragment = random.choice(larger_fragments)
                
                new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(fragment.smiles))
                if new_mol is not None:
                    mol = new_mol
                    self.generation_stats['hotspot_matches'][interaction_needed] += 1
                else:
                    break
        
        return self._sanitize_molecule(mol)

    def _scaffold_decoration_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Scaffold decoration using chemistry-aware attachment"""
        large_cores = [c for c in self.fragment_lib.fragments['cores'] 
                       if c.mw > 200 and c.ring_count >= 2 and getattr(c, 'attachment_points', 1) >= 3]
        if not large_cores:
            large_cores = [c for c in self.fragment_lib.fragments['cores'] if c.mw > 150]
        if not large_cores:
            large_cores = self.fragment_lib.fragments['cores']
        if not large_cores:
            return None
            
        scaffold = random.choice(large_cores)
        mol = Chem.MolFromSmiles(scaffold.smiles)
        if mol is None:
            return None

        substituents = self.fragment_lib.fragments['substituents']
        if not substituents:
            return self._sanitize_molecule(mol)
        
        max_decorations = min(7, max(5, len(target_interactions) * 2))
        for i in range(max_decorations):
            if random.random() < 0.8:
                interaction_type = target_interactions[i % len(target_interactions)]
                suitable_subs = [s for s in substituents if interaction_type in s.interaction_types]
                if suitable_subs:
                    sub = random.choice(self._select_larger_fragments(suitable_subs) or suitable_subs)
                    new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(sub.smiles))
                    if new_mol is not None:
                        mol = new_mol
        
        return self._sanitize_molecule(mol)

    def _pharmacophore_guided_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Pharmacophore-guided assembly with 3D spatial awareness"""
        if not self.pocket.pharmacophore_points:
            return self._multi_core_assembly(target_interactions)
        
        relevant_points = [p for p in self.pocket.pharmacophore_points 
                          if p.get('type', '') in target_interactions]
        
        if not relevant_points:
            relevant_points = self.pocket.pharmacophore_points[:6]

        flexible_cores = [c for c in self.fragment_lib.fragments['cores'] 
                         if getattr(c, 'attachment_points', 1) >= 3 and c.mw > 180]
        if not flexible_cores:
            flexible_cores = self.fragment_lib.fragments['cores']
        if not flexible_cores:
            return None

        core = random.choice(self._select_larger_fragments(flexible_cores) or flexible_cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        sorted_points = sorted(relevant_points, 
                              key=lambda p: p.get('strength', p.get('importance', 0.5)), 
                              reverse=True)

        # Try 3D-guided assembly for high-priority pharmacophore points
        use_3d_guidance = random.random() < 0.3  # 30% chance to use 3D guidance
        
        for point in sorted_points[:min(6, getattr(core, 'attachment_points', 2))]:
            point_type = point.get('type', 'hydrophobic')
            suitable_fragment_types = self.pharmacophore_fragment_map.get(point_type, [point_type])
            
            suitable_fragments = []
            for frag_type in suitable_fragment_types:
                suitable_fragments.extend(self.fragment_lib.get_fragments_for_interaction(frag_type))
            
            if suitable_fragments:
                fragment = self._score_and_select_fragment(mol, suitable_fragments, point_type)
                if fragment is None:
                    fragment = random.choice(suitable_fragments)
                
                # Use 3D-guided assembly if enabled and position available
                if use_3d_guidance and 'position' in point:
                    target_pos = np.array(point['position'])
                    new_mol = self._attach_fragment_at_position(
                        mol, fragment.smiles, target_pos, distance_threshold=6.0
                    )
                else:
                    new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(fragment.smiles))
                
                if new_mol is not None:
                    mol = new_mol
                    self.generation_stats['hotspot_matches'][point_type] += 1
                    
        return self._sanitize_molecule(mol)

    def _hotspot_guided_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Hotspot-guided assembly with 3D spatial awareness"""
        target_hotspots = self.high_priority_hotspots if self.high_priority_hotspots else self.pocket.hotspots[:3]
        
        if not target_hotspots:
            return None

        cores = [f for f in self.fragment_lib.fragments['cores'] if f.attachment_points >= 2]
        if not cores:
            cores = self.fragment_lib.fragments['cores']
        if not cores:
            return None

        core = random.choice(self._select_larger_fragments(cores) or cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        sorted_hotspots = sorted(target_hotspots, key=lambda h: h.strength, reverse=True)
        
        # Use 3D-guided assembly for hotspots
        use_3d_guidance = random.random() < 0.4  # 40% chance for hotspots

        for hotspot in sorted_hotspots[:min(4, getattr(core, 'attachment_points', 2))]:
            suitable_subs = self.fragment_lib.get_fragments_for_interaction(hotspot.interaction_type)
            if suitable_subs:
                sub = self._score_and_select_fragment(mol, suitable_subs, hotspot.interaction_type)
                if sub is None:
                    sub = random.choice(suitable_subs)
                
                # Use 3D-guided growth toward hotspot position
                if use_3d_guidance and hasattr(hotspot, 'position'):
                    new_mol = self._grow_toward_position(
                        mol, hotspot.position, hotspot.interaction_type, max_attempts=3
                    )
                else:
                    new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(sub.smiles))
                
                if new_mol is not None:
                    mol = new_mol
                    self.generation_stats['hotspot_matches'][hotspot.interaction_type] += 1
        
        return self._sanitize_molecule(mol)

    def _linking_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Linking assembly with chemistry-aware attachment"""
        cores = self.fragment_lib.fragments['cores']
        linkers = self.fragment_lib.fragments['linkers']
        if len(cores) < 2 or not linkers:
            return None
        
        larger_cores = self._select_larger_fragments(cores) or cores
        if len(larger_cores) < 2:
            return None
            
        core1, core2 = random.sample(larger_cores, 2)

        suitable_linkers = linkers
        for interaction in target_interactions:
            suitable_linkers = [l for l in linkers if interaction in l.interaction_types]
            if suitable_linkers:
                break
            
        if not suitable_linkers:
            suitable_linkers = linkers
            
        linker = random.choice(self._select_larger_fragments(suitable_linkers) or suitable_linkers)

        mol = self._link_fragments(core1.smiles, core2.smiles, linker.smiles)
        
        if mol:
            substituents = self.fragment_lib.fragments['substituents']
            if substituents:
                for _ in range(random.randint(1, 3)):
                    sub = random.choice(substituents)
                    new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(sub.smiles))
                    if new_mol is not None:
                        mol = new_mol
        
        return self._sanitize_molecule(mol)

    def _ring_closure_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Ring closure assembly"""
        cores = self.fragment_lib.fragments['cores']
        if not cores:
            return None
            
        core = random.choice(self._select_larger_fragments(cores) or cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        substituents = self.fragment_lib.fragments.get('substituents', [])
        for _ in range(random.randint(2, 4)):
            if substituents:
                sub = random.choice(substituents)
                new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(sub.smiles))
                if new_mol is not None:
                    mol = new_mol

        return self._sanitize_molecule(mol)

    def _conjugation_extension_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Conjugation extension"""
        cores = self.fragment_lib.fragments['cores']
        aryl_extenders = self.fragment_lib.fragments.get('linkers', []) + self.fragment_lib.fragments.get('substituents', [])
        if not cores or not aryl_extenders:
            return None
            
        core = random.choice(self._select_larger_fragments(cores) or cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        for _ in range(random.randint(2, 4)):
            extender = random.choice(aryl_extenders)
            new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(extender.smiles))
            if new_mol is not None:
                mol = new_mol
                
        return self._sanitize_molecule(mol)

    def _sp3_enrichment_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """sp3 enrichment"""
        cores = self.fragment_lib.fragments['cores']
        if not cores:
            return None
            
        core = random.choice(self._select_larger_fragments(cores) or cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        candidates: List[FragmentInfo] = []
        for tag in ("aliphatic", "donor", "acceptor", "sp3", "hydrogen_bond", "polar"):
            candidates += self.fragment_lib.get_fragments_for_interaction(tag)
        if not candidates:
            candidates = self.fragment_lib.fragments.get('substituents', [])

        for _ in range(random.randint(4, 6)):
            if candidates:
                frag = random.choice(candidates)
                new_mol = self._attach_fragments_by_points(mol, Chem.MolFromSmiles(frag.smiles))
                if new_mol is not None:
                    mol = new_mol
        
        return self._sanitize_molecule(mol)

    def _enumerative_reaction_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Reaction-based assembly"""
        if not self.reaction_templates:
            return None
        
        safe_reactions = ["amide", "esterification", "ether", "n_alkylation", "reductive_amination"]
        available_reactions = [r for r in safe_reactions if r in self.reaction_templates]
        
        if not available_reactions:
            return None
            
        rxn_name = random.choice(available_reactions)

        buckets = [
            self.fragment_lib.fragments.get('cores', []),
            self.fragment_lib.fragments.get('linkers', []),
            self.fragment_lib.fragments.get('substituents', []),
        ]
        
        reactant_mols: List[Chem.Mol] = []
        for bucket in buckets[:2]:
            if bucket:
                fi = random.choice(self._select_larger_fragments(bucket) or bucket)
                m = Chem.MolFromSmiles(fi.smiles)
                if m is not None:
                    m = self._sanitize_molecule(m)
                    if m:
                        reactant_mols.append(m)
                
        if len(reactant_mols) < 2:
            return None
            
        product = self._apply_reaction_safe(rxn_name, reactant_mols)
        return self._sanitize_molecule(product)

    # ========== HELPER METHODS ==========
    
    def _select_larger_fragments(self, fragments: List[FragmentInfo], prefer_large: bool = True) -> List[FragmentInfo]:
        """Select larger fragments"""
        if not fragments or not prefer_large or len(fragments) <= 3:
            return fragments
        
        sorted_frags = sorted(fragments, key=lambda f: getattr(f, 'mw', 0), reverse=True)
        cutoff = max(1, int(len(sorted_frags) * self.constraints.large_fragment_threshold))
        return sorted_frags[:cutoff]

    def _apply_reaction_safe(self, reaction_name: str, reactant_mols: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """Apply reaction safely"""
        try:
            if reaction_name not in self.reaction_templates:
                return None
            rxn = self.reaction_templates[reaction_name]
            needed = rxn.GetNumReactantTemplates()
            if needed == 0 or len(reactant_mols) < needed:
                return None

            prods = rxn.RunReactants(tuple(reactant_mols[:needed]))
            if not prods:
                return None

            best_mol = None
            best_qed = -1.0
            for tpl in prods:
                for p in tpl:
                    try:
                        sanitized = self._sanitize_molecule(p)
                        if sanitized is not None:
                            q = QED.qed(sanitized)
                            if q > best_qed:
                                best_qed = q
                                best_mol = sanitized
                    except:
                        continue
            return best_mol
        except:
            return None

    def _add_functional_group(self, mol: Chem.Mol, functional_group_type: str, 
                             attachment_preference: str = 'any') -> Optional[Chem.Mol]:
        """Add functional group"""
        if functional_group_type not in self.functional_group_library:
            return mol
        
        try:
            available_groups = self.functional_group_library[functional_group_type]
            if not available_groups:
                return mol
            
            functional_group = random.choice(available_groups)
            fg_smiles = functional_group['smiles']
            fg_mol = Chem.MolFromSmiles(fg_smiles)
            if fg_mol is None:
                return mol
            
            result = self._attach_fragments_by_points(mol, fg_mol)
            if result is not None:
                self.generation_stats['functional_group_additions'][functional_group['name']] += 1
                return result
            
            return mol
            
        except Exception as e:
            logger.debug(f"Error adding functional group {functional_group_type}: {e}")
            return mol

    def _add_pharmacophore_functional_groups(self, mol: Chem.Mol, 
                                           target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Add functional groups based on pocket requirements"""
        if mol is None or not target_interactions:
            return mol
        
        try:
            mol = self.pocket_fg_manager.add_targeted_functional_groups(
                mol, target_interactions, max_additions=5
            )
            return mol
        except Exception as e:
            logger.warning(f"Error in pocket-aware FG addition: {e}")
            return mol

    def _validate_pocket_satisfaction(self, mol: Chem.Mol, 
                                      target_interactions: List[str]) -> Tuple[float, Dict]:
        """Calculate pocket satisfaction using scorer if available"""
        if not self.pocket.pharmacophore_points:
            return 1.0, {}
        
        if self.scorer:
            try:
                score_obj = self.scorer.calculate_comprehensive_score(mol)
                pharmacophore_score = score_obj.pharmacophore_score
                
                unsatisfied = self.pocket_fg_manager.analyze_unsatisfied_requirements(mol, target_interactions)
                unsatisfied_types = defaultdict(int)
                for req in unsatisfied:
                    unsatisfied_types[req['type']] += 1
                
                return pharmacophore_score, dict(unsatisfied_types)
            except Exception as e:
                logger.debug(f"Could not use scorer for pocket satisfaction: {e}")
        
        total_points = len([p for p in self.pocket.pharmacophore_points 
                           if p.get('type', '') in target_interactions])
        
        if total_points == 0:
            return 1.0, {}
        
        satisfied = 0
        unsatisfied_types = defaultdict(int)
        
        for point in self.pocket.pharmacophore_points:
            point_type = point.get('type', '')
            if point_type not in target_interactions:
                continue
            
            point_position = np.array(point.get('position', [0, 0, 0]))
            
            if self.pocket_fg_manager._has_fg_near_position(mol, point_position, point_type):
                satisfied += 1
            else:
                unsatisfied_types[point_type] += 1
        
        satisfaction_score = satisfied / total_points if total_points > 0 else 0.0
        return satisfaction_score, dict(unsatisfied_types)

    def _validate_molecule(self, mol: Chem.Mol) -> bool:
        """Validate molecule"""
        if mol is None or mol.GetNumAtoms() == 0:
            return False
        try:
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles or any(char in smiles for char in ['A', '*', '[*]']):
                return False
                
            props = self._calculate_properties(mol)
            is_valid, reason = self.validator.validate_with_reason(mol, props)
            
            if not is_valid:
                if reason:
                    self.generation_stats['failure_reasons'][reason] += 1
                return False
                
            if self._has_toxic_alerts(mol):
                self.generation_stats['failure_reasons']['Toxic substructure'] += 1
                return False
                
            return True
        except Exception as e:
            self.generation_stats['failure_reasons'][f'Validation exception: {str(e)}'] += 1
            return False

    def _calculate_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate molecular properties"""
        try:
            return {
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'mw': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'qed': QED.qed(mol)
            }
        except:
            return {
                'heavy_atoms': 0, 'mw': 0, 'logp': 0, 'hbd': 0, 'hba': 0,
                'tpsa': 0, 'rotatable_bonds': 0, 'rings': 0, 'qed': 0
            }

    def _has_toxic_alerts(self, mol: Chem.Mol) -> bool:
        """Check for toxic substructures"""
        if not hasattr(self.fragment_lib, 'toxic_alerts'):
            return False
        for alert_smarts in self.fragment_lib.toxic_alerts:
            try:
                pattern = Chem.MolFromSmarts(alert_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
            except:
                continue
        return False

    def generate_structure_guided(self, target_interactions: List[str],
                                  n_molecules: int = 100,
                                  enable_elaboration: bool = True,
                                  elaboration_fraction: float = 0.5) -> List[Chem.Mol]:
        """
        Generate molecules with chemistry-aware attachment point handling, 3D guidance, and post-assembly elaboration
        
        Args:
            target_interactions: Target interaction types for the pocket
            n_molecules: Number of molecules to generate
            enable_elaboration: Whether to enable post-assembly elaboration
            elaboration_fraction: Fraction of molecules to elaborate (0.0 to 1.0)
            
        Returns:
            List of generated molecules
        """
        molecules: List[Chem.Mol] = []
        attempts = 0
        max_attempts = n_molecules * 100
        
        logger.info(f"Starting generation of {n_molecules} molecules with chemistry-aware + 3D-guided assembly")
        if enable_elaboration:
            logger.info(f"Post-assembly elaboration enabled for {elaboration_fraction*100:.0f}% of molecules")
        logger.info(f"Target interactions: {target_interactions}")
        logger.info(f"Pocket druggability: {self.pocket.druggability_score:.3f}, Volume: {self.pocket.volume:.1f}")

        while len(molecules) < n_molecules and attempts < max_attempts:
            attempts += 1
            self.generation_stats['total_attempts'] += 1
            
            if attempts > n_molecules * 10 and len(molecules) == 0:
                logger.error("No valid molecules generated after many attempts")
                break

            strategy = self._choose_assembly_strategy(target_interactions)
            self.generation_stats['strategy_counts'][strategy] += 1

            mol: Optional[Chem.Mol] = None
            try:
                if strategy == 'hotspot_guided':
                    mol = self._hotspot_guided_assembly(target_interactions)
                elif strategy == 'fragment_growing':
                    mol = self._fragment_growing_assembly(target_interactions)
                elif strategy == 'linking':
                    mol = self._linking_assembly(target_interactions)
                elif strategy == 'scaffold_decoration':
                    mol = self._scaffold_decoration_assembly(target_interactions)
                elif strategy == 'pharmacophore_guided':
                    mol = self._pharmacophore_guided_assembly(target_interactions)
                elif strategy == 'ring_closure':
                    mol = self._ring_closure_assembly(target_interactions)
                elif strategy == 'conjugation_extension':
                    mol = self._conjugation_extension_assembly(target_interactions)
                elif strategy == 'sp3_enrichment':
                    mol = self._sp3_enrichment_assembly(target_interactions)
                elif strategy == 'enumerative_reaction':
                    mol = self._enumerative_reaction_assembly(target_interactions)
                elif strategy == 'multi_core_assembly':
                    mol = self._multi_core_assembly(target_interactions)
                elif strategy == 'mandatory_linking_assembly':
                    mol = self._mandatory_linking_assembly(target_interactions)
                else:
                    mol = self._fragment_growing_assembly(target_interactions)

                if mol is not None:
                    mol = self._add_pharmacophore_functional_groups(mol, target_interactions)

                if mol is not None:
                    mol = self._sanitize_molecule(mol)

                if mol and self._validate_molecule(mol):
                    # POST-ASSEMBLY ELABORATION
                    if enable_elaboration and random.random() < elaboration_fraction:
                        elaborated_mol = self.elaboration_engine.elaborate_molecule(
                            mol, 
                            target_interactions,
                            max_additions=random.randint(3, 5),
                            min_score_improvement=0.01
                        )
                        
                        if elaborated_mol is not None and self._validate_molecule(elaborated_mol):
                            mol = elaborated_mol
                            self.generation_stats['elaborations_performed'] += 1
                            
                            # Check if elaboration improved the score
                            if self.scorer:
                                try:
                                    orig_score = self.scorer.calculate_comprehensive_score(mol).total_score
                                    elab_score = self.scorer.calculate_comprehensive_score(elaborated_mol).total_score
                                    if elab_score > orig_score:
                                        self.generation_stats['elaboration_improvements'] += 1
                                except:
                                    pass
                    
                    satisfaction_score, unsatisfied = self._validate_pocket_satisfaction(
                        mol, target_interactions
                    )
                    
                    self.generation_stats['pocket_satisfaction_scores'].append(satisfaction_score)
                    self.generation_stats['molecule_sizes'].append(mol.GetNumHeavyAtoms())
                    
                    molecules.append(mol)
                    self.generation_stats['successful_assemblies'] += 1
                    
                    if len(molecules) % 10 == 0:
                        avg_satisfaction = np.mean(self.generation_stats['pocket_satisfaction_scores'])
                        avg_size = np.mean(self.generation_stats['molecule_sizes'])
                        improvements = self.generation_stats['scorer_improvements']
                        chem_matches = self.generation_stats['chemistry_aware_matches']
                        guided_3d = self.generation_stats['3d_guided_assemblies']
                        elaborations = self.generation_stats['elaborations_performed']
                        logger.info(f"Generated {len(molecules)}/{n_molecules} molecules "
                                  f"(attempt {attempts}, satisfaction: {avg_satisfaction:.2%}, "
                                  f"size: {avg_size:.1f} atoms, scorer improvements: {improvements}, "
                                  f"chemistry-aware: {chem_matches}, 3D-guided: {guided_3d}, "
                                  f"elaborated: {elaborations})")
                elif mol:
                    self.generation_stats['validation_failures'] += 1

            except Exception as e:
                logger.debug(f"Assembly error with {strategy}: {str(e)}")
                continue
        
        logger.info(f"Generation complete: {len(molecules)} molecules from {attempts} attempts")
        if self.generation_stats['attachment_point_usage']:
            logger.info(f"Attachment point usage: {dict(self.generation_stats['attachment_point_usage'])}")
        if self.generation_stats['hotspot_matches']:
            logger.info(f"Hotspot matches: {dict(self.generation_stats['hotspot_matches'])}")
        logger.info(f"Chemistry-aware matches: {self.generation_stats['chemistry_aware_matches']}")
        logger.info(f"3D-guided assemblies: {self.generation_stats['3d_guided_assemblies']}")
        logger.info(f"3D optimization successes: {self.generation_stats['3d_optimization_successes']}")
        logger.info(f"Elaborations performed: {self.generation_stats['elaborations_performed']}")
        if self.generation_stats['elaborations_performed'] > 0:
            improvement_rate = self.generation_stats['elaboration_improvements'] / self.generation_stats['elaborations_performed']
            logger.info(f"Elaboration improvement rate: {improvement_rate:.1%}")
        
        return molecules

    def get_generation_statistics(self) -> Dict:
        """Get generation statistics"""
        stats = self.generation_stats.copy()
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_assemblies'] / stats['total_attempts']
            stats['validation_failure_rate'] = stats['validation_failures'] / stats['total_attempts']
            stats['3d_guided_rate'] = stats['3d_guided_assemblies'] / stats['successful_assemblies'] if stats['successful_assemblies'] > 0 else 0.0
            stats['elaboration_rate'] = stats['elaborations_performed'] / stats['successful_assemblies'] if stats['successful_assemblies'] > 0 else 0.0
        else:
            stats['success_rate'] = 0.0
            stats['validation_failure_rate'] = 0.0
            stats['3d_guided_rate'] = 0.0
            stats['elaboration_rate'] = 0.0
        
        if stats['elaborations_performed'] > 0:
            stats['elaboration_improvement_rate'] = stats['elaboration_improvements'] / stats['elaborations_performed']
        else:
            stats['elaboration_improvement_rate'] = 0.0
        
        if stats['failure_reasons']:
            sorted_failures = sorted(stats['failure_reasons'].items(), 
                                   key=lambda x: x[1], reverse=True)
            stats['top_failure_reasons'] = sorted_failures[:10]
        
        if stats['pocket_satisfaction_scores']:
            stats['avg_pocket_satisfaction'] = float(np.mean(stats['pocket_satisfaction_scores']))
            stats['min_pocket_satisfaction'] = float(np.min(stats['pocket_satisfaction_scores']))
            stats['max_pocket_satisfaction'] = float(np.max(stats['pocket_satisfaction_scores']))
        
        if stats['molecule_sizes']:
            stats['avg_molecule_size'] = float(np.mean(stats['molecule_sizes']))
            stats['min_molecule_size'] = int(np.min(stats['molecule_sizes']))
            stats['max_molecule_size'] = int(np.max(stats['molecule_sizes']))
        
        # Add elaboration engine statistics
        if hasattr(self, 'elaboration_engine'):
            stats['elaboration_engine_stats'] = self.elaboration_engine.get_statistics()
        
        return stats

    def reset_statistics(self):
        """Reset statistics"""
        self.generation_stats = {
            'total_attempts': 0,
            'successful_assemblies': 0,
            'validation_failures': 0,
            'strategy_counts': defaultdict(int),
            'functional_group_additions': defaultdict(int),
            'failure_reasons': defaultdict(int),
            'pocket_satisfaction_scores': [],
            'molecule_sizes': [],
            'attachment_point_usage': defaultdict(int),
            'scorer_improvements': 0,
            'hotspot_matches': defaultdict(int),
            'chemistry_aware_matches': 0,
            'compatibility_scores': [],
            '3d_guided_assemblies': 0,
            '3d_optimization_successes': 0,
            'elaborations_performed': 0,
            'elaboration_improvements': 0
        }
        self.pocket_fg_manager.addition_stats.clear()
        if hasattr(self, 'elaboration_engine'):
            self.elaboration_engine.elaboration_stats = {
                'molecules_elaborated': 0,
                'elaboration_attempts': 0,
                'successful_additions': 0,
                'score_improvements': 0,
                'property_violations': 0,
                'avg_additions_per_mol': [],
                'avg_score_delta': []
            }