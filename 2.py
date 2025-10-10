import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Union
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski, QED, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import LigandForgeConfig
from data_structures import PocketAnalysis, MolecularScore
from diversity_manager import EnhancedDiversityManager

logger = logging.getLogger(__name__)


class ScoringError(Exception):
    """Custom exception for scoring errors"""
    pass


class MultiObjectiveScorer:
    """Multi-objective scoring with 3D pharmacophore alignment - COMPLETE VERSION"""
    
    def __init__(self, config: LigandForgeConfig, pocket_analysis: PocketAnalysis, 
                 diversity_manager: EnhancedDiversityManager):
        self.config = self._validate_config(config)
        self.pocket = pocket_analysis
        self.diversity = diversity_manager
        self.reference_fingerprints = []
        
        # Enhanced caching with size limits and thread safety
        self._property_cache = {}
        self._fingerprint_cache = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = getattr(config, 'max_cache_size', 10000)
        
        # Scoring history for adaptive weights
        self.scoring_history = []
        self._history_lock = threading.RLock()
        self._max_history_size = getattr(config, 'max_history_size', 1000)
        
        # Performance tracking
        self._scoring_stats = {
            'total_molecules': 0,
            'cache_hits': 0,
            'errors': 0,
            'avg_score_time': 0.0,
            'avg_3d_alignment': 0.0,
            '3d_scoring_attempts': 0,
            '3d_scoring_successes': 0
        }
        self._stats_lock = threading.RLock()
        
        # Pre-compile SMARTS patterns for efficiency
        self._common_patterns = self._compile_common_patterns()
        
        # Thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
        
        # NEW: Extract and cache 3D pharmacophore targets
        self.pharmacophore_targets = self._extract_pharmacophore_targets()
        
        logger.info(f"Initialized scorer with {len(self.pharmacophore_targets)} 3D pharmacophore targets")
    
    def _validate_config(self, config: LigandForgeConfig) -> LigandForgeConfig:
        """Validate configuration object and set defaults"""
        required_attrs = {
            'reward_weights': {
                'pharmacophore': 0.30,  # Increased for 3D alignment
                'synthetic': 0.20,
                'drug_likeness': 0.20,
                'novelty': 0.15,
                'selectivity': 0.10,
                'water_displacement': 0.05
            },
            'lipinski_thresholds': {
                'mw': [0, 650],
                'logp': [-5, 5],
                'hbd': [0, 5],
                'hba': [0, 10],
                'tpsa': [0, 140]
            },
            'max_heavy_atoms': 50,
            'max_rotatable_bonds': 10,
            'max_rings': 6,
            'fingerprint_radius': 2,
            'fingerprint_bits': 1024
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(config, attr):
                setattr(config, attr, default_value)
                logger.warning(f"Config missing {attr}, using default: {default_value}")
        
        return config
    
    def _compile_common_patterns(self) -> Dict[str, Chem.Mol]:
        """Pre-compile common SMARTS patterns for efficiency"""
        patterns = {
            'hydroxyl': "[OH]",
            'amino': "[NH2]",
            'carboxyl': "[C](=O)[OH]",
            'amide': "[C](=O)[NH2]",
            'nitrile': "[C]#[N]",
            'sulfonamide': "[S](=O)(=O)[NH2]",
            'ether': "[O]([C])[C]",
            'ketone': "[C](=O)[C]",
            'aldehyde': "[C](=O)[H]",
            'ester': "[C](=O)[O][C]"
        }
        
        compiled = {}
        for name, smarts in patterns.items():
            try:
                compiled[name] = Chem.MolFromSmarts(smarts)
            except Exception as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")
        
        return compiled
    
    def _extract_pharmacophore_targets(self) -> List[Dict]:
        """NEW: Extract and structure 3D pharmacophore targets from pocket"""
        targets = []
        
        # Extract from pharmacophore_points
        if hasattr(self.pocket, 'pharmacophore_points') and self.pocket.pharmacophore_points:
            for point in self.pocket.pharmacophore_points:
                targets.append({
                    'position': np.array(point.get('position', [0, 0, 0])),
                    'type': point.get('type', '').lower(),
                    'strength': point.get('strength', point.get('importance', 1.0)),
                    'tolerance': point.get('tolerance', 2.0),
                    'features': point.get('features', [])
                })
        
        # Also extract from hotspots
        if hasattr(self.pocket, 'hotspots') and self.pocket.hotspots:
            for hotspot in self.pocket.hotspots:
                targets.append({
                    'position': hotspot.position,
                    'type': hotspot.interaction_type.lower(),
                    'strength': getattr(hotspot, 'strength', 1.0),
                    'tolerance': 2.5,
                    'features': getattr(hotspot, 'pharmacophore_features', [])
                })
        
        return targets
    
    def calculate_comprehensive_score(self, mol: Chem.Mol, 
                                    generation_round: int = 0,
                                    use_cache: bool = True) -> MolecularScore:
        """Calculate comprehensive multi-objective score with 3D pharmacophore alignment"""
        
        # Input validation
        if mol is None:
            logger.warning("Received None molecule for scoring")
            return self._create_zero_score(["None molecule"])
        
        if not isinstance(generation_round, int) or generation_round < 0:
            logger.warning(f"Invalid generation_round: {generation_round}, using 0")
            generation_round = 0
        
        try:
            # Basic molecule validation
            Chem.SanitizeMol(mol)
            if mol.GetNumAtoms() == 0:
                return self._create_zero_score(["Empty molecule"])
        except (Chem.KekulizeException, Chem.AtomValenceException, ValueError) as e:
            logger.warning(f"Molecule sanitization failed: {e}")
            return self._create_zero_score([f"Sanitization error: {str(e)}"])
        except Exception as e:
            logger.error(f"Unexpected error in molecule validation: {e}")
            return self._create_zero_score([f"Validation error: {str(e)}"])
        
        mol_hash = self._get_molecule_hash(mol)
        
        # Check cache if enabled
        if use_cache:
            with self._cache_lock:
                cached_result = self._property_cache.get(mol_hash, {}).get('score_result')
                if cached_result:
                    with self._stats_lock:
                        self._scoring_stats['cache_hits'] += 1
                    return cached_result
        
        start_time = time.time()
        
        try:
            # Calculate individual score components with error isolation
            scores = self._calculate_score_components(mol)
            
            # Calculate weighted total with validation
            weights = self._get_adaptive_weights(generation_round, scores)
            total_score = self._calculate_weighted_score(scores, weights)
            
            # Get molecular properties with caching
            properties = self._calculate_molecular_properties(mol)
            
            # Check for violations with detailed reporting
            violations = self._check_violations(mol, properties)
            
            # Apply penalties for violations with graduated scale
            penalty_factor = self._calculate_penalty_factor(violations)
            total_score *= penalty_factor
            
            # Ensure score is in valid range
            total_score = float(np.clip(total_score, 0.0, 1.0))
            
            # Create comprehensive score object
            molecular_score = MolecularScore(
                total_score=total_score,
                pharmacophore_score=float(scores['pharmacophore']),
                synthetic_score=float(scores['synthetic']),
                drug_likeness_score=float(scores['drug_likeness']),
                novelty_score=float(scores['novelty']),
                selectivity_score=float(scores['selectivity']),
                water_displacement_score=float(scores['water_displacement']),
                component_scores=scores,
                property_values=properties,
                violations=violations,
                confidence=self._calculate_confidence(scores, properties, violations)
            )
            
            # Cache the result
            if use_cache:
                self._cache_scoring_result(mol_hash, molecular_score, properties)
            
            # Store in history with size management
            self._update_scoring_history(generation_round, scores, total_score, properties)
            
            # Update performance stats
            scoring_time = time.time() - start_time
            self._update_performance_stats(scoring_time, scores)
            
            return molecular_score
            
        except Exception as e:
            with self._stats_lock:
                self._scoring_stats['errors'] += 1
            logger.error(f"Error calculating comprehensive score: {e}", exc_info=True)
            return self._create_zero_score([f"Scoring error: {str(e)}"])
    
    def _calculate_score_components(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate individual score components with error isolation"""
        scores = {}
        
        component_methods = {
            'pharmacophore': self._pharmacophore_score,
            'synthetic': self._synthetic_feasibility_score,
            'drug_likeness': self._drug_likeness_score,
            'novelty': self._novelty_score,
            'selectivity': self._selectivity_score,
            'water_displacement': self._water_displacement_score
        }
        
        for component, method in component_methods.items():
            try:
                score = method(mol)
                # Validate score range
                if not (0.0 <= score <= 1.0):
                    logger.warning(f"Score {component} out of range: {score}, clipping to [0,1]")
                    score = np.clip(score, 0.0, 1.0)
                scores[component] = float(score)
            except Exception as e:
                logger.warning(f"Error calculating {component} score: {e}")
                scores[component] = 0.0
        
        return scores
    
    def _pharmacophore_score(self, mol: Chem.Mol) -> float:
        """Enhanced pharmacophore scoring with 3D spatial alignment"""
        # Try 3D scoring first
        if self.pharmacophore_targets:
            try:
                with self._stats_lock:
                    self._scoring_stats['3d_scoring_attempts'] += 1
                
                score_3d = self._pharmacophore_score_3d(mol)
                
                with self._stats_lock:
                    self._scoring_stats['3d_scoring_successes'] += 1
                    # Update running average
                    alpha = 0.1
                    self._scoring_stats['avg_3d_alignment'] = (
                        alpha * score_3d + 
                        (1 - alpha) * self._scoring_stats['avg_3d_alignment']
                    )
                
                return score_3d
            except Exception as e:
                logger.debug(f"3D pharmacophore scoring failed, falling back to 2D: {e}")
        
        # Fallback to original 2D scoring
        return self._pharmacophore_score_2d(mol)
    
    def _pharmacophore_score_3d(self, mol: Chem.Mol) -> float:
        """NEW: 3D spatial pharmacophore alignment scoring"""
        if not self.pharmacophore_targets:
            return 0.5
        
        try:
            # Ensure molecule has 3D conformer
            if not mol.GetNumConformers():
                mol = Chem.AddHs(mol)
                # Try multiple embedding attempts
                for seed in range(3):
                    result = AllChem.EmbedMolecule(mol, randomSeed=42 + seed, useRandomCoords=True)
                    if result == 0:
                        break
                
                if result == 0:  # Success
                    # Optimize geometry
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    except:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                
                mol = Chem.RemoveHs(mol)
            
            if not mol.GetNumConformers():
                # Fallback to 2D if embedding fails
                return self._pharmacophore_score_2d(mol)
            
            # Extract molecular 3D features
            mol_features = self._extract_3d_pharmacophore_features(mol)
            
            if not mol_features:
                return 0.3
            
            # Calculate alignment with each target
            alignment_scores = []
            matched_targets = 0
            matched_critical = 0
            total_critical = 0
            
            for target in self.pharmacophore_targets:
                target_pos = target['position']
                target_type = target['type']
                target_strength = target['strength']
                tolerance = target['tolerance']
                
                is_critical = target_strength > 0.8
                if is_critical:
                    total_critical += 1
                
                # Find matching molecular features
                matching_features = [f for f in mol_features if f['type'] == target_type]
                
                if matching_features:
                    # Find closest feature
                    distances = [np.linalg.norm(f['position'] - target_pos) for f in matching_features]
                    min_dist = min(distances)
                    
                    # Score based on distance with tolerance
                    if min_dist < tolerance:
                        # Perfect match
                        feature_score = 1.0
                        matched_targets += 1.0
                        if is_critical:
                            matched_critical += 1
                    elif min_dist < tolerance * 2:
                        # Acceptable match
                        feature_score = 0.5 * (1.0 - (min_dist - tolerance) / tolerance)
                        matched_targets += 0.5
                        if is_critical:
                            matched_critical += 0.5
                    else:
                        # Too far
                        feature_score = 0.1
                    
                    # Weight by target strength
                    weighted_score = feature_score * target_strength
                    alignment_scores.append(weighted_score)
                else:
                    # No matching feature
                    alignment_scores.append(0.0)
            
            # Calculate overall alignment
            if alignment_scores:
                base_alignment = np.mean(alignment_scores)
            else:
                base_alignment = 0.0
            
            # Bonus for coverage (matching multiple targets)
            coverage_ratio = matched_targets / len(self.pharmacophore_targets)
            coverage_bonus = min(0.2, coverage_ratio * 0.3)
            
            # Critical target bonus/penalty
            if total_critical > 0:
                critical_ratio = matched_critical / total_critical
                if critical_ratio >= 0.8:
                    coverage_bonus += 0.1  # Bonus for matching critical targets
                elif critical_ratio < 0.5:
                    coverage_bonus -= 0.15  # Penalty for missing critical targets
            
            final_score = base_alignment + coverage_bonus
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.debug(f"3D pharmacophore scoring error: {e}")
            return self._pharmacophore_score_2d(mol)
    
    def _extract_3d_pharmacophore_features(self, mol: Chem.Mol) -> List[Dict]:
        """NEW: Extract 3D pharmacophore features from molecule"""
        if not mol.GetNumConformers():
            return []
        
        try:
            conformer = mol.GetConformer()
        except:
            return []
        
        features = []
        
        # H-bond donors
        hbd_pattern = Chem.MolFromSmarts('[N,O;H1,H2]')
        if hbd_pattern:
            matches = mol.GetSubstructMatches(hbd_pattern)
            for match in matches:
                try:
                    pos = conformer.GetAtomPosition(match[0])
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'hbd',
                        'atom_idx': match[0]
                    })
                except:
                    continue
        
        # Also match 'hbond_donor'
        for f in features[:]:
            if f['type'] == 'hbd':
                features.append({**f, 'type': 'hbond_donor'})
        
        # H-bond acceptors
        hba_pattern = Chem.MolFromSmarts('[N,O;H0]')
        if hba_pattern:
            matches = mol.GetSubstructMatches(hba_pattern)
            for match in matches:
                try:
                    pos = conformer.GetAtomPosition(match[0])
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'hba',
                        'atom_idx': match[0]
                    })
                except:
                    continue
        
        # Also match 'hbond_acceptor'
        for f in features[:]:
            if f['type'] == 'hba':
                features.append({**f, 'type': 'hbond_acceptor'})
        
        # Aromatic rings (centroids)
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            try:
                ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
                if all(atom.GetIsAromatic() for atom in ring_atoms) and len(ring) in [5, 6]:
                    positions = [conformer.GetAtomPosition(idx) for idx in ring]
                    centroid = np.mean([[p.x, p.y, p.z] for p in positions], axis=0)
                    features.append({
                        'position': centroid,
                        'type': 'aromatic',
                        'atom_indices': list(ring)
                    })
            except:
                continue
        
        # Hydrophobic centers
        hydrophobic_pattern = Chem.MolFromSmarts('[C;H3,H2,H1]')
        if hydrophobic_pattern:
            matches = mol.GetSubstructMatches(hydrophobic_pattern)
            if matches:
                try:
                    positions = [conformer.GetAtomPosition(m[0]) for m in matches]
                    coords = np.array([[p.x, p.y, p.z] for p in positions])
                    if len(coords) > 0:
                        centroid = np.mean(coords, axis=0)
                        features.append({
                            'position': centroid,
                            'type': 'hydrophobic',
                            'atom_indices': [m[0] for m in matches]
                        })
                except:
                    pass
        
        # Charged groups
        positive_pattern = Chem.MolFromSmarts('[N+,n+]')
        if positive_pattern:
            matches = mol.GetSubstructMatches(positive_pattern)
            for match in matches:
                try:
                    pos = conformer.GetAtomPosition(match[0])
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'positive',
                        'atom_idx': match[0]
                    })
                    # Also add as 'electrostatic'
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'electrostatic',
                        'atom_idx': match[0]
                    })
                except:
                    continue
        
        negative_pattern = Chem.MolFromSmarts('[O-,o-]')
        if negative_pattern:
            matches = mol.GetSubstructMatches(negative_pattern)
            for match in matches:
                try:
                    pos = conformer.GetAtomPosition(match[0])
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'negative',
                        'atom_idx': match[0]
                    })
                    # Also add as 'electrostatic'
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'electrostatic',
                        'atom_idx': match[0]
                    })
                except:
                    continue
        
        # Polar groups
        polar_pattern = Chem.MolFromSmarts('[N,O]')
        if polar_pattern:
            matches = mol.GetSubstructMatches(polar_pattern)
            for match in matches:
                try:
                    pos = conformer.GetAtomPosition(match[0])
                    features.append({
                        'position': np.array([pos.x, pos.y, pos.z]),
                        'type': 'polar',
                        'atom_idx': match[0]
                    })
                except:
                    continue
        
        return features
    
    def _pharmacophore_score_2d(self, mol: Chem.Mol) -> float:
        """Original 2D pharmacophore scoring (feature presence only)"""
        hotspots = getattr(self.pocket, 'hotspots', [])
        if not hotspots:
            return 0.5
        
        try:
            # Get molecular features
            mol_features = self._extract_molecular_features(mol)
            
            # Calculate feature matching score
            total_match_score = 0.0
            total_weight = 0.0
            matched_hotspots = 0
            
            for hotspot in hotspots:
                interaction_type = hotspot.interaction_type.lower()
                hotspot_strength = getattr(hotspot, 'strength', 1.0)
                
                # Enhanced feature matching
                feature_score = self._match_interaction_feature(
                    interaction_type, mol_features, mol
                )
                
                if feature_score > 0.1:
                    matched_hotspots += 1
                
                weighted_score = feature_score * hotspot_strength
                total_match_score += weighted_score
                total_weight += hotspot_strength
            
            # Normalize by total weight
            if total_weight > 0:
                pharmacophore_score = total_match_score / total_weight
            else:
                pharmacophore_score = 0.5
            
            # Bonus for multiple hotspot matches
            hotspot_coverage = matched_hotspots / len(hotspots) if hotspots else 0
            coverage_bonus = min(0.2, hotspot_coverage * 0.3)
            pharmacophore_score += coverage_bonus
            
            # Bonus for complementary feature combinations
            combination_bonus = self._calculate_feature_combination_bonus(mol_features)
            pharmacophore_score += combination_bonus
            
            return min(1.0, pharmacophore_score)
            
        except Exception as e:
            logger.warning(f"Error in 2D pharmacophore scoring: {e}")
            return 0.3
    
    # ... Continue with ALL remaining methods from original scoring.py ...
    # (I'll include the complete set in the next continuation due to length)
    
    def _match_interaction_feature(self, interaction_type: str, 
                                 mol_features: Dict, mol: Chem.Mol) -> float:
        """Enhanced interaction feature matching"""
        if interaction_type in ['hbd', 'hbond_donor']:
            hbd_count = mol_features.get('hbd_count', 0)
            return min(1.0, hbd_count / 2.0) if hbd_count > 0 else 0.0
            
        elif interaction_type in ['hba', 'hbond_acceptor']:
            hba_count = mol_features.get('hba_count', 0)
            return min(1.0, hba_count / 3.0) if hba_count > 0 else 0.0
            
        elif interaction_type == 'hydrophobic':
            hydrophobic_area = mol_features.get('hydrophobic_area', 0)
            return min(1.0, hydrophobic_area / 150.0) if hydrophobic_area > 0 else 0.0
            
        elif interaction_type == 'aromatic':
            aromatic_rings = mol_features.get('aromatic_rings', 0)
            return min(1.0, aromatic_rings / 2.0) if aromatic_rings > 0 else 0.0
            
        elif interaction_type == 'electrostatic':
            formal_charge = abs(mol_features.get('formal_charge', 0))
            if formal_charge > 0:
                return 0.9
            elif mol_features.get('hbd_count', 0) > 0 or mol_features.get('hba_count', 0) > 0:
                return 0.6
            else:
                return 0.1
                
        elif interaction_type in ['halogen', 'halogen_bond']:
            return self._check_halogen_bonding_capability(mol)
            
        elif interaction_type in ['metal', 'metal_coordination']:
            return self._check_metal_coordination_capability(mol)
        
        return 0.0
    
    def _check_halogen_bonding_capability(self, mol: Chem.Mol) -> float:
        """Check for halogen bonding capability"""
        try:
            halogen_count = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() in [9, 17, 35, 53]:  # F, Cl, Br, I
                    if any(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()):
                        halogen_count += 1
            
            return min(1.0, halogen_count / 2.0)
        except Exception:
            return 0.0
    
    def _check_metal_coordination_capability(self, mol: Chem.Mol) -> float:
        """Check for metal coordination capability"""
        try:
            coordination_score = 0.0
            
            coordinating_patterns = ['[N]', '[O]', '[S]', '[P]']
            for pattern in coordinating_patterns:
                try:
                    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                    coordination_score += len(matches) * 0.1
                except Exception:
                    continue
            
            return min(1.0, coordination_score)
        except Exception:
            return 0.0
    
    def _calculate_feature_combination_bonus(self, mol_features: Dict) -> float:
        """Calculate bonus for favorable feature combinations"""
        bonus = 0.0
        
        # Classic drug-like combination: HBD + HBA + Aromatic
        if (mol_features.get('hbd_count', 0) > 0 and 
            mol_features.get('hba_count', 0) > 0 and 
            mol_features.get('aromatic_rings', 0) > 0):
            bonus += 0.1
        
        # Balanced polarity
        hbd_count = mol_features.get('hbd_count', 0)
        hba_count = mol_features.get('hba_count', 1)
        if hba_count > 0:
            hbd_hba_ratio = hbd_count / hba_count
            if 0.3 <= hbd_hba_ratio <= 1.5:
                bonus += 0.05
        
        # Appropriate hydrophobic content
        hydrophobic_area = mol_features.get('hydrophobic_area', 0)
        aromatic_rings = mol_features.get('aromatic_rings', 0)
        if 50 <= hydrophobic_area <= 200 and aromatic_rings >= 1:
            bonus += 0.05
        
        return min(0.2, bonus)
    
    def _extract_molecular_features(self, mol: Chem.Mol) -> Dict:
        """Enhanced molecular feature extraction"""
        try:
            features = {
                'hbd_count': Lipinski.NumHDonors(mol),
                'hba_count': Lipinski.NumHAcceptors(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'hydrophobic_area': self._calculate_hydrophobic_area(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'positive_charge': 0,
                'negative_charge': 0
            }
            
            # Enhanced charge analysis
            for atom in mol.GetAtoms():
                charge = atom.GetFormalCharge()
                if charge > 0:
                    features['positive_charge'] += charge
                elif charge < 0:
                    features['negative_charge'] += abs(charge)
            
            # Additional features
            features['halogen_count'] = sum(
                1 for atom in mol.GetAtoms() 
                if atom.GetAtomicNum() in [9, 17, 35, 53]
            )
            
            features['sulfur_count'] = sum(
                1 for atom in mol.GetAtoms() 
                if atom.GetAtomicNum() == 16
            )
            
            features['nitrogen_count'] = sum(
                1 for atom in mol.GetAtoms() 
                if atom.GetAtomicNum() == 7
            )
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting molecular features: {e}")
            return {
                'hbd_count': 0, 'hba_count': 0, 'aromatic_rings': 0,
                'hydrophobic_area': 0, 'formal_charge': 0,
                'positive_charge': 0, 'negative_charge': 0,
                'halogen_count': 0, 'sulfur_count': 0, 'nitrogen_count': 0
            }
    
    def _calculate_hydrophobic_area(self, mol: Chem.Mol) -> float:
        """Enhanced hydrophobic surface area calculation"""
        try:
            hydrophobic_area = 0.0
            
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                
                if atomic_num == 6:
                    if atom.GetIsAromatic():
                        hydrophobic_area += 20.0
                    elif atom.GetHybridization() == Chem.HybridizationType.SP3:
                        hydrophobic_area += 15.0
                    else:
                        hydrophobic_area += 12.0
                
                elif atomic_num == 16:
                    hydrophobic_area += 18.0
                
                elif atomic_num in [17, 35, 53]:
                    hydrophobic_area += 10.0
            
            heavy_atoms = mol.GetNumHeavyAtoms()
            if heavy_atoms > 0:
                compactness_factor = min(1.5, heavy_atoms / 20.0)
                hydrophobic_area *= compactness_factor
            
            return hydrophobic_area
            
        except Exception as e:
            logger.warning(f"Error calculating hydrophobic area: {e}")
            return 0.0
    
    def _synthetic_feasibility_score(self, mol: Chem.Mol) -> float:
        """Enhanced synthetic feasibility assessment"""
        try:
            props = self._get_cached_properties(mol)
            
            score = 1.0
            
            complexity_factors = self._calculate_synthetic_complexity_factors(mol, props)
            
            for factor_name, factor_value in complexity_factors.items():
                if factor_name == 'ring_complexity':
                    score -= min(0.25, factor_value * 0.1)
                elif factor_name == 'stereocenter_penalty':
                    score -= min(0.2, factor_value * 0.05)
                elif factor_name == 'size_penalty':
                    score -= min(0.3, factor_value)
                elif factor_name == 'branching_penalty':
                    score -= min(0.2, factor_value)
                elif factor_name == 'heteroatom_penalty':
                    score -= min(0.15, factor_value)
                elif factor_name == 'functional_group_bonus':
                    score += min(0.15, factor_value)
                elif factor_name == 'sp3_bonus':
                    score += min(0.1, factor_value)
            
            problematic_penalty = self._check_problematic_substructures(mol)
            score -= problematic_penalty
            
            building_block_bonus = self._check_building_block_similarity(mol)
            score += building_block_bonus
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error in synthetic feasibility scoring: {e}")
            return 0.5
    
    def _calculate_synthetic_complexity_factors(self, mol: Chem.Mol, 
                                              props: Dict) -> Dict[str, float]:
        """Calculate detailed synthetic complexity factors"""
        factors = {}
        
        try:
            rings = props.get('rings', 0)
            aromatic_rings = props.get('aromatic_rings', 0)
            fused_rings = self._count_fused_rings(mol)
            factors['ring_complexity'] = rings + aromatic_rings * 0.5 + fused_rings * 2.0
            
            try:
                stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                factors['stereocenter_penalty'] = stereocenters
            except Exception:
                factors['stereocenter_penalty'] = 0
            
            heavy_atoms = props.get('heavy_atoms', 0)
            if heavy_atoms > 30:
                factors['size_penalty'] = (heavy_atoms - 30) * 0.02
            else:
                factors['size_penalty'] = 0
            
            rotatable_bonds = props.get('rotatable_bonds', 0)
            if rotatable_bonds > 8:
                factors['branching_penalty'] = (rotatable_bonds - 8) * 0.03
            else:
                factors['branching_penalty'] = 0
            
            heteroatom_count = self._count_heteroatoms(mol)
            factors['heteroatom_penalty'] = max(0, (heteroatom_count - 5) * 0.02)
            
            factors['functional_group_bonus'] = self._calculate_functional_group_bonus(mol)
            
            fsp3 = props.get('fsp3', rdMolDescriptors.CalcFractionCsp3(mol))
            factors['sp3_bonus'] = fsp3 * 0.1
            
        except Exception as e:
            logger.warning(f"Error calculating complexity factors: {e}")
            factors = {k: 0.0 for k in ['ring_complexity', 'stereocenter_penalty', 
                                       'size_penalty', 'branching_penalty', 
                                       'heteroatom_penalty', 'functional_group_bonus', 
                                       'sp3_bonus']}
        
        return factors
    
    def _count_fused_rings(self, mol: Chem.Mol) -> int:
        """Count fused ring systems"""
        try:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            
            fused_count = 0
            for i, ring1 in enumerate(atom_rings):
                for ring2 in atom_rings[i+1:]:
                    if set(ring1) & set(ring2):
                        fused_count += 1
            
            return fused_count
        except Exception:
            return 0
    
    def _count_heteroatoms(self, mol: Chem.Mol) -> int:
        """Count heteroatoms"""
        try:
            return sum(1 for atom in mol.GetAtoms() 
                      if atom.GetAtomicNum() not in [1, 6])
        except Exception:
            return 0
    
    def _calculate_functional_group_bonus(self, mol: Chem.Mol) -> float:
        """Calculate bonus for common functional groups"""
        bonus = 0.0
        
        try:
            for group_name, pattern_mol in self._common_patterns.items():
                if pattern_mol is not None:
                    matches = mol.GetSubstructMatches(pattern_mol)
                    if matches:
                        group_bonuses = {
                            'hydroxyl': 0.02, 'amino': 0.02, 'carboxyl': 0.03,
                            'amide': 0.03, 'nitrile': 0.01, 'ether': 0.02,
                            'ketone': 0.02, 'aldehyde': 0.01, 'ester': 0.02,
                            'sulfonamide': 0.01
                        }
                        bonus += group_bonuses.get(group_name, 0.01) * min(3, len(matches))
        except Exception as e:
            logger.warning(f"Error calculating FG bonus: {e}")
        
        return bonus
    
    def _check_problematic_substructures(self, mol: Chem.Mol) -> float:
        """Check for difficult-to-synthesize substructures"""
        penalty = 0.0
        
        try:
            problematic_patterns = {
                'peroxide': '[O][O]',
                'azide': '[N]=[N+]=[N-]',
                'nitroso': '[N]=O',
                'nitro_on_aromatic': 'c[N+](=O)[O-]',
                'cyclopropene': 'C1=CC1',
                'cyclobutene': 'C1=CCC1',
                'quaternary_carbon': '[C]([C])([C])([C])[C]',
            }
            
            for pattern_name, smarts in problematic_patterns.items():
                try:
                    pattern_mol = Chem.MolFromSmarts(smarts)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        if pattern_name in ['peroxide', 'azide']:
                            penalty += 0.3
                        elif pattern_name in ['cyclopropene', 'cyclobutene']:
                            penalty += 0.2
                        else:
                            penalty += 0.1
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error checking problematic substructures: {e}")
        
        return min(0.5, penalty)
    
    def _check_building_block_similarity(self, mol: Chem.Mol) -> float:
        """Check similarity to known building blocks"""
        try:
            building_block_patterns = [
                'c1ccccc1', 'c1ccc2ccccc2c1', 'c1cnccn1', 'c1cccnc1',
                'c1ccncc1', 'C1CCCCC1', 'C1CCCC1',
            ]
            
            bonus = 0.0
            for pattern in building_block_patterns:
                try:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        bonus += 0.02
                except Exception:
                    continue
            
            return min(0.1, bonus)
        except Exception:
            return 0.0
    
    def _drug_likeness_score(self, mol: Chem.Mol) -> float:
        """Enhanced drug-likeness assessment"""
        try:
            props = self._get_cached_properties(mol)
            
            qed_score = props.get('qed', 0.0)
            if qed_score == 0.0:
                try:
                    qed_score = QED.qed(mol)
                except Exception:
                    qed_score = 0.3
            
            ro5_score = self._calculate_enhanced_ro5_score(props)
            
            additional_scores = self._calculate_additional_drug_likeness_factors(mol, props)
            
            drug_likeness = (
                0.4 * qed_score + 
                0.3 * ro5_score + 
                0.15 * additional_scores['tpsa_score'] +
                0.10 * additional_scores['complexity_score'] +
                0.05 * additional_scores['charge_score']
            )
            
            return min(1.0, drug_likeness)
            
        except Exception as e:
            logger.warning(f"Error in drug-likeness scoring: {e}")
            return 0.3
    
    def _calculate_enhanced_ro5_score(self, props: Dict) -> float:
        """Calculate enhanced Rule of 5 score"""
        thresholds = self.config.lipinski_thresholds
        score = 1.0
        
        mw = props.get('mw', 0)
        if mw > thresholds['mw'][1]:
            excess = mw - thresholds['mw'][1]
            score -= min(0.3, excess / 200.0)
        elif mw < 150:
            score -= 0.1
        
        logp = props.get('logp', 0)
        if logp > thresholds['logp'][1]:
            excess = logp - thresholds['logp'][1]
            score -= min(0.25, excess / 2.0)
        elif logp < -1:
            score -= 0.15
        
        hbd = props.get('hbd', 0)
        if hbd > thresholds['hbd'][1]:
            score -= 0.2 * (hbd - thresholds['hbd'][1])
        
        hba = props.get('hba', 0)
        if hba > thresholds['hba'][1]:
            score -= 0.15 * (hba - thresholds['hba'][1])
        
        return max(0.0, score)
    
    def _calculate_additional_drug_likeness_factors(self, mol: Chem.Mol, 
                                                  props: Dict) -> Dict[str, float]:
        """Calculate additional drug-likeness factors"""
        scores = {}
        
        tpsa = props.get('tpsa', 0)
        if 20 <= tpsa <= 140:
            scores['tpsa_score'] = 1.0
        elif tpsa <= 20:
            scores['tpsa_score'] = 0.6
        elif tpsa <= 200:
            scores['tpsa_score'] = 0.8
        else:
            scores['tpsa_score'] = 0.4
        
        scores['complexity_score'] = self._calculate_molecular_complexity(mol)
        
        formal_charge = abs(props.get('formal_charge', 0))
        if formal_charge == 0:
            scores['charge_score'] = 1.0
        elif formal_charge == 1:
            scores['charge_score'] = 0.8
        elif formal_charge == 2:
            scores['charge_score'] = 0.6
        else:
            scores['charge_score'] = 0.4
        
        return scores
    
    def _calculate_molecular_complexity(self, mol: Chem.Mol) -> float:
        """Enhanced molecular complexity calculation"""
        try:
            complexity_factors = []
            
            rings = rdMolDescriptors.CalcNumRings(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            fused_rings = self._count_fused_rings(mol)
            
            ring_complexity = (rings * 0.08 + aromatic_rings * 0.04 + fused_rings * 0.15)
            complexity_factors.append(min(1.0, ring_complexity))
            
            try:
                stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                stereo_complexity = stereocenters * 0.12
                complexity_factors.append(min(1.0, stereo_complexity))
            except Exception:
                complexity_factors.append(0.0)
            
            elements = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
            hetero_complexity = (len(elements) - 1) * 0.08
            complexity_factors.append(min(1.0, hetero_complexity))
            
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            branch_complexity = rotatable_bonds * 0.04
            complexity_factors.append(min(1.0, branch_complexity))
            
            fg_complexity = self._calculate_functional_group_complexity(mol)
            complexity_factors.append(fg_complexity)
            
            heavy_atoms = mol.GetNumHeavyAtoms()
            size_complexity = min(1.0, max(0, heavy_atoms - 15) * 0.02)
            complexity_factors.append(size_complexity)
            
            avg_complexity = np.mean(complexity_factors)
            complexity_score = max(0.0, 1.0 - avg_complexity)
            
            if 0.3 <= avg_complexity <= 0.7:
                complexity_score += 0.1
            
            return min(1.0, complexity_score)
            
        except Exception as e:
            logger.warning(f"Error calculating molecular complexity: {e}")
            return 0.5
    
    def _calculate_functional_group_complexity(self, mol: Chem.Mol) -> float:
        """Calculate complexity based on functional groups"""
        try:
            complexity = 0.0
            
            complex_patterns = {
                'nitro': '[N+](=O)[O-]',
                'sulfonamide': '[S](=O)(=O)[NH2]',
                'phosphate': '[P](=O)([OH])([OH])[OH]',
                'quaternary_ammonium': '[N+]([C])([C])([C])[C]'
            }
            
            for pattern_name, smarts in complex_patterns.items():
                try:
                    pattern_mol = Chem.MolFromSmarts(smarts)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        matches = mol.GetSubstructMatches(pattern_mol)
                        complexity += len(matches) * 0.1
                except Exception:
                    continue
            
            return min(1.0, complexity)
            
        except Exception:
            return 0.0
    
    def _novelty_score(self, mol: Chem.Mol) -> float:
        """Enhanced novelty score"""
        try:
            if not self.reference_fingerprints:
                return 0.8
            
            base_novelty = self.diversity.calculate_novelty_score(mol, self.reference_fingerprints)
            scaffold_novelty = self._calculate_scaffold_novelty(mol)
            
            combined_novelty = 0.7 * base_novelty + 0.3 * scaffold_novelty
            
            return min(1.0, combined_novelty)
            
        except Exception as e:
            logger.warning(f"Error in novelty scoring: {e}")
            return 0.5
    
    def _calculate_scaffold_novelty(self, mol: Chem.Mol) -> float:
        """Calculate novelty based on molecular scaffold"""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                return 0.8
            
            scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(
                scaffold, radius=2, nBits=1024
            )
            
            max_similarity = 0.0
            for ref_fp in self.reference_fingerprints:
                try:
                    similarity = DataStructs.TanimotoSimilarity(scaffold_fp, ref_fp)
                    max_similarity = max(max_similarity, similarity)
                except Exception:
                    continue
            
            scaffold_novelty = 1.0 - max_similarity
            return scaffold_novelty
            
        except Exception as e:
            logger.warning(f"Error calculating scaffold novelty: {e}")
            return 0.5
    
    def _selectivity_score(self, mol: Chem.Mol) -> float:
        """Enhanced selectivity estimation"""
        try:
            props = self._get_cached_properties(mol)
            
            selectivity_factors = self._calculate_selectivity_factors(mol, props)
            
            factor_weights = {
                'size_selectivity': 0.25,
                'hydrophobicity_selectivity': 0.25,
                'polarity_selectivity': 0.20,
                'complexity_selectivity': 0.15,
                'charge_selectivity': 0.10,
                'stereochemistry_selectivity': 0.05
            }
            
            weighted_score = sum(
                factor_weights.get(factor, 0.1) * score 
                for factor, score in selectivity_factors.items()
            )
            
            return min(1.0, weighted_score)
            
        except Exception as e:
            logger.warning(f"Error in selectivity scoring: {e}")
            return 0.5
    
    def _calculate_selectivity_factors(self, mol: Chem.Mol, props: Dict) -> Dict[str, float]:
        """Calculate detailed selectivity factors"""
        factors = {}
        
        try:
            mw = props.get('mw', 0)
            if 250 <= mw <= 650:
                factors['size_selectivity'] = 0.9
            elif 200 <= mw <= 650:
                factors['size_selectivity'] = 0.7
            else:
                factors['size_selectivity'] = 0.4
            
            logp = props.get('logp', 0)
            if 1 <= logp <= 4:
                factors['hydrophobicity_selectivity'] = 0.9
            elif 0 <= logp <= 5:
                factors['hydrophobicity_selectivity'] = 0.7
            else:
                factors['hydrophobicity_selectivity'] = 0.4
            
            hbd = props.get('hbd', 0)
            hba = props.get('hba', 0)
            if hbd >= 1 and hba >= 2 and (hbd + hba) <= 8:
                factors['polarity_selectivity'] = 0.8
            elif (hbd + hba) <= 10:
                factors['polarity_selectivity'] = 0.6
            else:
                factors['polarity_selectivity'] = 0.3
            
            aromatic_rings = props.get('aromatic_rings', 0)
            rings = props.get('rings', 0)
            if 1 <= aromatic_rings <= 3 and rings <= 4:
                factors['complexity_selectivity'] = 0.8
            elif aromatic_rings <= 4 and rings <= 6:
                factors['complexity_selectivity'] = 0.6
            else:
                factors['complexity_selectivity'] = 0.4
            
            formal_charge = abs(props.get('formal_charge', 0))
            if formal_charge == 0:
                factors['charge_selectivity'] = 1.0
            elif formal_charge == 1:
                factors['charge_selectivity'] = 0.7
            else:
                factors['charge_selectivity'] = 0.4
            
            try:
                stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                if 1 <= stereocenters <= 3:
                    factors['stereochemistry_selectivity'] = 0.8
                elif stereocenters == 0:
                    factors['stereochemistry_selectivity'] = 0.6
                else:
                    factors['stereochemistry_selectivity'] = 0.4
            except Exception:
                factors['stereochemistry_selectivity'] = 0.6
            
        except Exception as e:
            logger.warning(f"Error calculating selectivity factors: {e}")
            factors = {k: 0.5 for k in ['size_selectivity', 'hydrophobicity_selectivity',
                                        'polarity_selectivity', 'complexity_selectivity',
                                        'charge_selectivity', 'stereochemistry_selectivity']}
        
        return factors
    
    def _water_displacement_score(self, mol: Chem.Mol) -> float:
        """Enhanced water displacement scoring"""
        try:
            water_sites = getattr(self.pocket, 'water_sites', [])
            if not water_sites:
                return 0.5
            
            displaceable_waters = [
                w for w in water_sites 
                if (getattr(w, 'replaceability_score', 0.5) > 0.6 and 
                    getattr(w, 'coordination_number', 3) <= 2)
            ]
            
            if not displaceable_waters:
                return 0.3
            
            displacement_features = self._calculate_water_displacement_features(mol)
            
            displacement_scores = []
            
            hydrophobic_score = min(1.0, displacement_features['hydrophobic_area'] / 150.0)
            displacement_scores.append(('hydrophobic', hydrophobic_score, 0.4))
            
            hbond_score = min(1.0, displacement_features['hbond_potential'] / 6.0)
            displacement_scores.append(('hbond', hbond_score, 0.4))
            
            pi_score = min(1.0, displacement_features['aromatic_area'] / 100.0)
            displacement_scores.append(('pi_interaction', pi_score, 0.2))
            
            total_displacement = sum(score * weight for _, score, weight in displacement_scores)
            
            water_factor = self._calculate_water_displacement_factor(displaceable_waters)
            
            final_score = total_displacement * water_factor
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning(f"Error in water displacement scoring: {e}")
            return 0.4
    
    def _calculate_water_displacement_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate features relevant to water displacement"""
        features = {}
        
        try:
            props = self._get_cached_properties(mol)
            
            features['hydrophobic_area'] = props.get('hydrophobic_area', 0)
            
            hbd = props.get('hbd', 0)
            hba = props.get('hba', 0)
            features['hbond_potential'] = hbd + hba
            
            aromatic_rings = props.get('aromatic_rings', 0)
            features['aromatic_area'] = aromatic_rings * 50.0
            
            formal_charge = abs(props.get('formal_charge', 0))
            features['electrostatic_potential'] = formal_charge
            
            heavy_atoms = props.get('heavy_atoms', 0)
            features['molecular_volume'] = heavy_atoms * 20.0
            
        except Exception as e:
            logger.warning(f"Error calculating displacement features: {e}")
            features = {
                'hydrophobic_area': 0, 'hbond_potential': 0, 'aromatic_area': 0,
                'electrostatic_potential': 0, 'molecular_volume': 0
            }
        
        return features
    
    def _calculate_water_displacement_factor(self, displaceable_waters: List) -> float:
        """Calculate factor based on displaceable water properties"""
        if not displaceable_waters:
            return 0.0
        
        try:
            base_factor = min(1.0, len(displaceable_waters) / 3.0)
            
            quality_scores = [
                getattr(w, 'replaceability_score', 0.5) 
                for w in displaceable_waters
            ]
            avg_quality = np.mean(quality_scores)
            quality_factor = (avg_quality - 0.5) * 2.0
            
            total_factor = 0.7 * base_factor + 0.3 * quality_factor
            
            return max(0.0, min(1.0, total_factor))
            
        except Exception:
            return 0.5
    
    def _calculate_weighted_score(self, scores: Dict[str, float], 
                                weights: Dict[str, float]) -> float:
        """Calculate weighted total score"""
        try:
            missing_weights = set(scores.keys()) - set(weights.keys())
            if missing_weights:
                logger.warning(f"Missing weights: {missing_weights}")
                equal_weight = 1.0 / len(scores)
                for component in missing_weights:
                    weights[component] = equal_weight
            
            total_weight = sum(weights.values())
            if total_weight == 0:
                logger.error("Total weight zero, using equal weights")
                weights = {k: 1.0/len(scores) for k in scores.keys()}
                total_weight = 1.0
            
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            weighted_score = sum(normalized_weights.get(key, 0.0) * score 
                               for key, score in scores.items())
            
            return weighted_score
            
        except Exception as e:
            logger.error(f"Error calculating weighted score: {e}")
            return np.mean(list(scores.values()))
    
    def _get_adaptive_weights(self, generation_round: int, 
                           scores: Dict[str, float]) -> Dict[str, float]:
        """Enhanced adaptive weight calculation"""
        base_weights = self.config.reward_weights.copy()
        
        # Phase-based adaptation
        if generation_round < 5:
            base_weights['drug_likeness'] *= 1.3
            base_weights['synthetic'] *= 1.3
            base_weights['novelty'] *= 0.7
            base_weights['selectivity'] *= 0.8
            
        elif generation_round < 15:
            base_weights['pharmacophore'] *= 1.1
            base_weights['novelty'] *= 1.1
            
        else:
            base_weights['pharmacophore'] *= 1.3  # Increased for 3D
            base_weights['selectivity'] *= 1.4
            base_weights['water_displacement'] *= 1.2
            base_weights['novelty'] *= 1.3
        
        # Adaptive adjustment
        with self._history_lock:
            if len(self.scoring_history) > 20:
                recent_scores = self.scoring_history[-20:]
                weight_adjustments = self._calculate_adaptive_adjustments(recent_scores)
                
                for component, adjustment in weight_adjustments.items():
                    if component in base_weights:
                        base_weights[component] *= adjustment
        
        # Normalize
        total_weight = sum(base_weights.values())
        if total_weight == 0:
            score_keys = ['pharmacophore', 'synthetic', 'drug_likeness', 
                         'novelty', 'selectivity', 'water_displacement']
            return {k: 1.0/len(score_keys) for k in score_keys}
        
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        return normalized_weights
    
    def _calculate_adaptive_adjustments(self, recent_scores: List[Dict]) -> Dict[str, float]:
        """Calculate adaptive weight adjustments"""
        adjustments = {}
        
        try:
            for component in ['pharmacophore', 'synthetic', 'drug_likeness', 
                            'novelty', 'selectivity', 'water_displacement']:
                component_scores = [entry['scores'][component] for entry in recent_scores]
                
                avg_score = np.mean(component_scores)
                score_std = np.std(component_scores)
                
                if avg_score < 0.3:
                    adjustments[component] = 1.4
                elif avg_score < 0.5:
                    adjustments[component] = 1.2
                elif avg_score > 0.8 and score_std < 0.1:
                    adjustments[component] = 0.9
                else:
                    adjustments[component] = 1.0
                    
                if component == 'novelty' and avg_score > 0.9:
                    adjustments[component] = 0.7
        
        except Exception as e:
            logger.warning(f"Error in adaptive adjustments: {e}")
            adjustments = {comp: 1.0 for comp in ['pharmacophore', 'synthetic', 
                          'drug_likeness', 'novelty', 'selectivity', 'water_displacement']}
        
        return adjustments
    
    def _calculate_molecular_properties(self, mol: Chem.Mol) -> Dict:
        """Enhanced molecular properties calculation"""
        if mol is None:
            return self._get_default_properties()
        
        mol_hash = self._get_molecule_hash(mol)
        with self._cache_lock:
            cached_props = self._property_cache.get(mol_hash, {}).get('properties')
            if cached_props:
                return cached_props
        
        props = {}
        
        try:
            descriptors = {
                'mw': lambda m: rdMolDescriptors.CalcExactMolWt(m),
                'logp': lambda m: Crippen.MolLogP(m),
                'hbd': lambda m: Lipinski.NumHDonors(m),
                'hba': lambda m: Lipinski.NumHAcceptors(m),
                'tpsa': lambda m: rdMolDescriptors.CalcTPSA(m),
                'rotatable_bonds': lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
                'aromatic_rings': lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
                'rings': lambda m: rdMolDescriptors.CalcNumRings(m),
                'heavy_atoms': lambda m: m.GetNumHeavyAtoms(),
                'formal_charge': lambda m: Chem.rdmolops.GetFormalCharge(m),
                'fsp3': lambda m: rdMolDescriptors.CalcFractionCsp3(m)
            }
            
            for prop_name, calc_func in descriptors.items():
                try:
                    props[prop_name] = calc_func(mol)
                except Exception as e:
                    logger.warning(f"Error calculating {prop_name}: {e}")
                    props[prop_name] = self._get_default_properties()[prop_name]
            
            try:
                props['qed'] = QED.qed(mol)
            except Exception as e:
                logger.warning(f"Error calculating QED: {e}")
                props['qed'] = 0.0
            
            props['hydrophobic_area'] = self._calculate_hydrophobic_area(mol)
            
            props = self._validate_properties(props)
            
            self._cache_properties(mol_hash, props)
            
            return props
            
        except Exception as e:
            logger.error(f"Error calculating molecular properties: {e}")
            return self._get_default_properties()
    
    def _get_default_properties(self) -> Dict:
        """Get default property values"""
        return {
            'mw': 0.0, 'logp': 0.0, 'hbd': 0, 'hba': 0, 'tpsa': 0.0,
            'rotatable_bonds': 0, 'aromatic_rings': 0, 'rings': 0,
            'heavy_atoms': 0, 'qed': 0.0, 'formal_charge': 0, 'fsp3': 0.0,
            'hydrophobic_area': 0.0
        }
    
    def _validate_properties(self, props: Dict) -> Dict:
        """Validate and clean property values"""
        ranges = {
            'mw': (0, 2000), 'logp': (-10, 15), 'hbd': (0, 20), 'hba': (0, 30),
            'tpsa': (0, 500), 'rotatable_bonds': (0, 50), 'aromatic_rings': (0, 10),
            'rings': (0, 15), 'heavy_atoms': (0, 200), 'qed': (0, 1),
            'formal_charge': (-10, 10), 'fsp3': (0, 1), 'hydrophobic_area': (0, 1000)
        }
        
        validated_props = {}
        defaults = self._get_default_properties()
        
        for prop_name, value in props.items():
            try:
                if not np.isfinite(value):
                    validated_props[prop_name] = defaults[prop_name]
                    continue
                
                if prop_name in ranges:
                    min_val, max_val = ranges[prop_name]
                    if min_val <= value <= max_val:
                        validated_props[prop_name] = value
                    else:
                        logger.warning(f"Property {prop_name} out of range: {value}")
                        validated_props[prop_name] = np.clip(value, min_val, max_val)
                else:
                    validated_props[prop_name] = value
                    
            except Exception as e:
                logger.warning(f"Error validating {prop_name}: {e}")
                validated_props[prop_name] = defaults.get(prop_name, 0.0)
        
        return validated_props
    
    def _cache_properties(self, mol_hash: str, props: Dict):
        """Cache properties with size management"""
        try:
            with self._cache_lock:
                if len(self._property_cache) >= self._max_cache_size:
                    keys_to_remove = list(self._property_cache.keys())[:self._max_cache_size // 4]
                    for key in keys_to_remove:
                        del self._property_cache[key]
                
                self._property_cache[mol_hash] = {'properties': props}
                
        except Exception as e:
            logger.warning(f"Error caching properties: {e}")
    
    def _cache_scoring_result(self, mol_hash: str, score: MolecularScore, props: Dict):
        """Cache complete scoring result"""
        try:
            with self._cache_lock:
                if mol_hash in self._property_cache:
                    self._property_cache[mol_hash]['score_result'] = score
                else:
                    self._property_cache[mol_hash] = {
                        'properties': props,
                        'score_result': score
                    }
        except Exception as e:
            logger.warning(f"Error caching score: {e}")
    
    def _update_scoring_history(self, generation_round: int, scores: Dict, 
                              total_score: float, properties: Dict):
        """Update scoring history"""
        try:
            with self._history_lock:
                if len(self.scoring_history) >= self._max_history_size:
                    del self.scoring_history[:self._max_history_size//2]
                
                self.scoring_history.append({
                    'generation_round': generation_round,
                    'scores': scores.copy(),
                    'total_score': total_score,
                    'properties': properties.copy(),
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.warning(f"Error updating history: {e}")
    
    def _update_performance_stats(self, scoring_time: float, scores: Dict):
        """Update performance statistics"""
        try:
            with self._stats_lock:
                self._scoring_stats['total_molecules'] += 1
                
                alpha = 0.1
                if self._scoring_stats['avg_score_time'] == 0:
                    self._scoring_stats['avg_score_time'] = scoring_time
                else:
                    self._scoring_stats['avg_score_time'] = (
                        alpha * scoring_time + 
                        (1 - alpha) * self._scoring_stats['avg_score_time']
                    )
                    
        except Exception as e:
            logger.warning(f"Error updating stats: {e}")
    
    def _get_cached_properties(self, mol: Chem.Mol) -> Dict:
        """Get cached properties"""
        return self._calculate_molecular_properties(mol)
    
    def _check_violations(self, mol: Chem.Mol, properties: Dict) -> List[str]:
        """Enhanced violation checking"""
        violations = []
        
        try:
            lipinski_violations = self._check_lipinski_violations(properties)
            violations.extend(lipinski_violations)
            
            config_violations = self._check_config_violations(properties)
            violations.extend(config_violations)
            
            structural_violations = self._check_structural_violations(mol)
            violations.extend(structural_violations)
            
            admet_violations = self._check_admet_violations(properties)
            violations.extend(admet_violations)
            
        except Exception as e:
            violations.append(f"Error checking violations: {str(e)}")
        
        return violations
    
    def _check_lipinski_violations(self, properties: Dict) -> List[str]:
        """Check Lipinski Rule of 5 violations"""
        violations = []
        thresholds = self.config.lipinski_thresholds
        
        try:
            if properties.get('mw', 0) > thresholds['mw'][1]:
                violations.append(f"MW {properties['mw']:.1f} > {thresholds['mw'][1]}")
            
            if properties.get('logp', 0) > thresholds['logp'][1]:
                violations.append(f"LogP {properties['logp']:.2f} > {thresholds['logp'][1]}")
            
            if properties.get('hbd', 0) > thresholds['hbd'][1]:
                violations.append(f"HBD {properties['hbd']} > {thresholds['hbd'][1]}")
            
            if properties.get('hba', 0) > thresholds['hba'][1]:
                violations.append(f"HBA {properties['hba']} > {thresholds['hba'][1]}")
            
            if properties.get('tpsa', 0) > thresholds.get('tpsa', [0, 140])[1]:
                violations.append(f"TPSA {properties['tpsa']:.1f} > {thresholds.get('tpsa', [0, 140])[1]}")
                
        except Exception as e:
            violations.append(f"Error checking Lipinski: {str(e)}")
        
        return violations
    
    def _check_config_violations(self, properties: Dict) -> List[str]:
        """Check configuration violations"""
        violations = []
        
        try:
            if properties.get('heavy_atoms', 0) > self.config.max_heavy_atoms:
                violations.append(f"Heavy atoms {properties['heavy_atoms']} > {self.config.max_heavy_atoms}")
            
            if properties.get('rotatable_bonds', 0) > self.config.max_rotatable_bonds:
                violations.append(f"Rotatable bonds {properties['rotatable_bonds']} > {self.config.max_rotatable_bonds}")
            
            if properties.get('rings', 0) > self.config.max_rings:
                violations.append(f"Rings {properties['rings']} > {self.config.max_rings}")
                
        except Exception as e:
            violations.append(f"Error checking config: {str(e)}")
        
        return violations
    
    def _check_structural_violations(self, mol: Chem.Mol) -> List[str]:
        """Check structural violations"""
        violations = []
        
        try:
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                violations.append("Disconnected structure")
            
            try:
                Chem.SanitizeMol(mol)
            except Chem.AtomValenceException:
                violations.append("Unusual atom valencies")
            except Exception:
                violations.append("Sanitization issue")
            
            if mol.GetNumHeavyAtoms() < 5:
                violations.append("Too small (< 5 heavy atoms)")
            
            if mol.GetNumHeavyAtoms() > 100:
                violations.append("Too large (> 100 heavy atoms)")
                
        except Exception as e:
            violations.append(f"Structural check error: {str(e)}")
        
        return violations
    
    def _check_admet_violations(self, properties: Dict) -> List[str]:
        """Check ADMET violations"""
        violations = []
        
        try:
            logp = properties.get('logp', 0)
            if logp > 6:
                violations.append(f"Very high LogP {logp:.2f}")
            elif logp < -3:
                violations.append(f"Very low LogP {logp:.2f}")
            
            tpsa = properties.get('tpsa', 0)
            if tpsa > 200:
                violations.append(f"Very high TPSA {tpsa:.1f}")
            elif tpsa < 10:
                violations.append(f"Very low TPSA {tpsa:.1f}")
            
            mw = properties.get('mw', 0)
            if mw > 800:
                violations.append(f"Very high MW {mw:.1f}")
            elif mw < 100:
                violations.append(f"Very low MW {mw:.1f}")
            
            rotatable_bonds = properties.get('rotatable_bonds', 0)
            if rotatable_bonds > 15:
                violations.append(f"Too many rotatable bonds {rotatable_bonds}")
                
        except Exception as e:
            violations.append(f"ADMET check error: {str(e)}")
        
        return violations
    
    def _calculate_penalty_factor(self, violations: List[str]) -> float:
        """Calculate penalty factor"""
        if not violations:
            return 1.0
        
        violation_weights = {
            'Molecular weight': 0.15, 'LogP': 0.10, 'HBD': 0.05, 'HBA': 0.05,
            'TPSA': 0.08, 'Heavy atoms': 0.12, 'Rotatable bonds': 0.08,
            'Rings': 0.10, 'Disconnected': 0.20, 'Sanitization': 0.25,
            'Invalid': 0.30
        }
        
        total_penalty = 0.0
        for violation in violations:
            penalty = 0.10
            for viol_type, weight in violation_weights.items():
                if viol_type.lower() in violation.lower():
                    penalty = weight
                    break
            total_penalty += penalty
        
        penalty_factor = max(0.05, 1.0 - total_penalty)
        return penalty_factor
    
    def _calculate_confidence(self, scores: Dict[str, float], 
                            properties: Dict, violations: List[str]) -> float:
        """Enhanced confidence calculation"""
        try:
            confidence_factors = []
            
            score_values = list(scores.values())
            score_variance = np.var(score_values)
            score_mean = np.mean(score_values)
            
            if score_variance < 0.1 and score_mean > 0.3:
                confidence_factors.append(0.9)
            elif score_variance < 0.2:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            property_confidence = self._calculate_property_confidence(properties)
            confidence_factors.append(property_confidence)
            
            violation_penalty = min(0.8, len(violations) * 0.1)
            violation_confidence = max(0.2, 1.0 - violation_penalty)
            confidence_factors.append(violation_confidence)
            
            qed = properties.get('qed', 0)
            if qed > 0:
                qed_confidence = min(1.0, qed + 0.3)
                confidence_factors.append(qed_confidence)
            
            if all(0 <= score <= 1 for score in score_values):
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
            
            weights = [0.25, 0.25, 0.20, 0.15, 0.15]
            if len(confidence_factors) != len(weights):
                return np.mean(confidence_factors)
            
            confidence = sum(w * f for w, f in zip(weights, confidence_factors))
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_property_confidence(self, properties: Dict) -> float:
        """Calculate confidence based on properties"""
        try:
            property_scores = []
            
            mw = properties.get('mw', 0)
            if 150 <= mw <= 600:
                property_scores.append(1.0)
            elif 100 <= mw <= 800:
                property_scores.append(0.8)
            else:
                property_scores.append(0.4)
            
            logp = properties.get('logp', 0)
            if -2 <= logp <= 6:
                property_scores.append(1.0)
            elif -4 <= logp <= 8:
                property_scores.append(0.7)
            else:
                property_scores.append(0.3)
            
            tpsa = properties.get('tpsa', 0)
            if 20 <= tpsa <= 140:
                property_scores.append(1.0)
            elif 0 <= tpsa <= 200:
                property_scores.append(0.8)
            else:
                property_scores.append(0.4)
            
            heavy_atoms = properties.get('heavy_atoms', 0)
            if 10 <= heavy_atoms <= 50:
                property_scores.append(1.0)
            elif 5 <= heavy_atoms <= 80:
                property_scores.append(0.8)
            else:
                property_scores.append(0.4)
            
            return np.mean(property_scores)
            
        except Exception:
            return 0.5
    
    def _create_zero_score(self, violations: List[str] = None) -> MolecularScore:
        """Create zero score for invalid molecules"""
        if violations is None:
            violations = ["Invalid molecule"]
        
        return MolecularScore(
            total_score=0.0,
            pharmacophore_score=0.0,
            synthetic_score=0.0,
            drug_likeness_score=0.0,
            novelty_score=0.0,
            selectivity_score=0.0,
            water_displacement_score=0.0,
            component_scores={
                'pharmacophore': 0.0, 'synthetic': 0.0, 'drug_likeness': 0.0,
                'novelty': 0.0, 'selectivity': 0.0, 'water_displacement': 0.0
            },
            property_values=self._get_default_properties(),
            violations=violations,
            confidence=0.0
        )
    
    def _get_molecule_hash(self, mol: Chem.Mol) -> str:
        """Get molecule hash for caching"""
        try:
            return Chem.MolToSmiles(mol, canonical=True, doRandom=False, allHsExplicit=False)
        except Exception:
            try:
                return Chem.MolToInchiKey(mol)
            except Exception:
                return f"mol_hash_{id(mol)}"
    
    def add_reference_molecules(self, reference_mols: List[Chem.Mol]):
        """Add reference molecules"""
        added_count = 0
        
        for mol in reference_mols:
            try:
                if mol is None:
                    continue
                
                Chem.SanitizeMol(mol)
                
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=self.config.fingerprint_radius, 
                    nBits=self.config.fingerprint_bits
                )
                
                self.reference_fingerprints.append(fp)
                added_count += 1
                
            except Exception as e:
                logger.warning(f"Error adding reference: {e}")
                continue
        
        logger.info(f"Added {added_count} reference molecules")
    
    def get_scoring_statistics(self) -> Dict:
        """Get comprehensive scoring statistics"""
        stats = {}
        
        try:
            with self._history_lock:
                if not self.scoring_history:
                    return {'message': 'No scoring history'}
                
                component_stats = {}
                for component in ['pharmacophore', 'synthetic', 'drug_likeness', 
                                'novelty', 'selectivity', 'water_displacement']:
                    scores = [entry['scores'][component] for entry in self.scoring_history]
                    component_stats[component] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores)),
                        'median': float(np.median(scores))
                    }
                
                stats['component_statistics'] = component_stats
                
                total_scores = [entry['total_score'] for entry in self.scoring_history]
                stats['overall_statistics'] = {
                    'mean_total_score': float(np.mean(total_scores)),
                    'std_total_score': float(np.std(total_scores)),
                    'min_total_score': float(np.min(total_scores)),
                    'max_total_score': float(np.max(total_scores)),
                    'total_molecules_scored': len(self.scoring_history)
                }
                
                high_scoring = sum(1 for score in total_scores if score > 0.7)
                medium_scoring = sum(1 for score in total_scores if 0.4 <= score <= 0.7)
                low_scoring = sum(1 for score in total_scores if score < 0.4)
                
                stats['score_distribution'] = {
                    'high_scoring_molecules': high_scoring,
                    'medium_scoring_molecules': medium_scoring,
                    'low_scoring_molecules': low_scoring,
                    'high_scoring_percentage': float(high_scoring / len(total_scores) * 100),
                    'medium_scoring_percentage': float(medium_scoring / len(total_scores) * 100),
                    'low_scoring_percentage': float(low_scoring / len(total_scores) * 100)
                }
            
            with self._stats_lock:
                stats['performance_statistics'] = self._scoring_stats.copy()
                stats['performance_statistics']['cache_hit_rate'] = (
                    self._scoring_stats['cache_hits'] / max(1, self._scoring_stats['total_molecules']) * 100
                )
                if self._scoring_stats['3d_scoring_attempts'] > 0:
                    stats['performance_statistics']['3d_success_rate'] = (
                        self._scoring_stats['3d_scoring_successes'] / 
                        self._scoring_stats['3d_scoring_attempts'] * 100
                    )
            
            with self._history_lock:
                if len(self.scoring_history) > 10:
                    total_scores = [entry['total_score'] for entry in self.scoring_history]
                    recent_scores = total_scores[-10:]
                    older_scores = total_scores[-20:-10] if len(total_scores) > 20 else total_scores[:-10]
                    
                    if older_scores:
                        trend = np.mean(recent_scores) - np.mean(older_scores)
                        stats['trend_analysis'] = {
                            'score_trend': float(trend),
                            'trend_direction': 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable'
                        }
        
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def clear_cache(self):
        """Clear caches"""
        with self._cache_lock:
            cache_size = len(self._property_cache)
            fingerprint_cache_size = len(self._fingerprint_cache)
            
            self._property_cache.clear()
            self._fingerprint_cache.clear()
            
            logger.info(f"Cleared {cache_size} property entries, {fingerprint_cache_size} fingerprint entries")
    
    def reset_scoring_history(self):
        """Reset scoring history"""
        with self._history_lock:
            history_size = len(self.scoring_history)
            
            if history_size > 0:
                summary = {
                    'total_molecules': history_size,
                    'final_avg_score': np.mean([entry['total_score'] for entry in self.scoring_history[-10:]]),
                    'reset_timestamp': time.time()
                }
                logger.info(f"Resetting scoring history: {summary}")
            
            self.scoring_history.clear()
        
        with self._stats_lock:
            self._scoring_stats = {
                'total_molecules': 0,
                'cache_hits': 0,
                'errors': 0,
                'avg_score_time': 0.0,
                'avg_3d_alignment': 0.0,
                '3d_scoring_attempts': 0,
                '3d_scoring_successes': 0
            }
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'scoring_statistics': self.get_scoring_statistics(),
            'performance_metrics': self._scoring_stats.copy(),
            'cache_statistics': {
                'property_cache_size': len(self._property_cache),
                'fingerprint_cache_size': len(self._fingerprint_cache),
                'cache_utilization': min(1.0, len(self._property_cache) / self._max_cache_size)
            },
            'configuration': {
                'max_cache_size': self._max_cache_size,
                'max_history_size': self._max_history_size,
                'reward_weights': self.config.reward_weights.copy(),
                'num_pharmacophore_targets': len(self.pharmacophore_targets)
            }
        }
        
        return report
    
    def shutdown(self):
        """Proper shutdown"""
        try:
            self._shutdown = True
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=True, timeout=30)
            logger.info("Scorer shutdown complete")
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Cleanup"""
        if not getattr(self, '_shutdown', True):
            self.shutdown()

