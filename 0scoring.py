"""
Multi-Objective Scoring Module
Comprehensive scoring system for generated molecules
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski, QED, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import LigandForgeConfig
from data_structures import PocketAnalysis, MolecularScore
from diversity_manager import EnhancedDiversityManager


class MultiObjectiveScorer:
    """Multi-objective scoring for generated molecules"""
    
    def __init__(self, config: LigandForgeConfig, pocket_analysis: PocketAnalysis, 
                 diversity_manager: EnhancedDiversityManager):
        self.config = config
        self.pocket = pocket_analysis
        self.diversity = diversity_manager
        self.reference_fingerprints = []
        
        # Cache for expensive calculations
        self._property_cache = {}
        self._fingerprint_cache = {}
        
        # Scoring history for adaptive weights
        self.scoring_history = []
    
    def calculate_comprehensive_score(self, mol: Chem.Mol, 
                                    generation_round: int = 0) -> MolecularScore:
        """Calculate comprehensive multi-objective score"""
        
        if mol is None:
            return self._create_zero_score()
        
        try:
            # Calculate individual score components
            scores = {
                'pharmacophore': self._pharmacophore_score(mol),
                'synthetic': self._synthetic_feasibility_score(mol),
                'drug_likeness': self._drug_likeness_score(mol),
                'novelty': self._novelty_score(mol),
                'selectivity': self._selectivity_score(mol),
                'water_displacement': self._water_displacement_score(mol)
            }
            
            # Calculate weighted total
            weights = self._get_adaptive_weights(generation_round, scores)
            total_score = sum(weights[key] * score for key, score in scores.items())
            
            # Get molecular properties
            properties = self._calculate_molecular_properties(mol)
            
            # Check for violations
            violations = self._check_violations(mol, properties)
            
            # Apply penalties for violations
            penalty_factor = max(0.1, 1.0 - 0.1 * len(violations))
            total_score *= penalty_factor
            
            # Create comprehensive score object
            molecular_score = MolecularScore(
                total_score=float(np.clip(total_score, 0.0, 1.0)),
                pharmacophore_score=float(scores['pharmacophore']),
                synthetic_score=float(scores['synthetic']),
                drug_likeness_score=float(scores['drug_likeness']),
                novelty_score=float(scores['novelty']),
                selectivity_score=float(scores['selectivity']),
                water_displacement_score=float(scores['water_displacement']),
                component_scores=scores,
                property_values=properties,
                violations=violations,
                confidence=self._calculate_confidence(scores, properties)
            )
            
            # Store in history for adaptive learning
            self.scoring_history.append({
                'generation_round': generation_round,
                'scores': scores,
                'total_score': total_score,
                'properties': properties
            })
            
            return molecular_score
            
        except Exception as e:
            warnings.warn(f"Error calculating score: {e}")
            return self._create_zero_score()
    
    def _pharmacophore_score(self, mol: Chem.Mol) -> float:
        """Score based on pharmacophore fit to binding site"""
        hotspots = getattr(self.pocket, 'hotspots', [])
        if not hotspots:
            return 0.5
        
        try:
            # Get molecular features
            mol_features = self._extract_molecular_features(mol)
            
            # Calculate feature matching score
            total_match_score = 0.0
            total_weight = 0.0
            
            for hotspot in hotspots:
                interaction_type = hotspot.interaction_type
                hotspot_strength = hotspot.strength
                
                # Check if molecule has complementary features
                feature_score = 0.0
                
                if interaction_type == 'hbd' and mol_features['hbd_count'] > 0:
                    feature_score = min(1.0, mol_features['hbd_count'] / 3.0)
                elif interaction_type == 'hba' and mol_features['hba_count'] > 0:
                    feature_score = min(1.0, mol_features['hba_count'] / 5.0)
                elif interaction_type == 'hydrophobic' and mol_features['hydrophobic_area'] > 0:
                    feature_score = min(1.0, mol_features['hydrophobic_area'] / 200.0)
                elif interaction_type == 'aromatic' and mol_features['aromatic_rings'] > 0:
                    feature_score = min(1.0, mol_features['aromatic_rings'] / 3.0)
                elif interaction_type == 'electrostatic':
                    if mol_features['formal_charge'] != 0:
                        feature_score = 0.8
                    elif mol_features['hbd_count'] > 0 or mol_features['hba_count'] > 0:
                        feature_score = 0.5
                
                # Weight by hotspot strength
                weighted_score = feature_score * hotspot_strength
                total_match_score += weighted_score
                total_weight += hotspot_strength
            
            # Normalize by total weight
            if total_weight > 0:
                pharmacophore_score = total_match_score / total_weight
            else:
                pharmacophore_score = 0.5
            
            # Bonus for complementary feature combinations
            if (mol_features['hbd_count'] > 0 and mol_features['hba_count'] > 0 and 
                mol_features['aromatic_rings'] > 0):
                pharmacophore_score *= 1.1
            
            return min(1.0, pharmacophore_score)
            
        except Exception:
            return 0.3
    
    def _synthetic_feasibility_score(self, mol: Chem.Mol) -> float:
        """Assess synthetic feasibility"""
        try:
            props = self._get_cached_properties(mol)
            
            # Start with base score
            score = 1.0
            
            # Penalize complexity factors
            rings = props.get('rings', 0)
            score -= min(0.3, 0.05 * max(0, rings - 3))
            
            # Penalize stereocenters
            try:
                chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                score -= min(0.2, 0.05 * chiral_centers)
            except:
                pass
            
            # Penalize very large molecules
            heavy_atoms = mol.GetNumHeavyAtoms()
            if heavy_atoms > 30:
                score -= min(0.3, 0.02 * (heavy_atoms - 30))
            
            # Penalize highly branched molecules
            rotatable_bonds = props.get('rotatable_bonds', 0)
            if rotatable_bonds > 8:
                score -= min(0.2, 0.03 * (rotatable_bonds - 8))
            
            # Bonus for sp3 character (3D-ness)
            try:
                fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
                score += 0.1 * min(0.7, fsp3)
            except:
                pass
            
            # Penalize complex ring systems
            try:
                ring_info = mol.GetRingInfo()
                fused_rings = sum(1 for ring in ring_info.AtomRings() if len(ring) > 6)
                score -= min(0.15, 0.05 * fused_rings)
            except:
                pass
            
            # Bonus for common functional groups
            common_patterns = [
                "[OH]", "[NH2]", "[C](=O)[OH]", "[C](=O)[NH2]", "[C]#[N]"
            ]
            for pattern in common_patterns:
                try:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        score += 0.02
                except:
                    continue
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _drug_likeness_score(self, mol: Chem.Mol) -> float:
        """Calculate drug-likeness score"""
        try:
            props = self._get_cached_properties(mol)
            
            # QED score (quantitative estimate of drug-likeness)
            qed_score = props.get('qed', 0.0)
            
            # Rule of 5 compliance
            ro5_violations = 0
            thresholds = self.config.lipinski_thresholds
            
            if props.get('mw', 0) > thresholds['mw'][1]:
                ro5_violations += 1
            if props.get('logp', 0) > thresholds['logp'][1]:
                ro5_violations += 1
            if props.get('hbd', 0) > thresholds['hbd'][1]:
                ro5_violations += 1
            if props.get('hba', 0) > thresholds['hba'][1]:
                ro5_violations += 1
            
            ro5_score = max(0.0, 1.0 - ro5_violations / 4.0)
            
            # Additional drug-likeness factors
            tpsa = props.get('tpsa', 0)
            tpsa_score = 1.0 if 20 <= tpsa <= 140 else 0.7
            
            # Molecular complexity score
            complexity_score = self._calculate_molecular_complexity(mol)
            
            # Combine scores
            drug_likeness = (0.5 * qed_score + 0.3 * ro5_score + 
                           0.1 * tpsa_score + 0.1 * complexity_score)
            
            return min(1.0, drug_likeness)
            
        except Exception:
            return 0.3
    
    def _novelty_score(self, mol: Chem.Mol) -> float:
        """Calculate novelty score"""
        try:
            return self.diversity.calculate_novelty_score(mol, self.reference_fingerprints)
        except Exception:
            return 0.5
    
    def _selectivity_score(self, mol: Chem.Mol) -> float:
        """Estimate selectivity potential"""
        try:
            # This is a simplified selectivity estimation
            # In practice, this would involve comparison to off-target models
            
            props = self._get_cached_properties(mol)
            
            # Factors that generally correlate with selectivity
            selectivity_factors = []
            
            # Moderate size (not too large, not too small)
            mw = props.get('mw', 0)
            if 250 <= mw <= 450:
                selectivity_factors.append(0.8)
            else:
                selectivity_factors.append(0.5)
            
            # Balanced hydrophobicity
            logp = props.get('logp', 0)
            if 1 <= logp <= 4:
                selectivity_factors.append(0.8)
            else:
                selectivity_factors.append(0.4)
            
            # Presence of specific functional groups
            if props.get('hbd', 0) >= 1 and props.get('hba', 0) >= 2:
                selectivity_factors.append(0.7)
            else:
                selectivity_factors.append(0.5)
            
            # Aromatic ring count (moderate is better)
            aromatic_rings = props.get('aromatic_rings', 0)
            if 1 <= aromatic_rings <= 2:
                selectivity_factors.append(0.8)
            else:
                selectivity_factors.append(0.5)
            
            return np.mean(selectivity_factors)
            
        except Exception:
            return 0.5
    
    def _water_displacement_score(self, mol: Chem.Mol) -> float:
        """Score based on ability to displace unfavorable waters"""
        try:
            water_sites = getattr(self.pocket, 'water_sites', [])
            if not water_sites:
                return 0.5
            
            # Count displaceable waters (high B-factor, low coordination)
            displaceable_waters = [w for w in water_sites 
                                 if w.replaceability_score > 0.6 and w.coordination_number <= 2]
            
            if not displaceable_waters:
                return 0.3
            
            # Score based on molecular features that can displace water
            props = self._get_cached_properties(mol)
            
            # Hydrophobic areas can displace water
            hydrophobic_score = min(1.0, props.get('hydrophobic_area', 0) / 150.0)
            
            # Hydrogen bonding groups can form better interactions
            hbond_score = min(1.0, (props.get('hbd', 0) + props.get('hba', 0)) / 6.0)
            
            # Aromatic groups can participate in favorable interactions
            aromatic_score = min(1.0, props.get('aromatic_rings', 0) / 2.0)
            
            # Combine factors
            displacement_potential = (0.4 * hydrophobic_score + 
                                    0.4 * hbond_score + 
                                    0.2 * aromatic_score)
            
            # Scale by number of displaceable waters
            water_factor = min(1.0, len(displaceable_waters) / 3.0)
            
            return displacement_potential * water_factor
            
        except Exception:
            return 0.4
    
    def _get_adaptive_weights(self, generation_round: int, scores: Dict[str, float]) -> Dict[str, float]:
        """Adapt weights based on generation progress and scores"""
        base_weights = self.config.reward_weights.copy()
        
        # Early rounds: focus on basic drug-likeness and synthetic feasibility
        if generation_round < 5:
            base_weights['drug_likeness'] *= 1.2
            base_weights['synthetic'] *= 1.2
            base_weights['novelty'] *= 0.8
        
        # Later rounds: emphasize specificity and novel interactions
        elif generation_round > 15:
            base_weights['pharmacophore'] *= 1.1
            base_weights['selectivity'] *= 1.3
            base_weights['novelty'] *= 1.2
        
        # Adaptive adjustment based on scoring history
        if len(self.scoring_history) > 10:
            recent_scores = self.scoring_history[-10:]
            
            # If pharmacophore scores are consistently low, increase weight
            avg_pharmacophore = np.mean([s['scores']['pharmacophore'] for s in recent_scores])
            if avg_pharmacophore < 0.4:
                base_weights['pharmacophore'] *= 1.3
            
            # If novelty is too high (too different), decrease weight
            avg_novelty = np.mean([s['scores']['novelty'] for s in recent_scores])
            if avg_novelty > 0.9:
                base_weights['novelty'] *= 0.8
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    def _calculate_molecular_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate comprehensive molecular properties"""
        if mol is None:
            return {}
        
        # Check cache first
        mol_hash = self._get_molecule_hash(mol)
        if mol_hash in self._property_cache:
            return self._property_cache[mol_hash]
        
        try:
            props = {
                'mw': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'rings': rdMolDescriptors.CalcNumRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'qed': QED.qed(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
                'hydrophobic_area': self._calculate_hydrophobic_area(mol)
            }
            
            # Cache the result
            self._property_cache[mol_hash] = props
            return props
            
        except Exception as e:
            warnings.warn(f"Error calculating molecular properties: {e}")
            return {
                'mw': 0, 'logp': 0, 'hbd': 0, 'hba': 0, 'tpsa': 0,
                'rotatable_bonds': 0, 'aromatic_rings': 0, 'rings': 0,
                'heavy_atoms': 0, 'qed': 0, 'formal_charge': 0, 'fsp3': 0,
                'hydrophobic_area': 0
            }
    
    def _get_cached_properties(self, mol: Chem.Mol) -> Dict:
        """Get cached properties or calculate if not cached"""
        return self._calculate_molecular_properties(mol)
    
    def _extract_molecular_features(self, mol: Chem.Mol) -> Dict:
        """Extract molecular features for pharmacophore matching"""
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
            
            # Count charged atoms
            for atom in mol.GetAtoms():
                charge = atom.GetFormalCharge()
                if charge > 0:
                    features['positive_charge'] += charge
                elif charge < 0:
                    features['negative_charge'] += abs(charge)
            
            return features
            
        except Exception:
            return {
                'hbd_count': 0, 'hba_count': 0, 'aromatic_rings': 0,
                'hydrophobic_area': 0, 'formal_charge': 0,
                'positive_charge': 0, 'negative_charge': 0
            }
    
    def _calculate_hydrophobic_area(self, mol: Chem.Mol) -> float:
        """Estimate hydrophobic surface area"""
        try:
            # Simple estimation based on carbon atoms
            carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            
            # Estimate surface area contribution per carbon (rough approximation)
            base_area = carbon_count * 15.0  # Square Angstroms per carbon
            
            # Adjust for aromaticity and branching
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            base_area += aromatic_rings * 30.0  # Bonus for aromatic rings
            
            # Adjust for molecular weight (larger molecules have more area)
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            mw_factor = min(2.0, mw / 200.0)
            
            return base_area * mw_factor
            
        except Exception:
            return 0.0
    
    def _calculate_molecular_complexity(self, mol: Chem.Mol) -> float:
        """Calculate molecular complexity score"""
        try:
            # Factors contributing to complexity
            complexity_factors = []
            
            # Ring complexity
            rings = rdMolDescriptors.CalcNumRings(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            ring_complexity = (rings * 0.1 + aromatic_rings * 0.05)
            complexity_factors.append(min(1.0, ring_complexity))
            
            # Stereocenter complexity
            try:
                stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                stereo_complexity = stereocenters * 0.2
                complexity_factors.append(min(1.0, stereo_complexity))
            except:
                complexity_factors.append(0.0)
            
            # Heteroatom diversity
            elements = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
            hetero_complexity = (len(elements) - 1) * 0.1  # Subtract 1 for carbon
            complexity_factors.append(min(1.0, hetero_complexity))
            
            # Branching complexity
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            branch_complexity = rotatable_bonds * 0.05
            complexity_factors.append(min(1.0, branch_complexity))
            
            # Invert complexity for scoring (simpler is better for synthesis)
            avg_complexity = np.mean(complexity_factors)
            return max(0.0, 1.0 - avg_complexity)
            
        except Exception:
            return 0.5
    
    def _check_violations(self, mol: Chem.Mol, properties: Dict) -> List[str]:
        """Check for rule violations"""
        violations = []
        
        try:
            # Lipinski violations
            thresholds = self.config.lipinski_thresholds
            
            if properties.get('mw', 0) > thresholds['mw'][1]:
                violations.append(f"Molecular weight > {thresholds['mw'][1]}")
            
            if properties.get('logp', 0) > thresholds['logp'][1]:
                violations.append(f"LogP > {thresholds['logp'][1]}")
            
            if properties.get('hbd', 0) > thresholds['hbd'][1]:
                violations.append(f"HBD > {thresholds['hbd'][1]}")
            
            if properties.get('hba', 0) > thresholds['hba'][1]:
                violations.append(f"HBA > {thresholds['hba'][1]}")
            
            if properties.get('tpsa', 0) > thresholds['tpsa'][1]:
                violations.append(f"TPSA > {thresholds['tpsa'][1]}")
            
            # Configuration violations
            if properties.get('heavy_atoms', 0) > self.config.max_heavy_atoms:
                violations.append(f"Heavy atoms > {self.config.max_heavy_atoms}")
            
            if properties.get('rotatable_bonds', 0) > self.config.max_rotatable_bonds:
                violations.append(f"Rotatable bonds > {self.config.max_rotatable_bonds}")
            
            if properties.get('rings', 0) > self.config.max_rings:
                violations.append(f"Rings > {self.config.max_rings}")
            
            # Check for problematic SMILES patterns
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                violations.append("Disconnected structure")
            
        except Exception as e:
            violations.append(f"Error checking violations: {e}")
        
        return violations
    
    def _calculate_confidence(self, scores: Dict[str, float], properties: Dict) -> float:
        """Calculate confidence in the scoring"""
        try:
            # High confidence when:
            # 1. All scores are reasonable (not extreme)
            # 2. Properties are in expected ranges
            # 3. No calculation errors occurred
            
            confidence_factors = []
            
            # Score consistency (no extreme values)
            score_variance = np.var(list(scores.values()))
            score_confidence = max(0.0, 1.0 - score_variance * 2.0)
            confidence_factors.append(score_confidence)
            
            # Property reasonableness
            mw = properties.get('mw', 0)
            if 150 <= mw <= 600:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.5)
            
            logp = properties.get('logp', 0)
            if -2 <= logp <= 6:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.6)
            
            # QED confidence
            qed = properties.get('qed', 0)
            if qed > 0:
                confidence_factors.append(min(1.0, qed + 0.3))
            else:
                confidence_factors.append(0.3)
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5
    
    def _create_zero_score(self) -> MolecularScore:
        """Create a zero score for invalid molecules"""
        return MolecularScore(
            total_score=0.0,
            pharmacophore_score=0.0,
            synthetic_score=0.0,
            drug_likeness_score=0.0,
            novelty_score=0.0,
            selectivity_score=0.0,
            water_displacement_score=0.0,
            violations=["Invalid molecule"],
            confidence=0.0
        )
    
    def _get_molecule_hash(self, mol: Chem.Mol) -> str:
        """Get hash for molecule caching"""
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return str(mol)
    
    def add_reference_molecules(self, reference_mols: List[Chem.Mol]):
        """Add reference molecules for novelty scoring"""
        for mol in reference_mols:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=self.config.fingerprint_radius, 
                    nBits=self.config.fingerprint_bits
                )
                self.reference_fingerprints.append(fp)
            except Exception:
                continue
    
    def get_scoring_statistics(self) -> Dict:
        """Get scoring statistics"""
        if not self.scoring_history:
            return {}
        
        stats = {}
        
        # Calculate average scores per component
        for component in ['pharmacophore', 'synthetic', 'drug_likeness', 'novelty', 'selectivity', 'water_displacement']:
            scores = [entry['scores'][component] for entry in self.scoring_history]
            stats[f'{component}_avg'] = np.mean(scores)
            stats[f'{component}_std'] = np.std(scores)
        
        # Overall statistics
        total_scores = [entry['total_score'] for entry in self.scoring_history]
        stats['total_avg'] = np.mean(total_scores)
        stats['total_std'] = np.std(total_scores)
        stats['total_molecules_scored'] = len(self.scoring_history)
        
        # Score distribution
        stats['high_scoring_molecules'] = sum(1 for score in total_scores if score > 0.7)
        stats['medium_scoring_molecules'] = sum(1 for score in total_scores if 0.4 <= score <= 0.7)
        stats['low_scoring_molecules'] = sum(1 for score in total_scores if score < 0.4)
        
        return stats
    
    def optimize_weights(self, target_molecules: List[Chem.Mol], 
                        optimization_target: str = 'balanced') -> Dict[str, float]:
        """Optimize scoring weights based on target molecules"""
        if not target_molecules:
            return self.config.reward_weights.copy()
        
        # Score all target molecules with current weights
        baseline_scores = []
        for mol in target_molecules:
            score = self.calculate_comprehensive_score(mol)
            baseline_scores.append(score.total_score)
        
        baseline_avg = np.mean(baseline_scores)
        
        # Try different weight combinations
        best_weights = self.config.reward_weights.copy()
        best_score = baseline_avg
        
        weight_components = list(self.config.reward_weights.keys())
        
        for component in weight_components:
            # Try increasing this component's weight
            test_weights = self.config.reward_weights.copy()
            test_weights[component] *= 1.5
            
            # Normalize
            total = sum(test_weights.values())
            test_weights = {k: v/total for k, v in test_weights.items()}
            
            # Test performance
            old_weights = self.config.reward_weights
            self.config.reward_weights = test_weights
            
            test_scores = []
            for mol in target_molecules:
                score = self.calculate_comprehensive_score(mol)
                test_scores.append(score.total_score)
            
            test_avg = np.mean(test_scores)
            
            if test_avg > best_score:
                best_score = test_avg
                best_weights = test_weights.copy()
            
            # Restore original weights
            self.config.reward_weights = old_weights
        
        return best_weights
    
    def clear_cache(self):
        """Clear property and fingerprint caches"""
        self._property_cache.clear()
        self._fingerprint_cache.clear()
    
    def reset_scoring_history(self):
        """Reset scoring history"""
        self.scoring_history.clear()
