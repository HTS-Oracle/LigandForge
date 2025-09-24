
import numpy as np
import hashlib
from typing import List, Dict, Set, Optional
from collections import defaultdict
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import DBSCAN

from config import LigandForgeConfig


class EnhancedDiversityManager:
    """Manage molecular diversity and prevent duplicate generation"""
    
    def __init__(self, config: LigandForgeConfig):
        self.config = config
        self.generated_fingerprints = []
        self.molecular_scaffolds = set()
        
        # Duplicate prevention features
        self.generated_smiles = set()  # Canonical SMILES
        self.generated_inchi_keys = set()  # InChI keys for tautomer-independent comparison
        self.generated_hashes = set()  # Hash-based deduplication
        self.scaffold_counts = defaultdict(int)  # Count molecules per scaffold
        
        # Statistics tracking
        self.generation_stats = {
            'total_generated': 0,
            'duplicates_prevented': 0,
            'unique_scaffolds': 0,
            'diversity_rejections': 0
        }
        
        # Reference fingerprints for novelty calculation
        self.reference_fingerprints = []
    
    def is_duplicate(self, mol: Chem.Mol) -> bool:
        """
        Comprehensive duplicate checking using multiple methods
        Returns True if molecule is a duplicate
        """
        try:
            # Method 1: Canonical SMILES comparison (fastest)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            if canonical_smiles in self.generated_smiles:
                self.generation_stats['duplicates_prevented'] += 1
                return True
            
            # Method 2: InChI Key comparison (handles tautomers)
            try:
                inchi_key = Chem.MolToInchiKey(mol)
                if inchi_key and inchi_key in self.generated_inchi_keys:
                    self.generation_stats['duplicates_prevented'] += 1
                    return True
            except:
                inchi_key = None  # InChI generation can fail for some molecules
            
            # Method 3: Molecular hash comparison (robust)
            mol_hash = self._calculate_molecular_hash(mol)
            if mol_hash in self.generated_hashes:
                self.generation_stats['duplicates_prevented'] += 1
                return True
            
            # If not a duplicate, add to tracking sets
            self.generated_smiles.add(canonical_smiles)
            if inchi_key:
                self.generated_inchi_keys.add(inchi_key)
            self.generated_hashes.add(mol_hash)
            
            return False
            
        except Exception as e:
            # If we can't process the molecule, consider it potentially duplicate
            warnings.warn(f"Could not check duplicate status for molecule: {e}")
            return True
    
    def is_diverse(self, mol: Chem.Mol, skip_duplicate_check: bool = False) -> bool:
        """
        Check if molecule is sufficiently diverse and not a duplicate
        """
        # First check for duplicates (unless skipped)
        if not skip_duplicate_check and self.is_duplicate(mol):
            return False
        
        # Generate fingerprint for diversity analysis
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.config.fingerprint_radius, nBits=self.config.fingerprint_bits
            )
        except:
            return False
        
        # Check similarity to existing molecules
        similar_count = 0
        for existing_fp in self.generated_fingerprints:
            similarity = DataStructs.TanimotoSimilarity(fp, existing_fp)
            if similarity > self.config.diversity_threshold:
                similar_count += 1
                if similar_count >= self.config.max_similar_molecules:
                    self.generation_stats['diversity_rejections'] += 1
                    return False
        
        # Check scaffold diversity with improved limits
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
            
            # Limit molecules per scaffold more strictly
            if self.scaffold_counts[scaffold_smiles] >= 3:  # Max 3 per scaffold
                self.generation_stats['diversity_rejections'] += 1
                return False
            
            # Add to tracking if diverse
            self.generated_fingerprints.append(fp)
            self.molecular_scaffolds.add(scaffold_smiles)
            self.scaffold_counts[scaffold_smiles] += 1
            
        except Exception:
            # If scaffold analysis fails, still allow the molecule but track fingerprint
            self.generated_fingerprints.append(fp)
        
        self.generation_stats['total_generated'] += 1
        return True
    
    def _calculate_molecular_hash(self, mol: Chem.Mol) -> str:
        """
        Calculate a robust hash for the molecule based on multiple descriptors
        """
        try:
            # Get various molecular descriptors
            descriptors = [
                mol.GetNumHeavyAtoms(),
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                round(rdMolDescriptors.CalcExactMolWt(mol), 2),
                round(rdMolDescriptors.CalcTPSA(mol), 1),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
            ]
            
            # Get atom and bond invariants
            atom_invariants = []
            for atom in mol.GetAtoms():
                atom_invariants.append((
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()) if atom.GetHybridization() else 0,
                    atom.GetIsAromatic()
                ))
            
            bond_invariants = []
            for bond in mol.GetBonds():
                bond_invariants.append((
                    int(bond.GetBondType()),
                    bond.GetIsAromatic(),
                    bond.GetIsConjugated()
                ))
            
            # Combine all information
            hash_input = str(descriptors) + str(sorted(atom_invariants)) + str(sorted(bond_invariants))
            
            # Create hash
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception:
            # Fallback to SMILES-based hash
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                return hashlib.md5(smiles.encode()).hexdigest()
            except:
                return hashlib.md5(str(mol).encode()).hexdigest()
    
    def add_reference_molecules(self, reference_mols: List[Chem.Mol]):
        """
        Add reference molecules to prevent generation of similar structures
        """
        for mol in reference_mols:
            try:
                # Add to duplicate tracking
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                self.generated_smiles.add(canonical_smiles)
                
                try:
                    inchi_key = Chem.MolToInchiKey(mol)
                    if inchi_key:
                        self.generated_inchi_keys.add(inchi_key)
                except:
                    pass
                
                mol_hash = self._calculate_molecular_hash(mol)
                self.generated_hashes.add(mol_hash)
                
                # Add to diversity tracking
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=self.config.fingerprint_radius, nBits=self.config.fingerprint_bits
                )
                self.generated_fingerprints.append(fp)
                self.reference_fingerprints.append(fp)
                
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                self.molecular_scaffolds.add(scaffold_smiles)
                self.scaffold_counts[scaffold_smiles] += 1
                
            except Exception as e:
                warnings.warn(f"Could not add reference molecule: {e}")
                continue
    
    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        stats = self.generation_stats.copy()
        stats['unique_scaffolds'] = len(self.molecular_scaffolds)
        stats['unique_smiles'] = len(self.generated_smiles)
        stats['unique_inchi_keys'] = len(self.generated_inchi_keys)
        stats['total_molecules_tracked'] = len(self.generated_fingerprints)
        
        total_attempts = stats['total_generated'] + stats['duplicates_prevented'] + stats['diversity_rejections']
        if total_attempts > 0:
            stats['diversity_rate'] = stats['total_generated'] / total_attempts
            stats['duplicate_rate'] = stats['duplicates_prevented'] / total_attempts
            stats['rejection_rate'] = stats['diversity_rejections'] / total_attempts
        else:
            stats['diversity_rate'] = 0.0
            stats['duplicate_rate'] = 0.0
            stats['rejection_rate'] = 0.0
        
        return stats
    
    def reset_tracking(self):
        """Reset all tracking for a new generation run"""
        self.generated_fingerprints = []
        self.molecular_scaffolds = set()
        self.generated_smiles = set()
        self.generated_inchi_keys = set()
        self.generated_hashes = set()
        self.scaffold_counts = defaultdict(int)
        self.generation_stats = {
            'total_generated': 0,
            'duplicates_prevented': 0,
            'unique_scaffolds': 0,
            'diversity_rejections': 0
        }
        # Keep reference fingerprints
    
    def cluster_and_select(self, molecules: List[Chem.Mol], 
                          n_clusters: int = None) -> List[Chem.Mol]:
        """
        Cluster molecules and select diverse representatives
        """
        if not molecules:
            return []
        
        # First remove duplicates using our enhanced method
        unique_molecules = []
        temp_diversity_manager = EnhancedDiversityManager(self.config)
        
        for mol in molecules:
            if not temp_diversity_manager.is_duplicate(mol):
                unique_molecules.append(mol)
        
        molecules = unique_molecules
        if not molecules:
            return []
        
        if n_clusters is None:
            n_clusters = min(self.config.cluster_count, len(molecules))
        
        # Calculate fingerprints
        fingerprints = []
        valid_molecules = []
        
        for mol in molecules:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=self.config.fingerprint_radius, nBits=self.config.fingerprint_bits
                )
                fingerprints.append(fp)
                valid_molecules.append(mol)
            except:
                continue
        
        if not valid_molecules:
            return []
        
        molecules = valid_molecules
        
        # Calculate distance matrix
        n_mols = len(molecules)
        distance_matrix = np.zeros((n_mols, n_mols))
        
        for i in range(n_mols):
            for j in range(i+1, n_mols):
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                distance = 1.0 - similarity
                distance_matrix[i][j] = distance_matrix[j][i] = distance
        
        # Perform clustering using DBSCAN for better cluster identification
        try:
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Select representatives from each cluster
            selected_indices = []
            cluster_representatives = {}
            
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise points (singletons)
                    selected_indices.append(i)
                else:
                    if label not in cluster_representatives:
                        cluster_representatives[label] = i
                    else:
                        # Choose the molecule with highest quality score
                        current_idx = cluster_representatives[label]
                        if self._calculate_quality_score(molecules[i]) > self._calculate_quality_score(molecules[current_idx]):
                            cluster_representatives[label] = i
            
            # Add cluster representatives
            selected_indices.extend(cluster_representatives.values())
            
        except Exception:
            # Fallback to distance-based selection
            selected_indices = self._distance_based_selection(distance_matrix, n_clusters, molecules)
        
        # Limit to requested number
        if len(selected_indices) > n_clusters:
            # Sort by quality and take top n_clusters
            quality_scores = [(idx, self._calculate_quality_score(molecules[idx])) for idx in selected_indices]
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in quality_scores[:n_clusters]]
        
        return [molecules[i] for i in selected_indices]
    
    def _distance_based_selection(self, distance_matrix: np.ndarray, n_clusters: int, 
                                molecules: List[Chem.Mol]) -> List[int]:
        """Fallback distance-based selection method"""
        selected_indices = []
        remaining_indices = list(range(len(molecules)))
        
        # Select first molecule (highest quality)
        quality_scores = [self._calculate_quality_score(mol) for mol in molecules]
        first_idx = remaining_indices[np.argmax([quality_scores[i] for i in remaining_indices])]
        
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select most distant molecules
        for _ in range(min(n_clusters - 1, len(remaining_indices))):
            if not remaining_indices:
                break
                
            max_min_distance = -1
            best_idx = None
            
            for candidate_idx in remaining_indices:
                min_distance = min(distance_matrix[candidate_idx][selected_idx] 
                                 for selected_idx in selected_indices)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _calculate_quality_score(self, mol: Chem.Mol) -> float:
        """Calculate a quality score for molecule selection"""
        try:
            from rdkit.Chem import QED
            qed_score = QED.qed(mol)
            
            # Add other quality metrics
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
            
            # Penalize extreme values
            mw_penalty = 1.0 if 200 <= mw <= 500 else 0.5
            logp_penalty = 1.0 if -2 <= logp <= 5 else 0.5
            
            return qed_score * mw_penalty * logp_penalty
            
        except:
            return 0.5  # Default score if calculation fails
    
    def calculate_novelty_score(self, mol: Chem.Mol, reference_fps: List = None) -> float:
        """Calculate novelty score against reference molecules"""
        if reference_fps is None:
            reference_fps = self.reference_fingerprints
        
        if not reference_fps:
            return 1.0
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.config.fingerprint_radius, nBits=self.config.fingerprint_bits
            )
            
            max_similarity = 0.0
            for ref_fp in reference_fps:
                similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
                max_similarity = max(max_similarity, similarity)
            
            return 1.0 - max_similarity
            
        except:
            return 0.0
    
    def calculate_population_diversity(self, molecules: List[Chem.Mol]) -> float:
        """Calculate diversity score for a population of molecules"""
        if len(molecules) < 2:
            return 0.0
        
        try:
            fingerprints = []
            for mol in molecules:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=self.config.fingerprint_radius, nBits=self.config.fingerprint_bits
                )
                fingerprints.append(fp)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(fingerprints)):
                for j in range(i+1, len(fingerprints)):
                    sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    similarities.append(sim)
            
            # Diversity is 1 - average similarity
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity
            
        except:
            return 0.0
    
    def get_scaffold_diversity_report(self) -> Dict:
        """Get detailed scaffold diversity report"""
        report = {
            'total_unique_scaffolds': len(self.molecular_scaffolds),
            'scaffold_distribution': dict(self.scaffold_counts),
            'most_common_scaffolds': [],
            'singleton_scaffolds': 0
        }
        
        # Find most common scaffolds
        sorted_scaffolds = sorted(self.scaffold_counts.items(), key=lambda x: x[1], reverse=True)
        report['most_common_scaffolds'] = sorted_scaffolds[:10]
        
        # Count singleton scaffolds
        report['singleton_scaffolds'] = sum(1 for count in self.scaffold_counts.values() if count == 1)
        
        return report
    
    def optimize_diversity_parameters(self, test_molecules: List[Chem.Mol]) -> Dict[str, float]:
        """Optimize diversity parameters based on test set"""
        if len(test_molecules) < 10:
            return {
                'optimal_threshold': self.config.diversity_threshold,
                'optimal_max_similar': self.config.max_similar_molecules
            }
        
        best_threshold = self.config.diversity_threshold
        best_diversity = 0.0
        
        # Test different thresholds
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            # Temporarily change threshold
            old_threshold = self.config.diversity_threshold
            self.config.diversity_threshold = threshold
            
            # Calculate diversity with this threshold
            selected = self.cluster_and_select(test_molecules, n_clusters=min(50, len(test_molecules)//2))
            diversity = self.calculate_population_diversity(selected)
            
            if diversity > best_diversity:
                best_diversity = diversity
                best_threshold = threshold
            
            # Restore original threshold
            self.config.diversity_threshold = old_threshold
        
        return {
            'optimal_threshold': best_threshold,
            'achieved_diversity': best_diversity,
            'recommended_max_similar': max(1, int(len(test_molecules) * 0.1))
        }
