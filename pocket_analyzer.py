
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import warnings

from config import LigandForgeConfig
from data_structures import (
    PocketAnalysis, InteractionHotspot, WaterSite, VoxelAnalysisResult,
    validate_position_array
)
from pdb_parser import PDBParser, PDBStructure, AtomRecord


class LigandForgeEnhancedAnalyzer:
    """Enhanced pocket analyzer with comprehensive binding site analysis"""
    
    def __init__(self, config: LigandForgeConfig):
        self.config = config
        self.pdb_parser = PDBParser()
    
    def analyze_binding_site_comprehensive(self, pdb_text: str, center: np.ndarray, 
                                         radius: float = 10.0) -> Dict:
        """Comprehensive binding site analysis returning both native and compatible formats"""
        
        center = validate_position_array(center, "center")
        
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        # Parse structure
        structure = self.pdb_parser.parse_pdb_structure(pdb_text)
        
        # Validate structure
        is_valid, issues = self.pdb_parser.validate_structure(structure)
        if not is_valid:
            warnings.warn(f"Structure validation issues: {'; '.join(issues)}")
        
        # Extract pocket atoms
        pocket_atoms = self.pdb_parser.extract_binding_site_atoms(structure, center, radius)
        
        if not pocket_atoms:
            raise ValueError(f"No atoms found within {radius} Å of center {center}")
        
        # Create interaction hotspots
        hotspots = self._identify_interaction_hotspots(pocket_atoms, center, radius)
        
        # Analyze water sites
        water_sites = self._analyze_water_sites(structure.waters, center, radius)
        
        # Calculate geometric properties
        volume = self._estimate_pocket_volume(pocket_atoms, center, radius)
        surface_area = self._estimate_surface_area(pocket_atoms, center, radius)
        
        # Calculate flexibility map
        flexibility_map = self._calculate_flexibility_map(pocket_atoms)
        
        # Identify hydrophobic patches
        hydrophobic_patches = self._identify_hydrophobic_patches(pocket_atoms, center, radius)
        
        # Create electrostatic potential grid (simplified)
        electrostatic_potential = self._create_electrostatic_grid(pocket_atoms, center, radius)
        
        # Calculate enhanced druggability score
        druggability_score = self._calculate_enhanced_druggability_score(
            hotspots, water_sites, volume, surface_area, pocket_atoms
        )
        
        # Calculate additional metrics
        field_complexity = self._estimate_field_complexity(hotspots, hydrophobic_patches)
        interaction_complementarity = self._calculate_interaction_complementarity(hotspots)
        
        # Create cavity shape descriptors
        cavity_shape_descriptor = self._calculate_cavity_shape_descriptors(
            pocket_atoms, center, radius
        )
        
        # Create pharmacophore points
        pharmacophore_points = self._generate_pharmacophore_points(hotspots)
        
        # Create native format (PocketAnalysis)
        native_analysis = PocketAnalysis(
            center=center,
            volume=volume,
            surface_area=surface_area,
            druggability_score=druggability_score,
            hotspots=hotspots,
            water_sites=water_sites,
            flexibility_map=flexibility_map,
            electrostatic_potential=electrostatic_potential,
            hydrophobic_patches=hydrophobic_patches,
            field_complexity=field_complexity,
            interaction_complementarity=interaction_complementarity,
            cavity_shape_descriptor=cavity_shape_descriptor,
            pharmacophore_points=pharmacophore_points
        )
        
        # Return both formats for compatibility
        return {
            'native_format': native_analysis,
            'compatible_format': native_analysis  # Same object, different reference
        }
    
    def _identify_interaction_hotspots(self, pocket_atoms: List[AtomRecord], 
                                     center: np.ndarray, radius: float) -> List[InteractionHotspot]:
        """Identify and characterize interaction hotspots"""
        hotspots = []
        
        # Group atoms by residue
        residue_groups = defaultdict(list)
        for atom in pocket_atoms:
            key = (atom.chain, atom.resnum, atom.resname)
            residue_groups[key].append(atom)
        
        # Analyze each residue for interaction potential
        for (chain, resnum, resname), atoms in residue_groups.items():
            residue_hotspots = self._analyze_residue_interactions(
                atoms, resname, chain, resnum, center
            )
            hotspots.extend(residue_hotspots)
        
        # Filter hotspots by distance and quality
        filtered_hotspots = []
        for hotspot in hotspots:
            distance = np.linalg.norm(hotspot.position - center)
            if distance <= radius and hotspot.strength > 0.3:
                filtered_hotspots.append(hotspot)
        
        # Sort by strength and remove duplicates
        filtered_hotspots.sort(key=lambda h: h.strength, reverse=True)
        
        return self._remove_duplicate_hotspots(filtered_hotspots)
    
    def _analyze_residue_interactions(self, atoms: List[AtomRecord], resname: str, 
                                    chain: str, resnum: int, center: np.ndarray) -> List[InteractionHotspot]:
        """Analyze interaction potential of a single residue"""
        hotspots = []
        
        # Get residue center
        positions = np.array([atom.position for atom in atoms])
        residue_center = np.mean(positions, axis=0)
        
        # Calculate flexibility (inverse of average B-factor)
        b_factors = [atom.b_factor for atom in atoms if atom.b_factor > 0]
        flexibility = 1.0 / (1.0 + np.mean(b_factors)) if b_factors else 0.5
        
        # Conservation score (simplified - could be enhanced with sequence alignment data)
        conservation = self._estimate_conservation(resname, atoms)
        
        # Analyze specific interaction types based on residue type
        if resname in ['ARG', 'LYS', 'HIS']:
            # Positive charged residues - electrostatic interactions
            for atom in atoms:
                if atom.name in ['NZ', 'NH1', 'NH2', 'ND1', 'NE2']:
                    strength = 0.8 if resname in ['ARG', 'LYS'] else 0.6
                    hotspot = InteractionHotspot(
                        position=atom.position,
                        interaction_type='electrostatic',
                        strength=strength,
                        residue_name=resname,
                        residue_id=resnum,
                        chain=chain,
                        flexibility=flexibility,
                        conservation=conservation,
                        pharmacophore_features=['positive_charge', 'hbd']
                    )
                    hotspots.append(hotspot)
        
        elif resname in ['ASP', 'GLU']:
            # Negative charged residues - electrostatic interactions
            for atom in atoms:
                if atom.name in ['OD1', 'OD2', 'OE1', 'OE2']:
                    hotspot = InteractionHotspot(
                        position=atom.position,
                        interaction_type='electrostatic',
                        strength=0.8,
                        residue_name=resname,
                        residue_id=resnum,
                        chain=chain,
                        flexibility=flexibility,
                        conservation=conservation,
                        pharmacophore_features=['negative_charge', 'hba']
                    )
                    hotspots.append(hotspot)
        
        elif resname in ['SER', 'THR', 'TYR']:
            # Hydroxyl-containing residues - hydrogen bonding
            for atom in atoms:
                if atom.name in ['OG', 'OG1', 'OH']:
                    hotspot = InteractionHotspot(
                        position=atom.position,
                        interaction_type='hbd',
                        strength=0.7,
                        residue_name=resname,
                        residue_id=resnum,
                        chain=chain,
                        flexibility=flexibility,
                        conservation=conservation,
                        pharmacophore_features=['hbd', 'hba']
                    )
                    hotspots.append(hotspot)
        
        elif resname in ['ASN', 'GLN']:
            # Amide-containing residues - hydrogen bonding
            for atom in atoms:
                if atom.name in ['OD1', 'OE1']:  # Oxygen acceptors
                    hotspot = InteractionHotspot(
                        position=atom.position,
                        interaction_type='hba',
                        strength=0.6,
                        residue_name=resname,
                        residue_id=resnum,
                        chain=chain,
                        flexibility=flexibility,
                        conservation=conservation,
                        pharmacophore_features=['hba']
                    )
                    hotspots.append(hotspot)
                elif atom.name in ['ND2', 'NE2']:  # Nitrogen donors
                    hotspot = InteractionHotspot(
                        position=atom.position,
                        interaction_type='hbd',
                        strength=0.6,
                        residue_name=resname,
                        residue_id=resnum,
                        chain=chain,
                        flexibility=flexibility,
                        conservation=conservation,
                        pharmacophore_features=['hbd']
                    )
                    hotspots.append(hotspot)
        
        elif resname in ['PHE', 'TYR', 'TRP']:
            # Aromatic residues - pi interactions
            aromatic_atoms = [a for a in atoms if a.name in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2']]
            if aromatic_atoms:
                aromatic_center = np.mean([a.position for a in aromatic_atoms], axis=0)
                hotspot = InteractionHotspot(
                    position=aromatic_center,
                    interaction_type='aromatic',
                    strength=0.7,
                    residue_name=resname,
                    residue_id=resnum,
                    chain=chain,
                    flexibility=flexibility,
                    conservation=conservation,
                    pharmacophore_features=['aromatic', 'hydrophobic']
                )
                hotspots.append(hotspot)
        
        elif resname in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO']:
            # Hydrophobic residues
            hydrophobic_atoms = [a for a in atoms if a.element == 'C' and a.name not in ['C', 'CA']]
            if hydrophobic_atoms:
                hydrophobic_center = np.mean([a.position for a in hydrophobic_atoms], axis=0)
                hotspot = InteractionHotspot(
                    position=hydrophobic_center,
                    interaction_type='hydrophobic',
                    strength=0.5,
                    residue_name=resname,
                    residue_id=resnum,
                    chain=chain,
                    flexibility=flexibility,
                    conservation=conservation,
                    pharmacophore_features=['hydrophobic']
                )
                hotspots.append(hotspot)
        
        elif resname == 'CYS':
            # Cysteine - potential metal coordination or disulfide
            for atom in atoms:
                if atom.name == 'SG':
                    hotspot = InteractionHotspot(
                        position=atom.position,
                        interaction_type='metal',
                        strength=0.4,
                        residue_name=resname,
                        residue_id=resnum,
                        chain=chain,
                        flexibility=flexibility,
                        conservation=conservation,
                        pharmacophore_features=['metal_coordination']
                    )
                    hotspots.append(hotspot)
        
        return hotspots
    
    def _analyze_water_sites(self, waters: List[AtomRecord], center: np.ndarray, 
                           radius: float) -> List[WaterSite]:
        """Analyze crystallographic water sites"""
        water_sites = []
        
        for water in waters:
            distance = np.linalg.norm(water.position - center)
            if distance <= radius:
                # Calculate replaceability based on B-factor and coordination
                replaceability_score = min(1.0, water.b_factor / 50.0)
                conservation_score = max(0.0, 1.0 - water.b_factor / 100.0)
                
                # Estimate hydrogen bonds (simplified)
                hydrogen_bonds = self._estimate_water_hydrogen_bonds(water, waters)
                
                water_site = WaterSite(
                    position=water.position,
                    b_factor=water.b_factor,
                    hydrogen_bonds=hydrogen_bonds,
                    replaceability_score=replaceability_score,
                    conservation_score=conservation_score,
                    coordination_number=len(hydrogen_bonds)
                )
                water_sites.append(water_site)
        
        return water_sites
    
    def _estimate_water_hydrogen_bonds(self, water: AtomRecord, 
                                     all_waters: List[AtomRecord]) -> List[str]:
        """Estimate hydrogen bonding partners for water molecule"""
        hydrogen_bonds = []
        
        # Simple distance-based estimation
        for other_water in all_waters:
            if other_water.serial != water.serial:
                distance = np.linalg.norm(water.position - other_water.position)
                if 2.5 <= distance <= 3.5:  # Typical H-bond distance range
                    hydrogen_bonds.append(f"HOH_{other_water.serial}")
        
        return hydrogen_bonds
    
    def _estimate_pocket_volume(self, pocket_atoms: List[AtomRecord], 
                              center: np.ndarray, radius: float) -> float:
        """Estimate pocket volume using atom-based calculation"""
        # Simple estimation based on number of atoms and their packing
        num_atoms = len(pocket_atoms)
        
        # Average volume per atom (rough estimate)
        avg_atom_volume = 20.0  # Cubic angstroms
        
        # Calculate occupied volume
        occupied_volume = num_atoms * avg_atom_volume
        
        # Total spherical volume
        total_volume = (4.0 / 3.0) * np.pi * (radius ** 3)
        
        # Pocket volume is the difference
        pocket_volume = max(0.0, total_volume - occupied_volume)
        
        return pocket_volume
    
    def _estimate_surface_area(self, pocket_atoms: List[AtomRecord], 
                             center: np.ndarray, radius: float) -> float:
        """Estimate pocket surface area"""
        # Simple estimation based on atom positions
        num_atoms = len(pocket_atoms)
        
        # Average surface area per atom
        avg_surface_per_atom = 15.0  # Square angstroms
        
        return num_atoms * avg_surface_per_atom
    
    def _calculate_flexibility_map(self, pocket_atoms: List[AtomRecord]) -> Dict[str, float]:
        """Calculate flexibility map based on B-factors"""
        flexibility_map = {}
        
        # Group by residue
        residue_groups = defaultdict(list)
        for atom in pocket_atoms:
            key = f"{atom.chain}_{atom.resnum}_{atom.resname}"
            residue_groups[key].append(atom)
        
        # Calculate average flexibility per residue
        for residue_key, atoms in residue_groups.items():
            b_factors = [atom.b_factor for atom in atoms if atom.b_factor > 0]
            if b_factors:
                avg_b_factor = np.mean(b_factors)
                # Convert B-factor to flexibility (higher B-factor = more flexible)
                flexibility = min(1.0, avg_b_factor / 50.0)
            else:
                flexibility = 0.5  # Default flexibility
            
            flexibility_map[residue_key] = flexibility
        
        return flexibility_map
    
    def _identify_hydrophobic_patches(self, pocket_atoms: List[AtomRecord], 
                                    center: np.ndarray, radius: float) -> List[np.ndarray]:
        """Identify hydrophobic patches in the pocket"""
        hydrophobic_patches = []
        
        # Find hydrophobic atoms
        hydrophobic_atoms = []
        for atom in pocket_atoms:
            if self._is_hydrophobic_atom(atom):
                hydrophobic_atoms.append(atom)
        
        if not hydrophobic_atoms:
            return hydrophobic_patches
        
        # Cluster hydrophobic atoms to form patches
        positions = np.array([atom.position for atom in hydrophobic_atoms])
        
        # Simple clustering based on distance
        clusters = self._cluster_positions(positions, max_distance=4.0)
        
        # Create patch centers
        for cluster in clusters:
            if len(cluster) >= 3:  # Minimum atoms for a patch
                patch_center = np.mean([positions[i] for i in cluster], axis=0)
                hydrophobic_patches.append(patch_center)
        
        return hydrophobic_patches
    
    def _is_hydrophobic_atom(self, atom: AtomRecord) -> bool:
        """Check if atom is hydrophobic"""
        hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'TYR', 'PRO']
        
        if atom.resname not in hydrophobic_residues:
            return False
        
        # Carbon atoms in side chains (excluding backbone)
        if atom.element == 'C' and atom.name not in ['C', 'CA', 'N', 'O']:
            return True
        
        # Sulfur in methionine
        if atom.resname == 'MET' and atom.name == 'SD':
            return True
        
        return False
    
    def _cluster_positions(self, positions: np.ndarray, max_distance: float) -> List[List[int]]:
        """Simple distance-based clustering"""
        n_points = len(positions)
        clusters = []
        visited = set()
        
        for i in range(n_points):
            if i in visited:
                continue
            
            cluster = [i]
            visited.add(i)
            
            # Find all points within max_distance
            for j in range(i + 1, n_points):
                if j in visited:
                    continue
                
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= max_distance:
                    cluster.append(j)
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _create_electrostatic_grid(self, pocket_atoms: List[AtomRecord], 
                                 center: np.ndarray, radius: float) -> np.ndarray:
        """Create simplified electrostatic potential grid"""
        # Simple grid for compatibility
        grid_size = 10
        grid = np.zeros((grid_size, grid_size, grid_size))
        
        # This is a simplified version - in practice would use proper electrostatic calculation
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Grid position
                    grid_pos = center + np.array([i-5, j-5, k-5]) * (radius / 5.0)
                    
                    # Calculate potential from nearby charged atoms
                    potential = 0.0
                    for atom in pocket_atoms:
                        if atom.resname in ['ARG', 'LYS']:
                            charge = 1.0
                        elif atom.resname in ['ASP', 'GLU']:
                            charge = -1.0
                        else:
                            charge = 0.0
                        
                        if charge != 0:
                            distance = np.linalg.norm(grid_pos - atom.position)
                            if distance > 0.1:  # Avoid singularities
                                potential += charge / distance
                    
                    grid[i, j, k] = potential
        
        return grid
    
    def _calculate_enhanced_druggability_score(self, hotspots: List[InteractionHotspot], 
                                             water_sites: List[WaterSite], 
                                             volume: float, surface_area: float,
                                             pocket_atoms: List[AtomRecord]) -> float:
        """Calculate enhanced druggability score"""
        
        # Base score from geometry (30%)
        geometry_score = min(1.0, (volume / 800.0) ** 0.5 * (surface_area / 600.0) ** 0.3)
        
        # Interaction diversity score (25%)
        interaction_types = set(h.interaction_type for h in hotspots)
        diversity_score = len(interaction_types) / 6.0  # Max 6 interaction types
        
        # Hotspot quality score (25%)
        if hotspots:
            avg_strength = np.mean([h.strength for h in hotspots])
            hotspot_score = avg_strength
        else:
            hotspot_score = 0.0
        
        # Water displacement opportunity (10%)
        displaceable_waters = sum(1 for w in water_sites if w.replaceability_score > 0.5)
        water_score = min(1.0, displaceable_waters / 3.0)
        
        # Pocket depth and shape (10%)
        shape_score = self._calculate_shape_score(pocket_atoms, volume)
        
        # Combine scores
        total_score = (
            0.3 * geometry_score + 
            0.25 * diversity_score + 
            0.25 * hotspot_score + 
            0.1 * water_score +
            0.1 * shape_score
        )
        
        return min(1.0, total_score)
    
    def _calculate_shape_score(self, pocket_atoms: List[AtomRecord], volume: float) -> float:
        """Calculate shape-based druggability contribution"""
        if not pocket_atoms:
            return 0.0
        
        positions = np.array([atom.position for atom in pocket_atoms])
        
        # Calculate bounding box
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate aspect ratio (depth vs width)
        max_dim = np.max(dimensions)
        min_dim = np.min(dimensions)
        aspect_ratio = max_dim / min_dim if min_dim > 0 else 1.0
        
        # Optimal aspect ratio is around 1.5-2.0 for druggability
        if 1.2 <= aspect_ratio <= 3.0:
            shape_score = 1.0 - abs(aspect_ratio - 1.75) / 1.75
        else:
            shape_score = 0.5
        
        return shape_score
    
    def _estimate_field_complexity(self, hotspots: List[InteractionHotspot], 
                                  hydrophobic_patches: List[np.ndarray]) -> float:
        """Estimate field complexity score"""
        # Based on diversity and distribution of interactions
        interaction_types = set(h.interaction_type for h in hotspots)
        type_diversity = len(interaction_types) / 6.0
        
        # Spatial distribution of hotspots
        if len(hotspots) > 1:
            positions = np.array([h.position for h in hotspots])
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distances.append(np.linalg.norm(positions[i] - positions[j]))
            
            if distances:
                spatial_diversity = min(1.0, np.std(distances) / 5.0)
            else:
                spatial_diversity = 0.0
        else:
            spatial_diversity = 0.0
        
        # Hydrophobic patch contribution
        patch_contribution = min(1.0, len(hydrophobic_patches) / 3.0)
        
        return 0.5 * type_diversity + 0.3 * spatial_diversity + 0.2 * patch_contribution
    
    def _calculate_interaction_complementarity(self, hotspots: List[InteractionHotspot]) -> float:
        """Calculate interaction complementarity score"""
        if not hotspots:
            return 0.0
        
        # Check for complementary pairs
        complementary_pairs = 0
        total_pairs = 0
        
        for i, hotspot1 in enumerate(hotspots):
            for j, hotspot2 in enumerate(hotspots[i+1:], i+1):
                total_pairs += 1
                
                # Check for complementarity
                type1, type2 = hotspot1.interaction_type, hotspot2.interaction_type
                
                if (type1 == 'hbd' and type2 == 'hba') or (type1 == 'hba' and type2 == 'hbd'):
                    complementary_pairs += 1
                elif (type1 == 'electrostatic' and type2 in ['hbd', 'hba']):
                    complementary_pairs += 0.5
                elif (type1 == 'aromatic' and type2 == 'hydrophobic'):
                    complementary_pairs += 0.3
        
        return complementary_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_cavity_shape_descriptors(self, pocket_atoms: List[AtomRecord], 
                                          center: np.ndarray, radius: float) -> Dict[str, float]:
        """Calculate cavity shape descriptors"""
        if not pocket_atoms:
            return {}
        
        positions = np.array([atom.position for atom in pocket_atoms])
        
        # Calculate various shape metrics
        descriptors = {}
        
        # Bounding box analysis
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        dimensions = max_coords - min_coords
        
        descriptors['length'] = float(np.max(dimensions))
        descriptors['width'] = float(np.median(dimensions))
        descriptors['height'] = float(np.min(dimensions))
        descriptors['aspect_ratio'] = float(descriptors['length'] / descriptors['width'] if descriptors['width'] > 0 else 1.0)
        
        # Distance from center analysis
        distances_from_center = np.linalg.norm(positions - center, axis=1)
        descriptors['max_distance'] = float(np.max(distances_from_center))
        descriptors['avg_distance'] = float(np.mean(distances_from_center))
        descriptors['distance_std'] = float(np.std(distances_from_center))
        
        # Sphericity estimation
        if len(positions) > 3:
            # Calculate convex hull volume vs actual volume
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(positions)
                convex_volume = hull.volume
                
                # Estimate actual volume from atom count
                estimated_volume = len(positions) * 20.0  # Rough atom volume
                
                descriptors['convexity'] = float(estimated_volume / convex_volume if convex_volume > 0 else 0.0)
            except:
                descriptors['convexity'] = 0.5
        else:
            descriptors['convexity'] = 0.0
        
        return descriptors
    
    def _generate_pharmacophore_points(self, hotspots: List[InteractionHotspot]) -> List[Dict]:
        """Generate pharmacophore points from hotspots"""
        pharmacophore_points = []
        
        for hotspot in hotspots:
            point = {
                'position': hotspot.position.tolist(),
                'type': hotspot.interaction_type,
                'strength': hotspot.strength,
                'residue': hotspot.residue_name,
                'features': hotspot.pharmacophore_features,
                'tolerance': 1.5  # Angstrom tolerance for pharmacophore matching
            }
            pharmacophore_points.append(point)
        
        return pharmacophore_points
    
    def _remove_duplicate_hotspots(self, hotspots: List[InteractionHotspot], 
                                 min_distance: float = 2.0) -> List[InteractionHotspot]:
        """Remove duplicate hotspots that are too close to each other"""
        if not hotspots:
            return hotspots
        
        filtered_hotspots = [hotspots[0]]  # Keep the first (highest strength)
        
        for hotspot in hotspots[1:]:
            is_duplicate = False
            
            for existing in filtered_hotspots:
                distance = np.linalg.norm(hotspot.position - existing.position)
                if distance < min_distance and hotspot.interaction_type == existing.interaction_type:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_hotspots.append(hotspot)
        
        return filtered_hotspots
    
    def _estimate_conservation(self, resname: str, atoms: List[AtomRecord]) -> float:
        """Estimate conservation score for residue (simplified)"""
        # This is a simplified version - in practice would use sequence alignment data
        
        # Highly conserved residues in binding sites
        high_conservation = ['CYS', 'PRO', 'GLY']
        medium_conservation = ['ARG', 'LYS', 'ASP', 'GLU', 'HIS', 'TRP', 'PHE', 'TYR']
        
        if resname in high_conservation:
            return 0.9
        elif resname in medium_conservation:
            return 0.7
        else:
            return 0.5
    
    def get_pocket_summary(self, analysis: PocketAnalysis) -> Dict[str, any]:
        """Get summary statistics for pocket analysis"""
        return {
            'druggability_score': analysis.druggability_score,
            'volume': analysis.volume,
            'surface_area': analysis.surface_area,
            'num_hotspots': len(analysis.hotspots),
            'num_water_sites': len(analysis.water_sites),
            'interaction_types': list(analysis.get_interaction_summary().keys()),
            'field_complexity': analysis.field_complexity,
            'shape_descriptors': analysis.cavity_shape_descriptor
        }
