"""
Molecular Assembly Module (fixed)
Structure-guided molecular assembly and generation

Fixed issues:
- Proper removal of dummy atoms to prevent "A" in SMILES
- Enhanced sanitization of products
- Better error handling for attachment points
- Improved validation of molecules before returning
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
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


class StructureGuidedAssembly:
    """Structure-guided molecular assembly"""

    def __init__(self, fragment_library: FragmentLibrary, pocket_analysis: PocketAnalysis,
                 config: LigandForgeConfig):
        self.fragment_lib = fragment_library
        self.pocket = pocket_analysis
        self.config = config
        self.reaction_templates = self._load_reaction_templates()
        self.pharmacophore_fragment_map = self._build_pharmacophore_fragment_map()
        self.generation_stats = {
            'total_attempts': 0,
            'successful_assemblies': 0,
            'validation_failures': 0,
            'strategy_counts': defaultdict(int)
        }

    def _remove_dummy_atoms(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove all dummy atoms (atomic number 0) from molecule"""
        if mol is None:
            return None
        
        try:
            # Create an editable molecule
            rw_mol = Chem.RWMol(mol)
            
            # Find all dummy atoms (atomic number 0)
            atoms_to_remove = []
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atoms_to_remove.append(atom.GetIdx())
            
            # Remove dummy atoms in reverse order to maintain indices
            for atom_idx in sorted(atoms_to_remove, reverse=True):
                rw_mol.RemoveAtom(atom_idx)
            
            # Get the resulting molecule
            result = rw_mol.GetMol()
            
            # Sanitize the molecule
            if result is not None:
                Chem.SanitizeMol(result)
                
                # Additional check: ensure no "A" atoms in SMILES
                smiles = Chem.MolToSmiles(result)
                if 'A' in smiles or '*' in smiles:
                    return None
                    
            return result
            
        except Exception as e:
            warnings.warn(f"Error removing dummy atoms: {e}")
            return None

    def _sanitize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Enhanced molecule sanitization"""
        if mol is None:
            return None
            
        try:
            # First remove dummy atoms
            mol = self._remove_dummy_atoms(mol)
            if mol is None:
                return None
            
            # Standard sanitization
            Chem.SanitizeMol(mol)
            
            # Check for empty molecule
            if mol.GetNumAtoms() == 0:
                return None
            
            # Check for disconnected fragments
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                # If disconnected, take the largest fragment
                fragments = smiles.split('.')
                largest_fragment = max(fragments, key=len)
                mol = Chem.MolFromSmiles(largest_fragment)
                if mol is not None:
                    Chem.SanitizeMol(mol)
            
            # Final check for problematic atoms
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if any(char in smiles for char in ['A', '*', '[*]']):
                    return None
                    
            return mol
            
        except Exception as e:
            warnings.warn(f"Error sanitizing molecule: {e}")
            return None

    # =========================
    # Reaction Template Registry
    # =========================
    def _load_reaction_templates(self) -> Dict[str, rdChemReactions.ChemicalReaction]:
        """Load extensive chemical reaction templates for molecular assembly"""
        reactions: Dict[str, rdChemReactions.ChemicalReaction] = {}
        try:
            # 1) Core set from original module (kept as-is)
            reactions["amide"] = rdChemReactions.ReactionFromSmarts(
                "[C:1](=[O:2])[OH:3].[N:4][H:5]>>[C:1](=[O:2])[N:4].[OH2:3]"
            )
            reactions["suzuki"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[c:3][B:4]([OH:5])[OH:6]>>[c:1][c:3].[Br,Cl,I:2][B:4]([OH:5])[OH:6]"
            )
            reactions["sulfonamide"] = rdChemReactions.ReactionFromSmarts(
                "[S:1](=[O:2])(=[O:3])[Cl:4].[N:5][H:6]>>[S:1](=[O:2])(=[O:3])[N:5].[Cl:4][H:6]"
            )
            reactions["urea"] = rdChemReactions.ReactionFromSmarts(
                "[N:1][H:2].[O:3]=[C:4]=[N:5][H:6]>>[N:1][C:4](=[O:3])[N:5].[H:2][H:6]"
            )
            reactions["ether"] = rdChemReactions.ReactionFromSmarts(
                "[C:1][OH:2].[C:3][Br,Cl,I:4]>>[C:1][O:2][C:3].[Br,Cl,I:4][H]"
            )
            reactions["reductive_amination"] = rdChemReactions.ReactionFromSmarts(
                "[C:1]=[O:2].[N:3][H:4]>>[C:1][N:3].[O:2][H:4]"
            )
            reactions["click"] = rdChemReactions.ReactionFromSmarts(
                "[C:1]#[C:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[c:2][n:3][n:4][n:5]1"
            )

            # 2) Expanded coupling reactions
            reactions["buchwald_hartwig"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[N:3]>>[c:1][N:3].[Br,Cl,I:2][H]"
            )
            reactions["heck"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[C:3]=[C:4]>>[c:1][C:3]=[C:4].[Br,Cl,I:2][H]"
            )
            reactions["sonogashira"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[C:3]#[C:4]>>[c:1][C:3]#[C:4].[Br,Cl,I:2][H]"
            )
            reactions["stille"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[c:3][Sn:4]>>[c:1][c:3].[Br,Cl,I:2][Sn:4]"
            )
            reactions["negishi"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[c:3][Zn:4]>>[c:1][c:3].[Br,Cl,I:2][Zn:4]"
            )
            reactions["kumada"] = rdChemReactions.ReactionFromSmarts(
                "[c:1][Br,Cl,I:2].[c:3][Mg:4]>>[c:1][c:3].[Br,Cl,I:2][Mg:4]"
            )

            # 3) Condensation and formation reactions
            reactions["esterification"] = rdChemReactions.ReactionFromSmarts(
                "[C:1](=[O:2])[OH:3].[O:4][H:5]>>[C:1](=[O:2])[O:4].[OH2:3]"
            )
            reactions["carbamate"] = rdChemReactions.ReactionFromSmarts(
                "[C:1](=[O:2])[Cl:3].[O:4][H].[N:5]>>[C:1](=[O:2])[O:4][N:5].[Cl:3][H]"
            )
            reactions["thioether"] = rdChemReactions.ReactionFromSmarts(
                "[C:1][Br,Cl,I:2].[S:3][H]>>[C:1][S:3].[Br,Cl,I:2][H]"
            )
            reactions["oxime"] = rdChemReactions.ReactionFromSmarts(
                "[C:1]=[O:2].[N:3]([H])[O:4][H]>>[C:1]=[N:3][O:4].[O:2][H]"
            )
            reactions["hydrazone"] = rdChemReactions.ReactionFromSmarts(
                "[C:1]=[O:2].[N:3][N:4]>>[C:1]=[N:3][N:4].[O:2][H]"
            )
            reactions["amidine"] = rdChemReactions.ReactionFromSmarts(
                "[C:1](=[O:2])[Cl:3].[N:4]=[C:5][N:6]>>[C:1](=[N:6])[N:4].[Cl:3][H]"
            )
            reactions["sulfonylation"] = rdChemReactions.ReactionFromSmarts(
                "[S:1](=[O:2])(=[O:3])[Cl:4].[O:5][H]>>[S:1](=[O:2])(=[O:3])[O:5].[Cl:4][H]"
            )
            reactions["michael_addition"] = rdChemReactions.ReactionFromSmarts(
                "[C:1]=[C:2][C:3](=[O:4])[O,N:5].[N,O,S:6][H]>>[C:1]([N,O,S:6])[C:2][C:3](=[O:4])[O,N:5]"
            )

            # Additional reactions - simplified for better success
            reactions["simple_coupling"] = rdChemReactions.ReactionFromSmarts(
                "[C:1][Br:2].[C:3][H:4]>>[C:1][C:3].[Br:2][H:4]"
            )

        except Exception as e:
            warnings.warn(f"Error loading reaction templates: {e}")
        return reactions

    def _build_pharmacophore_fragment_map(self) -> Dict[str, List[str]]:
        """Build mapping between pharmacophore types and suitable fragment interaction types"""
        return {
            'hydrophobic': ['aromatic', 'hydrophobic', 'aliphatic'],
            'aromatic': ['aromatic', 'pi_stacking', 'hydrophobic'],
            'hbond_donor': ['donor', 'hydrogen_bond', 'polar'],
            'hbond_acceptor': ['acceptor', 'hydrogen_bond', 'polar'],  
            'positive': ['basic', 'cation', 'charged'],
            'negative': ['acidic', 'anion', 'charged'],
            'polar': ['polar', 'hydrogen_bond', 'donor', 'acceptor'],
            'metal_coordination': ['metal_binding', 'coordination', 'polar'],
            'pi_stacking': ['aromatic', 'pi_stacking'],
            'halogen_bond': ['halogen', 'polar']
        }

    # ==================
    # Public API (kept)
    # ==================
    def generate_structure_guided(self, target_interactions: List[str],
                                  n_molecules: int = 100) -> List[Chem.Mol]:
        """Generate molecules targeting specific interactions"""
        molecules: List[Chem.Mol] = []
        attempts = 0
        max_attempts = n_molecules * 80  # Increased attempts for better yield

        self.generation_stats['total_attempts'] = 0
        self.generation_stats['successful_assemblies'] = 0
        self.generation_stats['validation_failures'] = 0

        while len(molecules) < n_molecules and attempts < max_attempts:
            attempts += 1
            self.generation_stats['total_attempts'] += 1

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

                # Always sanitize the molecule before validation
                if mol is not None:
                    mol = self._sanitize_molecule(mol)

                if mol and self._validate_molecule(mol):
                    molecules.append(mol)
                    self.generation_stats['successful_assemblies'] += 1
                elif mol:
                    self.generation_stats['validation_failures'] += 1

            except Exception as e:
                warnings.warn(f"Assembly error with {strategy}: {str(e)}")
                continue

        return molecules

    # ============================
    # Strategy selection / routing
    # ============================
    def _choose_assembly_strategy(self, target_interactions: List[str]) -> str:
        """Choose assembly strategy based on target interactions and pocket characteristics"""
        num_hotspots = len(self.pocket.hotspots)
        pocket_volume = self.pocket.volume
        interaction_diversity = len(set(h.interaction_type for h in self.pocket.hotspots))
        num_pharmacophore = len(self.pocket.pharmacophore_points)

        # Weighted choices incorporating pharmacophore information
        candidates: List[str] = []
        
        # Prioritize pharmacophore-guided when we have good pharmacophore data
        if num_pharmacophore >= 3:
            candidates += ['pharmacophore_guided'] * 3  # Higher weight
            
        if num_hotspots >= 3 and interaction_diversity >= 2:
            candidates += ['hotspot_guided', 'linking']
            if num_pharmacophore >= 2:
                candidates += ['pharmacophore_guided']
                
        if pocket_volume > 600 and 'hydrophobic' in target_interactions:
            candidates += ['scaffold_decoration', 'conjugation_extension']
            
        if len(self.pocket.pharmacophore_points) >= 3:
            candidates += ['pharmacophore_guided', 'fragment_growing']
            
        if len(target_interactions) == 1:
            candidates += ['fragment_growing', 'sp3_enrichment']

        # Always add some diversity routes
        candidates += ['ring_closure', 'enumerative_reaction']
        
        # Ensure pharmacophore_guided has good representation
        if num_pharmacophore >= 2:
            candidates += ['pharmacophore_guided'] * 2

        return random.choice(candidates) if candidates else random.choice([
            'linking', 'fragment_growing', 'pharmacophore_guided', 'enumerative_reaction'])

    # =============================
    # Existing strategies (fixed)
    # =============================
    def _hotspot_guided_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        target_hotspots = [h for h in self.pocket.hotspots if h.interaction_type in target_interactions]
        if not target_hotspots:
            target_hotspots = self.pocket.hotspots[:3]
        if not target_hotspots:
            return None

        cores = [f for f in self.fragment_lib.fragments['cores'] if f.attachment_points >= 2]
        if not cores:
            cores = self.fragment_lib.fragments['cores']
        if not cores:
            return None

        core = random.choice(cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        for i, hotspot in enumerate(target_hotspots[:min(3, getattr(core, 'attachment_points', 2))]):
            suitable_subs = self.fragment_lib.get_fragments_for_interaction(hotspot.interaction_type)
            if suitable_subs:
                sub = random.choice(suitable_subs)
                mol = self._attach_fragment_robust(mol, sub, attachment_point=i+1)
                if mol is None:
                    break
        
        return self._sanitize_molecule(mol)

    def _fragment_growing_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        suitable_cores: List[FragmentInfo] = []
        for interaction in target_interactions:
            cores = self.fragment_lib.get_fragments_for_interaction(interaction)
            suitable_cores.extend([c for c in cores if getattr(c, 'scaffold_type', '') == 'core'])
        if not suitable_cores:
            suitable_cores = self.fragment_lib.fragments['cores']
        if not suitable_cores:
            return None

        core = random.choice(suitable_cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        for _ in range(random.randint(2, 4)):  # Reduced from 5 to avoid overly complex molecules
            growth_sites = self._find_growth_sites(mol)
            if not growth_sites:
                break
            site = random.choice(growth_sites)
            interaction_needed = random.choice(target_interactions)
            suitable_fragments = self.fragment_lib.get_fragments_for_interaction(interaction_needed)
            if suitable_fragments:
                fragment = random.choice(suitable_fragments)
                new_mol = self._grow_at_site_robust(mol, fragment, site)
                if new_mol is not None:
                    mol = new_mol
                else:
                    break
        
        return self._sanitize_molecule(mol)

    def _linking_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        cores = self.fragment_lib.fragments['cores']
        linkers = self.fragment_lib.fragments['linkers']
        if len(cores) < 2 or not linkers:
            return None
        core1, core2 = random.sample(cores, 2)

        suitable_linkers: List[FragmentInfo] = []
        for interaction in target_interactions:
            suitable_linkers.extend([l for l in linkers if interaction in l.interaction_types])
        if not suitable_linkers:
            suitable_linkers = linkers
        linker = random.choice(suitable_linkers)

        mol = self._link_fragments_robust(core1, core2, linker)
        if mol and random.random() < 0.6:
            substituents = self.fragment_lib.fragments['substituents']
            if substituents:
                sub = random.choice(substituents)
                mol = self._add_substituent_robust(mol, sub)
        
        return self._sanitize_molecule(mol)

    def _scaffold_decoration_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        large_cores = [c for c in self.fragment_lib.fragments['cores'] if c.mw > 150 and c.ring_count >= 2]
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
        
        max_decorations = min(3, len(target_interactions))
        for _ in range(max_decorations):
            if random.random() < 0.7:
                interaction_type = random.choice(target_interactions)
                suitable_subs = [s for s in substituents if interaction_type in s.interaction_types]
                if suitable_subs:
                    sub = random.choice(suitable_subs)
                    new_mol = self._add_substituent_robust(mol, sub)
                    if new_mol is not None:
                        mol = new_mol
        
        return self._sanitize_molecule(mol)

    def _pharmacophore_guided_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Enhanced pharmacophore-guided assembly with systematic fragment mapping"""
        if not self.pocket.pharmacophore_points:
            return self._hotspot_guided_assembly(target_interactions)
            
        # Get pharmacophore points and map them to fragment types
        relevant_points = []
        for point in self.pocket.pharmacophore_points:
            point_type = point.get('type', '')
            if point_type in target_interactions:
                relevant_points.append(point)
        
        if not relevant_points:
            relevant_points = self.pocket.pharmacophore_points[:4]  # Take more points

        # Start with a flexible core that can support multiple attachments
        flexible_cores = [c for c in self.fragment_lib.fragments['cores'] 
                         if c.attachment_points >= 2 and 'aromatic' in c.interaction_types]
        if not flexible_cores:
            flexible_cores = self.fragment_lib.fragments['cores']
        if not flexible_cores:
            return None

        core = random.choice(flexible_cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        # For each pharmacophore point, find suitable fragments
        for i, point in enumerate(relevant_points[:min(4, getattr(core, 'attachment_points', 2))]):
            point_type = point.get('type', 'hydrophobic')
            
            # Use pharmacophore-fragment mapping
            suitable_fragment_types = self.pharmacophore_fragment_map.get(point_type, [point_type])
            
            suitable_fragments = []
            for frag_type in suitable_fragment_types:
                suitable_fragments.extend(self.fragment_lib.get_fragments_for_interaction(frag_type))
            
            if suitable_fragments:
                fragment = random.choice(suitable_fragments)
                if i == 0:
                    mol = self._add_substituent_robust(mol, fragment)
                else:
                    mol = self._attach_fragment_robust(mol, fragment, attachment_point=i+1)
                if mol is None:
                    break
                    
        # Add additional decorations based on pocket properties
        if mol and len(relevant_points) >= 3:
            substituents = self.fragment_lib.fragments.get('substituents', [])
            if substituents and random.random() < 0.5:
                # Add one more fragment targeting remaining interactions
                remaining_interactions = [ti for ti in target_interactions 
                                        if ti not in [p.get('type', '') for p in relevant_points]]
                if remaining_interactions:
                    interaction_type = random.choice(remaining_interactions)
                    suitable_subs = self.fragment_lib.get_fragments_for_interaction(interaction_type)
                    if suitable_subs:
                        sub = random.choice(suitable_subs)
                        new_mol = self._add_substituent_robust(mol, sub)
                        if new_mol is not None:
                            mol = new_mol
                            
        return self._sanitize_molecule(mol)

    # ==============================
    # New strategies (additive only)
    # ==============================
    def _ring_closure_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Form small/medium rings by decorating a linear core then attempting proxy closures."""
        cores = self.fragment_lib.fragments['cores']
        if not cores:
            return None
        core = random.choice(cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        # Add two tether-like substituents to enable ring formation attempts
        substituents = self.fragment_lib.fragments.get('substituents', [])
        for _ in range(2):
            if not substituents:
                break
            sub = random.choice(substituents)
            candidate = self._add_substituent_robust(mol, sub)
            if candidate is not None:
                mol = candidate

        return self._sanitize_molecule(mol)

    def _conjugation_extension_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Extend π-systems via coupling reactions to reach hydrophobic patches."""
        cores = self.fragment_lib.fragments['cores']
        aryl_extenders = self.fragment_lib.fragments.get('linkers', []) + self.fragment_lib.fragments.get('substituents', [])
        if not cores or not aryl_extenders:
            return None
        core = random.choice(cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        extender = random.choice(aryl_extenders)
        ext_mol = Chem.MolFromSmiles(extender.smiles)
        if ext_mol is None:
            return self._sanitize_molecule(mol)

        # Simple attachment approach (more reliable than complex reactions)
        product = self._simple_attach(mol, ext_mol)
        if product is not None:
            mol = product
                
        return self._sanitize_molecule(mol)

    def _sp3_enrichment_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Bias towards 3D (sp3) growth by adding aliphatic chains/cycles and heteroatom handles."""
        cores = self.fragment_lib.fragments['cores']
        if not cores:
            return None
        core = random.choice(cores)
        mol = Chem.MolFromSmiles(core.smiles)
        if mol is None:
            return None

        # Prefer fragments annotated with sp3/alkyl/hbonding interactions
        candidates: List[FragmentInfo] = []
        for tag in ("aliphatic", "donor", "acceptor", "sp3", "hydrogen_bond", "polar"):
            candidates += self.fragment_lib.get_fragments_for_interaction(tag)
        if not candidates:
            candidates = self.fragment_lib.fragments.get('substituents', [])

        for _ in range(random.randint(2, 3)):  # Reduced from 4 to 3
            if not candidates:
                break
            frag = random.choice(candidates)
            grown = self._add_substituent_robust(mol, frag)
            if grown is not None:
                mol = grown
        
        return self._sanitize_molecule(mol)

    def _enumerative_reaction_assembly(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Pick a random reaction template; map plausible reactants from library buckets and execute."""
        if not self.reaction_templates:
            return None
        
        # Use simpler, more reliable reactions
        safe_reactions = ["amide", "ether", "simple_coupling", "esterification"]
        available_reactions = [r for r in safe_reactions if r in self.reaction_templates]
        
        if not available_reactions:
            return None
            
        rxn_name = random.choice(available_reactions)

        # Heuristic selection of buckets to serve as reactants
        buckets = [
            self.fragment_lib.fragments.get('cores', []),
            self.fragment_lib.fragments.get('linkers', []),
            self.fragment_lib.fragments.get('substituents', []),
        ]
        # Flatten and sample a few candidates per role
        reactant_mols: List[Chem.Mol] = []
        for bucket in buckets[:2]:  # Only use first 2 buckets for simplicity
            if not bucket:
                continue
            fi = random.choice(bucket)
            m = Chem.MolFromSmiles(fi.smiles)
            if m is not None:
                reactant_mols.append(m)
                
        if len(reactant_mols) < 2:
            return None
            
        product = self._apply_reaction_safe(rxn_name, reactant_mols)
        return self._sanitize_molecule(product)

    # =========================
    # Robust attachment methods
    # =========================
    def _simple_attach(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[Chem.Mol]:
        """Simple attachment of two molecules without using dummy atoms"""
        try:
            # Find attachable atoms in both molecules
            sites1 = self._find_growth_sites(mol1)
            sites2 = self._find_growth_sites(mol2)
            
            if not sites1 or not sites2:
                return None
            
            site1 = random.choice(sites1)
            site2 = random.choice(sites2)
            
            # Combine molecules
            combined = Chem.CombineMols(mol1, mol2)
            rw_mol = Chem.RWMol(combined)
            
            # Add bond between the sites
            mol1_size = mol1.GetNumAtoms()
            rw_mol.AddBond(site1, mol1_size + site2, Chem.BondType.SINGLE)
            
            result = rw_mol.GetMol()
            return self._sanitize_molecule(result)
            
        except Exception:
            return None

    def _attach_fragment_robust(self, mol: Chem.Mol, fragment: FragmentInfo,
                               attachment_point: int = 1) -> Optional[Chem.Mol]:
        """Robust fragment attachment with fallback to simple methods"""
        try:
            # First try the original method
            result = self._attach_fragment(mol, fragment, attachment_point)
            if result is not None:
                result = self._sanitize_molecule(result)
                if result is not None:
                    return result
        except Exception:
            pass
        
        # Fallback to simple attachment
        try:
            fragment_mol = Chem.MolFromSmiles(fragment.smiles)
            if fragment_mol is None:
                return None
            return self._simple_attach(mol, fragment_mol)
        except Exception:
            return None

    def _add_substituent_robust(self, mol: Chem.Mol, substituent: FragmentInfo) -> Optional[Chem.Mol]:
        """Robust substituent addition"""
        try:
            result = self._add_substituent(mol, substituent)
            if result is not None:
                result = self._sanitize_molecule(result)
                if result is not None:
                    return result
        except Exception:
            pass
        
        # Fallback to simple attachment
        try:
            sub_mol = Chem.MolFromSmiles(substituent.smiles)
            if sub_mol is None:
                return None
            return self._simple_attach(mol, sub_mol)
        except Exception:
            return None

    def _grow_at_site_robust(self, mol: Chem.Mol, fragment: FragmentInfo, site_idx: int) -> Optional[Chem.Mol]:
        """Robust growth at specific site"""
        try:
            result = self._grow_at_site(mol, fragment, site_idx)
            if result is not None:
                result = self._sanitize_molecule(result)
                if result is not None:
                    return result
        except Exception:
            pass
        
        # Fallback to simple attachment
        try:
            fragment_mol = Chem.MolFromSmiles(fragment.smiles)
            if fragment_mol is None:
                return None
            return self._simple_attach(mol, fragment_mol)
        except Exception:
            return None

    def _link_fragments_robust(self, core1: FragmentInfo, core2: FragmentInfo,
                              linker: FragmentInfo) -> Optional[Chem.Mol]:
        """Robust fragment linking"""
        try:
            result = self._link_fragments(core1, core2, linker)
            if result is not None:
                result = self._sanitize_molecule(result)
                if result is not None:
                    return result
        except Exception:
            pass
        
        # Fallback: simple sequential attachment
        try:
            mol1 = Chem.MolFromSmiles(core1.smiles)
            mol2 = Chem.MolFromSmiles(core2.smiles)
            linker_mol = Chem.MolFromSmiles(linker.smiles)
            
            if None in [mol1, mol2, linker_mol]:
                return None
            
            # First attach core1 to linker
            intermediate = self._simple_attach(mol1, linker_mol)
            if intermediate is None:
                return None
                
            # Then attach core2
            result = self._simple_attach(intermediate, mol2)
            return result
            
        except Exception:
            return None

    # ============================
    # Original methods (unchanged)
    # ============================
    def _attach_fragment(self, mol: Chem.Mol, fragment: FragmentInfo,
                         attachment_point: int = 1) -> Optional[Chem.Mol]:
        try:
            rw_mol = Chem.RWMol(mol)
            fragment_mol = Chem.MolFromSmiles(fragment.smiles)
            if fragment_mol is None:
                return None

            core_attachment = None
            frag_attachment = None
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == attachment_point:
                    core_attachment = atom.GetIdx()
                    break
            for atom in fragment_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    frag_attachment = atom.GetIdx()
                    break
            if core_attachment is None or frag_attachment is None:
                return self._attach_fragment_alternative(mol, fragment)

            core_neighbor = None
            frag_neighbor = None
            for neighbor in rw_mol.GetAtomWithIdx(core_attachment).GetNeighbors():
                core_neighbor = neighbor.GetIdx(); break
            for neighbor in fragment_mol.GetAtomWithIdx(frag_attachment).GetNeighbors():
                frag_neighbor = neighbor.GetIdx(); break
            if core_neighbor is None or frag_neighbor is None:
                return None

            combined = Chem.CombineMols(rw_mol, fragment_mol)
            editable = Chem.RWMol(combined)
            core_size = rw_mol.GetNumAtoms()
            editable.AddBond(core_neighbor, core_size + frag_neighbor, Chem.BondType.SINGLE)
            atoms_to_remove = sorted([core_attachment, core_size + frag_attachment], reverse=True)
            for atom_idx in atoms_to_remove:
                editable.RemoveAtom(atom_idx)
            result = editable.GetMol()
            Chem.SanitizeMol(result)
            return result
        except Exception:
            return None

    def _attach_fragment_alternative(self, mol: Chem.Mol, fragment: FragmentInfo) -> Optional[Chem.Mol]:
        try:
            growth_sites = self._find_growth_sites(mol)
            if not growth_sites:
                return None
            growth_site = random.choice(growth_sites)
            return self._grow_at_site(mol, fragment, growth_site)
        except Exception:
            return None

    def _add_substituent(self, mol: Chem.Mol, substituent: FragmentInfo) -> Optional[Chem.Mol]:
        """Add a substituent to a random attachment point"""
        try:
            growth_sites = self._find_growth_sites(mol)
            if not growth_sites:
                return None
            site = random.choice(growth_sites)
            return self._grow_at_site(mol, substituent, site)
        except Exception:
            return None

    def _find_growth_sites(self, mol: Chem.Mol) -> List[int]:
        growth_sites: List[int] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [6, 7, 8]:
                current_valence = atom.GetTotalValence()
                max_valence = self._get_max_valence(atom.GetAtomicNum())
                if current_valence < max_valence:
                    growth_sites.append(atom.GetIdx())
        return growth_sites

    def _get_max_valence(self, atomic_num: int) -> int:
        valence_map = {6: 4, 7: 3, 8: 2, 16: 6, 15: 5, 9: 1, 17: 1, 35: 1, 53: 1}
        return valence_map.get(atomic_num, 4)

    def _grow_at_site(self, mol: Chem.Mol, fragment: FragmentInfo, site_idx: int) -> Optional[Chem.Mol]:
        """Growth method that avoids creating dummy atoms when possible"""
        try:
            fragment_mol = Chem.MolFromSmiles(fragment.smiles)
            if fragment_mol is None:
                return None
                
            # Try simple attachment first
            result = self._simple_attach(mol, fragment_mol)
            if result is not None:
                return result
            
            # Fallback to original method
            editable = Chem.RWMol(mol)
            carbon_idx = editable.AddAtom(Chem.Atom(6))
            editable.AddBond(site_idx, carbon_idx, Chem.BondType.SINGLE)
            dummy_idx = editable.AddAtom(Chem.Atom(0))
            editable.GetAtomWithIdx(dummy_idx).SetAtomMapNum(1)
            editable.AddBond(carbon_idx, dummy_idx, Chem.BondType.SINGLE)
            temp_mol = editable.GetMol()
            Chem.SanitizeMol(temp_mol)
            return self._attach_fragment(temp_mol, fragment, 1)
        except Exception:
            return None

    def _link_fragments(self, core1: FragmentInfo, core2: FragmentInfo,
                        linker: FragmentInfo) -> Optional[Chem.Mol]:
        try:
            mol1 = Chem.MolFromSmiles(core1.smiles)
            mol2 = Chem.MolFromSmiles(core2.smiles)
            linker_mol = Chem.MolFromSmiles(linker.smiles)
            if None in [mol1, mol2, linker_mol]:
                return None
            intermediate = self._attach_fragment_simple(mol1, linker_mol, 1, 1)
            if intermediate is None:
                return None
            final = self._attach_fragment_simple(intermediate, mol2, 2, 1)
            return final
        except Exception:
            return None

    def _attach_fragment_simple(self, mol1: Chem.Mol, mol2: Chem.Mol,
                                map1: int, map2: int) -> Optional[Chem.Mol]:
        try:
            rw1, rw2 = Chem.RWMol(mol1), Chem.RWMol(mol2)
            star1_idx = neighbor1_idx = star2_idx = neighbor2_idx = None
            for atom in rw1.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map1:
                    star1_idx = atom.GetIdx()
                    for neighbor in atom.GetNeighbors():
                        neighbor1_idx = neighbor.GetIdx(); break
                    break
            for atom in rw2.GetAtoms():
                if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map2:
                    star2_idx = atom.GetIdx()
                    for neighbor in atom.GetNeighbors():
                        neighbor2_idx = neighbor.GetIdx(); break
                    break
            if None in [star1_idx, neighbor1_idx, star2_idx, neighbor2_idx]:
                return None
            combined = Chem.CombineMols(rw1, rw2)
            editable = Chem.RWMol(combined)
            mol1_size = rw1.GetNumAtoms()
            editable.AddBond(neighbor1_idx, mol1_size + neighbor2_idx, Chem.BondType.SINGLE)
            for idx in sorted([star1_idx, mol1_size + star2_idx], reverse=True):
                editable.RemoveAtom(idx)
            result = editable.GetMol()
            Chem.SanitizeMol(result)
            return result
        except Exception:
            return None

    # ==================
    # Enhanced Reaction executor
    # ==================
    def _apply_reaction_safe(self, reaction_name: str, reactant_mols: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """Safe reaction application with enhanced sanitization"""
        try:
            if reaction_name not in self.reaction_templates:
                return None
            rxn = self.reaction_templates[reaction_name]
            # Pad/cut reactants to match expected count (use duplicates if needed)
            needed = rxn.GetNumReactantTemplates()
            if needed == 0:
                return None
            if len(reactant_mols) < needed:
                if not reactant_mols:
                    return None
                while len(reactant_mols) < needed:
                    reactant_mols.append(Chem.Mol(reactant_mols[-1]))
            elif len(reactant_mols) > needed:
                reactant_mols = reactant_mols[:needed]

            prods = rxn.RunReactants(tuple(reactant_mols))
            if not prods:
                return None

            # Flatten and sanitize products; pick highest QED as heuristic
            best_mol = None
            best_qed = -1.0
            for tpl in prods:
                for p in tpl:
                    try:
                        # Enhanced sanitization
                        sanitized = self._sanitize_molecule(p)
                        if sanitized is not None:
                            q = QED.qed(sanitized)
                            if q > best_qed:
                                best_qed = q
                                best_mol = sanitized
                    except Exception:
                        continue
            return best_mol
        except Exception:
            return None

    # ===================
    # Validation & stats
    # ===================
    def _validate_molecule(self, mol: Chem.Mol) -> bool:
        if mol is None:
            return False
        try:
            if mol.GetNumAtoms() == 0:
                return False
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                return False
            # Check for problematic characters
            if any(char in smiles for char in ['A', '*', '[*]']):
                return False
            props = self._calculate_properties(mol)
            if not (self.config.min_heavy_atoms <= props['heavy_atoms'] <= self.config.max_heavy_atoms):
                return False
            if props['rotatable_bonds'] > self.config.max_rotatable_bonds:
                return False
            if props['rings'] > self.config.max_rings:
                return False
            if not (self.config.min_molecular_weight <= props['mw'] <= self.config.max_molecular_weight):
                return False
            if self._has_toxic_alerts(mol):
                return False
            if props['logp'] > 6.0 or props['logp'] < -3.0:
                return False
            if props['tpsa'] > 200.0:
                return False
            return True
        except Exception:
            return False

    def _calculate_properties(self, mol: Chem.Mol) -> Dict:
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
        except Exception:
            return {
                'heavy_atoms': 0, 'mw': 0, 'logp': 0, 'hbd': 0, 'hba': 0,
                'tpsa': 0, 'rotatable_bonds': 0, 'rings': 0, 'qed': 0
            }

    def _has_toxic_alerts(self, mol: Chem.Mol) -> bool:
        if not hasattr(self.fragment_lib, 'toxic_alerts'):
            return False
        for alert_smarts in self.fragment_lib.toxic_alerts:
            try:
                pattern = Chem.MolFromSmarts(alert_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
            except Exception:
                continue
        return False

    def get_generation_statistics(self) -> Dict:
        stats = self.generation_stats.copy()
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_assemblies'] / stats['total_attempts']
            stats['validation_failure_rate'] = stats['validation_failures'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
            stats['validation_failure_rate'] = 0.0
        return stats

    def reset_statistics(self):
        self.generation_stats = {
            'total_attempts': 0,
            'successful_assemblies': 0,
            'validation_failures': 0,
            'strategy_counts': defaultdict(int)
        }
