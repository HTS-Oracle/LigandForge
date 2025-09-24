"""
PDB Parser Module
Handles parsing and validation of PDB structures for LigandForge pipeline
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

from data_structures import AtomicEnvironment, validate_position_array


@dataclass
class AtomRecord:
    """Individual atom record from PDB file"""
    serial: int
    name: str
    alt_loc: str
    resname: str
    chain: str
    resnum: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str
    charge: str
    record_type: str  # ATOM or HETATM
    
    @property
    def position(self) -> np.ndarray:
        """Get position as numpy array"""
        return np.array([self.x, self.y, self.z])
    
    @property
    def is_protein(self) -> bool:
        """Check if atom belongs to protein"""
        return self.record_type == "ATOM"
    
    @property
    def is_water(self) -> bool:
        """Check if atom is water"""
        return self.resname in ["HOH", "WAT", "H2O", "TIP", "SOL"]
    
    @property
    def is_ligand(self) -> bool:
        """Check if atom is ligand (non-water HETATM)"""
        return self.record_type == "HETATM" and not self.is_water


@dataclass
class PDBStructure:
    """Parsed PDB structure"""
    atoms: List[AtomRecord]
    waters: List[AtomRecord]
    ligands: Dict[str, List[AtomRecord]]
    header: Dict[str, str]
    resolution: Optional[float]
    space_group: Optional[str]
    unit_cell: Optional[List[float]]
    chains: Set[str]
    
    def get_protein_atoms(self) -> List[AtomRecord]:
        """Get all protein atoms"""
        return [atom for atom in self.atoms if atom.is_protein]
    
    def get_chain_atoms(self, chain: str) -> List[AtomRecord]:
        """Get atoms from specific chain"""
        return [atom for atom in self.atoms if atom.chain == chain]
    
    def get_residue_atoms(self, chain: str, resnum: int, icode: str = " ") -> List[AtomRecord]:
        """Get atoms from specific residue"""
        return [atom for atom in self.atoms 
                if atom.chain == chain and atom.resnum == resnum and atom.icode == icode]
    
    def get_center_of_mass(self) -> np.ndarray:
        """Calculate center of mass of all atoms"""
        positions = np.array([atom.position for atom in self.atoms])
        return np.mean(positions, axis=0)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of structure"""
        positions = np.array([atom.position for atom in self.atoms])
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        return min_coords, max_coords


class PDBParser:
    """PDB structure parsing and validation"""
    
    # Standard amino acid residues
    STANDARD_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    
    # Common nucleic acid residues
    NUCLEIC_RESIDUES = {
        'A', 'C', 'G', 'T', 'U', 'DA', 'DC', 'DG', 'DT', 'DU'
    }
    
    # Common water molecule names
    WATER_NAMES = {'HOH', 'WAT', 'H2O', 'TIP', 'SOL', 'TIP3', 'TIP4', 'SPC'}
    
    # Common ion names
    ION_NAMES = {
        'NA', 'CL', 'CA', 'MG', 'K', 'ZN', 'FE', 'MN', 'CO', 'NI', 'CU',
        'NA+', 'CL-', 'CA2+', 'MG2+', 'K+', 'ZN2+', 'FE2+', 'FE3+', 'MN2+'
    }
    
    def __init__(self, strict_parsing: bool = False, ignore_alt_locs: bool = True):
        """
        Initialize PDB parser
        
        Args:
            strict_parsing: If True, raise errors on malformed records
            ignore_alt_locs: If True, only use the first alternate location
        """
        self.strict_parsing = strict_parsing
        self.ignore_alt_locs = ignore_alt_locs
        self.warnings = []
    
    def parse_pdb_structure(self, pdb_text: str) -> PDBStructure:
        """Parse PDB structure from text"""
        lines = pdb_text.splitlines()
        
        atoms = []
        waters = []
        ligands = defaultdict(list)
        header = {}
        resolution = None
        space_group = None
        unit_cell = None
        chains = set()
        
        # Parse header information
        for line in lines:
            if line.startswith('HEADER'):
                header['title'] = line[10:50].strip()
                header['deposition_date'] = line[50:59].strip()
                header['pdb_code'] = line[62:66].strip()
            elif line.startswith('TITLE'):
                header['description'] = header.get('description', '') + line[10:].strip()
            elif line.startswith('REMARK   2'):
                resolution_match = re.search(r'RESOLUTION\.\s*(\d+\.\d+)', line)
                if resolution_match:
                    resolution = float(resolution_match.group(1))
            elif line.startswith('CRYST1'):
                unit_cell = self._parse_unit_cell(line)
                space_group = line[55:66].strip()
        
        # Parse atom records
        for line_num, line in enumerate(lines, 1):
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    atom = self._parse_atom_record(line, line_num)
                    if atom is None:
                        continue
                    
                    chains.add(atom.chain)
                    
                    if atom.is_protein:
                        atoms.append(atom)
                    elif atom.is_water:
                        waters.append(atom)
                    elif atom.is_ligand:
                        ligands[atom.resname].append(atom)
                    else:
                        # Other HETATM records (ions, cofactors, etc.)
                        if atom.resname in self.ION_NAMES:
                            ligands[atom.resname].append(atom)
                        else:
                            ligands[atom.resname].append(atom)
                            
                except Exception as e:
                    error_msg = f"Error parsing line {line_num}: {str(e)}"
                    if self.strict_parsing:
                        raise ValueError(error_msg)
                    else:
                        self.warnings.append(error_msg)
                        continue
        
        return PDBStructure(
            atoms=atoms,
            waters=waters,
            ligands=dict(ligands),
            header=header,
            resolution=resolution,
            space_group=space_group,
            unit_cell=unit_cell,
            chains=chains
        )
    
    def _parse_atom_record(self, line: str, line_num: int) -> Optional[AtomRecord]:
        """Parse a single ATOM or HETATM record"""
        if len(line) < 54:
            if self.strict_parsing:
                raise ValueError(f"Line too short: {line}")
            return None
        
        try:
            record_type = line[0:6].strip()
            serial = int(line[6:11].strip()) if line[6:11].strip() else 0
            name = line[12:16].strip()
            alt_loc = line[16:17].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resnum = int(line[22:26].strip()) if line[22:26].strip() else 0
            icode = line[26:27].strip()
            
            # Coordinates
            x = float(line[30:38].strip()) if line[30:38].strip() else 0.0
            y = float(line[38:46].strip()) if line[38:46].strip() else 0.0
            z = float(line[46:54].strip()) if line[46:54].strip() else 0.0
            
            # Optional fields
            occupancy = float(line[54:60].strip()) if len(line) > 54 and line[54:60].strip() else 1.0
            b_factor = float(line[60:66].strip()) if len(line) > 60 and line[60:66].strip() else 0.0
            element = line[76:78].strip() if len(line) > 76 else self._guess_element(name)
            charge = line[78:80].strip() if len(line) > 78 else ""
            
            # Handle alternate locations
            if self.ignore_alt_locs and alt_loc and alt_loc != 'A':
                return None
            
            # Validate coordinates
            if not all(np.isfinite([x, y, z])):
                if self.strict_parsing:
                    raise ValueError(f"Invalid coordinates: {x}, {y}, {z}")
                return None
            
            return AtomRecord(
                serial=serial,
                name=name,
                alt_loc=alt_loc,
                resname=resname,
                chain=chain,
                resnum=resnum,
                icode=icode,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                b_factor=b_factor,
                element=element,
                charge=charge,
                record_type=record_type
            )
            
        except (ValueError, IndexError) as e:
            if self.strict_parsing:
                raise ValueError(f"Error parsing atom record: {str(e)}")
            return None
    
    def _parse_unit_cell(self, line: str) -> Optional[List[float]]:
        """Parse CRYST1 record to extract unit cell parameters"""
        try:
            a = float(line[6:15].strip())
            b = float(line[15:24].strip())
            c = float(line[24:33].strip())
            alpha = float(line[33:40].strip())
            beta = float(line[40:47].strip())
            gamma = float(line[47:54].strip())
            return [a, b, c, alpha, beta, gamma]
        except (ValueError, IndexError):
            return None
    
    def _guess_element(self, atom_name: str) -> str:
        """Guess element from atom name"""
        # Remove digits and common prefixes
        clean_name = re.sub(r'\d+', '', atom_name)
        clean_name = clean_name.lstrip('0123456789')
        
        # Common element mappings
        if clean_name.startswith('CA') and len(clean_name) > 2:
            return 'C'  # CA is usually carbon alpha
        elif clean_name.startswith('CB') and len(clean_name) > 2:
            return 'C'  # CB is usually carbon beta
        elif clean_name.startswith('CG'):
            return 'C'
        elif clean_name.startswith('CD'):
            return 'C'
        elif clean_name.startswith('CE'):
            return 'C'
        elif clean_name.startswith('CZ'):
            return 'C'
        elif clean_name.startswith('C'):
            return 'C'
        elif clean_name.startswith('N'):
            return 'N'
        elif clean_name.startswith('O'):
            return 'O'
        elif clean_name.startswith('S'):
            return 'S'
        elif clean_name.startswith('P'):
            return 'P'
        elif clean_name.startswith('FE'):
            return 'Fe'
        elif clean_name.startswith('ZN'):
            return 'Zn'
        elif clean_name.startswith('MG'):
            return 'Mg'
        elif clean_name.startswith('CA'):
            return 'Ca'
        elif clean_name.startswith('MN'):
            return 'Mn'
        elif clean_name.startswith('CL'):
            return 'Cl'
        elif clean_name.startswith('BR'):
            return 'Br'
        elif clean_name.startswith('F'):
            return 'F'
        elif clean_name.startswith('H'):
            return 'H'
        else:
            # Default to first character if nothing matches
            return clean_name[0] if clean_name else 'C'
    
    def extract_binding_site_atoms(self, structure: PDBStructure, 
                                 center: np.ndarray, radius: float) -> List[AtomRecord]:
        """Extract atoms within binding site radius"""
        center = validate_position_array(center, "center")
        
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        binding_site_atoms = []
        
        for atom in structure.atoms:
            distance = np.linalg.norm(atom.position - center)
            if distance <= radius:
                binding_site_atoms.append(atom)
        
        return binding_site_atoms
    
    def create_atomic_environments(self, structure: PDBStructure, 
                                 center: np.ndarray, radius: float) -> List[AtomicEnvironment]:
        """Create AtomicEnvironment objects for binding site analysis"""
        binding_site_atoms = self.extract_binding_site_atoms(structure, center, radius)
        
        environments = []
        for atom in binding_site_atoms:
            partial_charge = self._estimate_partial_charge(atom)
            
            env = AtomicEnvironment(
                atom_id=atom.serial,
                position=atom.position,
                element=atom.element,
                residue_name=atom.resname,
                residue_id=atom.resnum,
                chain=atom.chain,
                partial_charge=partial_charge,
                is_hbd=self._is_hydrogen_bond_donor(atom),
                is_hba=self._is_hydrogen_bond_acceptor(atom),
                is_hydrophobic=self._is_hydrophobic(atom),
                is_aromatic=self._is_aromatic_atom(atom),
                van_der_waals_radius=self._get_vdw_radius(atom.element),
                formal_charge=self._get_formal_charge(atom),
                b_factor=atom.b_factor,
                occupancy=atom.occupancy
            )
            environments.append(env)
        
        return environments
    
    def validate_structure(self, structure: PDBStructure) -> Tuple[bool, List[str]]:
        """Validate parsed PDB structure"""
        issues = []
        
        # Check if structure has atoms
        if not structure.atoms:
            issues.append("No protein atoms found")
        
        # Check for missing coordinates
        for i, atom in enumerate(structure.atoms):
            if not np.all(np.isfinite(atom.position)):
                issues.append(f"Invalid coordinates for atom {i+1}")
        
        # Check for reasonable coordinate ranges
        if structure.atoms:
            positions = np.array([atom.position for atom in structure.atoms])
            coord_range = np.ptp(positions, axis=0)
            if np.any(coord_range > 1000):  # More than 1000 Å in any dimension
                issues.append("Coordinates span unusually large range")
            if np.any(coord_range < 1):  # Less than 1 Å in any dimension
                issues.append("Coordinates span unusually small range")
        
        # Check for duplicate atoms
        atom_keys = set()
        for atom in structure.atoms:
            key = (atom.chain, atom.resnum, atom.name, atom.alt_loc)
            if key in atom_keys:
                issues.append(f"Duplicate atom: {key}")
            atom_keys.add(key)
        
        # Check chain consistency
        if not structure.chains:
            issues.append("No chain identifiers found")
        
        # Check for reasonable B-factors
        b_factors = [atom.b_factor for atom in structure.atoms if atom.b_factor > 0]
        if b_factors:
            avg_b = np.mean(b_factors)
            if avg_b > 100:
                issues.append(f"Unusually high average B-factor: {avg_b:.1f}")
        
        return len(issues) == 0, issues
    
    def get_ligand_centers(self, structure: PDBStructure) -> Dict[str, np.ndarray]:
        """Get center of mass for each ligand"""
        centers = {}
        
        for ligand_name, atoms in structure.ligands.items():
            if atoms:
                positions = np.array([atom.position for atom in atoms])
                centers[ligand_name] = np.mean(positions, axis=0)
        
        return centers
    
    def find_binding_sites(self, structure: PDBStructure, 
                          min_cavity_size: float = 100.0) -> List[np.ndarray]:
        """Find potential binding sites using ligand positions and cavities"""
        binding_sites = []
        
        # Add ligand centers as potential binding sites
        ligand_centers = self.get_ligand_centers(structure)
        for center in ligand_centers.values():
            binding_sites.append(center)
        
        # TODO: Add cavity detection algorithm here
        # This would involve finding large empty spaces in the protein
        
        return binding_sites
    
    # Helper methods for chemical property assignment
    def _estimate_partial_charge(self, atom: AtomRecord) -> float:
        """Estimate partial charge based on atom type and chemical environment"""
        element = atom.element
        resname = atom.resname
        atom_name = atom.name
        
        # Enhanced charge assignment rules
        if element == 'N':
            if atom_name in ['NZ']:  # Lysine
                return 1.0
            elif atom_name in ['NH1', 'NH2']:  # Arginine
                return 0.5
            elif atom_name in ['N']:  # Backbone nitrogen
                return -0.3
            elif atom_name in ['ND1', 'NE2'] and resname == 'HIS':
                return -0.1
            else:
                return -0.2
        elif element == 'O':
            if atom_name in ['OE1', 'OE2']:  # Glutamate
                return -0.5
            elif atom_name in ['OD1', 'OD2']:  # Aspartate
                return -0.5
            elif atom_name in ['O']:  # Backbone carbonyl
                return -0.4
            elif atom_name in ['OH', 'OG', 'OG1']:  # Hydroxyl
                return -0.3
            else:
                return -0.2
        elif element == 'S':
            if atom_name == 'SG':  # Cysteine
                return -0.1
            else:
                return -0.05
        elif element == 'C':
            if atom_name in ['C']:  # Carbonyl carbon
                return 0.4
            elif resname in ['ARG'] and atom_name == 'CZ':
                return 0.2
            else:
                return 0.0
        else:
            return 0.0
    
    def _get_formal_charge(self, atom: AtomRecord) -> int:
        """Get formal charge of atom"""
        element = atom.element
        resname = atom.resname
        atom_name = atom.name
        
        if element == 'N' and atom_name == 'NZ' and resname == 'LYS':
            return 1
        elif element == 'N' and atom_name in ['NH1', 'NH2'] and resname == 'ARG':
            return 1
        elif element == 'O' and atom_name in ['OE1', 'OE2'] and resname == 'GLU':
            return -1
        elif element == 'O' and atom_name in ['OD1', 'OD2'] and resname == 'ASP':
            return -1
        else:
            return 0
    
    def _is_hydrogen_bond_donor(self, atom: AtomRecord) -> bool:
        """Enhanced HBD detection"""
        element = atom.element
        atom_name = atom.name
        resname = atom.resname
        
        if element == 'N':
            # Most nitrogens can donate, except some specific cases
            return atom_name not in ['NE2'] or resname != 'HIS'
        elif element == 'O':
            # Hydroxyl oxygens
            return atom_name in ['OH', 'OG', 'OG1', 'OD1', 'OE1', 'OW']
        elif element == 'S':
            return atom_name == 'SG'  # Cysteine sulfur
        
        return False
    
    def _is_hydrogen_bond_acceptor(self, atom: AtomRecord) -> bool:
        """Enhanced HBA detection"""
        element = atom.element
        return element in ['N', 'O', 'F', 'S']
    
    def _is_hydrophobic(self, atom: AtomRecord) -> bool:
        """Enhanced hydrophobic atom detection"""
        element = atom.element
        resname = atom.resname
        atom_name = atom.name
        
        if element == 'C':
            # Exclude polar carbons
            if atom_name in ['C']:  # Carbonyl carbons
                return False
            # Include aliphatic and aromatic carbons from hydrophobic residues
            if resname in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'TYR', 'PRO']:
                if atom_name not in ['CA', 'CB']:  # Exclude backbone carbons
                    return True
        elif element == 'S':
            if resname == 'MET':
                return True
        
        return False
    
    def _is_aromatic_atom(self, atom: AtomRecord) -> bool:
        """Detect aromatic atoms"""
        resname = atom.resname
        atom_name = atom.name
        
        aromatic_residues = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1'],
            'HIS': ['CG', 'CD2', 'CE1', 'ND1', 'NE2']
        }
        
        return resname in aromatic_residues and atom_name in aromatic_residues[resname]
    
    def _get_vdw_radius(self, element: str) -> float:
        """Get van der Waals radius for element"""
        vdw_radii = {
            "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
            "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
            "MG": 1.73, "CA": 2.31, "FE": 2.00, "ZN": 1.39, "MN": 1.61
        }
        return vdw_radii.get(element.upper(), 1.7)  # Default to carbon radius
    
    def get_warnings(self) -> List[str]:
        """Get list of parsing warnings"""
        return self.warnings.copy()
    
    def clear_warnings(self):
        """Clear warning list"""
        self.warnings.clear()


# Utility functions
def download_pdb(pdb_id: str) -> str:
    """Download PDB file from RCSB PDB"""
    import urllib.request
    import urllib.error
    
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except urllib.error.URLError as e:
        raise ValueError(f"Failed to download PDB {pdb_id}: {str(e)}")


def generate_sample_pdb() -> str:
    """Generate a sample PDB structure for demonstration"""
    return """HEADER    TRANSFERASE/DNA                         20-MAY-03   1ABC              
TITLE     SAMPLE KINASE STRUCTURE FOR LIGANDFORGE DEMONSTRATION                 
REMARK   2 RESOLUTION.    2.10 ANGSTROMS.                                       
ATOM      1  N   ALA A   1      -8.901   4.127  -0.555  1.00 11.99           N  
ATOM      2  CA  ALA A   1      -8.608   3.135  -1.618  1.00 11.99           C  
ATOM      3  C   ALA A   1      -7.221   2.458  -1.897  1.00 11.99           C  
ATOM      4  O   ALA A   1      -6.632   1.943  -1.084  1.00 11.99           O  
ATOM      5  CB  ALA A   1      -9.062   3.700  -2.969  1.00 11.99           C  
ATOM      6  N   SER A   2      -6.842   2.458  -3.157  1.00 11.99           N  
ATOM      7  CA  SER A   2      -5.618   1.840  -3.618  1.00 11.99           C  
ATOM      8  C   SER A   2      -5.997   0.573  -4.378  1.00 11.99           C  
ATOM      9  O   SER A   2      -7.115   0.340  -4.835  1.00 11.99           O  
ATOM     10  CB  SER A   2      -4.729   2.735  -4.469  1.00 11.99           C  
ATOM     11  OG  SER A   2      -5.470   3.465  -5.443  1.00 11.99           O  
ATOM     12  N   ASP A   3      -5.032  -0.317  -4.470  1.00 11.99           N  
ATOM     13  CA  ASP A   3      -5.246  -1.605  -5.156  1.00 11.99           C  
ATOM     14  C   ASP A   3      -4.163  -1.897  -6.208  1.00 11.99           C  
ATOM     15  O   ASP A   3      -3.044  -1.388  -6.206  1.00 11.99           O  
ATOM     16  CB  ASP A   3      -5.260  -2.805  -4.203  1.00 11.99           C  
ATOM     17  CG  ASP A   3      -6.406  -2.790  -3.195  1.00 11.99           C  
ATOM     18  OD1 ASP A   3      -7.575  -3.035  -3.551  1.00 11.99           O  
ATOM     19  OD2 ASP A   3      -6.192  -2.537  -1.991  1.00 11.99           O  
HETATM   20  O   HOH A   4      -4.123   0.234  -0.567  1.00 22.50           O  
HETATM   21  O   HOH A   5      -2.123   2.234   1.567  1.00 25.50           O  
HETATM   22  O   HOH A   6      -1.123  -1.234   2.567  1.00 30.50           O  
HETATM   23  C1  ATP A 501      -3.123   3.234   3.567  1.00 15.50           C  
HETATM   24  C2  ATP A 501      -2.123   4.234   4.567  1.00 16.50           C  
HETATM   25  N1  ATP A 501      -1.123   5.234   5.567  1.00 17.50           N  
END                                                                             """
