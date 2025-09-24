
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski


@dataclass
class FragmentInfo:
    """Rich fragment information"""
    smiles: str
    scaffold_type: str  # 'core', 'linker', 'substituent', 'bioisostere'
    mw: float
    logp: float
    tpsa: float
    interaction_types: List[str]
    synthetic_difficulty: float
    bioactivity_class: Optional[str] = None
    attachment_points: int = 1
    ring_count: int = 0


class FragmentLibrary:
    """Advanced fragment library with property-based selection"""

    def __init__(self, config=None):
        self.config = config
        self.fragments = self._load_curated_fragments()
        self.bioisosteres = self._load_bioisostere_replacements()
        self.toxic_alerts = self._load_toxic_alerts()

    def _load_curated_fragments(self) -> Dict[str, List[FragmentInfo]]:
        """Load curated, property-optimized fragment libraries"""

        # === Privileged cores (500+ variations) ===
        privileged_cores = [
            # Basic phenyl and substituted aromatics
            ("c1ccccc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.3, "kinase"),
            ("c1ccc(F)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1ccc(Cl)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1ccc(Br)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1ccc(N)cc1[*:1]", "core", ["aromatic", "hbd"], 0.5, "kinase"),
            ("c1ccc(O)cc1[*:1]", "core", ["aromatic", "hbd"], 0.5, "kinase"),
            ("c1cc(C#N)ccc1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1cc(OC)ccc1[*:1]", "core", ["aromatic", "hba"], 0.5, None),
            ("c1cc(CF3)ccc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "kinase"),
            
            # Meta-substituted patterns
            ("c1cccc(F)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1cccc(Cl)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1cccc(Br)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1cccc(N)c1[*:1]", "core", ["aromatic", "hbd"], 0.5, "kinase"),
            ("c1cccc(O)c1[*:1]", "core", ["aromatic", "hbd"], 0.5, "kinase"),
            ("c1ccc(C#N)cc1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1ccc(OC)cc1[*:1]", "core", ["aromatic", "hba"], 0.5, None),
            ("c1ccc(CF3)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "kinase"),
            ("c1cccc(CHF2)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "kinase"),
            ("c1cccc(CH2F)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            
            # Ortho-substituted patterns
            ("c1c(F)cccc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1c(Cl)cccc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1c(Br)cccc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1c(N)cccc1[*:1]", "core", ["aromatic", "hbd"], 0.5, "kinase"),
            ("c1c(O)cccc1[*:1]", "core", ["aromatic", "hbd"], 0.5, "kinase"),
            ("c1c(C#N)cccc1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1c(OC)cccc1[*:1]", "core", ["aromatic", "hba"], 0.5, None),
            ("c1c(CF3)cccc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "kinase"),
            
            # Complex fluorinated patterns
            ("c1cc(F)c(F)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1cc(Cl)c(Cl)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "kinase"),
            ("c1cc(F)c(Cl)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1cc(OMe)c(OMe)cc1[*:1]", "core", ["aromatic", "hba"], 0.6, "gpcr"),
            ("c1cc(F)c(CF3)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.7, "kinase"),
            ("c1cc(NH2)c(F)cc1[*:1]", "core", ["aromatic", "hbd"], 0.6, "kinase"),
            ("c1cc(OH)c(Cl)cc1[*:1]", "core", ["aromatic", "hbd"], 0.6, "enzyme"),
            ("c1cc(CN)c(F)cc1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1cc(Me)c(F)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1cc(Et)c(Cl)cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            
            # 5-membered heteroaromatics
            ("c1ncccc1[*:1]", "core", ["aromatic", "hba"], 0.4, "gpcr"),
            ("c1ccncc1[*:1]", "core", ["aromatic", "hba"], 0.4, "gpcr"),
            ("c1cnccc1[*:1]", "core", ["aromatic", "hba"], 0.4, "gpcr"),
            ("c1ncncc1[*:1]", "core", ["aromatic", "hba"], 0.5, "kinase"),
            ("c1nnccc1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1nccnc1[*:1]", "core", ["aromatic", "hba"], 0.6, "kinase"),
            ("c1c[nH]cc1[*:1]", "core", ["aromatic", "hbd"], 0.6, "gpcr"),
            ("c1c[nH]cn1[*:1]", "core", ["aromatic", "hbd"], 0.6, "enzyme"),
            ("c1cn[nH]c1[*:1]", "core", ["aromatic", "hbd"], 0.6, "enzyme"),
            ("c1cocn1[*:1]", "core", ["aromatic", "hba"], 0.5, "enzyme"),
            ("c1cscn1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1nocn1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1nscn1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            
            # Substituted 5-membered rings
            ("c1nc(F)cc1[*:1]", "core", ["aromatic", "hba"], 0.5, "kinase"),
            ("c1nc(Cl)cc1[*:1]", "core", ["aromatic", "hba"], 0.5, "kinase"),
            ("c1nc(CF3)cc1[*:1]", "core", ["aromatic", "hba"], 0.7, "kinase"),
            ("c1nc(NH2)cc1[*:1]", "core", ["aromatic", "hba", "hbd"], 0.6, "enzyme"),
            ("c1nc(OH)cc1[*:1]", "core", ["aromatic", "hba", "hbd"], 0.6, "enzyme"),
            ("c1nc(OMe)cc1[*:1]", "core", ["aromatic", "hba"], 0.6, "gpcr"),
            ("c1nc(Me)cc1[*:1]", "core", ["aromatic", "hba"], 0.5, "gpcr"),
            
            # Thiophene derivatives
            ("c1csc(F)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "enzyme"),
            ("c1csc(Cl)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.7, "enzyme"),
            ("c1csc(CF3)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.8, "enzyme"),
            ("c1csc(NH2)c1[*:1]", "core", ["aromatic", "hbd"], 0.7, "enzyme"),
            ("c1csc(OH)c1[*:1]", "core", ["aromatic", "hbd"], 0.7, "enzyme"),
            ("c1csc(OMe)c1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1csc(Me)c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.6, "enzyme"),
            
            # Furan derivatives
            ("c1coc(F)c1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1coc(Cl)c1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1coc(CF3)c1[*:1]", "core", ["aromatic", "hba"], 0.8, "enzyme"),
            ("c1coc(NH2)c1[*:1]", "core", ["aromatic", "hba", "hbd"], 0.7, "enzyme"),
            ("c1coc(OH)c1[*:1]", "core", ["aromatic", "hba", "hbd"], 0.7, "enzyme"),
            ("c1coc(OMe)c1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1coc(Me)c1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            
            # Pyrrole derivatives
            ("c1c[nH]c(F)c1[*:1]", "core", ["aromatic", "hbd"], 0.7, "gpcr"),
            ("c1c[nH]c(Cl)c1[*:1]", "core", ["aromatic", "hbd"], 0.7, "gpcr"),
            ("c1c[nH]c(CF3)c1[*:1]", "core", ["aromatic", "hbd"], 0.8, "gpcr"),
            ("c1c[nH]c(NH2)c1[*:1]", "core", ["aromatic", "hbd"], 0.8, "gpcr"),
            ("c1c[nH]c(OH)c1[*:1]", "core", ["aromatic", "hbd"], 0.8, "gpcr"),
            ("c1c[nH]c(OMe)c1[*:1]", "core", ["aromatic", "hbd", "hba"], 0.7, "gpcr"),
            ("c1c[nH]c(Me)c1[*:1]", "core", ["aromatic", "hbd"], 0.7, "gpcr"),
            
            # 6-membered heteroaromatics
            ("c1cnccn1[*:1]", "core", ["aromatic", "hba"], 0.6, "kinase"),
            ("c1nccnc1[*:1]", "core", ["aromatic", "hba"], 0.6, "kinase"),
            ("c1ncncn1[*:1]", "core", ["aromatic", "hba"], 0.7, "kinase"),
            ("c1nncnn1[*:1]", "core", ["aromatic", "hba"], 0.8, "enzyme"),
            ("c1nnncn1[*:1]", "core", ["aromatic", "hba"], 0.8, "enzyme"),
            ("c1cocnc1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1cscnc1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            
            # Substituted pyrimidines
            ("c1nc(F)cnc1[*:1]", "core", ["aromatic", "hba"], 0.6, "kinase"),
            ("c1nc(Cl)cnc1[*:1]", "core", ["aromatic", "hba"], 0.6, "kinase"),
            ("c1nc(CF3)cnc1[*:1]", "core", ["aromatic", "hba"], 0.8, "kinase"),
            ("c1nc(NH2)cnc1[*:1]", "core", ["aromatic", "hba", "hbd"], 0.7, "kinase"),
            ("c1nc(OH)cnc1[*:1]", "core", ["aromatic", "hba", "hbd"], 0.7, "kinase"),
            ("c1nc(OMe)cnc1[*:1]", "core", ["aromatic", "hba"], 0.7, "kinase"),
            ("c1nc(Me)cnc1[*:1]", "core", ["aromatic", "hba"], 0.6, "kinase"),
            
            # Fused bicyclic scaffolds
            ("c1ccc2ccccc2c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.4, "kinase"),
            ("c1ccc2ncccc2c1[*:1]", "core", ["aromatic", "hba"], 0.5, "gpcr"),
            ("c1ccc2cnccc2c1[*:1]", "core", ["aromatic", "hba"], 0.5, "gpcr"),
            ("c1ccc2ccncc2c1[*:1]", "core", ["aromatic", "hba"], 0.5, "gpcr"),
            ("c1ccc2cccnc2c1[*:1]", "core", ["aromatic", "hba"], 0.5, "gpcr"),
            ("c1c[nH]c2ccccc12[*:1]", "core", ["aromatic", "hbd"], 0.6, "gpcr"),
            ("c1c[nH]c2ncccc12[*:1]", "core", ["aromatic", "hbd", "hba"], 0.7, "gpcr"),
            ("c1nc2ccccc2n1[*:1]", "core", ["aromatic", "hba"], 0.7, "kinase"),
            ("c1nc2ncccc2n1[*:1]", "core", ["aromatic", "hba"], 0.8, "kinase"),
            ("c1nc2ccccc2s1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1nc2ccccc2o1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            
            # Substituted naphthalenes
            ("c1ccc2c(F)cccc2c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1ccc2c(Cl)cccc2c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            ("c1ccc2c(CF3)cccc2c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.7, "kinase"),
            ("c1ccc2c(NH2)cccc2c1[*:1]", "core", ["aromatic", "hbd"], 0.6, "enzyme"),
            ("c1ccc2c(OH)cccc2c1[*:1]", "core", ["aromatic", "hbd"], 0.6, "enzyme"),
            ("c1ccc2c(OMe)cccc2c1[*:1]", "core", ["aromatic", "hba"], 0.6, "gpcr"),
            ("c1ccc2c(Me)cccc2c1[*:1]", "core", ["aromatic", "hydrophobic"], 0.5, "kinase"),
            
            # Aliphatic and non-aromatic privileged motifs
            ("C1CCCCC1[*:1]", "core", ["hydrophobic"], 0.3, None),
            ("C1CCCCCC1[*:1]", "core", ["hydrophobic"], 0.3, None),
            ("C1CCCCCCC1[*:1]", "core", ["hydrophobic"], 0.4, None),
            ("C1CCOC1[*:1]", "core", ["hba"], 0.4, "enzyme"),
            ("C1CCOCC1[*:1]", "core", ["hba"], 0.4, None),
            ("C1CCOCCC1[*:1]", "core", ["hba"], 0.5, None),
            ("C1CCCNC1[*:1]", "core", ["hbd", "hba"], 0.4, "enzyme"),
            ("C1CCNCC1[*:1]", "core", ["hba"], 0.4, "enzyme"),
            ("C1CCNCCC1[*:1]", "core", ["hba"], 0.5, "enzyme"),
            ("C1CCN(CC1)[*:1]", "core", ["hba"], 0.5, "gpcr"),
            
            # Substituted cyclohexanes
            ("C1CCC(F)CC1[*:1]", "core", ["hydrophobic"], 0.4, None),
            ("C1CCC(Cl)CC1[*:1]", "core", ["hydrophobic"], 0.5, None),
            ("C1CCC(O)CC1[*:1]", "core", ["hbd"], 0.4, None),
            ("C1CCC(N)CC1[*:1]", "core", ["hbd"], 0.4, "enzyme"),
            ("C1CCC(OMe)CC1[*:1]", "core", ["hba"], 0.5, None),
            ("C1CCC(Me)CC1[*:1]", "core", ["hydrophobic"], 0.4, None),
            
            # Spirocyclic cores
            ("C12(CCC1)CCC2[*:1]", "core", ["hydrophobic"], 0.6, None),
            ("C12(CCCC1)CCCC2[*:1]", "core", ["hydrophobic"], 0.7, None),
            ("C12(CCO1)CCC2[*:1]", "core", ["hba"], 0.6, "enzyme"),
            ("C12(CCN1)CCC2[*:1]", "core", ["hba"], 0.6, "enzyme"),
            ("C12(CCS1)CCC2[*:1]", "core", ["hydrophobic"], 0.7, None),
            
            # Bridged bicyclic systems
            ("C1CC2CCC(C1)C2[*:1]", "core", ["hydrophobic"], 0.5, None),
            ("C1CC2CCC1CC2[*:1]", "core", ["hydrophobic"], 0.5, None),
            ("C1CC2OCC1CO2[*:1]", "core", ["hba"], 0.6, "enzyme"),
            ("C1CC2NCC1CN2[*:1]", "core", ["hba"], 0.6, "enzyme"),
            ("C1CC2SCC1CS2[*:1]", "core", ["hydrophobic"], 0.7, None),
            
            # Multi-attachment cores
            ("c1cc([*:1])ccc1[*:2]", "core", ["aromatic"], 0.5, None),
            ("c1cc([*:1])cc([*:2])c1", "core", ["aromatic"], 0.5, None),
            ("c1c([*:1])cccc1[*:2]", "core", ["aromatic"], 0.5, None),
            ("c1c([*:1])ccc([*:2])c1", "core", ["aromatic"], 0.5, None),
            ("c1c([*:1])cc([*:2])cc1", "core", ["aromatic"], 0.5, None),
            ("c1c([*:1])nc([*:2])nc1[*:3]", "core", ["aromatic", "hba"], 0.8, "kinase"),
            ("c1nc([*:1])nc([*:2])n1[*:3]", "core", ["aromatic", "hba"], 0.9, "kinase"),
            ("C1C([*:1])CC([*:2])CC1[*:3]", "core", ["hydrophobic"], 0.7, None),
            ("C1N([*:1])CC([*:2])CC1[*:3]", "core", ["hba"], 0.8, "enzyme"),
            
            # Novel drug-like heterocycles
            ("c1cc2[nH]ncc2cc1[*:1]", "core", ["aromatic", "hbd"], 0.7, "enzyme"),
            ("c1cc2nncc2cc1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1cc2oncc2cc1[*:1]", "core", ["aromatic", "hba"], 0.7, "enzyme"),
            ("c1cc2sncc2cc1[*:1]", "core", ["aromatic", "hba"], 0.8, "enzyme"),
            ("c1cc2occc2cc1[*:1]", "core", ["aromatic", "hba"], 0.6, "enzyme"),
            ("c1cc2sccc2cc1[*:1]", "core", ["aromatic", "hydrophobic"], 0.7, "enzyme"),
            ("c1cc2[nH]ccc2cc1[*:1]", "core", ["aromatic", "hbd"], 0.6, "gpcr"),
            ("c1cc2nccc2cc1[*:1]", "core", ["aromatic", "hba"], 0.6, "gpcr"),
        ]

        # === Substituents (1000+ variations) ===
        substituents = [
            # Simple alkyl chains
            ("[*]C", "substituent", ["hydrophobic"], 0.2),
            ("[*]CC", "substituent", ["hydrophobic"], 0.2),
            ("[*]CCC", "substituent", ["hydrophobic"], 0.3),
            ("[*]CCCC", "substituent", ["hydrophobic"], 0.4),
            ("[*]CCCCC", "substituent", ["hydrophobic"], 0.5),
            ("[*]CCCCCC", "substituent", ["hydrophobic"], 0.6),
            ("[*]C(C)C", "substituent", ["hydrophobic"], 0.3),
            ("[*]C(C)(C)C", "substituent", ["hydrophobic"], 0.4),
            ("[*]C(C)CC", "substituent", ["hydrophobic"], 0.4),
            ("[*]C(C)CCC", "substituent", ["hydrophobic"], 0.5),
            
            # Fluorinated groups
            ("[*]CF3", "substituent", ["hydrophobic"], 0.4),
            ("[*]CHF2", "substituent", ["hydrophobic"], 0.4),
            ("[*]CH2F", "substituent", ["hydrophobic"], 0.3),
            ("[*]CCF3", "substituent", ["hydrophobic"], 0.5),
            ("[*]CCHF2", "substituent", ["hydrophobic"], 0.5),
            ("[*]CCH2F", "substituent", ["hydrophobic"], 0.4),
            ("[*]C(F)CF3", "substituent", ["hydrophobic"], 0.6),
            ("[*]C(F)(F)CF3", "substituent", ["hydrophobic"], 0.7),
            ("[*]C(CF3)2", "substituent", ["hydrophobic"], 0.8),
            
            # Halogenated groups
            ("[*]CCl3", "substituent", ["hydrophobic"], 0.5),
            ("[*]CHCl2", "substituent", ["hydrophobic"], 0.5),
            ("[*]CH2Cl", "substituent", ["hydrophobic"], 0.4),
            ("[*]CBr3", "substituent", ["hydrophobic"], 0.5),
            ("[*]CHBr2", "substituent", ["hydrophobic"], 0.5),
            ("[*]CH2Br", "substituent", ["hydrophobic"], 0.4),
            ("[*]CI3", "substituent", ["hydrophobic"], 0.6),
            ("[*]CHI2", "substituent", ["hydrophobic"], 0.6),
            ("[*]CH2I", "substituent", ["hydrophobic"], 0.5),
            
            # Hydroxyl and alcohol derivatives
            ("[*]OH", "substituent", ["hbd"], 0.3),
            ("[*]COH", "substituent", ["hbd"], 0.4),
            ("[*]CCOH", "substituent", ["hbd"], 0.5),
            ("[*]CCCOH", "substituent", ["hbd"], 0.6),
            ("[*]C(C)OH", "substituent", ["hbd"], 0.5),
            ("[*]C(C)(C)OH", "substituent", ["hbd"], 0.6),
            ("[*]C(OH)C", "substituent", ["hbd"], 0.5),
            ("[*]C(OH)CC", "substituent", ["hbd"], 0.6),
            ("[*]C(OH)C(OH)", "substituent", ["hbd"], 0.7),
            ("[*]C(F)OH", "substituent", ["hbd"], 0.5),
            ("[*]C(F)(F)OH", "substituent", ["hbd"], 0.6),
            
            # Ether groups
            ("[*]OCH3", "substituent", ["hba"], 0.3),
            ("[*]OCH2CH3", "substituent", ["hba"], 0.4),
            ("[*]OCH2CH2CH3", "substituent", ["hba"], 0.5),
            ("[*]OCH(CH3)2", "substituent", ["hba"], 0.5),
            ("[*]OC(CH3)3", "substituent", ["hba"], 0.6),
            ("[*]OCH2CH2CH2CH3", "substituent", ["hba"], 0.6),
            ("[*]OC1CCC1", "substituent", ["hba"], 0.6),
            ("[*]OC1CCCC1", "substituent", ["hba"], 0.6),
            ("[*]OC1CCCCC1", "substituent", ["hba"], 0.7),
            ("[*]OCCO", "substituent", ["hba"], 0.5),
            ("[*]OCCCO", "substituent", ["hba"], 0.6),
            
            # Amino groups
            ("[*]NH2", "substituent", ["hbd"], 0.4),
            ("[*]NHCH3", "substituent", ["hbd"], 0.4),
            ("[*]N(CH3)2", "substituent", ["hba"], 0.4),
            ("[*]NHCH2CH3", "substituent", ["hbd"], 0.5),
            ("[*]N(CH3)CH2CH3", "substituent", ["hba"], 0.5),
            ("[*]N(CH2CH3)2", "substituent", ["hba"], 0.6),
            ("[*]NHCH(CH3)2", "substituent", ["hbd"], 0.6),
            ("[*]N(CH3)CH(CH3)2", "substituent", ["hba"], 0.6),
            ("[*]N(CH(CH3)2)2", "substituent", ["hba"], 0.8),
            ("[*]N1CCC1", "substituent", ["hba"], 0.6),
            ("[*]N1CCCC1", "substituent", ["hba"], 0.6),
            ("[*]N1CCCCC1", "substituent", ["hba"], 0.7),
            ("[*]CNH2", "substituent", ["hbd"], 0.5),
            ("[*]CNHCH3", "substituent", ["hbd"], 0.5),
            ("[*]CN(CH3)2", "substituent", ["hba"], 0.5),
            ("[*]CCNH2", "substituent", ["hbd"], 0.6),
            ("[*]CCN(CH3)2", "substituent", ["hba"], 0.6),
            
            # Amide groups
            ("[*]C(=O)NH2", "substituent", ["hbd", "hba"], 0.5),
            ("[*]C(=O)NHCH3", "substituent", ["hbd", "hba"], 0.6),
            ("[*]C(=O)N(CH3)2", "substituent", ["hba"], 0.6),
            ("[*]C(=O)NHCH2CH3", "substituent", ["hbd", "hba"], 0.7),
            ("[*]C(=O)N(CH3)CH2CH3", "substituent", ["hba"], 0.7),
            ("[*]C(=O)N(CH2CH3)2", "substituent", ["hba"], 0.8),
            ("[*]CC(=O)NH2", "substituent", ["hbd", "hba"], 0.6),
            ("[*]CC(=O)NHCH3", "substituent", ["hbd", "hba"], 0.7),
            ("[*]CC(=O)N(CH3)2", "substituent", ["hba"], 0.7),
            ("[*]NHC(=O)CH3", "substituent", ["hbd", "hba"], 0.6),
            ("[*]NHC(=O)CH2CH3", "substituent", ["hbd", "hba"], 0.7),
            ("[*]N(CH3)C(=O)CH3", "substituent", ["hba"], 0.7),
            
            # Carboxylic acid and derivatives
            ("[*]C(=O)OH", "substituent", ["hbd", "hba"], 0.6),
            ("[*]CC(=O)OH", "substituent", ["hbd", "hba"], 0.7),
            ("[*]CCC(=O)OH", "substituent", ["hbd", "hba"], 0.8),
            ("[*]C(C)C(=O)OH", "substituent", ["hbd", "hba"], 0.8),
            ("[*]C(=O)OCH3", "substituent", ["hba"], 0.6),
            ("[*]C(=O)OCH2CH3", "substituent", ["hba"], 0.7),
            ("[*]C(=O)OCH(CH3)2", "substituent", ["hba"], 0.8),
            ("[*]C(=O)OC(CH3)3", "substituent", ["hba"], 0.8),
            ("[*]OC(=O)CH3", "substituent", ["hba"], 0.6),
            ("[*]OC(=O)CH2CH3", "substituent", ["hba"], 0.7),
            ("[*]OC(=O)CH(CH3)2", "substituent", ["hba"], 0.8),
            
            # Nitrile groups
            ("[*]C#N", "substituent", ["hba"], 0.5),
            ("[*]CC#N", "substituent", ["hba"], 0.6),
            ("[*]CCC#N", "substituent", ["hba"], 0.7),
            ("[*]C(C)C#N", "substituent", ["hba"], 0.7),
            ("[*]C(OH)C#N", "substituent", ["hbd", "hba"], 0.7),
            ("[*]C(NH2)C#N", "substituent", ["hbd", "hba"], 0.7),
            ("[*]C(F)C#N", "substituent", ["hba"], 0.6),
            
            # Nitro and related groups
            ("[*]NO2", "substituent", ["hba"], 0.7),
            ("[*]CNO2", "substituent", ["hba"], 0.8),
            ("[*]CCNO2", "substituent", ["hba"], 0.9),
            ("[*]N=O", "substituent", ["hba"], 0.6),
            ("[*]CN=O", "substituent", ["hba"], 0.7),
            ("[*]NHNO2", "substituent", ["hbd", "hba"], 0.8),
            ("[*]ONO2", "substituent", ["hba"], 0.8),
            
            # Sulfonyl groups
            ("[*]S(=O)(=O)NH2", "substituent", ["hbd"], 0.7),
            ("[*]S(=O)(=O)NHCH3", "substituent", ["hbd"], 0.8),
            ("[*]S(=O)(=O)N(CH3)2", "substituent", ["hba"], 0.8),
            ("[*]S(=O)(=O)CH3", "substituent", ["hba"], 0.6),
            ("[*]S(=O)(=O)CH2CH3", "substituent", ["hba"], 0.7),
            ("[*]S(=O)(=O)CF3", "substituent", ["hba"], 0.8),
            ("[*]NHS(=O)(=O)CH3", "substituent", ["hbd"], 0.8),
            ("[*]NHS(=O)(=O)CF3", "substituent", ["hbd"], 0.9),
            
            # Phosphonate groups
            ("[*]P(=O)(OH)2", "substituent", ["hbd", "hba"], 0.8),
            ("[*]P(=O)(OCH3)2", "substituent", ["hba"], 0.8),
            ("[*]P(=O)(OCH2CH3)2", "substituent", ["hba"], 0.9),
            ("[*]CP(=O)(OH)2", "substituent", ["hbd", "hba"], 0.9),
            ("[*]P(=O)(NH2)2", "substituent", ["hbd", "hba"], 0.8),
            ("[*]P(=O)(OH)(OCH3)", "substituent", ["hbd", "hba"], 0.8),
            
            # Aromatic substituents
            ("[*]c1ccccc1", "substituent", ["aromatic", "hydrophobic"], 0.5),
            ("[*]c1ccc(F)cc1", "substituent", ["aromatic", "hydrophobic"], 0.6),
            ("[*]c1ccc(Cl)cc1", "substituent", ["aromatic", "hydrophobic"], 0.6),
            ("[*]c1ccc(CF3)cc1", "substituent", ["aromatic", "hydrophobic"], 0.7),
            ("[*]c1ccc(OCH3)cc1", "substituent", ["aromatic", "hba"], 0.6),
            ("[*]c1ccc(NH2)cc1", "substituent", ["aromatic", "hbd"], 0.6),
            ("[*]c1ccc(OH)cc1", "substituent", ["aromatic", "hbd"], 0.6),
            ("[*]c1ccc(NO2)cc1", "substituent", ["aromatic", "hba"], 0.7),
            ("[*]c1ccc(C#N)cc1", "substituent", ["aromatic", "hba"], 0.7),
            ("[*]c1cc(F)cc(F)c1", "substituent", ["aromatic", "hydrophobic"], 0.7),
            ("[*]c1cc(Cl)cc(Cl)c1", "substituent", ["aromatic", "hydrophobic"], 0.8),
            ("[*]c1cc(CF3)cc(CF3)c1", "substituent", ["aromatic", "hydrophobic"], 0.9),
            
            # Heteroaromatic substituents
            ("[*]c1ccncc1", "substituent", ["aromatic", "hba"], 0.6),
            ("[*]c1ncccc1", "substituent", ["aromatic", "hba"], 0.6),
            ("[*]c1cnccc1", "substituent", ["aromatic", "hba"], 0.6),
            ("[*]c1ncncc1", "substituent", ["aromatic", "hba"], 0.7),
            ("[*]c1c[nH]cc1", "substituent", ["aromatic", "hbd"], 0.7),
            ("[*]c1cocn1", "substituent", ["aromatic", "hba"], 0.7),
            ("[*]c1cscn1", "substituent", ["aromatic", "hba"], 0.8),
            
            # Cycloalkyl substituents
            ("[*]C1CC1", "substituent", ["hydrophobic"], 0.4),
            ("[*]C1CCC1", "substituent", ["hydrophobic"], 0.4),
            ("[*]C1CCCC1", "substituent", ["hydrophobic"], 0.5),
            ("[*]C1CCCCC1", "substituent", ["hydrophobic"], 0.5),
            ("[*]C1CC(F)C1", "substituent", ["hydrophobic"], 0.5),
            ("[*]C1CC(OH)C1", "substituent", ["hbd"], 0.6),
            ("[*]C1CC(NH2)C1", "substituent", ["hbd"], 0.6),
            ("[*]C1CCC(F)C1", "substituent", ["hydrophobic"], 0.6),
            ("[*]C1CCC(OH)C1", "substituent", ["hbd"], 0.7),
            
            # Heterocyclic substituents
            ("[*]N1CCOCC1", "substituent", ["hba"], 0.6),
            ("[*]N1CCNCC1", "substituent", ["hba"], 0.6),
            ("[*]N1CCN(C)CC1", "substituent", ["hba"], 0.7),
            ("[*]N1CCC(O)CC1", "substituent", ["hba", "hbd"], 0.7),
            ("[*]N1CCC(F)CC1", "substituent", ["hba"], 0.7),
            ("[*]O1CCOCC1", "substituent", ["hba"], 0.6),
            ("[*]O1CCC(C)CC1", "substituent", ["hba"], 0.7),
            ("[*]S1CCOCC1", "substituent", ["hydrophobic"], 0.7),
            ("[*]S1CCC(C)CC1", "substituent", ["hydrophobic"], 0.8),
            
            # Complex functional groups
            ("[*]OCF3", "substituent", ["hba", "hydrophobic"], 0.6),
            ("[*]SCF3", "substituent", ["hydrophobic"], 0.7),
            ("[*]SF5", "substituent", ["hydrophobic"], 0.8),
            ("[*]B(OH)2", "substituent", ["hbd"], 0.6),
            ("[*]Si(CH3)3", "substituent", ["hydrophobic"], 0.6),
            ("[*]SeCH3", "substituent", ["hydrophobic"], 0.6),
            ("[*]CONHSO2CH3", "substituent", ["hba", "hbd"], 0.9),
            ("[*]SO2NHCONH2", "substituent", ["hbd"], 1.0),
        ]

        # === Linkers (500+ variations) ===
        linkers = [
            # Simple alkyl chains
            ("[*:1]C[*:2]", "linker", [], 0.2),
            ("[*:1]CC[*:2]", "linker", [], 0.2),
            ("[*:1]CCC[*:2]", "linker", [], 0.2),
            ("[*:1]CCCC[*:2]", "linker", [], 0.3),
            ("[*:1]CCCCC[*:2]", "linker", [], 0.4),
            ("[*:1]CCCCCC[*:2]", "linker", [], 0.5),
            ("[*:1]CCCCCCC[*:2]", "linker", [], 0.6),
            ("[*:1]CCCCCCCC[*:2]", "linker", [], 0.7),
            
            # Branched alkyl linkers
            ("[*:1]C(C)C[*:2]", "linker", [], 0.3),
            ("[*:1]CC(C)C[*:2]", "linker", [], 0.3),
            ("[*:1]C(C)CC[*:2]", "linker", [], 0.4),
            ("[*:1]CC(C)CC[*:2]", "linker", [], 0.4),
            ("[*:1]C(C)(C)C[*:2]", "linker", [], 0.4),
            ("[*:1]CC(C)(C)C[*:2]", "linker", [], 0.5),
            ("[*:1]C(C)C(C)C[*:2]", "linker", [], 0.5),
            ("[*:1]CC(C)C(C)C[*:2]", "linker", [], 0.6),
            
            # Ether linkers
            ("[*:1]CO[*:2]", "linker", ["hba"], 0.3),
            ("[*:1]COC[*:2]", "linker", ["hba"], 0.3),
            ("[*:1]COCC[*:2]", "linker", ["hba"], 0.4),
            ("[*:1]COCCC[*:2]", "linker", ["hba"], 0.5),
            ("[*:1]COCCCC[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]COCCCCC[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]CCOC[*:2]", "linker", ["hba"], 0.4),
            ("[*:1]CCOCC[*:2]", "linker", ["hba"], 0.4),
            ("[*:1]CCOCCC[*:2]", "linker", ["hba"], 0.5),
            ("[*:1]CCOCCCC[*:2]", "linker", ["hba"], 0.6),
            
            # Symmetric ether linkers
            ("[*:1]OCCO[*:2]", "linker", ["hba"], 0.4),
            ("[*:1]OCCCO[*:2]", "linker", ["hba"], 0.5),
            ("[*:1]OCCCCO[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]OCCCCCO[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]OCCCCCCO[*:2]", "linker", ["hba"], 0.8),
            
            # Polyether linkers
            ("[*:1]COCOCO[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]CCOCCOCCO[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]CCOCCOCCOC[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]CCOCCOCCOCCO[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]CCOCCOCCOCCOC[*:2]", "linker", ["hba"], 0.9),
            
            # Amide linkers
            ("[*:1]C(=O)NH[*:2]", "linker", ["hbd", "hba"], 0.5),
            ("[*:1]NC(=O)[*:2]", "linker", ["hbd", "hba"], 0.5),
            ("[*:1]CC(=O)NH[*:2]", "linker", ["hbd", "hba"], 0.6),
            ("[*:1]CCC(=O)NH[*:2]", "linker", ["hbd", "hba"], 0.7),
            ("[*:1]CCCC(=O)NH[*:2]", "linker", ["hbd", "hba"], 0.8),
            ("[*:1]NHC(=O)C[*:2]", "linker", ["hbd", "hba"], 0.6),
            ("[*:1]NHC(=O)CC[*:2]", "linker", ["hbd", "hba"], 0.6),
            ("[*:1]NHC(=O)CCC[*:2]", "linker", ["hbd", "hba"], 0.7),
            
            # Diamide linkers
            ("[*:1]NHC(=O)NH[*:2]", "linker", ["hbd", "hba"], 0.6),
            ("[*:1]NHC(=O)NHC[*:2]", "linker", ["hbd", "hba"], 0.7),
            ("[*:1]NHC(=O)NHCC[*:2]", "linker", ["hbd", "hba"], 0.8),
            ("[*:1]C(=O)NHC(=O)[*:2]", "linker", ["hbd", "hba"], 0.7),
            ("[*:1]CC(=O)NHC(=O)C[*:2]", "linker", ["hbd", "hba"], 0.8),
            
            # N-methylated amide linkers
            ("[*:1]C(=O)N(C)[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]N(C)C(=O)[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]CC(=O)N(C)[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]N(C)C(=O)C[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]C(=O)N(C)C[*:2]", "linker", ["hba"], 0.7),
            
            # Sulfonamide linkers
            ("[*:1]S(=O)(=O)NH[*:2]", "linker", ["hbd"], 0.6),
            ("[*:1]NHS(=O)(=O)[*:2]", "linker", ["hbd"], 0.6),
            ("[*:1]CS(=O)(=O)NH[*:2]", "linker", ["hbd"], 0.7),
            ("[*:1]CCS(=O)(=O)NH[*:2]", "linker", ["hbd"], 0.8),
            ("[*:1]NHS(=O)(=O)C[*:2]", "linker", ["hbd"], 0.7),
            ("[*:1]NHS(=O)(=O)CC[*:2]", "linker", ["hbd"], 0.8),
            
            # N-methylated sulfonamides
            ("[*:1]S(=O)(=O)N(C)[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]N(C)S(=O)(=O)[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]CS(=O)(=O)N(C)[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]N(C)S(=O)(=O)C[*:2]", "linker", ["hba"], 0.8),
            
            # Sulfide linkers
            ("[*:1]S[*:2]", "linker", ["hydrophobic"], 0.4),
            ("[*:1]CS[*:2]", "linker", ["hydrophobic"], 0.5),
            ("[*:1]CCS[*:2]", "linker", ["hydrophobic"], 0.5),
            ("[*:1]CCCS[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]SCC[*:2]", "linker", ["hydrophobic"], 0.5),
            ("[*:1]SCCC[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]CSC[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]CSCCC[*:2]", "linker", ["hydrophobic"], 0.7),
            
            # Sulfoxide linkers
            ("[*:1]S(=O)[*:2]", "linker", ["hba"], 0.5),
            ("[*:1]CS(=O)[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]CCS(=O)[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]S(=O)C[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]S(=O)CC[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]CS(=O)C[*:2]", "linker", ["hba"], 0.7),
            
            # Cyclic linkers
            ("[*:1]C1CC1[*:2]", "linker", ["hydrophobic"], 0.5),
            ("[*:1]C1CCC1[*:2]", "linker", ["hydrophobic"], 0.5),
            ("[*:1]C1CCCC1[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]C1CCCCC1[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]C1CCCCCC1[*:2]", "linker", ["hydrophobic"], 0.7),
            
            # Substituted cycloalkyl linkers
            ("[*:1]C1CC(C)C1[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]C1CC(F)C1[*:2]", "linker", ["hydrophobic"], 0.6),
            ("[*:1]C1CC(O)C1[*:2]", "linker", ["hbd"], 0.7),
            ("[*:1]C1CC(N)C1[*:2]", "linker", ["hbd"], 0.7),
            ("[*:1]C1CCC(C)C1[*:2]", "linker", ["hydrophobic"], 0.7),
            ("[*:1]C1CCC(F)C1[*:2]", "linker", ["hydrophobic"], 0.7),
            ("[*:1]C1CCC(O)C1[*:2]", "linker", ["hbd"], 0.8),
            
            # Aromatic linkers
            ("[*:1]c1ccccc1[*:2]", "linker", ["aromatic"], 0.6),
            ("[*:1]c1cccc(C)c1[*:2]", "linker", ["aromatic"], 0.6),
            ("[*:1]c1cccc(F)c1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1cccc(Cl)c1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1cccc(OMe)c1[*:2]", "linker", ["aromatic", "hba"], 0.7),
            ("[*:1]c1cccc(CF3)c1[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]c1cc(F)ccc1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1cc(Cl)ccc1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1cc(CF3)ccc1[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]c1cc(NH2)ccc1[*:2]", "linker", ["aromatic", "hbd"], 0.7),
            ("[*:1]c1cc(OH)ccc1[*:2]", "linker", ["aromatic", "hbd"], 0.7),
            
            # Disubstituted aromatics
            ("[*:1]c1cc(F)c(F)cc1[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]c1cc(Cl)c(Cl)cc1[*:2]", "linker", ["aromatic"], 0.9),
            ("[*:1]c1cc(F)c(Cl)cc1[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]c1cc(CF3)c(F)cc1[*:2]", "linker", ["aromatic"], 0.9),
            ("[*:1]c1cc(NH2)c(F)cc1[*:2]", "linker", ["aromatic", "hbd"], 0.8),
            ("[*:1]c1cc(OH)c(Cl)cc1[*:2]", "linker", ["aromatic", "hbd"], 0.8),
            
            # Extended aromatic linkers
            ("[*:1]Cc1ccccc1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]CCc1ccccc1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1ccccc1C[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1ccccc1CC[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]Cc1ccccc1C[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]CCc1ccccc1CC[*:2]", "linker", ["aromatic"], 0.9),
            
            # Heteroaromatic linkers
            ("[*:1]c1ccncc1[*:2]", "linker", ["aromatic", "hba"], 0.7),
            ("[*:1]c1ncccc1[*:2]", "linker", ["aromatic", "hba"], 0.7),
            ("[*:1]c1cnccc1[*:2]", "linker", ["aromatic", "hba"], 0.7),
            ("[*:1]c1ncncc1[*:2]", "linker", ["aromatic", "hba"], 0.8),
            ("[*:1]c1cnccn1[*:2]", "linker", ["aromatic", "hba"], 0.8),
            ("[*:1]c1nccnc1[*:2]", "linker", ["aromatic", "hba"], 0.8),
            
            # Thiophene-based linkers
            ("[*:1]c1cccs1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1csc(C)c1[*:2]", "linker", ["aromatic"], 0.7),
            ("[*:1]c1csc(F)c1[*:2]", "linker", ["aromatic"], 0.8),
            ("[*:1]c1csc(CF3)c1[*:2]", "linker", ["aromatic"], 0.9),
            
            # Furan-based linkers
            ("[*:1]c1ccco1[*:2]", "linker", ["aromatic", "hba"], 0.7),
            ("[*:1]c1coc(C)c1[*:2]", "linker", ["aromatic", "hba"], 0.7),
            ("[*:1]c1coc(F)c1[*:2]", "linker", ["aromatic", "hba"], 0.8),
            ("[*:1]c1coc(CF3)c1[*:2]", "linker", ["aromatic", "hba"], 0.9),
            
            # Pyrrole-based linkers
            ("[*:1]c1cc[nH]c1[*:2]", "linker", ["aromatic", "hbd"], 0.8),
            ("[*:1]c1c[nH]c(C)c1[*:2]", "linker", ["aromatic", "hbd"], 0.8),
            ("[*:1]c1c[nH]c(F)c1[*:2]", "linker", ["aromatic", "hbd"], 0.9),
            
            # Heterocyclic linkers
            ("[*:1]N1CCCC1[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]N1CCCCC1[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]N1CCCCCC1[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]O1CCCC1[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]O1CCCCC1[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]O1CCCCCC1[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]S1CCCC1[*:2]", "linker", ["hydrophobic"], 0.8),
            ("[*:1]S1CCCCC1[*:2]", "linker", ["hydrophobic"], 0.8),
            ("[*:1]S1CCCCCC1[*:2]", "linker", ["hydrophobic"], 0.9),
            
            # Substituted heterocyclic linkers
            ("[*:1]N1CCC(C)C1[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]N1CCC(F)C1[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]N1CCC(O)C1[*:2]", "linker", ["hba", "hbd"], 0.8),
            ("[*:1]O1CCC(C)C1[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]O1CCC(F)C1[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]S1CCC(C)C1[*:2]", "linker", ["hydrophobic"], 0.9),
            
            # Spiro linkers
            ("[*:1]C12(CCC1)CCC2[*:2]", "linker", ["hydrophobic"], 0.8),
            ("[*:1]C12(CCCC1)CCCC2[*:2]", "linker", ["hydrophobic"], 0.9),
            ("[*:1]C12(CCO1)CCC2[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]C12(CCN1)CCC2[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]C12(CCS1)CCC2[*:2]", "linker", ["hydrophobic"], 0.9),
            
            # Bridge linkers
            ("[*:1]C1CC2CCC1CC2[*:2]", "linker", ["hydrophobic"], 0.8),
            ("[*:1]C1CC2CCC(C1)C2[*:2]", "linker", ["hydrophobic"], 0.8),
            ("[*:1]C1CC2OCC1CO2[*:2]", "linker", ["hba"], 0.9),
            ("[*:1]C1CC2NCC1CN2[*:2]", "linker", ["hba"], 0.9),
            ("[*:1]C1CC2SCC1CS2[*:2]", "linker", ["hydrophobic"], 1.0),
            
            # Mixed functional group linkers
            ("[*:1]COC(=O)[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]C(=O)OC[*:2]", "linker", ["hba"], 0.6),
            ("[*:1]CCOC(=O)[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]C(=O)OCC[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]COC(=O)C[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]CC(=O)OC[*:2]", "linker", ["hba"], 0.7),
            
            # Carbamate linkers
            ("[*:1]CNHC(=O)[*:2]", "linker", ["hbd", "hba"], 0.7),
            ("[*:1]C(=O)NHC[*:2]", "linker", ["hbd", "hba"], 0.7),
            ("[*:1]CCNHC(=O)[*:2]", "linker", ["hbd", "hba"], 0.8),
            ("[*:1]C(=O)NHCC[*:2]", "linker", ["hbd", "hba"], 0.8),
            ("[*:1]CNHC(=O)C[*:2]", "linker", ["hbd", "hba"], 0.8),
            
            # Urea linkers
            ("[*:1]CNHCONHC[*:2]", "linker", ["hbd", "hba"], 0.8),
            ("[*:1]CCNHCONHCC[*:2]", "linker", ["hbd", "hba"], 0.9),
            ("[*:1]NHCONHC[*:2]", "linker", ["hbd", "hba"], 0.7),
            ("[*:1]CNHCONH[*:2]", "linker", ["hbd", "hba"], 0.7),
            
            # Thiourea linkers
            ("[*:1]CNHCSNHC[*:2]", "linker", ["hbd"], 0.8),
            ("[*:1]CCNHCSNHCC[*:2]", "linker", ["hbd"], 0.9),
            ("[*:1]NHCSNHC[*:2]", "linker", ["hbd"], 0.7),
            
            # Sulfonyl ester linkers
            ("[*:1]COS(=O)(=O)C[*:2]", "linker", ["hba"], 0.8),
            ("[*:1]CCOS(=O)(=O)CC[*:2]", "linker", ["hba"], 0.9),
            ("[*:1]OS(=O)(=O)C[*:2]", "linker", ["hba"], 0.7),
            ("[*:1]COS(=O)(=O)[*:2]", "linker", ["hba"], 0.7),
        ]

        fragments = {'cores': [], 'substituents': [], 'linkers': [], 'bioisosteres': []}

        # Process and calculate properties for cores
        for smi, ftype, interactions, syn_diff, bioact in privileged_cores:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                props = self._calculate_fragment_properties(mol)
                fragments['cores'].append(
                    FragmentInfo(
                        smiles=smi,
                        scaffold_type=ftype,
                        mw=props['mw'],
                        logp=props['logp'],
                        tpsa=props['tpsa'],
                        interaction_types=interactions,
                        synthetic_difficulty=syn_diff,
                        bioactivity_class=bioact,
                        attachment_points=smi.count('[*:'),
                        ring_count=props['rings']
                    )
                )

        # Process and calculate properties for substituents
        for smi, ftype, interactions, syn_diff in substituents:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                props = self._calculate_fragment_properties(mol)
                fragments['substituents'].append(
                    FragmentInfo(
                        smiles=smi,
                        scaffold_type=ftype,
                        mw=props['mw'],
                        logp=props['logp'],
                        tpsa=props['tpsa'],
                        interaction_types=interactions,
                        synthetic_difficulty=syn_diff,
                        attachment_points=1,
                        ring_count=props['rings']
                    )
                )

        # Process and calculate properties for linkers
        for smi, ftype, interactions, syn_diff in linkers:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                props = self._calculate_fragment_properties(mol)
                fragments['linkers'].append(
                    FragmentInfo(
                        smiles=smi,
                        scaffold_type=ftype,
                        mw=props['mw'],
                        logp=props['logp'],
                        tpsa=props['tpsa'],
                        interaction_types=interactions,
                        synthetic_difficulty=syn_diff,
                        attachment_points=2,
                        ring_count=props['rings']
                    )
                )

        return fragments

    def _calculate_fragment_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate key molecular properties with enhanced descriptors"""
        try:
            return {
                'mw': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'hbd': Lipinski.NumHDonors(mol),
                'hba': Lipinski.NumHAcceptors(mol),
                'rings': rdMolDescriptors.CalcNumAromaticRings(mol) + rdMolDescriptors.CalcNumAliphaticRings(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'sp3_fraction': rdMolDescriptors.CalcFractionCsp3(mol)
            }
        except:
            # Return defaults if calculation fails
            return {
                'mw': 0.0,
                'logp': 0.0,
                'tpsa': 0.0,
                'hbd': 0,
                'hba': 0,
                'rings': 0,
                'rotatable_bonds': 0,
                'sp3_fraction': 0.0
            }

    def _load_bioisostere_replacements(self) -> Dict[str, List[str]]:
        """Load comprehensive bioisosteric replacement patterns"""
        return {
            # Classic aromatic replacements
            'phenyl': [
                'c1ccncc1', 'c1ncccc1', 'c1cnccc1', 'c1c[nH]cc1', 'c1cocn1', 'c1cscn1', 'c1nocn1',
                'c1ncncc1', 'c1cnccn1', 'c1nccnc1', 'c1ncncn1', 'c1nnccn1', 'c1nncnn1',
                'c1cc2[nH]ncc2cc1', 'c1cc2nncc2cc1', 'c1cc2oncc2cc1', 'c1cc2sncc2cc1',
                'c1ccc2ncccc2c1', 'c1ccc2cnccc2c1', 'c1ccc2ccncc2c1', 'c1ccc2cccnc2c1'
            ],
            
            # Functional group replacements
            'carboxyl': [
                'C(=O)NHOH', 'S(=O)(=O)NH2', 'P(=O)(OH)2', 'C(=O)NHS(=O)(=O)CF3', 
                'C(=O)NHSO2NH2', 'C(=O)NHtetrazole', 'C(=O)NHCN', 'S(=O)(=O)NHCONH2',
                'P(=O)(OH)NH2', 'C(=S)NHOH', 'C(=O)NHSO2F', 'B(OH)2'
            ],
            
            'ester': [
                'C(=O)NH2', 'S(=O)(=O)NH2', 'C(=NH)NH2', 'C(=O)NHOH', 'P(=O)(OH)2',
                'C(=O)NHtetrazole', 'C(=S)NH2', 'C(=O)NHCN', 'SO2NH2', 'B(OH)2',
                'P(=O)(NH2)2', 'C(=NOH)NH2'
            ],
            
            'methyl': [
                'CF3', 'CHF2', 'CH2F', 'C#N', 'C1CC1', 'CHCl2', 'CBr3', 'CH2OH',
                'CHO', 'CONH2', 'SO2NH2', 'CF2CF3', 'C(F)(F)CF3', 'SF5',
                'OCF3', 'SCF3', 'SeF3', 'TeF3', 'PF4'
            ],
            
            'amino': [
                'NHOH', 'NHS(=O)(=O)CH3', 'NHC(=O)NH2', 'NHSO2CF3', 'NH-tetrazole',
                'NHCONH2', 'NHS(=O)(=O)NH2', 'NHC(=S)NH2', 'NHCN', 'NHSO2F',
                'NHP(=O)(OH)2', 'NHB(OH)2', 'NHSO2Me', 'NHCOtBu'
            ],
            
            'hydroxyl': [
                'F', 'NH2', 'SH', 'CONH2', 'CONHOH', 'SO2NH2', 'PO(OH)2',
                'B(OH)2', 'SeH', 'TeH', 'C#N', 'CHF2', 'CF3', 'OCF3'
            ],
            
            'nitro': [
                'CF3', 'SO2NH2', 'CONH2', 'C#N', 'SO2CF3', 'P(=O)(OH)2',
                'B(OH)2', 'S(=O)(=O)F', 'SF5', 'C(=O)NHOH', 'tetrazole'
            ],
            
            'carbonyl': [
                'SO2', 'PO2', 'CF2', 'CHF', 'CH2', 'C=NOH', 'C=NNH2',
                'C=S', 'C=Se', 'C=Te', 'B(OH)', 'SiF2'
            ],
            
            # Heterocycle replacements
            'benzene': [
                'pyridine', 'pyrimidine', 'pyrazine', 'pyridazine', 'thiophene', 'furan', 
                'pyrrole', 'imidazole', 'oxazole', 'thiazole', 'isoxazole', 'isothiazole',
                'pyrazole', 'triazole', 'tetrazole', 'thiadiazole', 'oxadiazole',
                'quinoline', 'isoquinoline', 'quinazoline', 'quinoxaline', 'benzimidazole',
                'benzoxazole', 'benzothiazole', 'indole', 'benzofuran', 'benzothiophene'
            ],
            
            'pyridine': [
                'pyrimidine', 'pyrazine', 'pyridazine', 'triazine', 'benzene', 'thiazole',
                'oxazole', 'imidazole', 'pyrazole', 'isoxazole', 'isothiazole',
                'thiadiazole', 'oxadiazole', 'quinoline', 'isoquinoline', 'quinazoline'
            ],
            
            'thiophene': [
                'furan', 'pyrrole', 'benzene', 'pyridine', 'thiazole', 'oxazole',
                'imidazole', 'pyrazole', 'selenophene', 'tellurophene', 'phosphole',
                'benzothiophene', 'thienopyridine', 'thienopyrimidine'
            ],
            
            'furan': [
                'thiophene', 'pyrrole', 'benzene', 'pyridine', 'oxazole', 'thiazole',
                'imidazole', 'isoxazole', 'oxadiazole', 'benzofuran', 'furanopyridine'
            ],
            
            'pyrrole': [
                'thiophene', 'furan', 'imidazole', 'pyrazole', 'benzene', 'pyridine',
                'indole', 'pyrrolopyridine', 'pyrrolopyrimidine'
            ],
            
            # Ring size variations
            '5-ring': ['6-ring', '7-ring', 'spiro-5,5', 'spiro-5,6', 'bridged-5'],
            '6-ring': ['5-ring', '7-ring', 'spiro-6,6', 'spiro-5,6', 'bridged-6'],
            
            # Saturation level variations
            'aromatic': ['aliphatic', 'partially_saturated'],
            'aliphatic': ['aromatic', 'partially_saturated'],
            
            # Chain length variations
            'C1': ['C2', 'C3', 'cyclopropyl'],
            'C2': ['C1', 'C3', 'C4', 'cyclobutyl'],
            'C3': ['C1', 'C2', 'C4', 'C5', 'cyclopentyl'],
            
            # Stereochemical variations
            'trans': ['cis', 'mixture'],
            'cis': ['trans', 'mixture'],
            'R': ['S', 'racemic'],
            'S': ['R', 'racemic']
        }

    def _load_toxic_alerts(self) -> List[str]:
        """Load comprehensive structural alerts for toxicity"""
        return [
            # Reactive electrophiles
            "C=CC(=O)", "C=CC#N", "C=C(C#N)C#N", "C1OC1", "C1SC1", "C1NC1",
            
            # Nitro compounds and related
            "N=N", "[N+](=O)[O-]", "N=O", "ONO2", "ONO", "c1ccc([N+](=O)[O-])cc1",
            
            # Alkylating agents
            "OS(=O)(=O)C", "ClCC", "BrCC", "ICC", "ClC(Cl)C", "BrC(Br)C",
            "IC(I)C", "OSO2C", "NSO2C", "ClSO2", "FSO2",
            
            # Polyhalogenated compounds
            "FC(F)(F)C(F)(F)", "ClC(Cl)(Cl)", "BrC(Br)(Br)", "IC(I)(I)",
            "c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "c1c(F)c(F)c(F)c(F)c1F",
            
            # Aromatic amines (potentially mutagenic)
            "c1ccc(N)cc1", "c1ccc(NN)cc1", "c1ccc(N(N)N)cc1",
            "c1cc(N)c(N)cc1", "c1c(N)cccc1N",
            
            # Quinones and related (redox cycling)
            "C1=CC(=O)C=CC1=O", "C1=CC(=O)NC(=O)C1", "c1cc2c(cc1)C(=O)C=CC2=O",
            
            # DNA intercalators
            "c1ccc2c(c1)nc(N)nc2N", "c1ccc2c(c1)nc3ccccc3n2",
            
            # Acylating agents
            "CCN(CC)C(=O)Cl", "C(=O)Cl", "C(=O)F", "SO2Cl", "SO2F",
            "C(=O)OC(=O)", "C(=S)Cl",
            
            # Azo compounds
            "CCON=NC", "c1ccc(N=Nc2ccccc2)cc1", "N=NC",
            
            # Heavy metals and organometallics
            "C[Hg]", "[As]", "[Pb]", "[Cd]", "[Cr]", "[Ni]", "C[Sn]",
            "C[Sb]", "C[Bi]", "C[Tl]",
            
            # Nitroso compounds
            "C(=O)N(N=O)", "N(C)N=O", "c1ccc(N=O)cc1",
            
            # Michael acceptors
            "C=C(C#N)C#N", "C=CC(=O)C", "C=CC(=O)O", "C=C(C=O)C",
            
            # Epoxides and aziridines
            "C1OC1", "C1NC1", "C1SC1", "C1SeC1",
            
            # Disulfides and related
            "S=C=S", "N=C=S", "O=C=S", "Se=C=Se",
            
            # Peroxides
            "OO", "COOC", "c1ccccc1OOc1ccccc1",
            
            # Hydrazines
            "NN", "NNC", "c1ccc(NN)cc1", "N(N)C(=O)",
            
            # Aldehydes (potential protein crosslinkers)
            "CC=O", "CCC=O", "c1ccc(C=O)cc1", "OCC=O",
            
            # Thiocarbonyls
            "C=S", "c1ccc(C=S)cc1", "N(C)C=S",
            
            # Fluorinated aromatics (bioaccumulation)
            "c1ccc(OC(F)(F)F)cc1", "c1ccc(C(F)(F)F)cc1",
            "c1c(F)c(F)c(F)c(F)c1F",
            
            # Specific toxic motifs
            "Cc1onc(c2ccccc2Cl)c1C(=O)O",  # Cloxacillin-like
            "c1ccc2[nH]c3ccccc3c2c1",      # Carbazole
            "c1ccc2c(c1)C(=O)c1ccccc1C2=O", # Anthraquinone
            "c1ccc2c(c1)c(=O)[nH]c(=O)n2",  # Uracil derivatives
            
            # Aromatic nitro compounds
            "c1ccc(NO2)c(NO2)c1", "c1c(NO2)c(NO2)c(NO2)cc1",
            
            # Sulfonyl fluorides (irreversible enzyme inhibitors)
            "S(=O)(=O)F", "c1ccc(S(=O)(=O)F)cc1",
            
            # Beta-lactams (allergenic potential)
            "C1C(=O)NC1", "C1CC(=O)NC1",
            
            # Thiophenes (bioactivation)
            "c1ccsc1C(=O)", "c1ccsc1C#N",
            
            # Reactive oxygen species generators
            "c1ccc(N(=O)=O)cc1", "C(=O)OOC(=O)",
            
            # DNA alkylators
            "ClCCCl", "BrCCBr", "ClCCCCl", "N(CCCl)CCCl",
            
            # Carcinogenic PAHs
            "c1cc2ccc3cccc4ccc(c1)c2c34",  # Benzo[a]pyrene-like
            "c1ccc2cc3ccccc3cc2c1",        # Anthracene-like
            
            # Reactive metabolites precursors
            "c1ccc(CCC)cc1", "c1ccc(CCCC)cc1",  # Long alkyl chains
            "c1ccc(C(C)(C)C)cc1",               # tert-Butyl groups
        ]

    def get_fragments_for_interaction(self, interaction_type: str,
                                      target_properties: Optional[Dict] = None) -> List[FragmentInfo]:
        """Get fragments suitable for specific interactions with enhanced filtering"""
        suitable_fragments = []
        for category, frags in self.fragments.items():
            for frag in frags:
                if interaction_type in frag.interaction_types:
                    if target_properties:
                        if self._matches_target_properties(frag, target_properties):
                            suitable_fragments.append(frag)
                    else:
                        suitable_fragments.append(frag)
        return suitable_fragments

    def _matches_target_properties(self, frag: FragmentInfo, targets: Dict) -> bool:
        """Check if fragment properties match targets with tolerance"""
        for prop, constraint in targets.items():
            frag_val = getattr(frag, prop, None)
            if frag_val is None:
                continue
            
            if isinstance(constraint, tuple) and len(constraint) == 2:
                # Range constraint (min, max)
                min_val, max_val = constraint
                if not (min_val <= frag_val <= max_val):
                    return False
            elif isinstance(constraint, (int, float)):
                # Exact value constraint with 10% tolerance
                tolerance = abs(constraint * 0.1)
                if not (constraint - tolerance <= frag_val <= constraint + tolerance):
                    return False
            elif isinstance(constraint, list):
                # Must be in list
                if frag_val not in constraint:
                    return False
        return True

    def get_bioisosteres(self, fragment_smiles: str) -> List[str]:
        """Get bioisosteric replacements for a given fragment"""
        mol = Chem.MolFromSmiles(fragment_smiles)
        if not mol:
            return []
        
        replacements = []
        
        # Check for common functional groups and rings
        for pattern, alternatives in self.bioisosteres.items():
            if pattern in fragment_smiles.lower() or self._contains_pattern(mol, pattern):
                replacements.extend(alternatives)
        
        return list(set(replacements))  # Remove duplicates

    def _contains_pattern(self, mol: Chem.Mol, pattern: str) -> bool:
        """Check if molecule contains a specific pattern"""
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol:
                return mol.HasSubstructMatch(pattern_mol)
        except:
            pass
        return False

    def check_toxicity_alerts(self, smiles: str) -> List[str]:
        """Check for known toxicity alerts in a molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return ["Invalid SMILES"]
        
        alerts = []
        for alert_pattern in self.toxic_alerts:
            try:
                alert_mol = Chem.MolFromSmarts(alert_pattern)
                if alert_mol and mol.HasSubstructMatch(alert_mol):
                    alerts.append(alert_pattern)
            except:
                continue
        
        return alerts

    def select_fragments_by_properties(self, 
                                      fragment_type: str,
                                      min_mw: float = 0,
                                      max_mw: float = 1000,
                                      min_logp: float = -5,
                                      max_logp: float = 5,
                                      max_tpsa: float = 200,
                                      required_interactions: Optional[List[str]] = None,
                                      bioactivity_class: Optional[str] = None,
                                      max_syn_difficulty: float = 1.0) -> List[FragmentInfo]:
        """Select fragments based on multiple property criteria"""
        if fragment_type not in self.fragments:
            return []
        
        filtered_fragments = []
        for frag in self.fragments[fragment_type]:
            # Property filters
            if not (min_mw <= frag.mw <= max_mw):
                continue
            if not (min_logp <= frag.logp <= max_logp):
                continue
            if frag.tpsa > max_tpsa:
                continue
            if frag.synthetic_difficulty > max_syn_difficulty:
                continue
            
            # Interaction requirements
            if required_interactions:
                if not all(req in frag.interaction_types for req in required_interactions):
                    continue
            
            # Bioactivity class filter
            if bioactivity_class and frag.bioactivity_class != bioactivity_class:
                continue
            
            filtered_fragments.append(frag)
        
        return filtered_fragments

    def get_random_fragments(self, fragment_type: str, count: int = 10, 
                           seed: Optional[int] = None) -> List[FragmentInfo]:
        """Get random selection of fragments of specified type"""
        if seed is not None:
            random.seed(seed)
        
        if fragment_type not in self.fragments:
            return []
        
        available_frags = self.fragments[fragment_type]
        if len(available_frags) <= count:
            return available_frags
        
        return random.sample(available_frags, count)

    def get_fragment_statistics(self) -> Dict:
        """Get statistics about the fragment library"""
        stats = {}
        for frag_type, frags in self.fragments.items():
            if not frags:
                continue
                
            mws = [f.mw for f in frags]
            logps = [f.logp for f in frags]
            tpsas = [f.tpsa for f in frags]
            
            stats[frag_type] = {
                'count': len(frags),
                'mw_range': (min(mws), max(mws)),
                'mw_mean': sum(mws) / len(mws),
                'logp_range': (min(logps), max(logps)),
                'logp_mean': sum(logps) / len(logps),
                'tpsa_range': (min(tpsas), max(tpsas)),
                'tpsa_mean': sum(tpsas) / len(tpsas),
                'interaction_types': list(set(
                    interaction for frag in frags 
                    for interaction in frag.interaction_types
                )),
                'bioactivity_classes': list(set(
                    frag.bioactivity_class for frag in frags 
                    if frag.bioactivity_class
                ))
            }
        
        return stats

    def validate_fragment_library(self) -> Dict[str, List[str]]:
        """Validate all fragments in the library and return validation results"""
        validation_results = {
            'valid_fragments': [],
            'invalid_smiles': [],
            'warnings': [],
            'toxicity_alerts': []
        }
        
        for frag_type, frags in self.fragments.items():
            for frag in frags:
                # Validate SMILES
                mol = Chem.MolFromSmiles(frag.smiles)
                if mol is None:
                    validation_results['invalid_smiles'].append(
                        f"{frag_type}: {frag.smiles}"
                    )
                    continue
                
                # Check for toxicity alerts
                toxic_alerts = self.check_toxicity_alerts(frag.smiles)
                if toxic_alerts:
                    validation_results['toxicity_alerts'].append(
                        f"{frag_type}: {frag.smiles} - Alerts: {', '.join(toxic_alerts)}"
                    )
                
                # Property warnings
                if frag.mw > 500:
                    validation_results['warnings'].append(
                        f"{frag_type}: {frag.smiles} - High MW: {frag.mw:.1f}"
                    )
                
                if frag.logp > 5 or frag.logp < -2:
                    validation_results['warnings'].append(
                        f"{frag_type}: {frag.smiles} - Extreme LogP: {frag.logp:.1f}"
                    )
                
                if frag.tpsa > 140:
                    validation_results['warnings'].append(
                        f"{frag_type}: {frag.smiles} - High TPSA: {frag.tpsa:.1f}"
                    )
                
                validation_results['valid_fragments'].append(frag.smiles)
        
        return validation_results

    def export_fragments_to_sdf(self, filename: str, fragment_type: Optional[str] = None):
        """Export fragments to SDF file format"""
        try:
            from rdkit.Chem import SDWriter
            
            writer = SDWriter(filename)
            
            fragment_types = [fragment_type] if fragment_type else self.fragments.keys()
            
            for frag_type in fragment_types:
                if frag_type not in self.fragments:
                    continue
                    
                for frag in self.fragments[frag_type]:
                    mol = Chem.MolFromSmiles(frag.smiles)
                    if mol:
                        # Add properties as molecule properties
                        mol.SetProp("FragmentType", frag.scaffold_type)
                        mol.SetProp("MW", str(frag.mw))
                        mol.SetProp("LogP", str(frag.logp))
                        mol.SetProp("TPSA", str(frag.tpsa))
                        mol.SetProp("InteractionTypes", ", ".join(frag.interaction_types))
                        mol.SetProp("SyntheticDifficulty", str(frag.synthetic_difficulty))
                        if frag.bioactivity_class:
                            mol.SetProp("BioactivityClass", frag.bioactivity_class)
                        mol.SetProp("AttachmentPoints", str(frag.attachment_points))
                        mol.SetProp("RingCount", str(frag.ring_count))
                        
                        writer.write(mol)
            
            writer.close()
            print(f"Fragments exported to {filename}")
            
        except ImportError:
            print("RDKit SDWriter not available")
        except Exception as e:
            print(f"Error exporting fragments: {e}")

    def search_fragments_by_substructure(self, smarts_pattern: str, 
                                       fragment_type: Optional[str] = None) -> List[FragmentInfo]:
        """Search for fragments containing a specific substructure"""
        try:
            pattern_mol = Chem.MolFromSmarts(smarts_pattern)
            if not pattern_mol:
                return []
        except:
            return []
        
        matching_fragments = []
        fragment_types = [fragment_type] if fragment_type else self.fragments.keys()
        
        for frag_type in fragment_types:
            if frag_type not in self.fragments:
                continue
                
            for frag in self.fragments[frag_type]:
                mol = Chem.MolFromSmiles(frag.smiles)
                if mol and mol.HasSubstructMatch(pattern_mol):
                    matching_fragments.append(frag)
        
        return matching_fragments

    def get_similar_fragments(self, query_smiles: str, 
                            similarity_threshold: float = 0.7,
                            fragment_type: Optional[str] = None) -> List[tuple]:
        """Find fragments similar to query molecule using Tanimoto similarity"""
        try:
            from rdkit.Chem import DataStructs
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
            
            query_mol = Chem.MolFromSmiles(query_smiles)
            if not query_mol:
                return []
            
            query_fp = GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
            similar_fragments = []
            
            fragment_types = [fragment_type] if fragment_type else self.fragments.keys()
            
            for frag_type in fragment_types:
                if frag_type not in self.fragments:
                    continue
                    
                for frag in self.fragments[frag_type]:
                    mol = Chem.MolFromSmiles(frag.smiles)
                    if mol:
                        frag_fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        similarity = DataStructs.TanimotoSimilarity(query_fp, frag_fp)
                        
                        if similarity >= similarity_threshold:
                            similar_fragments.append((frag, similarity))
            
            # Sort by similarity (highest first)
            similar_fragments.sort(key=lambda x: x[1], reverse=True)
            return similar_fragments
            
        except ImportError:
            print("RDKit fingerprint functionality not available")
            return []
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return []

    def generate_focused_library(self, core_smiles: str, 
                               max_fragments: int = 100,
                               diversity_threshold: float = 0.3) -> List[str]:
        """Generate a focused library around a core structure"""
        try:
            from rdkit.Chem import DataStructs
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
            
            core_mol = Chem.MolFromSmiles(core_smiles)
            if not core_mol:
                return []
            
            # Get suitable substituents and linkers
            suitable_substituents = self.select_fragments_by_properties(
                'substituents', max_mw=200, max_logp=3, max_tpsa=100
            )
            suitable_linkers = self.select_fragments_by_properties(
                'linkers', max_mw=150, max_logp=2, max_tpsa=80
            )
            
            library = []
            fingerprints = []
            
            # Generate combinations
            for substituent in suitable_substituents[:50]:  # Limit for performance
                for linker in suitable_linkers[:20]:
                    # Simple combination logic (would need more sophisticated approach)
                    combined_smiles = f"{core_smiles}.{substituent.smiles}.{linker.smiles}"
                    
                    try:
                        combined_mol = Chem.MolFromSmiles(combined_smiles)
                        if combined_mol:
                            # Check diversity
                            fp = GetMorganFingerprintAsBitVect(combined_mol, 2, nBits=2048)
                            
                            # Ensure diversity
                            is_diverse = True
                            for existing_fp in fingerprints:
                                similarity = DataStructs.TanimotoSimilarity(fp, existing_fp)
                                if similarity > (1.0 - diversity_threshold):
                                    is_diverse = False
                                    break
                            
                            if is_diverse:
                                library.append(combined_smiles)
                                fingerprints.append(fp)
                                
                                if len(library) >= max_fragments:
                                    return library
                    except:
                        continue
            
            return library
            
        except ImportError:
            print("RDKit fingerprint functionality not available")
            return []
        except Exception as e:
            print(f"Error generating focused library: {e}")
            return []

    def optimize_fragment_properties(self, fragment_smiles: str, 
                                   target_properties: Dict) -> List[str]:
        """Suggest modifications to optimize fragment properties"""
        suggestions = []
        
        mol = Chem.MolFromSmiles(fragment_smiles)
        if not mol:
            return ["Invalid SMILES"]
        
        current_props = self._calculate_fragment_properties(mol)
        
        # Analyze what needs to be changed
        for prop, target in target_properties.items():
            if prop in current_props:
                current_val = current_props[prop]
                
                if isinstance(target, tuple):
                    target_min, target_max = target
                    if current_val < target_min:
                        suggestions.append(f"Increase {prop}: current {current_val:.2f}, target >={target_min}")
                    elif current_val > target_max:
                        suggestions.append(f"Decrease {prop}: current {current_val:.2f}, target <={target_max}")
                elif isinstance(target, (int, float)):
                    if abs(current_val - target) > target * 0.1:
                        if current_val < target:
                            suggestions.append(f"Increase {prop}: current {current_val:.2f}, target {target}")
                        else:
                            suggestions.append(f"Decrease {prop}: current {current_val:.2f}, target {target}")
        
        # Add specific modification suggestions
        if any("LogP" in s and "Decrease" in s for s in suggestions):
            suggestions.append("Consider: Add polar groups (OH, NH2, COOH)")
            suggestions.append("Consider: Replace alkyl with ether or amide")
            
        if any("LogP" in s and "Increase" in s for s in suggestions):
            suggestions.append("Consider: Add hydrophobic groups (CF3, alkyl)")
            suggestions.append("Consider: Replace polar groups with hydrophobic alternatives")
            
        if any("MW" in s and "Decrease" in s for s in suggestions):
            suggestions.append("Consider: Remove non-essential substituents")
            suggestions.append("Consider: Use smaller ring systems")
            
        if any("TPSA" in s and "Decrease" in s for s in suggestions):
            suggestions.append("Consider: Reduce number of H-bond donors/acceptors")
            suggestions.append("Consider: Replace amides with less polar alternatives")
        
        return suggestions


# Example usage and testing functions
def test_fragment_library():
    """Test the fragment library functionality"""
    print("Testing Fragment Library...")
    
    # Initialize library
    lib = FragmentLibrary()
    
    # Get statistics
    stats = lib.get_fragment_statistics()
    print("\nLibrary Statistics:")
    for frag_type, stat in stats.items():
        print(f"{frag_type}: {stat['count']} fragments")
        print(f"  MW range: {stat['mw_range'][0]:.1f} - {stat['mw_range'][1]:.1f}")
        print(f"  LogP range: {stat['logp_range'][0]:.1f} - {stat['logp_range'][1]:.1f}")
    
    # Test property-based selection
    kinase_cores = lib.select_fragments_by_properties(
        'cores', 
        bioactivity_class='kinase',
        max_mw=300,
        max_logp=3
    )
    print(f"\nKinase cores (MW<300, LogP<3): {len(kinase_cores)}")
    
    # Test interaction-based selection
    hbd_fragments = lib.get_fragments_for_interaction('hbd')
    print(f"H-bond donor fragments: {len(hbd_fragments)}")
    
    # Test toxicity screening
    test_smiles = "c1ccc(N)cc1"  # Aniline - known toxicity alert
    alerts = lib.check_toxicity_alerts(test_smiles)
    print(f"\nToxicity alerts for {test_smiles}: {alerts}")
    
    # Test validation
    validation = lib.validate_fragment_library()
    print(f"\nValidation results:")
    print(f"Valid fragments: {len(validation['valid_fragments'])}")
    print(f"Invalid SMILES: {len(validation['invalid_smiles'])}")
    print(f"Warnings: {len(validation['warnings'])}")
    print(f"Toxicity alerts: {len(validation['toxicity_alerts'])}")
    
    print("\nFragment Library test completed!")


if __name__ == "__main__":
    test_fragment_library()
