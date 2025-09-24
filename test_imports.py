# test_imports.py
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Testing basic imports...")
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    print("✓ Basic dependencies work")
    
    print("Testing config...")
    from config import LigandForgeConfig
    print("✓ Config imported")
    
    print("Testing data structures...")
    from data_structures import PocketVoxel, InteractionHotspot
    print("✓ Data structures imported")
    
    print("Testing PDB parser...")
    from pdb_parser import PDBParser, generate_sample_pdb
    print("✓ PDB parser imported")
    
    print("Testing fragment library...")
    from fragment_library7 import FragmentLibrary
    print("✓ Fragment library imported")
    
    print("All imports successful!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Check that all files are in the correct location")

except Exception as e:
    print(f"Other error: {e}")
