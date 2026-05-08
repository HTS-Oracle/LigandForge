LigandForge Technical Deployment and Installation Guide
This guide provides the technical specifications and deployment procedures for LigandForge, a structure-guided de novo ligand generation platform. As a bioinformatics software engineer or DevOps specialist, use this document to ensure binary compatibility, resource optimization, and stable orchestration across local, cloud, or high-performance computing (HPC) environments.

1. Project Overview and Repository Access
LigandForge is a modular platform that integrates structural validation, binding-site characterization, voxel-based property grid construction, and chemistry-aware fragment assembly. It is architected to bridge the gap between macromolecular structural data and experimentally feasible leads.
GitHub Repository: https://github.com/HTS-Oracle/LigandForge
Live Proof-of-Concept: https://ligandforge.onrender.com
Software Stack: Python (Core logic), Streamlit (Frontend/State Management), and RDKit (Cheminformatics Engine).

2. Software Prerequisites
The application requires Python 3.10+ and strict adherence to the following dependency versions to ensure binary compatibility with RDKit:
Cheminformatics: rdkit==2025.3.6
Web Framework: streamlit>=1.49.0
Scientific Stack:
numpy>=1.24.0
scipy>=1.15.0
pandas>=2.3.0
scikit-learn>=1.7.0 (Required for DBSCAN-based diversity clustering)
Visualization & Utilities:
plotly>=6.0.0
matplotlib>=3.10.0
pillow>=11.0.0
graphviz (For retrosynthetic tree rendering)
psutil>=7.0.0 (For real-time resource telemetry)

3. Environment Setup and Installation
   
3.1 Local Environment Initialisation
Execute the following commands to initialize the environment and resolve dependencies:
# Clone the repository
git clone https://github.com/HTS-Oracle/LigandForge.git
cd LigandForge

# Establish a virtual environment for dependency isolation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt


3.2 Component Verification
Before orchestrating a full deployment, use the test_imports.py script to verify the discoverability of local modules and the integrity of C-extensions (RDKit/NumPy):
python test_imports.py
Note for Admins: test_imports.py explicitly appends the current working directory to sys.path. In Docker or HPC environments, ensure that the application root is included in PYTHONPATH to prevent ModuleNotFoundError during headless execution.

3.3 Optional Module Integration
Retrosynthesis: The retrosynthesis_module is loaded conditionally. If the RDKit Fragments module or internal retrosynthetic logic fails to load, the UI will degrade gracefully, disabling the synthetic feasibility tab.
Performance Monitoring: The application checks the PSUTIL_AVAILABLE flag. If psutil is missing, real-time memory telemetry is disabled, but the core generation pipeline remains functional.


5. LocalExecution
Launch the web-based interface using the Streamlit CLI:
streamlit run app.py
