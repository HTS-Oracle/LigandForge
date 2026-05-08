LigandForge Technical Deployment and Installation Guide
This guide provides the technical specifications and deployment procedures for LigandForge.

1. Project Overview and Repository Access

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

# Clone the repository
git clone https://github.com/HTS-Oracle/LigandForge.git
cd LigandForge

# Establish a virtual environment for dependency isolation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt


3.2 Component Verification
python test_imports.py


4. LocalExecution
Launch the web-based interface using the Streamlit CLI:
streamlit run app.py
