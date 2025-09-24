"""
Updated Streamlit Application
Main Streamlit interface using the modular LigandForge backend
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# LigandForge imports
from pipeline import LigandForgePipeline
from config import LigandForgeConfig, ConfigPresets
from visualization import (
    create_score_distribution_plot, create_property_space_plot,
    create_convergence_plot, create_radar_plot, create_diversity_analysis_plot,
    create_property_correlation_matrix, create_molecular_weight_distribution,
    create_lipinski_compliance_plot, generate_summary_statistics,
    create_sdf_from_molecules
)
from pdb_parser import generate_sample_pdb


def create_streamlit_app():
    """Main Streamlit application with modular backend"""
    
    # Page configuration
    st.set_page_config(
        page_title="LigandForge 2.0",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧬 LigandForge 2.0</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Advanced AI-Driven Structure-Based Drug Design Platform**
    
    Combining structure-based drug design, AI-guided fragment assembly, 
    and multi-objective optimization for intelligent molecular design.
    """)
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'config' not in st.session_state:
        st.session_state.config = LigandForgeConfig()
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        main_interface()
    
    with col2:
        sidebar_info()


def setup_sidebar():
    """Setup sidebar configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    # Preset configurations
    with st.sidebar.expander("🎯 Preset Configurations"):
        preset_type = st.selectbox(
            "Choose preset configuration:",
            ["Custom", "Kinase Inhibitors", "GPCR Ligands", "Fragment-Based", 
             "Lead Optimization", "Fast Screening", "High Quality"]
        )
        
        if st.button("Apply Preset"):
            if preset_type != "Custom":
                config_map = {
                    "Kinase Inhibitors": ConfigPresets.kinase_focused(),
                    "GPCR Ligands": ConfigPresets.gpcr_focused(),
                    "Fragment-Based": ConfigPresets.fragment_based(),
                    "Lead Optimization": ConfigPresets.lead_optimization(),
                    "Fast Screening": ConfigPresets.fast_screening(),
                    "High Quality": ConfigPresets.high_quality()
                }
                st.session_state.config = config_map[preset_type]
                st.success(f"Applied {preset_type} preset configuration!")
    
    # Generation parameters
    with st.sidebar.expander("🔧 Generation Parameters"):
        st.session_state.config.max_heavy_atoms = st.slider(
            "Max Heavy Atoms", 15, 50, st.session_state.config.max_heavy_atoms
        )
        st.session_state.config.min_heavy_atoms = st.slider(
            "Min Heavy Atoms", 10, 25, st.session_state.config.min_heavy_atoms
        )
        st.session_state.config.max_rings = st.slider(
            "Max Rings", 1, 8, st.session_state.config.max_rings
        )
        st.session_state.config.diversity_threshold = st.slider(
            "Diversity Threshold", 0.3, 0.9, st.session_state.config.diversity_threshold
        )
    
    # Scoring weights
    with st.sidebar.expander("🎯 Scoring Weights"):
        total_weight = 0.0
        weights = {}
        
        for component in st.session_state.config.reward_weights.keys():
            weights[component] = st.slider(
                component.replace('_', ' ').title(),
                0.0, 1.0, st.session_state.config.reward_weights[component],
                key=f"weight_{component}"
            )
            total_weight += weights[component]
        
        if st.button("Normalize Weights"):
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in weights.items()}
                st.session_state.config.reward_weights = normalized_weights
                st.success("Weights normalized!")
    
    # Optimization parameters
    with st.sidebar.expander("🧠 Optimization Parameters"):
        opt_method = st.selectbox(
            "Optimization Method",
            ["Hybrid (RL → GA)", "Genetic Algorithm", "Reinforcement Learning"]
        )
        
        # GA parameters
        st.subheader("Genetic Algorithm")
        ga_population = st.number_input("Population Size", 50, 500, 150, step=10)
        ga_generations = st.number_input("Generations", 10, 100, 30, step=5)
        ga_crossover = st.slider("Crossover Rate", 0.0, 1.0, 0.7)
        ga_mutation = st.slider("Mutation Rate", 0.0, 1.0, 0.3)
        ga_elitism = st.number_input("Elitism", 0, 50, 10, step=1)
        
        # RL parameters
        st.subheader("Reinforcement Learning")
        rl_iterations = st.number_input("RL Iterations", 5, 50, 20, step=1)
        st.session_state.config.exploration_factor = st.slider(
            "Exploration Factor", 0.0, 0.5, st.session_state.config.exploration_factor
        )
        
        return {
            'opt_method': opt_method,
            'ga_population': ga_population,
            'ga_generations': ga_generations,
            'ga_crossover': ga_crossover,
            'ga_mutation': ga_mutation,
            'ga_elitism': ga_elitism,
            'rl_iterations': rl_iterations
        }


def main_interface():
    """Main interface components"""
    st.header("📂 Input Structure")
    
    # Structure input
    input_method = st.radio(
        "Input Method:", 
        ["Upload PDB File", "PDB ID", "Sample Structure"],
        horizontal=True
    )
    
    pdb_text = get_pdb_input(input_method)
    
    if not pdb_text:
        st.info("Please provide a PDB structure to proceed.")
        return
    
    # Binding site selection
    st.header("🎯 Binding Site Selection")
    center, radius = setup_binding_site_selection(pdb_text)
    
    if center is None:
        st.warning("Please specify a binding site center.")
        return
    
    # Target interactions
    st.header("🔗 Target Interactions")
    target_interactions = st.multiselect(
        "Select target interaction types:",
        ["hbd", "hba", "hydrophobic", "aromatic", "electrostatic"],
        default=["hbd", "hba", "hydrophobic"],
        help="Choose the types of interactions you want to target"
    )
    
    if not target_interactions:
        st.warning("Please select at least one target interaction type.")
        return
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        n_initial = st.number_input("Initial Population Size", 50, 1000, 200, step=10)
        include_voxel = st.checkbox("Include Voxel Analysis", value=False)
        binding_radius = st.number_input("Binding Site Radius (Å)", 5.0, 20.0, radius, step=0.5)
    
    # Run pipeline
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        run_button = st.button("🚀 Run LigandForge", type="primary", use_container_width=True)
    
    if run_button:
        run_pipeline(pdb_text, center, target_interactions, binding_radius, n_initial, include_voxel)
    
    # Display results
    if st.session_state.results:
        display_results()


def get_pdb_input(input_method: str) -> Optional[str]:
    """Get PDB input based on selected method"""
    pdb_text = None
    
    if input_method == "Upload PDB File":
        uploaded_file = st.file_uploader("Upload PDB file", type=['pdb'])
        if uploaded_file:
            pdb_text = uploaded_file.read().decode('utf-8')
            st.success(f"Loaded PDB file: {uploaded_file.name}")
    
    elif input_method == "PDB ID":
        col1, col2 = st.columns([2, 1])
        with col1:
            pdb_id = st.text_input("Enter PDB ID (e.g., 1abc):")
        with col2:
            fetch_button = st.button("Fetch PDB")
        
        if fetch_button and pdb_id:
            with st.spinner("Fetching PDB structure..."):
                try:
                    from pdb_parser import download_pdb
                    pdb_text = download_pdb(pdb_id)
                    st.success(f"Successfully fetched PDB {pdb_id.upper()}")
                except Exception as e:
                    st.error(f"Failed to fetch PDB: {e}")
    
    elif input_method == "Sample Structure":
        if st.button("Load Sample Structure"):
            pdb_text = generate_sample_pdb()
            st.success("Loaded sample kinase structure")
    
    return pdb_text


def setup_binding_site_selection(pdb_text: str) -> tuple:
    """Setup binding site selection interface"""
    
    # Parse structure to find ligands
    try:
        from pdb_parser import PDBParser
        parser = PDBParser()
        structure = parser.parse_pdb_structure(pdb_text)
        
        ligand_names = list(structure.ligands.keys()) if structure.ligands else []
        ligand_centers = {}
        
        if ligand_names:
            for name, atoms in structure.ligands.items():
                if atoms:
                    positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
                    ligand_centers[name] = positions.mean(axis=0)
    
    except Exception as e:
        st.error(f"Error parsing structure: {e}")
        return None, 10.0
    
    # Binding site determination method
    method = st.radio(
        "Binding site determination:",
        ["Manual Coordinates", "Co-crystallized Ligand"],
        horizontal=True
    )
    
    center = None
    radius = 10.0
    
    if method == "Co-crystallized Ligand":
        if not ligand_names:
            st.warning("No ligands found in structure. Please use manual coordinates.")
        else:
            selected_ligand = st.selectbox("Select ligand:", ligand_names)
            if selected_ligand in ligand_centers:
                center = ligand_centers[selected_ligand]
                st.info(f"Center from {selected_ligand}: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] Å")
    
    else:  # Manual coordinates
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.number_input("Center X (Å)", value=0.0, format="%.2f")
        with col2:
            y = st.number_input("Center Y (Å)", value=0.0, format="%.2f")
        with col3:
            z = st.number_input("Center Z (Å)", value=0.0, format="%.2f")
        
        center = np.array([x, y, z])
    
    radius = st.slider("Binding site radius (Å)", 5.0, 20.0, 10.0, 0.5)
    
    return center, radius


def run_pipeline(pdb_text: str, center: np.ndarray, target_interactions: List[str], 
                radius: float, n_initial: int, include_voxel: bool):
    """Run the LigandForge pipeline"""
    
    # Initialize pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Initializing LigandForge pipeline..."):
            st.session_state.pipeline = LigandForgePipeline(st.session_state.config)
    
    # Get optimization parameters from sidebar
    opt_params = setup_sidebar()
    
    # Determine optimization method
    method_map = {
        "Genetic Algorithm": "ga",
        "Reinforcement Learning": "rl",
        "Hybrid (RL → GA)": "hybrid"
    }
    opt_method = method_map.get(opt_params['opt_method'], 'hybrid')
    
    # Run pipeline
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        start_time = time.time()
        
        # Update progress
        status_text.text("🔬 Analyzing binding site...")
        progress_bar.progress(20)
        
        # Run pipeline
        results = st.session_state.pipeline.run_full_pipeline(
            pdb_text=pdb_text,
            center=center,
            target_interactions=target_interactions,
            optimization_method=opt_method,
            radius=radius,
            n_initial=n_initial,
            include_voxel_analysis=include_voxel,
            population_size=opt_params['ga_population'],
            generations=opt_params['ga_generations'],
            crossover_rate=opt_params['ga_crossover'],
            mutation_rate=opt_params['ga_mutation'],
            elitism=opt_params['ga_elitism'],
            rl_iterations=opt_params['rl_iterations']
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Pipeline completed successfully!")
        
        execution_time = time.time() - start_time
        
        # Store results
        st.session_state.results = results
        
        # Display success message
        st.success(f"Pipeline completed in {execution_time:.1f} seconds!")
        
        # Show quick statistics
        molecules = results.generation_results.molecules
        scores = results.generation_results.scores
        
        if molecules and scores:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Molecules Generated", len(molecules))
            with col2:
                best_score = max(s.total_score for s in scores)
                st.metric("Best Score", f"{best_score:.3f}")
            with col3:
                avg_score = np.mean([s.total_score for s in scores])
                st.metric("Average Score", f"{avg_score:.3f}")
            with col4:
                druggability = results.pocket_analysis.druggability_score
                st.metric("Pocket Druggability", f"{druggability:.3f}")
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("")
        st.error(f"Pipeline failed: {str(e)}")
        st.exception(e)


def display_results():
    """Display pipeline results"""
    if not st.session_state.results:
        return
    
    results = st.session_state.results
    molecules = results.generation_results.molecules
    scores = results.generation_results.scores
    
    # Create results DataFrame
    results_data = []
    for i, (mol, score) in enumerate(zip(molecules, scores)):
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski, QED
            
            row = {
                'ID': f"MOL_{i+1:03d}",
                'SMILES': Chem.MolToSmiles(mol),
                'MW': rdMolDescriptors.CalcExactMolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol),
                'QED': QED.qed(mol),
                'Total_Score': score.total_score,
                'Pharmacophore_Score': score.pharmacophore_score,
                'Drug_Likeness_Score': score.drug_likeness_score,
                'Synthetic_Score': score.synthetic_score,
                'Novelty_Score': score.novelty_score,
                'Selectivity_Score': score.selectivity_score,
                'Water_Displacement_Score': score.water_displacement_score,
                'Violations': "; ".join(score.violations) if score.violations else "",
                'Confidence': score.confidence
            }
            results_data.append(row)
        except Exception as e:
            st.warning(f"Error processing molecule {i+1}: {e}")
            continue
    
    if not results_data:
        st.error("No valid results to display")
        return
    
    results_df = pd.DataFrame(results_data).sort_values('Total_Score', ascending=False).reset_index(drop=True)
    
    # Results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Results Overview", "🧬 Top Molecules", "📈 Visualizations", 
        "🔬 Binding Site Analysis", "📋 Detailed Report"
    ])
    
    with tab1:
        display_results_overview(results_df, results)
    
    with tab2:
        display_top_molecules(results_df, molecules, scores)
    
    with tab3:
        display_visualizations(results_df, molecules, results)
    
    with tab4:
        display_binding_site_analysis(results)
    
    with tab5:
        display_detailed_report(results)


def display_results_overview(df: pd.DataFrame, results):
    """Display results overview"""
    st.header("📊 Results Overview")
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generation Statistics")
        gen_stats = results.generation_results.generation_statistics
        
        st.write(f"**Optimization Method:** {gen_stats.get('optimization_method', 'N/A')}")
        st.write(f"**Initial Molecules:** {gen_stats.get('initial_molecules', 'N/A')}")
        st.write(f"**Valid Seeds:** {gen_stats.get('valid_seeds', 'N/A')}")
        st.write(f"**Final Molecules:** {len(df)}")
        
        # Diversity statistics
        div_stats = results.generation_results.diversity_statistics
        if div_stats:
            st.write(f"**Unique Scaffolds:** {div_stats.get('unique_scaffolds', 'N/A')}")
            st.write(f"**Diversity Rate:** {div_stats.get('diversity_rate', 0):.3f}")
    
    with col2:
        st.subheader("Score Distribution")
        if 'Total_Score' in df.columns:
            scores = df['Total_Score']
            st.write(f"**Best Score:** {scores.max():.3f}")
            st.write(f"**Average Score:** {scores.mean():.3f}")
            st.write(f"**Score Range:** {scores.min():.3f} - {scores.max():.3f}")
            
            # Score categories
            high_quality = sum(scores > 0.7)
            medium_quality = sum((scores >= 0.4) & (scores <= 0.7))
            low_quality = sum(scores < 0.4)
            
            st.write(f"**High Quality (>0.7):** {high_quality}")
            st.write(f"**Medium Quality (0.4-0.7):** {medium_quality}")
            st.write(f"**Low Quality (<0.4):** {low_quality}")
    
    # Data table
    st.subheader("Results Table")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_all = st.checkbox("Show all columns", value=False)
    with col2:
        max_rows = st.selectbox("Rows to display:", [10, 25, 50, 100], index=0)
    with col3:
        sort_by = st.selectbox("Sort by:", 
                              ['Total_Score', 'MW', 'LogP', 'QED', 'Drug_Likeness_Score'])
    
    # Select columns to display
    if show_all:
        display_df = df.head(max_rows).sort_values(sort_by, ascending=False)
    else:
        key_columns = ['ID', 'SMILES', 'MW', 'LogP', 'QED', 'Total_Score', 'Violations']
        display_df = df[key_columns].head(max_rows).sort_values(sort_by, ascending=False)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download options
    st.subheader("Download Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"ligandforge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col2:
        try:
            sdf_data = create_sdf_from_molecules(
                results.generation_results.molecules[:len(df)], 
                df.to_dict('records')
            )
            st.download_button(
                "📥 Download SDF",
                sdf_data,
                f"ligandforge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sdf",
                "chemical/x-mdl-sdfile"
            )
        except Exception as e:
            st.error(f"SDF generation failed: {e}")
    
    with col3:
        try:
            report_text = results.pipeline.create_report() if hasattr(results, 'pipeline') else "Report not available"
            st.download_button(
                "📥 Download Report",
                report_text,
                f"ligandforge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )
        except:
            pass


def display_top_molecules(df: pd.DataFrame, molecules, scores):
    """Display top molecules with detailed analysis"""
    st.header("🧬 Top Molecules")
    
    # Select molecule to analyze
    mol_options = [f"{row['ID']} (Score: {row['Total_Score']:.3f})" 
                   for _, row in df.head(20).iterrows()]
    
    if not mol_options:
        st.warning("No molecules available for analysis")
        return
    
    selected_mol = st.selectbox("Select molecule for detailed analysis:", mol_options)
    mol_idx = int(selected_mol.split()[0].split('_')[1]) - 1
    
    if mol_idx < len(molecules):
        mol = molecules[mol_idx]
        score = scores[mol_idx]
        row_data = df.iloc[mol_idx]
        
        # Molecule information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Molecular Information")
            st.write(f"**SMILES:** `{row_data['SMILES']}`")
            st.write(f"**Molecular Weight:** {row_data['MW']:.2f} Da")
            st.write(f"**LogP:** {row_data['LogP']:.2f}")
            st.write(f"**QED:** {row_data['QED']:.3f}")
            st.write(f"**TPSA:** {row_data['TPSA']:.1f} Ų")
            
            if row_data['Violations']:
                st.warning(f"**Violations:** {row_data['Violations']}")
        
        with col2:
            # Score breakdown radar plot
            score_dict = {
                'Pharmacophore': score.pharmacophore_score,
                'Drug-likeness': score.drug_likeness_score,
                'Synthetic': score.synthetic_score,
                'Novelty': score.novelty_score,
                'Selectivity': score.selectivity_score
            }
            
            radar_fig = create_radar_plot(score_dict, f"Scores for {row_data['ID']}")
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
        
        # Detailed scores
        st.subheader("Score Breakdown")
        score_cols = st.columns(6)
        
        score_names = ['Total', 'Pharmacophore', 'Drug-likeness', 'Synthetic', 'Novelty', 'Selectivity']
        score_values = [
            score.total_score, score.pharmacophore_score, score.drug_likeness_score,
            score.synthetic_score, score.novelty_score, score.selectivity_score
        ]
        
        for i, (name, value) in enumerate(zip(score_names, score_values)):
            with score_cols[i]:
                st.metric(name, f"{value:.3f}")
    
    # Top molecules table
    st.subheader("Top 20 Molecules")
    top_molecules = df.head(20)[['ID', 'SMILES', 'MW', 'LogP', 'QED', 'Total_Score']]
    st.dataframe(top_molecules, use_container_width=True)


def display_visualizations(df: pd.DataFrame, molecules, results):
    """Display various visualizations"""
    st.header("📈 Visualizations")
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Score Distributions", "Property Space", "Convergence", "Diversity Analysis"
    ])
    
    with viz_tab1:
        st.subheader("Score Distributions")
        score_fig = create_score_distribution_plot(df)
        if score_fig:
            st.plotly_chart(score_fig, use_container_width=True)
        else:
            st.warning("Could not create score distribution plot")
        
        # Molecular weight distribution
        mw_fig = create_molecular_weight_distribution(df)
        if mw_fig:
            st.plotly_chart(mw_fig, use_container_width=True)
        
        # Lipinski compliance
        lipinski_fig = create_lipinski_compliance_plot(df)
        if lipinski_fig:
            st.plotly_chart(lipinski_fig, use_container_width=True)
    
    with viz_tab2:
        st.subheader("Chemical Property Space")
        prop_fig = create_property_space_plot(df)
        if prop_fig:
            st.plotly_chart(prop_fig, use_container_width=True)
        else:
            st.warning("Could not create property space plot")
        
        # Property correlation matrix
        corr_fig = create_property_correlation_matrix(df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
    
    with viz_tab3:
        st.subheader("Optimization Convergence")
        if hasattr(results, 'optimization_history') and results.optimization_history:
            conv_fig = create_convergence_plot(results.optimization_history)
            if conv_fig:
                st.plotly_chart(conv_fig, use_container_width=True)
            else:
                st.info("Convergence data not available for visualization")
        else:
            st.info("No optimization history available")
    
    with viz_tab4:
        st.subheader("Chemical Diversity Analysis")
        if molecules and len(molecules) > 1:
            div_fig = create_diversity_analysis_plot(molecules[:50])  # Limit for performance
            if div_fig:
                st.plotly_chart(div_fig, use_container_width=True)
            else:
                st.info("Could not create diversity analysis plot")
        else:
            st.info("Insufficient molecules for diversity analysis")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary_stats = generate_summary_statistics(df)
    if summary_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Molecular Weight**")
            if 'MW' in summary_stats:
                mw_stats = summary_stats['MW']
                st.write(f"Mean: {mw_stats['mean']:.1f} Da")
                st.write(f"Range: {mw_stats['min']:.1f} - {mw_stats['max']:.1f} Da")
            
            st.write("**LogP**")
            if 'LogP' in summary_stats:
                logp_stats = summary_stats['LogP']
                st.write(f"Mean: {logp_stats['mean']:.2f}")
                st.write(f"Range: {logp_stats['min']:.2f} - {logp_stats['max']:.2f}")
        
        with col2:
            st.write("**Drug-likeness**")
            if 'QED' in summary_stats:
                qed_stats = summary_stats['QED']
                st.write(f"Mean QED: {qed_stats['mean']:.3f}")
            
            if 'lipinski_compliance' in summary_stats:
                compliance = summary_stats['lipinski_compliance']
                st.write(f"Lipinski Compliance: {compliance['compliance_rate']:.1%}")


def display_binding_site_analysis(results):
    """Display binding site analysis"""
    st.header("🔬 Binding Site Analysis")
    
    pocket = results.pocket_analysis
    
    # Pocket summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Druggability Score", f"{pocket.druggability_score:.3f}")
    with col2:
        st.metric("Pocket Volume", f"{pocket.volume:.1f} Ų")
    with col3:
        st.metric("Surface Area", f"{pocket.surface_area:.1f} Ų")
    
    # Interaction analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Interaction Hotspots")
        if pocket.hotspots:
            hotspot_data = []
            for i, hotspot in enumerate(pocket.hotspots):
                hotspot_data.append({
                    'Index': i+1,
                    'Type': hotspot.interaction_type,
                    'Strength': f"{hotspot.strength:.3f}",
                    'Residue': hotspot.residue_name,
                    'Conservation': f"{hotspot.conservation:.3f}"
                })
            
            hotspot_df = pd.DataFrame(hotspot_data)
            st.dataframe(hotspot_df, use_container_width=True)
            
            # Interaction type summary
            interaction_summary = pocket.get_interaction_summary()
            if interaction_summary:
                st.write("**Interaction Type Summary:**")
                for interaction, count in interaction_summary.items():
                    st.write(f"- {interaction}: {count}")
        else:
            st.info("No interaction hotspots identified")
    
    with col2:
        st.subheader("Water Sites")
        if pocket.water_sites:
            water_data = []
            for i, water in enumerate(pocket.water_sites):
                water_data.append({
                    'Index': i+1,
                    'B-factor': f"{water.b_factor:.1f}",
                    'Replaceability': f"{water.replaceability_score:.3f}",
                    'Conservation': f"{water.conservation_score:.3f}",
                    'H-bonds': len(water.hydrogen_bonds)
                })
            
            water_df = pd.DataFrame(water_data)
            st.dataframe(water_df, use_container_width=True)
            
            # Water summary
            replaceable = sum(1 for w in pocket.water_sites if w.replaceability_score > 0.5)
            st.write(f"**Displaceable waters:** {replaceable}/{len(pocket.water_sites)}")
        else:
            st.info("No water sites found")
    
    # Additional analysis
    if hasattr(results, 'voxel_analysis') and results.voxel_analysis:
        st.subheader("Voxel Field Analysis")
        voxel = results.voxel_analysis
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Field Complexity", f"{voxel.field_complexity_score:.3f}")
        with col2:
            st.metric("Hotspot Voxels", len(voxel.hotspot_voxels))
        with col3:
            st.metric("Hydrophobic Moment", f"{voxel.hydrophobic_moment:.2f}")


def display_detailed_report(results):
    """Display detailed text report"""
    st.header("📋 Detailed Report")
    
    # Generate report
    if hasattr(st.session_state.pipeline, 'create_report'):
        report_text = st.session_state.pipeline.create_report()
        st.text_area("Full Report", report_text, height=600)
    else:
        st.info("Detailed report not available")
    
    # Execution metadata
    if hasattr(results, 'execution_metadata'):
        st.subheader("Execution Metadata")
        metadata = results.execution_metadata
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Run ID:** {metadata.get('run_id', 'N/A')}")
            st.write(f"**Execution Time:** {metadata.get('execution_time', 0):.2f} seconds")
            st.write(f"**Method:** {metadata.get('optimization_method', 'N/A')}")
        
        with col2:
            st.write(f"**Target Interactions:** {', '.join(metadata.get('target_interactions', []))}")
            if 'structure_info' in metadata:
                struct_info = metadata['structure_info']
                st.write(f"**Structure Atoms:** {struct_info.get('num_atoms', 'N/A')}")
                st.write(f"**Resolution:** {struct_info.get('resolution', 'N/A')} Å")


def sidebar_info():
    """Sidebar information and tips"""
    st.header("💡 Tips & Info")
    
    st.markdown("""
    ### 🎯 Optimization Methods
    - **Hybrid**: Combines RL exploration with GA refinement
    - **GA**: Population-based evolutionary optimization
    - **RL**: Policy-gradient molecular optimization
    
    ### 🔗 Interaction Types
    - **HBD**: Hydrogen bond donor
    - **HBA**: Hydrogen bond acceptor  
    - **Hydrophobic**: Van der Waals interactions
    - **Aromatic**: π-π stacking
    - **Electrostatic**: Charge interactions
    
    ### 📊 Score Components
    - **Pharmacophore**: Binding site complementarity
    - **Drug-likeness**: ADMET properties
    - **Synthetic**: Ease of synthesis
    - **Novelty**: Chemical uniqueness
    - **Selectivity**: Target specificity
    """)
    
    if st.session_state.results:
        st.success("✅ Results available!")
        
        # Quick statistics
        molecules = st.session_state.results.generation_results.molecules
        if molecules:
            st.write(f"**Molecules:** {len(molecules)}")
            
            scores = [s.total_score for s in st.session_state.results.generation_results.scores]
            st.write(f"**Best Score:** {max(scores):.3f}")
            st.write(f"**Avg Score:** {np.mean(scores):.3f}")


if __name__ == "__main__":
    create_streamlit_app()