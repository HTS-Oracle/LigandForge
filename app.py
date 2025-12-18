import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import re
import zipfile
import io
import base64
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

# Core imports with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    st.warning("psutil not available - performance monitoring disabled")

# LigandForge imports with error handling
try:
    from pipeline import LigandForgePipeline
    from config import LigandForgeConfig, ConfigPresets
    from visualization import (
        create_score_distribution_plot, create_property_space_plot,
        create_convergence_plot, create_radar_plot, create_diversity_analysis_plot,
        create_property_correlation_matrix, create_molecular_weight_distribution,
        create_lipinski_compliance_plot, generate_summary_statistics,
        create_sdf_from_molecules, generate_2d_structure_image,
        add_structure_images_to_dataframe, create_enhanced_results_display,
        create_molecular_grid_image
    )
    from pdb_parser import generate_sample_pdb
    LIGANDFORGE_AVAILABLE = True
except Exception as e:
    LIGANDFORGE_AVAILABLE = False
    st.error(f"LigandForge modules not available: {e}")

# Retrosynthesis optional module
try:
    from retrosynthesis_module import RetrosyntheticAnalyzer, RetrosyntheticRoute
    RETROSYNTHESIS_AVAILABLE = True
except Exception as e:
    RETROSYNTHESIS_AVAILABLE = False
    st.warning(f"Retrosynthesis module not available: {e}")

# Optional visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available - some visualizations disabled")

# RDKit optional
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski, QED, Draw, AllChem
    from PIL import Image
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    st.warning("RDKit not available - molecular analysis limited")

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False

# ------------------------
# Helper / fallback functions (safe stubs)
# ------------------------

def create_retrosynthetic_comparison(routes: List, analyzed_molecules: List[int]):
    """Fallback for retrosynthetic comparison visualization"""
    if not PLOTLY_AVAILABLE:
        st.info("Retrosynthesis comparison requires Plotly.")
        return
    try:
        # Simple radar chart stub if routes provide metrics, else info
        molecule_ids = analyzed_molecules
        categories = ["Steps", "Yield", "Feasibility", "Difficulty", "Complexity"]
        fig = go.Figure()
        for i, route in enumerate(routes[:5]):  # limit
            values = [
                1.0 - min(1.0, getattr(route, "total_steps", 5)/10),
                getattr(route, "total_yield_estimate", 0.5),
                getattr(route, "route_feasibility", 0.5),
                1.0 - getattr(route, "overall_difficulty", 0.5),
                1.0 - getattr(route, "synthetic_complexity_score", 0.5)
            ]
            fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=f"MOL_{molecule_ids[i]+1:03d}"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=500)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not create comparison chart: {e}")

# Enhanced Retrosynthesis Visualization Module

def display_retrosynthetic_route(route, mol_idx: int):
    """Enhanced display for a single retrosynthetic route with full details"""
    st.markdown(f"### 📋 Route Overview for Molecule {mol_idx+1}")
    
    # Route metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", getattr(route, 'total_steps', 'N/A'))
    with col2:
        total_yield = getattr(route, 'total_yield_estimate', 0)
        st.metric("Overall Yield", f"{total_yield*100:.1f}%")
    with col3:
        feasibility = getattr(route, 'route_feasibility', 0)
        st.metric("Feasibility", f"{feasibility:.2f}/1.0")
    with col4:
        difficulty = getattr(route, 'overall_difficulty', 0)
        st.metric("Difficulty", f"{difficulty:.2f}/1.0")
    
    # Starting materials
    st.markdown("#### 🧪 Starting Materials")
    starting_materials = getattr(route, 'starting_materials', [])
    if starting_materials:
        for i, sm in enumerate(starting_materials, 1):
            col1, col2 = st.columns([1, 3])
            with col1:
                if RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(sm)
                        if mol:
                            img_data = generate_2d_structure_image(sm, size=(150, 150))
                            if img_data:
                                st.markdown(f'<img src="{img_data}" style="max-width:100%;">', unsafe_allow_html=True)
                    except Exception:
                        pass
            with col2:
                st.markdown(f"""
                **Starting Material {i}**  
                `{sm}`
                """)
    else:
        st.info("No starting materials identified")
    
    st.markdown("---")
    
    # Detailed synthetic steps
    st.markdown("#### ⚗️ Synthetic Steps (Forward Direction)")
    
    steps = getattr(route, 'steps', [])
    if not steps:
        st.warning("No synthetic steps available")
        return
    
    for step in steps:
        step_num = getattr(step, 'step_number', '?')
        reaction_type = getattr(step, 'reaction_type', 'Unknown')
        
        # Step header with color coding based on difficulty
        difficulty = getattr(step, 'difficulty', 0)
        if difficulty < 0.25:
            color = "#28a745"  # green
            difficulty_label = "Easy"
        elif difficulty < 0.35:
            color = "#ffc107"  # yellow
            difficulty_label = "Moderate"
        else:
            color = "#dc3545"  # red
            difficulty_label = "Challenging"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                    border-left: 4px solid {color}; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 20px 0;">
            <h4 style="margin: 0; color: {color};">
                Step {step_num}: {reaction_type}
            </h4>
            <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px;">
                {difficulty_label}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Get reactants and product for comprehensive scheme
        reactant_1 = getattr(step, 'reactant_1', None)
        reactant_2 = getattr(step, 'reactant_2', None)
        product = getattr(step, 'product', None)
        
        # Reaction arrow with conditions
        st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
        
        # Create a comprehensive reaction scheme view
        st.markdown("#### 🔬 Complete Reaction Scheme")
        
        # Create three-column layout for full reaction visualization
        scheme_col1, scheme_col2, scheme_col3 = st.columns([3, 2, 3])
        
        with scheme_col1:
            st.markdown("**Reactants**")
            # Show reactant 1
            reactant_1_displayed = False
            if reactant_1 and RDKIT_AVAILABLE:
                try:
                    mol = Chem.MolFromSmiles(reactant_1)
                    if mol:
                        img_data = generate_2d_structure_image(reactant_1, size=(250, 200))
                        if img_data:
                            st.markdown(f"""
                            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background: #f9f9f9;">
                                <img src="{img_data}" style="max-width: 100%;">
                                <p style="font-size: 11px; color: #666; text-align: center; margin: 5px 0;">Reactant 1</p>
                            </div>
                            """, unsafe_allow_html=True)
                            reactant_1_displayed = True
                except Exception as e:
                    st.caption(f"Structure error: {e}")
            
            # Fallback to SMILES if structure couldn't be displayed
            if not reactant_1_displayed and reactant_1:
                st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background: #f9f9f9;">
                    <p style="font-size: 11px; color: #666; text-align: center; margin: 5px 0; font-weight: bold;">Reactant 1</p>
                    <code style="font-size: 10px; word-wrap: break-word; display: block; padding: 5px; background: white; border-radius: 3px;">{reactant_1}</code>
                </div>
                """, unsafe_allow_html=True)
            elif not reactant_1:
                st.info("No reactant 1 data")
            
            # Show reactant 2 if exists
            if reactant_2:
                st.markdown("<div style='text-align: center; margin: 10px 0;'><strong style='font-size: 24px;'>+</strong></div>", unsafe_allow_html=True)
                reactant_2_displayed = False
                if RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(reactant_2)
                        if mol:
                            img_data = generate_2d_structure_image(reactant_2, size=(250, 200))
                            if img_data:
                                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background: #f9f9f9;">
                                    <img src="{img_data}" style="max-width: 100%;">
                                    <p style="font-size: 11px; color: #666; text-align: center; margin: 5px 0;">Reactant 2</p>
                                </div>
                                """, unsafe_allow_html=True)
                                reactant_2_displayed = True
                    except Exception as e:
                        st.caption(f"Structure error: {e}")
                
                # Fallback for reactant 2
                if not reactant_2_displayed:
                    st.markdown(f"""
                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background: #f9f9f9;">
                        <p style="font-size: 11px; color: #666; text-align: center; margin: 5px 0; font-weight: bold;">Reactant 2</p>
                        <code style="font-size: 10px; word-wrap: break-word; display: block; padding: 5px; background: white; border-radius: 3px;">{reactant_2}</code>
                    </div>
                    """, unsafe_allow_html=True)
        
        with scheme_col2:
            st.markdown("**Reaction**")
            # Show reaction arrow with conditions
            st.markdown(f"""
            <div style="text-align: center; padding: 20px 10px;">
                <div style="font-size: 48px; color: #2196F3; font-weight: bold;">→</div>
                <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <p style="font-size: 11px; margin: 5px 0; color: #1976d2;"><strong>{reaction_type}</strong></p>
                    <hr style="margin: 5px 0; border: none; border-top: 1px solid #90caf9;">
                    <p style="font-size: 10px; margin: 3px 0;">Reagents:</p>
            """, unsafe_allow_html=True)
            
            reagents = getattr(step, 'reagents', [])
            for reagent in reagents[:3]:  # Show max 3
                st.markdown(f"<p style='font-size: 9px; margin: 2px 0; color: #555;'>• {reagent}</p>", unsafe_allow_html=True)
            
            conditions = getattr(step, 'conditions', 'N/A')
            st.markdown(f"""
                    <p style="font-size: 10px; margin: 5px 0 3px 0;">Conditions:</p>
                    <p style="font-size: 9px; margin: 2px 0; color: #555;">{conditions}</p>
                </div>
                <p style="font-size: 12px; color: #4CAF50; font-weight: bold; margin: 10px 0;">
                    Yield: {getattr(step, 'yield_estimate', 0)*100:.0f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Try to show key reagent structures
            if reagents and RDKIT_AVAILABLE:
                reagent_smiles_map = {
                    'NaBH4': '[Na+].[BH4-]',
                    'LiAlH4': '[Li+].[AlH4-]',
                    'NaBH(OAc)3': 'CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]',
                    'NaCNBH3': '[Na+].[BH3-]C#N',
                    'H2SO4': 'OS(O)(=O)=O',
                    'NaOH': '[Na+].[OH-]',
                    'K2CO3': '[K+].[K+].[O-]C([O-])=O',
                    'Cs2CO3': '[Cs+].[Cs+].[O-]C([O-])=O',
                    'NaH': '[Na+].[H-]',
                    'mCPBA': 'Cc1cccc(c1)C(=O)OO',
                }
                
                for reagent in reagents[:1]:  # Show first reagent structure
                    reagent_key = reagent.split('/')[0].strip()
                    if reagent_key in reagent_smiles_map:
                        try:
                            reagent_smiles = reagent_smiles_map[reagent_key]
                            mol = Chem.MolFromSmiles(reagent_smiles)
                            if mol:
                                img_data = generate_2d_structure_image(reagent_smiles, size=(120, 100))
                                if img_data:
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 5px; margin: 5px 0; background: white; border-radius: 5px;">
                                        <img src="{img_data}" style="max-width: 100px;">
                                        <p style="font-size: 9px; color: #888; margin: 2px 0;">{reagent_key}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except Exception:
                            pass
        
        with scheme_col3:
            st.markdown("**Product**")
            # Show product
            product_displayed = False
            if product and RDKIT_AVAILABLE:
                try:
                    mol = Chem.MolFromSmiles(product)
                    if mol:
                        img_data = generate_2d_structure_image(product, size=(250, 200))
                        if img_data:
                            st.markdown(f"""
                            <div style="border: 2px solid #2196F3; padding: 10px; border-radius: 8px; background: #f9f9f9;">
                                <img src="{img_data}" style="max-width: 100%;">
                                <p style="font-size: 11px; color: #666; text-align: center; margin: 5px 0;">Product</p>
                            </div>
                            """, unsafe_allow_html=True)
                            product_displayed = True
                except Exception as e:
                    st.caption(f"Structure error: {e}")
            
            # Fallback to SMILES
            if not product_displayed and product:
                st.markdown(f"""
                <div style="border: 2px solid #2196F3; padding: 10px; border-radius: 8px; background: #f9f9f9;">
                    <p style="font-size: 11px; color: #666; text-align: center; margin: 5px 0; font-weight: bold;">Product</p>
                    <code style="font-size: 10px; word-wrap: break-word; display: block; padding: 5px; background: white; border-radius: 3px;">{product}</code>
                </div>
                """, unsafe_allow_html=True)
            elif not product:
                st.info("No product data")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Reagents and conditions with visual representation
        st.markdown("**Reaction Conditions:**")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            reagents = getattr(step, 'reagents', [])
            if reagents:
                st.markdown("**Reagents:**")
                for reagent in reagents:
                    st.markdown(f"• {reagent}")
                    
                # Try to visualize common reagents if possible
                if RDKIT_AVAILABLE and len(reagents) > 0:
                    reagent_smiles_map = {
                        'NaBH4': '[Na+].[BH4-]',
                        'LiAlH4': '[Li+].[AlH4-]',
                        'H2SO4': 'OS(O)(=O)=O',
                        'NaOH': '[Na+].[OH-]',
                        'HCl': 'Cl',
                        'MeOH': 'CO',
                        'EtOH': 'CCO',
                        'THF': 'C1CCOC1',
                        'DCM': 'ClCCl',
                        'DMF': 'CN(C)C=O',
                        'AcOH': 'CC(O)=O',
                        'acetone': 'CC(C)=O'
                    }
                    
                    # Try to visualize first reagent if it's recognizable
                    for reagent in reagents[:2]:  # Show max 2 reagents
                        reagent_key = reagent.split('/')[0].strip()
                        if reagent_key in reagent_smiles_map:
                            try:
                                reagent_smiles = reagent_smiles_map[reagent_key]
                                mol = Chem.MolFromSmiles(reagent_smiles)
                                if mol:
                                    img_data = generate_2d_structure_image(reagent_smiles, size=(100, 100))
                                    if img_data:
                                        st.markdown(f"""
                                        <div style="text-align: center; padding: 5px;">
                                            <img src="{img_data}" style="max-width: 80px;">
                                            <p style="font-size: 10px; color: #888; margin: 2px 0;">{reagent_key}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            except Exception:
                                pass
        
        with col2:
            st.markdown("<div style='text-align: center; padding: 40px 0;'><h3>→</h3></div>", unsafe_allow_html=True)
        
        with col3:
            conditions = getattr(step, 'conditions', 'N/A')
            st.markdown(f"**Conditions:**  \n{conditions}")
            
            # Show temperature and solvent info prominently
            if 'C' in conditions or '°' in conditions:
                st.markdown("🌡️ *Temperature controlled*")
            if any(solvent in conditions for solvent in ['THF', 'DCM', 'DMF', 'MeOH', 'EtOH', 'H2O']):
                st.markdown("🧪 *Solvent required*")
        
        # Yield information
        yield_estimate = getattr(step, 'yield_estimate', 0)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Expected Yield:** {yield_estimate*100:.0f}%")
        with col2:
            st.markdown(f"**Difficulty:** {difficulty:.2f}/1.0")
        
        st.markdown("---")
    
    # Key intermediates
    key_intermediates = getattr(route, 'key_intermediates', [])
    if key_intermediates:
        st.markdown("#### 🔑 Key Intermediates")
        for i, intermediate in enumerate(key_intermediates[:3], 1):
            with st.expander(f"Intermediate {i}"):
                st.code(intermediate, language="smiles")
                if RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(intermediate)
                        if mol:
                            img_data = generate_2d_structure_image(intermediate, size=(200, 200))
                            if img_data:
                                st.markdown(f'<div style="text-align: center;"><img src="{img_data}" style="max-width: 100%;"></div>', unsafe_allow_html=True)
                    except Exception:
                        pass
    
    # Critical reactions warning
    critical_reactions = getattr(route, 'critical_reactions', [])
    if critical_reactions:
        st.markdown("#### ⚠️ Critical Steps (Require Special Attention)")
        for critical in critical_reactions:
            st.warning(critical)
    
    # Summary statistics
    st.markdown("#### 📊 Route Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        complexity = getattr(route, 'synthetic_complexity_score', 0)
        st.metric("Synthetic Complexity", f"{complexity:.2f}/1.0")
    with col2:
        st.metric("Number of Steps", len(steps))
    with col3:
        avg_yield_per_step = (total_yield ** (1/len(steps))) if len(steps) > 0 else 0
        st.metric("Avg Yield/Step", f"{avg_yield_per_step*100:.1f}%")


def create_retrosynthetic_comparison(routes: List, analyzed_molecules: List[int]):
    """Enhanced comparison visualization for multiple routes"""
    if not PLOTLY_AVAILABLE:
        st.info("Route comparison requires Plotly.")
        return
    
    try:
        # Radar chart comparison
        st.markdown("#### 📊 Multi-Route Comparison")
        
        categories = ['Feasibility', 'Yield', 'Simplicity', 'Low Difficulty', 'Low Complexity']
        
        fig = go.Figure()
        
        for i, (route, mol_idx) in enumerate(zip(routes[:5], analyzed_molecules[:5])):
            # Normalize values to 0-1 scale
            values = [
                getattr(route, "route_feasibility", 0.5),
                getattr(route, "total_yield_estimate", 0.5),
                1.0 - min(1.0, getattr(route, "total_steps", 5)/10),
                1.0 - getattr(route, "overall_difficulty", 0.5),
                1.0 - getattr(route, "synthetic_complexity_score", 0.5)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"MOL_{mol_idx+1:03d}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=500,
            title="Synthetic Route Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        st.markdown("#### 📋 Detailed Comparison Table")
        
        comparison_data = []
        for route, mol_idx in zip(routes, analyzed_molecules):
            comparison_data.append({
                'Molecule': f"MOL_{mol_idx+1:03d}",
                'Steps': getattr(route, 'total_steps', 'N/A'),
                'Overall Yield (%)': f"{getattr(route, 'total_yield_estimate', 0)*100:.1f}",
                'Feasibility': f"{getattr(route, 'route_feasibility', 0):.2f}",
                'Difficulty': f"{getattr(route, 'overall_difficulty', 0):.2f}",
                'Complexity': f"{getattr(route, 'synthetic_complexity_score', 0):.2f}",
                'Critical Steps': len(getattr(route, 'critical_reactions', []))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Step distribution chart
        st.markdown("#### 📈 Step Distribution")
        
        fig = go.Figure()
        
        molecules_list = [f"MOL_{mol_idx+1:03d}" for mol_idx in analyzed_molecules]
        steps_list = [getattr(route, 'total_steps', 0) for route in routes]
        
        fig.add_trace(go.Bar(
            x=molecules_list,
            y=steps_list,
            marker_color='lightblue',
            text=steps_list,
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Number of Synthetic Steps per Route",
            xaxis_title="Molecule",
            yaxis_title="Number of Steps",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not create comparison chart: {e}")
# ------------------------
# Custom Exceptions
# ------------------------

class LigandForgeError(Exception):
    """Custom exception with user-friendly messages"""
    def __init__(self, message: str, technical_details: Optional[str] = None,
                 suggestions: Optional[List[str]] = None):
        self.message = message
        self.technical_details = technical_details
        self.suggestions = suggestions or []
        super().__init__(self.message)

class ValidationError(LigandForgeError):
    pass

class PipelineError(LigandForgeError):
    pass

# ------------------------
# App State & Validators & UI Components
# ------------------------

class AppState:
    """Centralized application state management"""
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        defaults = {
            'pipeline': None,
            'results': None,
            'config': LigandForgeConfig() if LIGANDFORGE_AVAILABLE else None,
            'session_start': time.time(),
            'checkpoints': {},
            'progress_tracker': {
                'current_stage': '',
                'stage_progress': 0,
                'overall_progress': 0,
                'start_time': None,
                'stage_times': {},
                'eta': None
            },
            'validation_errors': [],
            'performance_metrics': {
                'memory_usage': [],
                'execution_times': [],
                'cache_stats': {}
            },
            'opt_params': {
                'opt_method': 'Hybrid (RL -> GA)',
                'ga_population': 150,
                'ga_generations': 30,
                'ga_crossover': 0.7,
                'ga_mutation': 0.3,
                'ga_elitism': 10,
                'rl_iterations': 20
            },
            'display_options': {
                'show_structures': True,
                'structure_size': 150,
                'max_structures_grid': 20,
                'structure_format': 'png'
            }
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

class InputValidator:
    @staticmethod
    def clean_smiles(smiles: str) -> str:
        if not smiles:
            return ""
        cleaned = re.sub(r'[^A-Za-z0-9\[\]()=#+\-@/\\.\s]', '', smiles)
        return cleaned.strip()

    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float, name: str):
        if not min_val <= value <= max_val:
            raise ValidationError(f"{name} must be between {min_val} and {max_val}",
                                  suggestions=[f"Use a value between {min_val} and {max_val}"])

    @staticmethod
    def validate_pdb_content(pdb_text: str) -> str:
        if not pdb_text or not pdb_text.strip():
            raise ValidationError("PDB content is empty", suggestions=["Upload a valid PDB file"])
        if "ATOM" not in pdb_text and "HETATM" not in pdb_text:
            raise ValidationError("Invalid PDB format - no ATOM/HETATM records found",
                                  suggestions=["Ensure the file is in PDB format"])
        valid_records = {'ATOM', 'HETATM', 'HEADER', 'TITLE', 'COMPND', 'SOURCE',
                         'KEYWDS', 'EXPDTA', 'REVDAT', 'REMARK', 'SEQRES', 'HET',
                         'HETNAM', 'HETSYN', 'FORMUL', 'SHEET', 'HELIX', 'LINK',
                         'SSBOND', 'SITE', 'CRYST1', 'SCALE', 'MASTER', 'END'}
        lines = pdb_text.split('\n')
        cleaned_lines = [line for line in lines if any(line.startswith(record) for record in valid_records)]
        return '\n'.join(cleaned_lines)

    @staticmethod
    def validate_pdb_id(pdb_id: str) -> bool:
        if not pdb_id or len(pdb_id) != 4:
            return False
        return re.match(r'^[0-9][a-zA-Z0-9]{3}$', pdb_id) is not None

    @staticmethod
    def validate_file_size(uploaded_file, max_size_mb: int = 50) -> bool:
        if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size_mb * 1024 * 1024:
            st.error(f"File too large. Maximum size: {max_size_mb}MB")
            return False
        return True

class UIComponents:
    @staticmethod
    def display_error_with_context(error: Exception):
        if isinstance(error, LigandForgeError):
            st.error(f"Error: {error.message}")
            if error.suggestions:
                st.markdown("**Suggestions:**")
                for suggestion in error.suggestions:
                    st.write(f"• {suggestion}")
            if error.technical_details:
                with st.expander("Technical Details"):
                    st.code(str(error.technical_details))
        else:
            st.error(f"Error: {str(error)}")

    @staticmethod
    def create_progress_tracker():
        """Simple ongoing indicator with hourglass"""
        tracker = st.session_state.progress_tracker
        if tracker.get('start_time'):
            elapsed = time.time() - tracker['start_time']
            # Animated hourglass using different unicode characters
            hourglass_frames = ['⏳', '⌛']
            frame = hourglass_frames[int(elapsed) % 2]
            st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                </div>
                    <h2 style="color: #1f77b4;">{frame} Ongoing...</h2>
                    <p style="color: #666; font-size: 14px;">Elapsed: {elapsed:.0f}s</p>
                </div>
            """, unsafe_allow_html=True)                

    @staticmethod
    def create_metric_container(title: str, value: str, delta: Optional[str] = None):
        delta_html = f"<br><small style='color: #888;'>{delta}</small>" if delta else ""
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f0f2f6, #e8ecf0);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 4px solid #1f77b4;">
                <strong>{title}</strong><br>
                <span style="font-size: 1.2em;">{value}</span>
                {delta_html}
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def display_molecular_structure_grid(molecules: List, scores: List, max_display: int = 12):
        if not RDKIT_AVAILABLE or not molecules:
            st.info("Molecular structure display requires RDKit")
            return
        try:
            display_molecules = molecules[:max_display]
            display_scores = scores[:max_display]
            cols_per_row = 4
            rows = (len(display_molecules) + cols_per_row - 1) // cols_per_row
            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    mol_idx = row * cols_per_row + col_idx
                    if mol_idx < len(display_molecules):
                        mol = display_molecules[mol_idx]
                        score = display_scores[mol_idx]
                        with cols[col_idx]:
                            try:
                                smiles = Chem.MolToSmiles(mol)
                                img_data = generate_2d_structure_image(smiles, size=(150,150))
                                if img_data:
                                    st.markdown(f"""
                                        <div style="text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                                            <img src="{img_data}" style="max-width: 100%; height: auto;">
                                            <p style="font-size: 12px; margin: 5px 0;"><strong>MOL_{mol_idx+1:03d}</strong></p>
                                            <p style="font-size: 11px; color: #666;">Score: {score.total_score:.3f}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.write(f"MOL_{mol_idx+1:03d}")
                                    st.write(f"Score: {score.total_score:.3f}")
                            except Exception as e:
                                st.write(f"MOL_{mol_idx+1:03d}")
                                st.write(f"Score: {score.total_score:.3f}")
                                st.caption(f"Display error: {e}")
        except Exception as e:
            st.error(f"Error displaying molecular grid: {e}")

# ------------------------
# Performance Monitor
# ------------------------

class PerformanceMonitor:
    @staticmethod
    def update_performance_metrics():
        if not PSUTIL_AVAILABLE:
            return
        try:
            memory_usage = psutil.virtual_memory().percent
            st.session_state.performance_metrics['memory_usage'].append({
                'timestamp': time.time(),
                'usage': memory_usage
            })
            if len(st.session_state.performance_metrics['memory_usage']) > 100:
                st.session_state.performance_metrics['memory_usage'] = st.session_state.performance_metrics['memory_usage'][-100:]
        except Exception:
            pass

# ------------------------
# Input Handler
# ------------------------

class InputHandler:
    @staticmethod
    def get_pdb_input(input_method: str) -> Optional[str]:
        validator = InputValidator()
        pdb_text = None
        if input_method == "Upload PDB File":
            pdb_text = InputHandler._handle_file_upload(validator)
        elif input_method == "PDB ID":
            pdb_text = InputHandler._handle_pdb_id(validator)
        elif input_method == "Sample Structure":
            pdb_text = InputHandler._handle_sample_structure(validator)
        return pdb_text

    @staticmethod
    def _handle_file_upload(validator: InputValidator) -> Optional[str]:
        uploaded_file = st.file_uploader("Upload PDB file", type=['pdb'], help="Upload a protein structure in PDB format")
        if uploaded_file:
            if not validator.validate_file_size(uploaded_file):
                return None
            try:
                pdb_text = uploaded_file.read().decode('utf-8')
                pdb_text = validator.validate_pdb_content(pdb_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"Loaded PDB file: {uploaded_file.name}")
                with col2:
                    file_size = getattr(uploaded_file, 'size', 0) / 1024
                    st.info(f"File size: {file_size:.1f} KB")
                return pdb_text
            except Exception as e:
                UIComponents.display_error_with_context(e)
                return None
        return None

    @staticmethod
    def _handle_pdb_id(validator: InputValidator) -> Optional[str]:
        col1, col2 = st.columns([2,1])
        with col1:
            pdb_id = st.text_input("Enter PDB ID (e.g., 1abc):", help="4-character PDB identifier")
        with col2:
            fetch_button = st.button("Fetch PDB")
        if pdb_id and not validator.validate_pdb_id(pdb_id):
            st.error("Invalid PDB ID format. Should be 4 characters (e.g., 1abc)")
            return None
        if fetch_button and pdb_id and validator.validate_pdb_id(pdb_id):
            try:
                with st.spinner("Fetching PDB structure..."):
                    if LIGANDFORGE_AVAILABLE:
                        from pdb_parser import download_pdb
                        pdb_text = download_pdb(pdb_id.upper())
                        pdb_text = validator.validate_pdb_content(pdb_text)
                        st.success(f"Successfully fetched PDB {pdb_id.upper()}")
                        return pdb_text
                    else:
                        st.error("PDB download not available - LigandForge modules missing")
                        return None
            except Exception as e:
                UIComponents.display_error_with_context(e)
                return None
        return None

    @staticmethod
    def _handle_sample_structure(validator: InputValidator) -> Optional[str]:
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Load Sample Structure"):
                try:
                    if LIGANDFORGE_AVAILABLE:
                        pdb_text = generate_sample_pdb()
                        pdb_text = validator.validate_pdb_content(pdb_text)
                        st.success("Loaded sample kinase structure")
                        return pdb_text
                    else:
                        st.error("Sample structure not available - LigandForge modules missing")
                        return None
                except Exception as e:
                    UIComponents.display_error_with_context(e)
                    return None
        with col2:
            st.info("Use this to test the pipeline with a pre-configured structure")
        return None

# ------------------------
# Configuration Manager 
# ------------------------

class ConfigurationManager:
    @staticmethod
    def setup_sidebar():
        st.sidebar.header("Configuration")
        if not LIGANDFORGE_AVAILABLE:
            st.sidebar.error("LigandForge modules not available")
            return
        ConfigurationManager._setup_presets()
        ConfigurationManager._setup_generation_params()
        ConfigurationManager._setup_scoring_weights()
        ConfigurationManager._setup_optimization_params()
        ConfigurationManager._setup_display_options()
        ConfigurationManager._setup_config_management()

        # Retrosynthesis section (fixed variable usage)
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Retrosynthesis")

        if RETROSYNTHESIS_AVAILABLE:
            st.sidebar.success("✓ Retrosynthesis module available")
            auto_retro = st.sidebar.checkbox(
                "Auto-analyze top molecules",
                value=False,
                help="Automatically analyze retrosynthetic routes for top 5 molecules after generation"
            )
            st.session_state['auto_retro_enabled'] = bool(auto_retro)
            st.session_state['auto_retro_count'] = 5 if auto_retro else 0
        else:
            st.sidebar.warning("⚠ Retrosynthesis module not available")
            st.sidebar.caption("Install retrosynthesis_module.py to enable")

    @staticmethod
    def _setup_display_options():
        st.sidebar.subheader("Display Options")
        display_opts = st.session_state.display_options
        display_opts['show_structures'] = st.sidebar.checkbox(
            "Show 2D Structures",
            value=display_opts['show_structures'],
            help="Display 2D molecular structures in results table"
        )
        if display_opts['show_structures']:
            display_opts['structure_size'] = st.sidebar.slider(
                "Structure Size (px)", 100, 300, display_opts['structure_size'],
                help="Size of displayed molecular structures"
            )
            display_opts['max_structures_grid'] = st.sidebar.number_input(
                "Max Grid Structures", 4, 24, display_opts['max_structures_grid'],
                help="Maximum number of structures to show in grid view"
            )
            display_opts['structure_format'] = st.sidebar.selectbox(
                "Structure Format",
                ['png', 'svg'],
                index=0 if display_opts['structure_format'] == 'png' else 1,
                help="Format for molecular structure images"
            )

    @staticmethod
    def _setup_presets():
        with st.sidebar.expander("Preset Configurations"):
            preset_type = st.selectbox(
                "Choose preset configuration:",
                ["Custom", "Kinase Inhibitors", "GPCR Ligands", "Fragment-Based",
                 "Lead Optimization", "Fast Screening", "High Quality"]
            )
            if st.button("Apply Preset") and preset_type != "Custom":
                try:
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
                    st.experimental_rerun()
                except Exception as e:
                    UIComponents.display_error_with_context(ValidationError(
                        f"Failed to apply {preset_type} preset",
                        technical_details=str(e),
                        suggestions=["Try a different preset", "Use custom configuration"]
                    ))

    @staticmethod
    def _setup_generation_params():
        with st.sidebar.expander("Generation Parameters"):
            config = st.session_state.config
            if not config:
                st.sidebar.warning("Config object not loaded. Apply a preset or load config.")
                return
            max_atoms = st.slider("Max Heavy Atoms", 15, 80, getattr(config, "max_heavy_atoms", 30),
                                  help="Maximum number of heavy (non-hydrogen) atoms")
            min_atoms = st.slider("Min Heavy Atoms", 5, 40, getattr(config, "min_heavy_atoms", 10),
                                  help="Minimum number of heavy (non-hydrogen) atoms")
            if max_atoms < min_atoms:
                st.error("Max atoms must be greater than min atoms")
            else:
                setattr(config, "max_heavy_atoms", max_atoms)
                setattr(config, "min_heavy_atoms", min_atoms)
            config.max_rings = st.slider("Max Rings", 1, 8, getattr(config, "max_rings", 3),
                                         help="Maximum number of ring systems")
            config.diversity_threshold = st.slider("Diversity Threshold", 0.3, 0.9, getattr(config, "diversity_threshold", 0.4),
                                                   help="Minimum structural diversity required")

    @staticmethod
    def _setup_scoring_weights():
        with st.sidebar.expander("Scoring Weights"):
            config = st.session_state.config
            if not config:
                st.sidebar.warning("Config object not loaded.")
                return
            st.write("**Weight Distribution:**")
            weights = {}
            total_weight = 0.0
            for component in getattr(config, "reward_weights", {}).keys():
                weight_value = st.slider(component.replace('_', ' ').title(), 0.0, 1.0, config.reward_weights[component], step=0.05)
                weights[component] = weight_value
                total_weight += weight_value
            st.write(f"**Total Weight:** {total_weight:.2f}")
            if abs(total_weight - 1.0) > 0.1:
                st.warning(f"Weights should sum to ~1.0 (current: {total_weight:.2f})")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Normalize Weights"):
                    if total_weight > 0:
                        normalized_weights = {k: v/total_weight for k, v in weights.items()}
                        config.reward_weights = normalized_weights
                        st.success("Weights normalized!")
                        st.experimental_rerun()
            with col2:
                if st.button("Reset to Default"):
                    config.reward_weights = LigandForgeConfig().reward_weights
                    st.success("Weights reset to default!")
                    st.experimental_rerun()

    @staticmethod
    def _setup_optimization_params():
        with st.sidebar.expander("Optimization Parameters"):
            opt_params = st.session_state.opt_params
            opt_params['opt_method'] = st.selectbox(
                "Optimization Method",
                ["Hybrid (RL -> GA)", "Genetic Algorithm", "Reinforcement Learning"],
                index=["Hybrid (RL -> GA)", "Genetic Algorithm", "Reinforcement Learning"].index(opt_params['opt_method'])
            )
            st.subheader("Genetic Algorithm")
            population = st.number_input("Population Size", 50, 500, opt_params['ga_population'], step=10)
            generations = st.number_input("Generations", 10, 100, opt_params['ga_generations'], step=5)
            estimated_time = (population * generations) / 1000
            st.caption(f"Estimated time: ~{estimated_time:.1f} minutes")
            opt_params['ga_population'] = int(population)
            opt_params['ga_generations'] = int(generations)
            opt_params['ga_crossover'] = st.slider("Crossover Rate", 0.0, 1.0, float(opt_params['ga_crossover']))
            opt_params['ga_mutation'] = st.slider("Mutation Rate", 0.0, 1.0, float(opt_params['ga_mutation']))
            opt_params['ga_elitism'] = st.number_input("Elitism", 0, 50, opt_params['ga_elitism'], step=1)
            st.subheader("Reinforcement Learning")
            opt_params['rl_iterations'] = st.number_input("RL Iterations", 5, 200, opt_params['rl_iterations'], step=1)

    @staticmethod
    def _setup_config_management():
        st.sidebar.subheader("Configuration Management")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Save Config"):
                try:
                    config_dict = {
                        'config': st.session_state.config.__dict__.copy() if st.session_state.config else {},
                        'opt_params': st.session_state.opt_params.copy(),
                        'display_options': st.session_state.display_options.copy(),
                        'timestamp': datetime.now().isoformat(),
                        'version': '2.0'
                    }
                    config_json = json.dumps(config_dict, indent=2, default=str)
                    st.download_button("Download Config", config_json,
                                       f"ligandforge_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                       "application/json")
                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")
        with col2:
            uploaded_config = st.file_uploader("Load Config", type=['json'])
            if uploaded_config:
                try:
                    config_data = json.load(uploaded_config)
                    if 'config' in config_data and 'opt_params' in config_data:
                        if st.session_state.config:
                            for key, value in config_data['config'].items():
                                if hasattr(st.session_state.config, key):
                                    setattr(st.session_state.config, key, value)
                        st.session_state.opt_params.update(config_data['opt_params'])
                        if 'display_options' in config_data:
                            st.session_state.display_options.update(config_data['display_options'])
                        st.success("Configuration loaded!")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid configuration file format")
                except Exception as e:
                    st.error(f"Failed to load configuration: {e}")

# ------------------------
# Main Application
# ------------------------

class LigandForgeApp:
    def __init__(self):
        self.state = AppState()
        self.setup_page_config()
        self.add_custom_css()

    def setup_page_config(self):
        st.set_page_config(page_title="LigandForge 2.0", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

    def add_custom_css(self):
        st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
        .metric-container { background: linear-gradient(135deg, #f0f2f6, #e8ecf0); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #1f77b4; }
        .molecule-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 15px; margin: 20px 0; }
        .molecule-card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; text-align:center; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .molecule-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); transform: translateY(-2px); transition: all 0.3s ease; }
        .structure-img { max-width: 100%; height: auto; border-radius: 4px; }
        .success-box { background: linear-gradient(135deg,#d4edda,#c3e6cb); border:1px solid #c3e6cb; border-radius:5px; padding:1rem; margin:1rem 0; border-left:4px solid #28a745; }
        .error-box { background: linear-gradient(135deg,#f8d7da,#f5c6cb); border:1px solid #f5c6cb; border-radius:5px; padding:1rem; margin:1rem 0; border-left:4px solid #dc3545; }
        .info-box { background: linear-gradient(135deg,#d1ecf1,#bee5eb); border:1px solid #bee5eb; border-radius:5px; padding:1rem; margin:1rem 0; border-left:4px solid #17a2b8; }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        st.markdown('<h1 class="main-header">LigandForge 2.0</h1>', unsafe_allow_html=True)
        st.markdown("**AI-Driven Structure-Based Drug Design Platform developed by Hossam Nada**")


        # Main navigation tabs
        main_tab1, main_tab2 = st.tabs(["🔬 Application", "ℹ️ About"])
        
        with main_tab1:
            self.run_application()
        
        with main_tab2:
            self.display_about_page()

    def run_application(self):
        """Main application interface"""
        if not LIGANDFORGE_AVAILABLE:
            st.error("LigandForge core modules are not available. Please install required dependencies.")
            st.stop()
        
        PerformanceMonitor.update_performance_metrics()
        ConfigurationManager.setup_sidebar()
        
        col1, col2 = st.columns([3,1])
        with col1:
            self.main_interface()
        with col2:
            self.sidebar_info()

    def display_about_page(self):
        """Comprehensive About page"""
        
        # Hero section
        st.markdown("""
        <div class="about-section">
            <h1>🧬 LigandForge 2.0</h1>
            <h3>Next-Generation AI-Powered Drug Design Platform</h3>
            <p style="font-size: 1.1em; margin-top: 1rem;">
                LigandForge combines cutting-edge artificial intelligence, cheminformatics, 
                and computational chemistry to accelerate the drug discovery process.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Overview
        st.markdown("## 📖 Overview")
        st.markdown("""
        LigandForge 2.0 is a comprehensive computational platform designed for structure-based 
        drug design. It leverages advanced machine learning algorithms, genetic algorithms, and 
        reinforcement learning to generate novel drug-like molecules optimized for specific 
        protein targets.
        
        The platform integrates multiple computational approaches to:
        - Generate diverse chemical libraries
        - Optimize molecular properties
        - Predict synthetic accessibility
        - Analyze retrosynthetic routes
        - Evaluate drug-likeness and pharmacological properties
        """)
        
        # Key Features
        st.markdown("## ✨ Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 Structure-Based Design</h3>
                <p>Direct input of protein structures (PDB format) for targeted ligand generation. 
                Automatic binding site detection and pharmacophore mapping.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 AI Optimization</h3>
                <p>Hybrid optimization using Genetic Algorithms and Reinforcement Learning. 
                Multi-objective scoring with customizable weight distributions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Comprehensive Analysis</h3>
                <p>Detailed molecular property calculations, drug-likeness predictions, 
                and interactive visualization of chemical space.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>⚗️ Retrosynthetic Planning</h3>
                <p>Automated retrosynthetic analysis for generated molecules. 
                Step-by-step synthesis routes with reagent suggestions and yield predictions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🎨 Interactive Visualizations</h3>
                <p>2D/3D molecular structures, property distribution plots, 
                correlation matrices, and comparative analysis charts.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>💾 Multiple Export Formats</h3>
                <p>Export results as CSV, JSON, SDF, or Excel. 
                Save configurations for reproducible workflows.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Stack
        st.markdown("## 🔧 Technical Stack")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <span class="tech-badge">Python 3.8+</span>
            <span class="tech-badge">RDKit</span>
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Plotly</span>
            <span class="tech-badge">NumPy</span>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">SciPy</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Workflow
        st.markdown("## 🔄 Workflow")
        
        workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
        
        with workflow_col1:
            st.markdown("""
            ### 1️⃣ Input
            - Upload protein structure (PDB)
            - Define binding site
            - Select target interactions
            - Configure parameters
            """)
        
        with workflow_col2:
            st.markdown("""
            ### 2️⃣ Generation
            - AI-driven molecule creation
            - Property optimization
            - Diversity screening
            - Iterative refinement
            """)
        
        with workflow_col3:
            st.markdown("""
            ### 3️⃣ Analysis
            - Score ranking
            - Property evaluation
            - Retrosynthetic analysis
            - Export results
            """)
        
        # Capabilities
        st.markdown("## 🎓 Capabilities")
        
        cap_tab1, cap_tab2, cap_tab3, cap_tab4 = st.tabs([
            "Molecular Generation", 
            "Optimization Methods", 
            "Property Prediction", 
            "Analysis Tools"
        ])
        
        with cap_tab1:
            st.markdown("""
            ### Molecular Generation
            
            **De Novo Design:**
            - Fragment-based assembly
            - Scaffold hopping
            - Bioisosteric replacement
            - Structure-activity relationship (SAR) exploration
            
            **Diversity Control:**
            - Tanimoto similarity filtering
            - Maximum common substructure (MCS) analysis
            - Chemical space exploration
            - Pharmacophore-guided generation
            
            **Constraints:**
            - Molecular weight limits
            - Ring system control
            - Functional group requirements
            - Synthetic accessibility filtering
            """)
        
        with cap_tab2:
            st.markdown("""
            ### Optimization Methods
            
            **Genetic Algorithm (GA):**
            - Population-based evolution
            - Crossover and mutation operators
            - Elitism for best candidates
            - Multi-generational refinement
            
            **Reinforcement Learning (RL):**
            - Policy gradient methods
            - Reward shaping for desired properties
            - Exploration-exploitation balance
            - Continuous improvement
            
            **Hybrid Approach:**
            - RL for initial exploration
            - GA for final optimization
            - Best of both strategies
            - Enhanced convergence
            """)
        
        with cap_tab3:
            st.markdown("""
            ### Property Prediction
            
            **Physicochemical Properties:**
            - Molecular weight
            - LogP (lipophilicity)
            - Topological polar surface area (TPSA)
            - Hydrogen bond donors/acceptors
            - Rotatable bonds
            
            **Drug-likeness Metrics:**
            - Lipinski's Rule of Five
            - Veber's rules
            - QED (Quantitative Estimate of Drug-likeness)
            - Synthetic accessibility score
            
            **ADMET Predictions:**
            - Blood-brain barrier permeability
            - Human intestinal absorption
            - CYP450 interactions
            - Toxicity predictions
            """)
        
        with cap_tab4:
            st.markdown("""
            ### Analysis Tools
            
            **Visualization:**
            - 2D structure rendering
            - Property distribution plots
            - Chemical space mapping
            - Correlation matrices
            - Radar charts for multi-property comparison
            
            **Retrosynthetic Analysis:**
            - Automated route planning
            - Reagent suggestions
            - Yield predictions
            - Difficulty assessment
            - Starting material identification
            
            **Comparison Tools:**
            - Side-by-side molecule comparison
            - Batch property analysis
            - Statistical summaries
            - Export to multiple formats
            """)
        
        # Use Cases
        st.markdown("## 🏥 Use Cases")
        
        st.markdown("""
        ### Drug Discovery
        - Lead identification and optimization
        - Fragment-to-lead evolution
        - Scaffold hopping for IP protection
        - Multi-parameter optimization
        
        ### Chemical Biology
        - Tool compound design
        - Probe molecule development
        - Target validation compounds
        - Selective inhibitor design
        
        ### Medicinal Chemistry
        - Hit-to-lead optimization
        - ADMET property improvement
        - Synthetic accessibility enhancement
        - Structure-activity relationship exploration
        
        ### Academic Research
        - Computational chemistry education
        - Method development and benchmarking
        - Virtual screening campaigns
        - Chemical space exploration
        """)
        
        # Getting Started
        st.markdown("## 🚀 Getting Started")
        
        with st.expander("📝 Quick Start Guide", expanded=False):
            st.markdown("""
            ### Step 1: Prepare Your Input
            1. Obtain a protein structure in PDB format
            2. Identify the binding site (coordinates or use co-crystallized ligand)
            3. Determine target interaction types
            
            ### Step 2: Configure Parameters
            1. Choose a preset configuration or customize settings
            2. Select optimization method (GA, RL, or Hybrid)
            3. Adjust scoring weights for your goals
            4. Set population size and generations
            
            ### Step 3: Run Generation
            1. Click "Run LigandForge"
            2. Monitor progress in real-time
            3. Wait for optimization to complete
            
            ### Step 4: Analyze Results
            1. Review top-scoring molecules
            2. Examine property distributions
            3. Optionally run retrosynthetic analysis
            4. Export results in desired format
            
            ### Step 5: Iterate
            1. Refine parameters based on results
            2. Adjust scoring weights
            3. Re-run with updated configuration
            4. Compare iterations
            """)
        
        with st.expander("⚙️ Configuration Tips", expanded=False):
            st.markdown("""
            ### Optimization Method Selection
            - **Fast Screening**: Use GA with smaller population (50-100)
            - **High Quality**: Use Hybrid with larger population (200+)
            - **Exploration**: Use RL for diverse chemical space
            
            ### Scoring Weight Adjustment
            - **Drug-like molecules**: Increase drug-likeness weight (0.3-0.4)
            - **Synthetic feasibility**: Increase synthetic score weight (0.25-0.35)
            - **Target binding**: Increase pharmacophore weight (0.3-0.4)
            - **Novelty**: Increase novelty score for unique structures (0.15-0.25)
            
            ### Performance Optimization
            - Start with smaller populations for testing
            - Use preset configurations as starting points
            - Monitor memory usage for large runs
            - Save configurations for reproducibility
            """)
        
        # System Requirements
        st.markdown("## 💻 System Requirements")
        
        req_col1, req_col2 = st.columns(2)
        
        with req_col1:
            st.markdown("""
            ### Minimum Requirements
            - **CPU**: 4 cores, 2.0 GHz
            - **RAM**: 8 GB
            - **Storage**: 2 GB free space
            - **OS**: Windows 10, macOS 10.14, or Linux
            - **Python**: 3.8 or higher
            """)
        
        with req_col2:
            st.markdown("""
            ### Recommended Configuration
            - **CPU**: 8+ cores, 3.0 GHz
            - **RAM**: 16 GB or more
            - **Storage**: 10 GB free space (for larger datasets)
            - **GPU**: Optional, for accelerated computations
            - **Python**: 3.9 or 3.10
            """)
        
        # Dependencies Status
        st.markdown("## 📦 Dependencies Status")
        
        status_data = {
            'Module': ['LigandForge Core', 'RDKit', 'Plotly', 'Retrosynthesis', 'Performance Monitoring'],
            'Status': [
                '✅ Available' if LIGANDFORGE_AVAILABLE else '❌ Not Available',
                '✅ Available' if RDKIT_AVAILABLE else '❌ Not Available',
                '✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available',
                '✅ Available' if RETROSYNTHESIS_AVAILABLE else '❌ Not Available',
                '✅ Available' if PSUTIL_AVAILABLE else '❌ Not Available'
            ],
            'Purpose': [
                'Core pipeline functionality',
                'Molecular structure handling',
                'Interactive visualizations',
                'Synthetic route planning',
                'System resource monitoring'
            ]
        }
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)
        

    def sidebar_info(self):
        st.header("Session Info")
        st.write(f"Session start: {datetime.fromtimestamp(st.session_state.session_start).strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Pipeline initialized: {'Yes' if st.session_state.pipeline else 'No'}")
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory().percent
            st.metric("Memory %", f"{mem}%")

    def main_interface(self):
        st.header("Input Structure")
        input_method = st.radio("Input Method:", ["Upload PDB File", "PDB ID"], horizontal=True)
        pdb_text = InputHandler.get_pdb_input(input_method)
        if not pdb_text:
            st.info("Please provide a PDB structure to proceed.")
            return

        st.header("Binding Site Selection")
        try:
            center, radius = self.setup_binding_site_selection(pdb_text)
        except Exception as e:
            UIComponents.display_error_with_context(e)
            return
        if center is None:
            st.warning("Please specify a binding site center.")
            return

        st.header("Target Interactions")
        target_interactions = st.multiselect(
            "Select target interaction types:",
            ["hbd", "hba", "hydrophobic", "aromatic", "electrostatic"],
            default=["hbd", "hba", "hydrophobic"]
        )
        if not target_interactions:
            st.warning("Please select at least one target interaction type.")
            return

        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                n_initial = st.number_input("Initial Population Size", 50, 1000, 200, step=10)
                include_voxel = st.checkbox("Include Voxel Analysis", value=False)
            with col2:
                binding_radius = st.number_input("Binding Site Radius (Å)", 5.0, 20.0, radius, step=0.5)

        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            run_button = st.button("Run LigandForge", type="primary")
        if run_button:
            self.run_pipeline(pdb_text, center, target_interactions, binding_radius, n_initial, include_voxel)

        if st.session_state.results:
            self.display_results()

    def get_atom_coordinates(self, atom):
        try:
            if hasattr(atom, '__getitem__'):
                return [float(atom['x']), float(atom['y']), float(atom['z'])]
            elif hasattr(atom, 'x') and hasattr(atom, 'y') and hasattr(atom, 'z'):
                return [float(atom.x), float(atom.y), float(atom.z)]
            elif hasattr(atom, 'coord'):
                coord = atom.coord
                if isinstance(coord, (list, tuple, np.ndarray)) and len(coord) >= 3:
                    return [float(coord[0]), float(coord[1]), float(coord[2])]
            elif hasattr(atom, 'position'):
                pos = atom.position
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 3:
                    return [float(pos[0]), float(pos[1]), float(pos[2])]
            else:
                raise ValidationError(f"Could not extract coordinates from atom: {type(atom)}",
                                      suggestions=["Check PDB file format"])
        except Exception as e:
            raise ValidationError("Error extracting atom coordinates", technical_details=str(e), suggestions=["Check PDB file format"])

    def setup_binding_site_selection(self, pdb_text: str) -> Tuple[Optional[np.ndarray], float]:
        ligand_names = []
        ligand_centers = {}
        try:
            if LIGANDFORGE_AVAILABLE:
                from pdb_parser import PDBParser
                parser = PDBParser()
                structure = parser.parse_pdb_structure(pdb_text)
                ligand_names = list(getattr(structure, "ligands", {}).keys()) if getattr(structure, "ligands", None) else []
                if ligand_names:
                    for name, atoms in structure.ligands.items():
                        if atoms:
                            try:
                                positions = []
                                for atom in atoms:
                                    coords = self.get_atom_coordinates(atom)
                                    positions.append(coords)
                                if positions:
                                    positions = np.array(positions)
                                    ligand_centers[name] = positions.mean(axis=0)
                                    st.success(f"Found ligand {name} with {len(positions)} atoms")
                            except Exception as e:
                                st.warning(f"Error processing ligand {name}: {e}")
                                continue
        except Exception:
            st.info("Using manual coordinate input as fallback")
            ligand_names = []
            ligand_centers = {}

        col1, col2 = st.columns([2,1])
        with col1:
            method = st.radio("Binding site determination:", ["Manual Coordinates", "Co-crystallized Ligand"], horizontal=True)
        with col2:
            if ligand_names:
                st.success(f"Found {len(ligand_names)} ligand(s)")
            else:
                st.info("No ligands detected")

        center = None
        radius = 10.0

        if method == "Co-crystallized Ligand":
            if not ligand_names:
                st.warning("No ligands found in structure. Please use manual coordinates.")
                method = "Manual Coordinates"
            else:
                col1, col2 = st.columns([2,1])
                with col1:
                    selected_ligand = st.selectbox("Select ligand:", ligand_names)
                with col2:
                    if selected_ligand in ligand_centers:
                        center = ligand_centers[selected_ligand]
                        st.info(f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] Å")
                    else:
                        st.error(f"Could not find center for ligand {selected_ligand}")

        if method == "Manual Coordinates" or center is None:
            st.subheader("Manual Coordinate Entry")
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            with col1:
                x = st.number_input("Center X (Å)", value=0.0, format="%.2f", step=0.1)
            with col2:
                y = st.number_input("Center Y (Å)", value=0.0, format="%.2f", step=0.1)
            with col3:
                z = st.number_input("Center Z (Å)", value=0.0, format="%.2f", step=0.1)
            with col4:
                if st.button("Validate Center"):
                    center_coords = np.array([x, y, z])
                    if np.linalg.norm(center_coords) > 200:
                        st.warning("Center is very far from origin")
                    else:
                        st.success("Center coordinates look reasonable")
            center = np.array([x, y, z])

        col1, col2 = st.columns([2,1])
        with col1:
            radius = st.slider("Binding site radius (Å)", 5.0, 20.0, 10.0, 0.5)
        with col2:
            volume = (4/3) * np.pi * radius**3
            st.metric("Volume", f"{volume:.0f} Å³")
        return center, radius

    def run_pipeline(self, pdb_text: str, center: np.ndarray, target_interactions: List[str],
                     radius: float, n_initial: int, include_voxel: bool):
        try:
            if st.session_state.pipeline is None:
                with st.spinner("Initializing LigandForge pipeline..."):
                    st.session_state.pipeline = LigandForgePipeline(st.session_state.config)
            progress_container = st.empty()
            st.session_state.progress_tracker.update({
                'start_time': time.time(),
                'current_stage': 'Initializing...',
                'overall_progress': 0.05
            })
            progress_placeholder = st.empty()
            with progress_placeholder:
                st.markdown("### ⏳ Running: Don't close this tab")
            opt_params = st.session_state.opt_params
            method_map = {
                "Genetic Algorithm": "ga",
                "Reinforcement Learning": "rl",
                "Hybrid (RL -> GA)": "hybrid"
            }
            opt_method = method_map.get(opt_params['opt_method'], 'hybrid')

            def update_progress(stage: str, progress: float):
                st.session_state.progress_tracker.update({
                    'current_stage': stage,
                    'overall_progress': progress
                })
                with progress_container:
                    UIComponents.create_progress_tracker()

            #update_progress("Analyzing binding site...", 0.10)
            #time.sleep(0.1)
            #update_progress("Generating initial molecules...", 0.30)
            #time.sleep(0.1)
            #update_progress(f"Running {opt_method.upper()} optimization...", 0.50)

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

            update_progress("Pipeline completed successfully!", 1.0)
            execution_time = time.time() - st.session_state.progress_tracker['start_time']
            results.execution_metadata = {
                'execution_time': execution_time,
                'optimization_method': opt_method,
                'target_interactions': target_interactions,
                'timestamp': datetime.now(),
                'config_used': st.session_state.config.__dict__.copy() if st.session_state.config else {},
                'opt_params_used': opt_params.copy()
            }
            st.session_state.results = results
            st.success(f"Pipeline completed successfully in {execution_time:.1f} seconds!")

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
                    success_rate = len([s for s in scores if s.total_score > 0.5]) / len(scores) * 100
                    st.metric("Success Rate", f"{success_rate:.0f}%")
            # Auto retrosynthesis if enabled
            if st.session_state.get('auto_retro_enabled') and st.session_state.get('auto_retro_count', 0) > 0:
                try:
                    analyzer = RetrosyntheticAnalyzer() if RETROSYNTHESIS_AVAILABLE else None
                    if analyzer and molecules:
                        top_k = min(st.session_state['auto_retro_count'], len(molecules))
                        routes = []
                        for i in range(top_k):
                            try:
                                route = analyzer.analyze_molecule(molecules[i], {})
                                routes.append(route)
                            except Exception:
                                continue
                        if routes:
                            st.session_state['retro_routes'] = routes
                            st.session_state['retro_molecules'] = list(range(top_k))
                except Exception:
                    pass

        except Exception as e:
            st.session_state.progress_tracker.update({'current_stage': 'Error occurred', 'overall_progress': 0})
            UIComponents.display_error_with_context(PipelineError(
                "Pipeline execution failed",
                technical_details=str(e),
                suggestions=[
                    "Check your input parameters",
                    "Try with a smaller population size",
                    "Use a different optimization method",
                    "Verify your PDB structure is valid"
                ]
            ))

    # ---- Display Results (single authoritative implementation with retrosynthesis tab) ----
    def display_results(self):
        if not st.session_state.results:
            return
        results = st.session_state.results
        molecules = results.generation_results.molecules
        scores = results.generation_results.scores
        if not molecules or not scores:
            st.error("No valid results to display")
            return
        results_data = []
        for i, (mol, score) in enumerate(zip(molecules, scores)):
            try:
                if RDKIT_AVAILABLE:
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
                        'Novelty_Score': score.novelty_score
                    }
                else:
                    row = {
                        'ID': f"MOL_{i+1:03d}",
                        'Total_Score': score.total_score,
                        'Pharmacophore_Score': score.pharmacophore_score,
                        'Drug_Likeness_Score': score.drug_likeness_score,
                        'Synthetic_Score': score.synthetic_score,
                        'Novelty_Score': score.novelty_score
                    }
                results_data.append(row)
            except Exception as e:
                st.warning(f"Error processing molecule {i+1}: {e}")
                continue
        if not results_data:
            st.error("No valid results to display")
            return
        results_df = pd.DataFrame(results_data).sort_values('Total_Score', ascending=False).reset_index(drop=True)

        # 6 tabs including retrosynthesis
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Results Overview", "Top Molecules", "Structure Grid", "Visualizations", "Retrosynthesis", "Analysis Report"
        ])
        with tab1:
            self.display_results_overview(results_df, results, molecules)
        with tab2:
            self.display_top_molecules(results_df, molecules, scores)
        with tab3:
            self.display_structure_grid(molecules, scores)
        with tab4:
            self.display_visualizations(results_df, molecules, results)
        with tab5:
            self.display_retrosynthesis_tab(molecules, scores)
        with tab6:
            self.display_analysis_report(results)

    # ---- Results Overview ----
    def display_results_overview(self, df: pd.DataFrame, results, molecules):
        st.header("Results Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generation Statistics")
            gen_stats = getattr(results.generation_results, "generation_statistics", {})
            UIComponents.create_metric_container("Generation Method", gen_stats.get('optimization_method', 'N/A'))
            UIComponents.create_metric_container("Molecules Generated", str(len(df)))
            try:
                initial_molecules = gen_stats.get('initial_molecules', len(df))
                UIComponents.create_metric_container("Success Rate", f"{len(df)/initial_molecules*100:.1f}%")
            except Exception:
                pass
        with col2:
            st.subheader("Score Distribution")
            if 'Total_Score' in df.columns:
                scores = df['Total_Score']
                UIComponents.create_metric_container("Best Score", f"{scores.max():.3f}")
                UIComponents.create_metric_container("Average Score", f"{scores.mean():.3f}")
                UIComponents.create_metric_container("Score Range", f"{scores.min():.3f} - {scores.max():.3f}")
        st.subheader("Results Table")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_rows = st.selectbox("Rows to display:", [10, 25, 50], index=0)
        with col2:
            sort_by = st.selectbox("Sort by:", [col for col in df.columns if col != 'ID'], index=0)
        with col3:
            ascending = st.checkbox("Ascending order", value=False)
        with col4:
            show_structures = st.checkbox("Show 2D Structures", value=st.session_state.display_options['show_structures'])
        display_df = df.head(max_rows).sort_values(sort_by, ascending=ascending)
        if show_structures and RDKIT_AVAILABLE and 'SMILES' in display_df.columns:
            try:
                enhanced_df, _ = create_enhanced_results_display(display_df, molecules)
                st.dataframe(enhanced_df, use_container_width=True, height=400)
            except Exception:
                st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.dataframe(display_df, use_container_width=True, height=400)
        self.display_export_options(df, results)

    # ---- Structure Grid ----
    def display_structure_grid(self, molecules: List, scores: List):
        st.header("Molecular Structure Grid")
        if not RDKIT_AVAILABLE:
            st.warning("Structure grid requires RDKit. Please install RDKit to view molecular structures.")
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            max_display = st.slider("Max molecules to display:", 4, 24, st.session_state.display_options['max_structures_grid'])
        with col2:
            structure_size = st.slider("Structure size:", 100, 300, st.session_state.display_options['structure_size'])
        with col3:
            sort_by_score = st.checkbox("Sort by score", value=True)
        if sort_by_score:
            mol_score_pairs = list(zip(molecules, scores))
            mol_score_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
            display_molecules = [pair[0] for pair in mol_score_pairs[:max_display]]
            display_scores = [pair[1] for pair in mol_score_pairs[:max_display]]
        else:
            display_molecules = molecules[:max_display]
            display_scores = scores[:max_display]
        UIComponents.display_molecular_structure_grid(display_molecules, display_scores, max_display)
        st.subheader("Detailed Molecule View")
        mol_options = [f"MOL_{i+1:03d} - Score: {score.total_score:.3f}" for i, score in enumerate(display_scores)]
        if mol_options:
            selected_mol = st.selectbox("Select molecule for detailed view:", mol_options)
            mol_idx = int(selected_mol.split()[0].split('_')[1]) - 1
            if mol_idx < len(display_molecules):
                mol = display_molecules[mol_idx]
                score = display_scores[mol_idx]
                col1, col2 = st.columns([1,1])
                with col1:
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        img_data = generate_2d_structure_image(smiles, size=(400,400))
                        if img_data:
                            st.markdown(f"""
                                <div style="text-align:center; border:2px solid #1f77b4; padding:20px; border-radius:10px; background:white;">
                                    <img src="{img_data}" style="max-width:100%; height:auto;">
                                    <h4 style="margin:10px 0; color:#1f77b4;">MOL_{mol_idx+1:03d}</h4>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Could not generate structure image")
                    except Exception as e:
                        st.error(f"Error displaying structure: {e}")
                with col2:
                    st.subheader("Molecular Properties")
                    try:
                        mw = rdMolDescriptors.CalcExactMolWt(mol)
                        logp = Crippen.MolLogP(mol)
                        hbd = Lipinski.NumHDonors(mol)
                        hba = Lipinski.NumHAcceptors(mol)
                        tpsa = rdMolDescriptors.CalcTPSA(mol)
                        qed = QED.qed(mol)
                        st.write(f"**SMILES:** `{Chem.MolToSmiles(mol)}`")
                        st.write(f"**Molecular Weight:** {mw:.2f} Da")
                        st.write(f"**LogP:** {logp:.2f}")
                        st.write(f"**H-Bond Donors:** {hbd}")
                        st.write(f"**H-Bond Acceptors:** {hba}")
                        st.write(f"**TPSA:** {tpsa:.2f}")
                        st.write(f"**QED:** {qed:.3f}")
                        st.subheader("Score Breakdown")
                        st.write(f"**Total Score:** {score.total_score:.3f}")
                        st.write(f"**Pharmacophore:** {score.pharmacophore_score:.3f}")
                        st.write(f"**Drug-likeness:** {score.drug_likeness_score:.3f}")
                        st.write(f"**Synthetic:** {score.synthetic_score:.3f}")
                        st.write(f"**Novelty:** {score.novelty_score:.3f}")
                    except Exception as e:
                        st.error(f"Error calculating properties: {e}")

    # ---- Top molecules ----
    def display_top_molecules(self, df: pd.DataFrame, molecules, scores):
        st.header("Top Molecules")
        if len(df) == 0:
            st.warning("No molecules available for analysis")
            return
        col1, col2 = st.columns([2,1])
        with col1:
            mol_options = [f"{row['ID']} - Score: {row['Total_Score']:.3f}" for _, row in df.head(10).iterrows()]
            selected_mol = st.selectbox("Select molecule for detailed analysis:", mol_options)
        with col2:
            mol_idx = int(selected_mol.split()[0].split('_')[1]) - 1
            if mol_idx < len(molecules) and RDKIT_AVAILABLE:
                try:
                    mol = molecules[mol_idx]
                    smiles = Chem.MolToSmiles(mol)
                    img_data = generate_2d_structure_image(smiles, size=(150,150))
                    if img_data:
                        st.markdown(f"""<div style="text-align:center; border:1px solid #ddd; padding:10px; border-radius:5px;">
                                        <img src="{img_data}" style="max-width:100%; height:auto;">
                                        <p style="font-size:12px; margin:5px 0;">Preview</p></div>""", unsafe_allow_html=True)
                except Exception:
                    st.info("Preview not available")
        if mol_idx < len(molecules):
            mol = molecules[mol_idx]
            score = scores[mol_idx]
            row_data = df.iloc[mol_idx] if mol_idx < len(df) else {}
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Molecular Information")
                if RDKIT_AVAILABLE and 'SMILES' in row_data:
                    st.code(f"SMILES: {row_data['SMILES']}")
                    if 'MW' in row_data:
                        st.write(f"**Molecular Weight:** {row_data['MW']:.2f} Da")
                    if 'LogP' in row_data:
                        st.write(f"**LogP:** {row_data['LogP']:.2f}")
                    if 'QED' in row_data:
                        st.write(f"**QED:** {row_data['QED']:.3f}")
                st.subheader("Score Breakdown")
                score_data = {
                    'Total': score.total_score,
                    'Pharmacophore': score.pharmacophore_score,
                    'Drug-likeness': score.drug_likeness_score,
                    'Synthetic': score.synthetic_score,
                    'Novelty': score.novelty_score
                }
                for name, value in score_data.items():
                    st.write(f"**{name} Score:** {value:.3f}")
            with col2:
                if PLOTLY_AVAILABLE:
                    try:
                        fig = go.Figure(data=go.Scatterpolar(r=list(score_data.values()), theta=list(score_data.keys()), fill='toself', name='Score Profile'))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False, title="Score Profile", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.write("Score visualization not available")
        st.subheader("Top Molecules Ranking")
        ranking_metric = st.selectbox("Rank by:", [col for col in df.columns if 'Score' in col], index=0)
        top_n = st.number_input("Show top N:", 5, 50, 10)
        top_molecules = df.nlargest(top_n, ranking_metric)
        top_molecules_display = top_molecules.copy()
        top_molecules_display.insert(0, 'Rank', range(1, len(top_molecules_display) + 1))
        st.dataframe(top_molecules_display, use_container_width=True)

    # ---- Visualizations ----
    def display_visualizations(self, df: pd.DataFrame, molecules, results):
        st.header("Visualizations")
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly not available - visualizations disabled")
            return
        if 'Total_Score' in df.columns:
            try:
                fig = create_score_distribution_plot(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create score distribution plot: {e}")
        if RDKIT_AVAILABLE and all(col in df.columns for col in ['MW', 'LogP', 'Total_Score']):
            try:
                fig = create_property_space_plot(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create property space plot: {e}")
        score_columns = [col for col in df.columns if 'Score' in col]
        if len(score_columns) > 1:
            try:
                fig = create_property_correlation_matrix(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create correlation matrix: {e}")

    # ---- Analysis Report ----
    def display_analysis_report(self, results):
        st.header("Analysis Report")
        if st.button("Generate Detailed Report"):
            try:
                report_sections = []
                report_sections.extend(["="*80, "LIGANDFORGE 2.0 - ANALYSIS REPORT", "="*80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""])
                molecules = results.generation_results.molecules
                scores = results.generation_results.scores
                if molecules and scores:
                    best_score = max(s.total_score for s in scores)
                    avg_score = np.mean([s.total_score for s in scores])
                    success_rate = len([s for s in scores if s.total_score > 0.5]) / len(scores) * 100
                    report_sections.extend(["EXECUTIVE SUMMARY", "-"*50, f"Total Molecules Generated: {len(molecules)}", f"Best Score Achieved: {best_score:.3f}", f"Average Score: {avg_score:.3f}", f"Success Rate (Score >0.5): {success_rate:.1f}%", ""])
                if hasattr(results, 'execution_metadata'):
                    metadata = results.execution_metadata
                    report_sections.extend(["CONFIGURATION DETAILS", "-"*50, f"Optimization Method: {metadata.get('optimization_method', 'N/A')}", f"Execution Time: {metadata.get('execution_time', 0):.1f} seconds", f"Target Interactions: {', '.join(metadata.get('target_interactions', []))}", ""])
                if molecules and scores:
                    report_sections.extend(["TOP MOLECULES", "-"*50])
                    mol_score_pairs = list(zip(molecules, scores))
                    mol_score_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
                    for i, (mol, score) in enumerate(mol_score_pairs[:5], 1):
                        try:
                            if RDKIT_AVAILABLE:
                                smiles = Chem.MolToSmiles(mol)
                                report_sections.extend([f"Molecule {i}:", f"  SMILES: {smiles}", f"  Total Score: {score.total_score:.3f}", ""])
                            else:
                                report_sections.extend([f"Molecule {i}:", f"  Total Score: {score.total_score:.3f}", ""])
                        except Exception:
                            report_sections.append(f"  Error processing molecule {i}")
                report_sections.extend(["="*80, "End of Report", "="*80])
                full_report = "\n".join(report_sections)
                st.text_area("Generated Report", full_report, height=600)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button("Download Report", full_report, f"ligandforge_report_{timestamp}.txt", "text/plain")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")
        else:
            st.info("Click 'Generate Detailed Report' to create a comprehensive analysis report")

    # ---- Retrosynthesis tab ----
    def display_retrosynthesis_tab(self, molecules: List, scores: List):
        st.header("🔬 Retrosynthetic Analysis")
        if not RETROSYNTHESIS_AVAILABLE:
            st.error("Retrosynthesis module not available. Please ensure retrosynthesis_module.py is installed.")
            return
        if not RDKIT_AVAILABLE:
            st.error("RDKit is required for retrosynthetic analysis.")
            return
        if not molecules:
            st.warning("No molecules available for retrosynthetic analysis.")
            return
        try:
            analyzer = RetrosyntheticAnalyzer()
        except Exception as e:
            st.error(f"Failed to initialize retrosynthetic analyzer: {e}")
            return
        st.subheader("Select Molecules for Analysis")
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            selection_method = st.radio("Selection method:", ["Top N molecules", "Specific molecule", "All molecules (top 10)"], horizontal=True)
        with col2:
            if selection_method == "Top N molecules":
                n_molecules = st.number_input("Number of molecules:", 1, min(20, len(molecules)), 5)
            elif selection_method == "Specific molecule":
                n_molecules = 1
        with col3:
            if selection_method != "All molecules (top 10)":
                analyze_button = st.button("Analyze Routes", type="primary")
            else:
                analyze_button = st.button("Analyze Top 10", type="primary")
                n_molecules = min(10, len(molecules))
        selected_indices = []
        if selection_method == "Specific molecule":
            mol_options = [f"MOL_{i+1:03d} - Score: {score.total_score:.3f}" for i, score in enumerate(scores[:20])]
            selected_mol = st.selectbox("Select molecule:", mol_options)
            selected_idx = int(selected_mol.split()[0].split('_')[1]) - 1
            selected_indices = [selected_idx]
        else:
            selected_indices = list(range(min(n_molecules, len(molecules))))
        if analyze_button:
            with st.spinner(f"Analyzing retrosynthetic routes for {len(selected_indices)} molecule(s)..."):
                routes = []
                analyzed_molecules = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, idx in enumerate(selected_indices):
                    try:
                        status_text.text(f"Analyzing molecule {i+1}/{len(selected_indices)}...")
                        mol = molecules[idx]
                        metadata = None
                        if hasattr(mol, 'GetPropsAsDict'):
                            metadata = mol.GetPropsAsDict()
                        route = analyzer.analyze_molecule(mol, metadata)
                        routes.append(route)
                        analyzed_molecules.append(idx)
                        progress_bar.progress((i+1)/len(selected_indices))
                    except Exception as e:
                        st.warning(f"Error analyzing molecule {idx+1}: {e}")
                        continue
                progress_bar.empty()
                status_text.empty()
                if not routes:
                    st.error("No routes could be generated. Please check your molecules.")
                    return
                st.session_state['retro_routes'] = routes
                st.session_state['retro_molecules'] = analyzed_molecules
                st.success(f"Successfully analyzed {len(routes)} molecule(s)!")
        if 'retro_routes' in st.session_state and st.session_state['retro_routes']:
            routes = st.session_state['retro_routes']
            analyzed_molecules = st.session_state['retro_molecules']
            st.markdown("---")
            st.subheader("📊 Route Analysis Results")
            if len(routes) > 1:
                st.markdown("### Route Comparison")
                create_retrosynthetic_comparison(routes, analyzed_molecules)
                st.markdown("---")
            for i, (route, mol_idx) in enumerate(zip(routes, analyzed_molecules)):
                with st.expander(f"🔬 Molecule {mol_idx+1} Retrosynthetic Route ({getattr(route,'total_steps', '?')} steps)", expanded=(len(routes) == 1)):
                    display_retrosynthetic_route(route, mol_idx)
            st.markdown("---")
            st.subheader("💾 Export Retrosynthetic Routes")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📄 Export Text Summary"):
                    summaries = []
                    for route, mol_idx in zip(routes, analyzed_molecules):
                        summaries.append("\n" + "="*80 + "\n")
                        summaries.append(f"MOLECULE {mol_idx+1} RETROSYNTHETIC ROUTE\n")
                        summaries.append("="*80 + "\n")
                        try:
                            summaries.append(analyzer.generate_route_summary(route))
                        except Exception:
                            summaries.append(str(route))
                    full_summary = "\n".join(summaries)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button("Download Text Report", full_summary, f"retrosynthetic_routes_{timestamp}.txt", "text/plain")
            with col2:
                if st.button("📊 Export Excel"):
                    excel_data = export_routes_to_excel(routes, analyzed_molecules)
                    if excel_data:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button("Download Excel File", excel_data, f"retrosynthetic_routes_{timestamp}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col3:
                if st.button("🎨 Export Graphviz"):
                    dot_strings = []
                    for route, mol_idx in zip(routes, analyzed_molecules):
                        try:
                            dot = analyzer.visualize_route_graphviz(route)
                            if dot:
                                dot_strings.append(f"// Molecule {mol_idx+1}\n{dot}\n\n")
                        except Exception:
                            continue
                    if dot_strings:
                        combined_dot = "".join(dot_strings)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button("Download DOT Files", combined_dot, f"retrosynthetic_routes_{timestamp}.dot", "text/plain")

    # ---- Export Options used in results overview ----
    def display_export_options(self, df: pd.DataFrame, results):
        st.subheader("Export Options")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Export CSV"):
                try:
                    csv_data = df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button("Download CSV", csv_data, f"ligandforge_results_{timestamp}.csv", "text/csv")
                except Exception as e:
                    st.error(f"CSV export failed: {e}")
        with col2:
            if st.button("Export JSON"):
                try:
                    export_data = {'timestamp': datetime.now().isoformat(), 'results': df.to_dict('records'), 'metadata': getattr(results, 'execution_metadata', {})}
                    json_data = json.dumps(export_data, indent=2, default=str)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button("Download JSON", json_data, f"ligandforge_results_{timestamp}.json", "application/json")
                except Exception as e:
                    st.error(f"JSON export failed: {e}")
        with col3:
            if st.button("Export SDF") and RDKIT_AVAILABLE:
                try:
                    molecules = results.generation_results.molecules
                    results_data = df.to_dict('records')
                    sdf_data = create_sdf_from_molecules(molecules, results_data)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button("Download SDF", sdf_data, f"ligandforge_structures_{timestamp}.sdf", "chemical/x-mdl-sdfile")
                except Exception as e:
                    st.error(f"SDF export failed: {e}")
        with col4:
            if st.button("Export Config"):
                try:
                    config_data = {
                        'config': st.session_state.config.__dict__.copy() if st.session_state.config else {},
                        'opt_params': st.session_state.opt_params.copy(),
                        'display_options': st.session_state.display_options.copy(),
                        'timestamp': datetime.now().isoformat()
                    }
                    config_json = json.dumps(config_data, indent=2, default=str)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button("Download Config", config_json, f"ligandforge_config_{timestamp}.json", "application/json")
                except Exception as e:
                    st.error(f"Config export failed: {e}")

# ------------------------
# Entry point
# ------------------------

def main():
    try:
        app = LigandForgeApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your installation and dependencies")

if __name__ == "__main__":
    main()

