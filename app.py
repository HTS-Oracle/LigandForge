"""
Enhanced LigandForge Streamlit Application
Now with 2D molecular structure visualization in results tables
"""

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
except ImportError as e:
    LIGANDFORGE_AVAILABLE = False
    st.error(f"LigandForge modules not available: {e}")

# Optional visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available - some visualizations disabled")

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski, QED
    from PIL import Image
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("RDKit not available - molecular analysis limited")


class LigandForgeError(Exception):
    """Custom exception with user-friendly messages"""
    def __init__(self, message: str, technical_details: Optional[str] = None, 
                 suggestions: Optional[List[str]] = None):
        self.message = message
        self.technical_details = technical_details
        self.suggestions = suggestions or []
        super().__init__(self.message)


class ValidationError(LigandForgeError):
    """Input validation errors"""
    pass


class PipelineError(LigandForgeError):
    """Pipeline execution errors"""
    pass


# Configuration and State Management
class AppState:
    """Centralized application state management"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state with defaults"""
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


# Input Validation
class InputValidator:
    """Centralized input validation"""
    
    @staticmethod
    def clean_smiles(smiles: str) -> str:
        """Clean SMILES string"""
        if not smiles:
            return ""
        cleaned = re.sub(r'[^A-Za-z0-9\[\]()=#+\-@/\\.\s]', '', smiles)
        return cleaned.strip()
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float, name: str):
        """Validate numeric inputs"""
        if not min_val <= value <= max_val:
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val}",
                suggestions=[f"Use a value between {min_val} and {max_val}"]
            )
    
    @staticmethod
    def validate_pdb_content(pdb_text: str) -> str:
        """Validate and clean PDB content"""
        if not pdb_text or not pdb_text.strip():
            raise ValidationError(
                "PDB content is empty",
                suggestions=["Upload a valid PDB file", "Use the sample structure"]
            )
        
        if "ATOM" not in pdb_text and "HETATM" not in pdb_text:
            raise ValidationError(
                "Invalid PDB format - no ATOM/HETATM records found",
                suggestions=[
                    "Ensure the file is in PDB format",
                    "Check file encoding (should be UTF-8)",
                    "Try a different PDB file"
                ]
            )
        
        # Basic sanitization
        valid_records = {'ATOM', 'HETATM', 'HEADER', 'TITLE', 'COMPND', 'SOURCE', 
                        'KEYWDS', 'EXPDTA', 'REVDAT', 'REMARK', 'SEQRES', 'HET',
                        'HETNAM', 'HETSYN', 'FORMUL', 'SHEET', 'HELIX', 'LINK',
                        'SSBOND', 'SITE', 'CRYST1', 'SCALE', 'MASTER', 'END'}
        
        lines = pdb_text.split('\n')
        cleaned_lines = [line for line in lines 
                        if any(line.startswith(record) for record in valid_records)]
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def validate_pdb_id(pdb_id: str) -> bool:
        """Validate PDB ID format"""
        if not pdb_id or len(pdb_id) != 4:
            return False
        return re.match(r'^[0-9][a-zA-Z0-9]{3}$', pdb_id) is not None
    
    @staticmethod
    def validate_file_size(uploaded_file, max_size_mb: int = 50) -> bool:
        """Validate uploaded file size"""
        if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size_mb * 1024 * 1024:
            st.error(f"File too large. Maximum size: {max_size_mb}MB")
            return False
        return True


# UI Components
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def display_error_with_context(error: Exception):
        """Enhanced error display with actionable suggestions"""
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
        """Multi-stage progress tracking with ETA"""
        tracker = st.session_state.progress_tracker
        
        if tracker['start_time'] and tracker['overall_progress'] > 0:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.progress(tracker['overall_progress'])
                st.caption(tracker['current_stage'])
            
            with col2:
                elapsed = time.time() - tracker['start_time']
                if tracker['overall_progress'] > 0.05:
                    eta = (elapsed / tracker['overall_progress']) * (1 - tracker['overall_progress'])
                    st.metric("ETA", f"{eta:.0f}s")
                else:
                    st.metric("Elapsed", f"{elapsed:.0f}s")
    
    @staticmethod
    def create_metric_container(title: str, value: str, delta: Optional[str] = None):
        """Create styled metric container"""
        delta_html = f"<br><small style='color: #888;'>{delta}</small>" if delta else ""
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f0f2f6, #e8ecf0);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        ">
            <strong>{title}</strong><br>
            <span style="font-size: 1.2em;">{value}</span>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_molecular_structure_grid(molecules: List, scores: List, max_display: int = 12):
        """Display a grid of molecular structures with scores"""
        if not RDKIT_AVAILABLE or not molecules:
            st.info("Molecular structure display requires RDKit")
            return
        
        try:
            # Limit number of molecules to display
            display_molecules = molecules[:max_display]
            display_scores = scores[:max_display]
            
            # Create grid layout
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
                            # Generate SMILES for structure display
                            try:
                                smiles = Chem.MolToSmiles(mol)
                                img_data = generate_2d_structure_image(smiles, size=(150, 150))
                                
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


# Performance Monitoring
class PerformanceMonitor:
    """Monitor application performance"""
    
    @staticmethod
    def update_performance_metrics():
        """Update performance metrics if psutil available"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            memory_usage = psutil.virtual_memory().percent
            st.session_state.performance_metrics['memory_usage'].append({
                'timestamp': time.time(),
                'usage': memory_usage
            })
            
            # Keep only last 100 measurements
            if len(st.session_state.performance_metrics['memory_usage']) > 100:
                st.session_state.performance_metrics['memory_usage'] = \
                    st.session_state.performance_metrics['memory_usage'][-100:]
        except Exception:
            pass  # Silently fail if monitoring unavailable


# Input Handlers
class InputHandler:
    """Handle different input methods"""
    
    @staticmethod
    def get_pdb_input(input_method: str) -> Optional[str]:
        """Get PDB input based on selected method"""
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
        """Handle PDB file upload"""
        uploaded_file = st.file_uploader(
            "Upload PDB file", 
            type=['pdb'],
            help="Upload a protein structure in PDB format"
        )
        
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
        """Handle PDB ID input"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pdb_id = st.text_input(
                "Enter PDB ID (e.g., 1abc):",
                help="4-character PDB identifier"
            )
        
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
        """Handle sample structure"""
        col1, col2 = st.columns([1, 1])
        
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


# Configuration Manager (keeping the same from original)
class ConfigurationManager:
    """Manage application configuration"""
    
    @staticmethod
    def setup_sidebar():
        """Setup sidebar configuration with display options"""
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
    
    @staticmethod
    def _setup_display_options():
        """Setup display options for molecular structures"""
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
        """Setup preset configurations"""
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
                    st.rerun()
                except Exception as e:
                    UIComponents.display_error_with_context(ValidationError(
                        f"Failed to apply {preset_type} preset",
                        technical_details=str(e),
                        suggestions=["Try a different preset", "Use custom configuration"]
                    ))
    
    @staticmethod
    def _setup_generation_params():
        """Setup generation parameters"""
        with st.sidebar.expander("Generation Parameters"):
            config = st.session_state.config
            
            max_atoms = st.slider(
                "Max Heavy Atoms", 15, 50, config.max_heavy_atoms,
                help="Maximum number of heavy (non-hydrogen) atoms"
            )
            min_atoms = st.slider(
                "Min Heavy Atoms", 10, 25, config.min_heavy_atoms,
                help="Minimum number of heavy (non-hydrogen) atoms"
            )
            
            if max_atoms < min_atoms:
                st.error("Max atoms must be greater than min atoms")
            else:
                config.max_heavy_atoms = max_atoms
                config.min_heavy_atoms = min_atoms
            
            config.max_rings = st.slider(
                "Max Rings", 1, 8, config.max_rings,
                help="Maximum number of ring systems"
            )
            config.diversity_threshold = st.slider(
                "Diversity Threshold", 0.3, 0.9, config.diversity_threshold,
                help="Minimum structural diversity required"
            )
    
    @staticmethod
    def _setup_scoring_weights():
        """Setup scoring weights"""
        with st.sidebar.expander("Scoring Weights"):
            config = st.session_state.config
            st.write("**Weight Distribution:**")
            
            weights = {}
            total_weight = 0.0
            
            for component in config.reward_weights.keys():
                weight_value = st.slider(
                    component.replace('_', ' ').title(),
                    0.0, 1.0, config.reward_weights[component],
                    step=0.05,
                    help=f"Importance of {component.replace('_', ' ')} in scoring"
                )
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
                        st.rerun()
            
            with col2:
                if st.button("Reset to Default"):
                    config.reward_weights = LigandForgeConfig().reward_weights
                    st.success("Weights reset to default!")
                    st.rerun()
    
    @staticmethod
    def _setup_optimization_params():
        """Setup optimization parameters"""
        with st.sidebar.expander("Optimization Parameters"):
            opt_params = st.session_state.opt_params
            
            opt_params['opt_method'] = st.selectbox(
                "Optimization Method",
                ["Hybrid (RL -> GA)", "Genetic Algorithm", "Reinforcement Learning"],
                index=["Hybrid (RL -> GA)", "Genetic Algorithm", "Reinforcement Learning"].index(opt_params['opt_method']),
                help="Choose optimization strategy for molecular generation"
            )
            
            # GA parameters
            st.subheader("Genetic Algorithm")
            population = st.number_input(
                "Population Size", 50, 500, opt_params['ga_population'], step=10,
                help="Number of molecules per generation"
            )
            generations = st.number_input(
                "Generations", 10, 100, opt_params['ga_generations'], step=5,
                help="Number of evolutionary cycles"
            )
            
            # Show estimated time
            estimated_time = (population * generations) / 1000
            st.caption(f"Estimated time: ~{estimated_time:.1f} minutes")
            
            opt_params['ga_population'] = population
            opt_params['ga_generations'] = generations
            
            opt_params['ga_crossover'] = st.slider(
                "Crossover Rate", 0.0, 1.0, opt_params['ga_crossover'],
                help="Probability of combining parent molecules"
            )
            opt_params['ga_mutation'] = st.slider(
                "Mutation Rate", 0.0, 1.0, opt_params['ga_mutation'],
                help="Probability of random modifications"
            )
            opt_params['ga_elitism'] = st.number_input(
                "Elitism", 0, 50, opt_params['ga_elitism'], step=1,
                help="Number of best molecules to preserve each generation"
            )
            
            # RL parameters
            st.subheader("Reinforcement Learning")
            opt_params['rl_iterations'] = st.number_input(
                "RL Iterations", 5, 50, opt_params['rl_iterations'], step=1,
                help="Number of policy improvement cycles"
            )
    
    @staticmethod
    def _setup_config_management():
        """Setup configuration save/load"""
        st.sidebar.subheader("Configuration Management")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Save Config"):
                try:
                    config_dict = {
                        'config': st.session_state.config.__dict__.copy(),
                        'opt_params': st.session_state.opt_params.copy(),
                        'display_options': st.session_state.display_options.copy(),
                        'timestamp': datetime.now().isoformat(),
                        'version': '2.0'
                    }
                    config_json = json.dumps(config_dict, indent=2, default=str)
                    st.download_button(
                        "Download Config",
                        config_json,
                        f"ligandforge_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")
        
        with col2:
            uploaded_config = st.file_uploader("Load Config", type=['json'])
            
            if uploaded_config:
                try:
                    config_data = json.load(uploaded_config)
                    
                    if 'config' in config_data and 'opt_params' in config_data:
                        for key, value in config_data['config'].items():
                            if hasattr(st.session_state.config, key):
                                setattr(st.session_state.config, key, value)
                        
                        st.session_state.opt_params.update(config_data['opt_params'])
                        
                        if 'display_options' in config_data:
                            st.session_state.display_options.update(config_data['display_options'])
                        
                        st.success("Configuration loaded!")
                        st.rerun()
                    else:
                        st.error("Invalid configuration file format")
                except Exception as e:
                    st.error(f"Failed to load configuration: {e}")


# Main Application (keeping most of the same structure but enhancing results display)
class LigandForgeApp:
    """Main application class with enhanced structure visualization"""
    
    def __init__(self):
        self.state = AppState()
        self.setup_page_config()
        self.add_custom_css()
    
    def setup_page_config(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="LigandForge 2.0",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def add_custom_css(self):
        """Add custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .metric-container {
            background: linear-gradient(135deg, #f0f2f6, #e8ecf0);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        }
        .molecule-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .molecule-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .molecule-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        .structure-img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .success-box {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #28a745;
        }
        .error-box {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #dc3545;
        }
        .info-box {
            background: linear-gradient(135deg, #d1ecf1, #bee5eb);
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #17a2b8;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<h1 class="main-header">LigandForge 2.0</h1>', unsafe_allow_html=True)
        st.markdown("""
        **Advanced AI-Driven Structure-Based Drug Design Platform**
        
        Combining structure-based drug design, AI-guided fragment assembly, 
        and multi-objective optimization for intelligent molecular design.
        """)
        
        # Check if core modules are available
        if not LIGANDFORGE_AVAILABLE:
            st.error("LigandForge core modules are not available. Please install required dependencies.")
            st.stop()
        
        # Monitor performance
        PerformanceMonitor.update_performance_metrics()
        
        # Setup sidebar
        ConfigurationManager.setup_sidebar()
        
        # Main interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.main_interface()
        
        with col2:
            self.sidebar_info()
    
    def main_interface(self):
        """Main interface components"""
        st.header("Input Structure")
        
        # Structure input
        input_method = st.radio(
            "Input Method:", 
            ["Upload PDB File", "PDB ID", "Sample Structure"],
            horizontal=True,
            help="Choose how to provide the protein structure"
        )
        
        pdb_text = InputHandler.get_pdb_input(input_method)
        
        if not pdb_text:
            st.info("Please provide a PDB structure to proceed.")
            return
        
        # Binding site selection
        st.header("Binding Site Selection")
        try:
            center, radius = self.setup_binding_site_selection(pdb_text)
        except Exception as e:
            UIComponents.display_error_with_context(e)
            return
        
        if center is None:
            st.warning("Please specify a binding site center.")
            return
        
        # Target interactions
        st.header("Target Interactions")
        target_interactions = st.multiselect(
            "Select target interaction types:",
            ["hbd", "hba", "hydrophobic", "aromatic", "electrostatic"],
            default=["hbd", "hba", "hydrophobic"],
            help="Choose the types of interactions you want to target for optimization"
        )
        
        if not target_interactions:
            st.warning("Please select at least one target interaction type.")
            return
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                n_initial = st.number_input(
                    "Initial Population Size", 50, 1000, 200, step=10,
                    help="Number of initial molecules to generate"
                )
                include_voxel = st.checkbox(
                    "Include Voxel Analysis", value=False,
                    help="Perform detailed 3D field analysis (slower but more accurate)"
                )
            
            with col2:
                binding_radius = st.number_input(
                    "Binding Site Radius (Å)", 5.0, 20.0, radius, step=0.5,
                    help="Radius around the binding site center to consider"
                )
        
        # Run button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            run_button = st.button(
                "Run LigandForge", 
                type="primary", 
                use_container_width=True,
                help="Start the molecular generation pipeline"
            )
        
        if run_button:
            self.run_pipeline(pdb_text, center, target_interactions, binding_radius, n_initial, include_voxel)
        
        # Display results
        if st.session_state.results:
            self.display_results()
    
    def get_atom_coordinates(self, atom):
        """Safely extract coordinates from an atom object"""
        try:
            # Try dictionary-style access first
            if hasattr(atom, '__getitem__'):
                return [float(atom['x']), float(atom['y']), float(atom['z'])]
            # Try attribute-style access
            elif hasattr(atom, 'x') and hasattr(atom, 'y') and hasattr(atom, 'z'):
                return [float(atom.x), float(atom.y), float(atom.z)]
            # Try coordinate attribute
            elif hasattr(atom, 'coord'):
                coord = atom.coord
                if isinstance(coord, (list, tuple, np.ndarray)) and len(coord) >= 3:
                    return [float(coord[0]), float(coord[1]), float(coord[2])]
            # Try position attribute  
            elif hasattr(atom, 'position'):
                pos = atom.position
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 3:
                    return [float(pos[0]), float(pos[1]), float(pos[2])]
            else:
                raise ValidationError(
                    f"Could not extract coordinates from atom: {type(atom)}",
                    suggestions=["Check PDB file format", "Try a different structure"]
                )
        except Exception as e:
            raise ValidationError(
                "Error extracting atom coordinates",
                technical_details=str(e),
                suggestions=["Check PDB file format", "Use a different structure"]
            )
    
    def setup_binding_site_selection(self, pdb_text: str) -> Tuple[Optional[np.ndarray], float]:
        """Setup binding site selection interface"""
        ligand_names = []
        ligand_centers = {}
        
        try:
            # Parse structure to find ligands
            if LIGANDFORGE_AVAILABLE:
                from pdb_parser import PDBParser
                parser = PDBParser()
                structure = parser.parse_pdb_structure(pdb_text)
                
                ligand_names = list(structure.ligands.keys()) if structure.ligands else []
                
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
        
        except Exception as e:
            st.info("Using manual coordinate input as fallback")
            ligand_names = []
            ligand_centers = {}
        
        # Binding site determination method
        col1, col2 = st.columns([2, 1])
        
        with col1:
            method = st.radio(
                "Binding site determination:",
                ["Manual Coordinates", "Co-crystallized Ligand"],
                horizontal=True,
                help="Choose how to define the binding site center"
            )
        
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
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_ligand = st.selectbox(
                        "Select ligand:", 
                        ligand_names,
                        help="Choose which ligand to use as binding site center"
                    )
                
                with col2:
                    if selected_ligand in ligand_centers:
                        center = ligand_centers[selected_ligand]
                        st.info(f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] Å")
                    else:
                        st.error(f"Could not find center for ligand {selected_ligand}")
        
        if method == "Manual Coordinates" or center is None:
            st.subheader("Manual Coordinate Entry")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                x = st.number_input(
                    "Center X (Å)", value=0.0, format="%.2f", step=0.1,
                    help="X coordinate of binding site center"
                )
            with col2:
                y = st.number_input(
                    "Center Y (Å)", value=0.0, format="%.2f", step=0.1,
                    help="Y coordinate of binding site center"
                )
            with col3:
                z = st.number_input(
                    "Center Z (Å)", value=0.0, format="%.2f", step=0.1,
                    help="Z coordinate of binding site center"
                )
            with col4:
                if st.button("Validate Center"):
                    center_coords = np.array([x, y, z])
                    if np.linalg.norm(center_coords) > 200:
                        st.warning("Center is very far from origin")
                    else:
                        st.success("Center coordinates look reasonable")
            
            center = np.array([x, y, z])
        
        # Radius selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            radius = st.slider(
                "Binding site radius (Å)", 5.0, 20.0, 10.0, 0.5,
                help="Radius around the center to consider as binding site"
            )
        
        with col2:
            volume = (4/3) * np.pi * radius**3
            st.metric("Volume", f"{volume:.0f} Å³")
        
        return center, radius
    
    def run_pipeline(self, pdb_text: str, center: np.ndarray, target_interactions: List[str], 
                    radius: float, n_initial: int, include_voxel: bool):
        """Run the LigandForge pipeline with progress tracking"""
        
        try:
            # Initialize pipeline
            if st.session_state.pipeline is None:
                with st.spinner("Initializing LigandForge pipeline..."):
                    st.session_state.pipeline = LigandForgePipeline(st.session_state.config)
            
            # Update progress tracker
            st.session_state.progress_tracker.update({
                'start_time': time.time(),
                'current_stage': 'Initializing...',
                'overall_progress': 0.05
            })
            
            # Create progress tracker display
            progress_container = st.container()
            with progress_container:
                UIComponents.create_progress_tracker()
            
            # Get optimization parameters
            opt_params = st.session_state.opt_params
            
            # Determine optimization method
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
            
            # Pipeline stages
            update_progress("Analyzing binding site...", 0.10)
            time.sleep(0.1)
            
            update_progress("Generating initial molecules...", 0.30)
            time.sleep(0.1)
            
            update_progress(f"Running {opt_method.upper()} optimization...", 0.50)
            
            # Run the actual pipeline
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
            
            # Store results with metadata
            results.execution_metadata = {
                'execution_time': execution_time,
                'optimization_method': opt_method,
                'target_interactions': target_interactions,
                'timestamp': datetime.now(),
                'config_used': st.session_state.config.__dict__.copy(),
                'opt_params_used': opt_params.copy()
            }
            
            st.session_state.results = results
            
            # Success message
            st.success(f"Pipeline completed successfully in {execution_time:.1f} seconds!")
            
            # Quick statistics
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
            
        except Exception as e:
            st.session_state.progress_tracker.update({
                'current_stage': 'Error occurred',
                'overall_progress': 0
            })
            
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
    
    def display_results(self):
        """Display pipeline results with enhanced 2D structure visualization"""
        if not st.session_state.results:
            return
        
        results = st.session_state.results
        molecules = results.generation_results.molecules
        scores = results.generation_results.scores
        
        if not molecules or not scores:
            st.error("No valid results to display")
            return
        
        # Create results DataFrame
        results_data = []
        for i, (mol, score) in enumerate(zip(molecules, scores)):
            try:
                if RDKIT_AVAILABLE:
                    # Calculate molecular properties
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
                    # Fallback without RDKit
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
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Results Overview", "Top Molecules", "Structure Grid", "Visualizations", "Analysis Report"
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
            self.display_analysis_report(results)
    
    def display_results_overview(self, df: pd.DataFrame, results, molecules):
        """Display results overview with enhanced structure display"""
        st.header("Results Overview")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generation Statistics")
            gen_stats = results.generation_results.generation_statistics
            
            UIComponents.create_metric_container(
                "Generation Method", 
                gen_stats.get('optimization_method', 'N/A')
            )
            UIComponents.create_metric_container(
                "Molecules Generated", 
                str(len(df))
            )
            UIComponents.create_metric_container(
                "Success Rate", 
                f"{len(df)/gen_stats.get('initial_molecules', len(df))*100:.1f}%"
            )
        
        with col2:
            st.subheader("Score Distribution")
            if 'Total_Score' in df.columns:
                scores = df['Total_Score']
                
                UIComponents.create_metric_container(
                    "Best Score", 
                    f"{scores.max():.3f}"
                )
                UIComponents.create_metric_container(
                    "Average Score", 
                    f"{scores.mean():.3f}"
                )
                UIComponents.create_metric_container(
                    "Score Range", 
                    f"{scores.min():.3f} - {scores.max():.3f}"
                )
        
        # Enhanced results table with structure display
        st.subheader("Results Table")
        
        # Display options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_rows = st.selectbox("Rows to display:", [10, 25, 50], index=0)
        with col2:
            sort_by = st.selectbox("Sort by:", 
                                  [col for col in df.columns if col != 'ID'], 
                                  index=0)
        with col3:
            ascending = st.checkbox("Ascending order", value=False)
        with col4:
            show_structures = st.checkbox("Show 2D Structures", 
                                        value=st.session_state.display_options['show_structures'])
        
        # Prepare display dataframe
        display_df = df.head(max_rows).sort_values(sort_by, ascending=ascending)
        
        if show_structures and RDKIT_AVAILABLE and 'SMILES' in display_df.columns:
            # Add structure images to dataframe
            enhanced_df, _ = create_enhanced_results_display(display_df, molecules)
            
            # Configure column display
            column_config = {}
            if 'Structure' in enhanced_df.columns:
                column_config['Structure'] = st.column_config.ImageColumn(
                    "2D Structure",
                    help="Molecular structure",
                    width="medium"
                )
            
            st.dataframe(
                enhanced_df, 
                use_container_width=True, 
                height=400,
                column_config=column_config
            )
        else:
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Export options
        self.display_export_options(df, results)
    
    def display_structure_grid(self, molecules: List, scores: List):
        """Display molecular structures in a grid layout"""
        st.header("Molecular Structure Grid")
        
        if not RDKIT_AVAILABLE:
            st.warning("Structure grid requires RDKit. Please install RDKit to view molecular structures.")
            return
        
        # Grid display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_display = st.slider(
                "Max molecules to display:", 4, 24, 
                st.session_state.display_options['max_structures_grid']
            )
        
        with col2:
            structure_size = st.slider(
                "Structure size:", 100, 300, 
                st.session_state.display_options['structure_size']
            )
        
        with col3:
            sort_by_score = st.checkbox("Sort by score", value=True)
        
        # Sort molecules by score if requested
        if sort_by_score:
            mol_score_pairs = list(zip(molecules, scores))
            mol_score_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
            display_molecules = [pair[0] for pair in mol_score_pairs[:max_display]]
            display_scores = [pair[1] for pair in mol_score_pairs[:max_display]]
        else:
            display_molecules = molecules[:max_display]
            display_scores = scores[:max_display]
        
        # Display grid
        UIComponents.display_molecular_structure_grid(
            display_molecules, display_scores, max_display
        )
        
        # Individual molecule selection for detailed view
        st.subheader("Detailed Molecule View")
        
        mol_options = [f"MOL_{i+1:03d} - Score: {score.total_score:.3f}" 
                      for i, score in enumerate(display_scores)]
        
        if mol_options:
            selected_mol = st.selectbox(
                "Select molecule for detailed view:", 
                mol_options,
                help="Choose a molecule to view in larger size with details"
            )
            
            mol_idx = int(selected_mol.split()[0].split('_')[1]) - 1
            
            if mol_idx < len(display_molecules):
                mol = display_molecules[mol_idx]
                score = display_scores[mol_idx]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Large structure display
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        img_data = generate_2d_structure_image(smiles, size=(400, 400))
                        
                        if img_data:
                            st.markdown(f"""
                            <div style="text-align: center; border: 2px solid #1f77b4; padding: 20px; border-radius: 10px; background: white;">
                                <img src="{img_data}" style="max-width: 100%; height: auto;">
                                <h4 style="margin: 10px 0; color: #1f77b4;">MOL_{mol_idx+1:03d}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Could not generate structure image")
                    except Exception as e:
                        st.error(f"Error displaying structure: {e}")
                
                with col2:
                    # Molecular details
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
                        st.write(f"**TPSA:** {tpsa:.2f} Ų")
                        st.write(f"**QED:** {qed:.3f}")
                        
                        st.subheader("Score Breakdown")
                        st.write(f"**Total Score:** {score.total_score:.3f}")
                        st.write(f"**Pharmacophore:** {score.pharmacophore_score:.3f}")
                        st.write(f"**Drug-likeness:** {score.drug_likeness_score:.3f}")
                        st.write(f"**Synthetic:** {score.synthetic_score:.3f}")
                        st.write(f"**Novelty:** {score.novelty_score:.3f}")
                        
                    except Exception as e:
                        st.error(f"Error calculating properties: {e}")
    
    def display_top_molecules(self, df: pd.DataFrame, molecules, scores):
        """Display top molecules analysis with enhanced visualization"""
        st.header("Top Molecules")
        
        if len(df) == 0:
            st.warning("No molecules available for analysis")
            return
        
        # Enhanced molecule selection with preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            mol_options = [f"{row['ID']} - Score: {row['Total_Score']:.3f}" 
                          for _, row in df.head(10).iterrows()]
            
            selected_mol = st.selectbox(
                "Select molecule for detailed analysis:", 
                mol_options,
                help="Choose a molecule to view detailed properties and scores"
            )
        
        with col2:
            # Quick preview of selected molecule
            mol_idx = int(selected_mol.split()[0].split('_')[1]) - 1
            
            if mol_idx < len(molecules) and RDKIT_AVAILABLE:
                mol = molecules[mol_idx]
                try:
                    smiles = Chem.MolToSmiles(mol)
                    img_data = generate_2d_structure_image(smiles, size=(150, 150))
                    
                    if img_data:
                        st.markdown(f"""
                        <div style="text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                            <img src="{img_data}" style="max-width: 100%; height: auto;">
                            <p style="font-size: 12px; margin: 5px 0;">Preview</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception:
                    st.info("Preview not available")
        
        if mol_idx < len(molecules):
            mol = molecules[mol_idx]
            score = scores[mol_idx]
            row_data = df.iloc[mol_idx]
            
            # Detailed molecule information
            col1, col2 = st.columns([2, 1])
            
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
                
                # Score breakdown
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
                # Score visualization if available
                if PLOTLY_AVAILABLE:
                    try:
                        fig = go.Figure(data=go.Scatterpolar(
                            r=list(score_data.values()),
                            theta=list(score_data.keys()),
                            fill='toself',
                            name='Score Profile'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=False,
                            title="Score Profile",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        # Fallback to simple display
                        st.write("Score visualization not available")
        
        # Top molecules ranking
        st.subheader("Top Molecules Ranking")
        
        ranking_metric = st.selectbox(
            "Rank by:",
            [col for col in df.columns if 'Score' in col],
            index=0
        )
        
        top_n = st.number_input("Show top N:", 5, 20, 10)
        top_molecules = df.nlargest(top_n, ranking_metric)
        
        # Add ranking column
        top_molecules_display = top_molecules.copy()
        top_molecules_display.insert(0, 'Rank', range(1, len(top_molecules_display) + 1))
        
        st.dataframe(top_molecules_display, use_container_width=True)
    
    def display_visualizations(self, df: pd.DataFrame, molecules, results):
        """Display visualizations"""
        st.header("Visualizations")
        
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly not available - visualizations disabled")
            return
        
        # Score distribution
        if 'Total_Score' in df.columns:
            try:
                fig = create_score_distribution_plot(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create score distribution plot: {e}")
        
        # Property space plot if RDKit available
        if RDKIT_AVAILABLE and all(col in df.columns for col in ['MW', 'LogP', 'Total_Score']):
            try:
                fig = create_property_space_plot(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create property space plot: {e}")
        
        # Score correlation matrix
        score_columns = [col for col in df.columns if 'Score' in col]
        if len(score_columns) > 1:
            try:
                fig = create_property_correlation_matrix(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create correlation matrix: {e}")
    
    def display_analysis_report(self, results):
        """Display analysis report"""
        st.header("Analysis Report")
        
        # Generate report
        if st.button("Generate Detailed Report"):
            try:
                report_sections = []
                
                # Header
                report_sections.extend([
                    "=" * 80,
                    "LIGANDFORGE 2.0 - ANALYSIS REPORT",
                    "=" * 80,
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ""
                ])
                
                # Executive Summary
                molecules = results.generation_results.molecules
                scores = results.generation_results.scores
                
                if molecules and scores:
                    best_score = max(s.total_score for s in scores)
                    avg_score = np.mean([s.total_score for s in scores])
                    success_rate = len([s for s in scores if s.total_score > 0.5]) / len(scores) * 100
                    
                    report_sections.extend([
                        "EXECUTIVE SUMMARY",
                        "-" * 50,
                        f"Total Molecules Generated: {len(molecules)}",
                        f"Best Score Achieved: {best_score:.3f}",
                        f"Average Score: {avg_score:.3f}",
                        f"Success Rate (Score >0.5): {success_rate:.1f}%",
                        ""
                    ])
                
                # Configuration details
                if hasattr(results, 'execution_metadata'):
                    metadata = results.execution_metadata
                    report_sections.extend([
                        "CONFIGURATION DETAILS",
                        "-" * 50,
                        f"Optimization Method: {metadata.get('optimization_method', 'N/A')}",
                        f"Execution Time: {metadata.get('execution_time', 0):.1f} seconds",
                        f"Target Interactions: {', '.join(metadata.get('target_interactions', []))}",
                        ""
                    ])
                
                # Top molecules
                if molecules and scores:
                    report_sections.extend([
                        "TOP MOLECULES",
                        "-" * 50
                    ])
                    
                    mol_score_pairs = list(zip(molecules, scores))
                    mol_score_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
                    
                    for i, (mol, score) in enumerate(mol_score_pairs[:5], 1):
                        try:
                            if RDKIT_AVAILABLE:
                                smiles = Chem.MolToSmiles(mol)
                                report_sections.extend([
                                    f"Molecule {i}:",
                                    f"  SMILES: {smiles}",
                                    f"  Total Score: {score.total_score:.3f}",
                                    ""
                                ])
                            else:
                                report_sections.extend([
                                    f"Molecule {i}:",
                                    f"  Total Score: {score.total_score:.3f}",
                                    ""
                                ])
                        except Exception:
                            report_sections.append(f"  Error processing molecule {i}")
                
                report_sections.extend([
                    "=" * 80,
                    "End of Report",
                    "=" * 80
                ])
                
                full_report = "\n".join(report_sections)
                
                # Display report
                st.text_area(
                    "Generated Report", 
                    full_report, 
                    height=600
                )
                
                # Download button
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    "Download Report",
                    full_report,
                    f"ligandforge_report_{timestamp}.txt",
                    "text/plain"
                )
                
            except Exception as e:
                st.error(f"Failed to generate report: {e}")
        else:
            st.info("Click 'Generate Detailed Report' to create a comprehensive analysis report")
    
    def display_export_options(self, df: pd.DataFrame, results):
        """Display export options"""
        st.subheader("Export Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV export
            if st.button("Export CSV"):
                try:
                    csv_data = df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        f"ligandforge_results_{timestamp}.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"CSV export failed: {e}")
        
        with col2:
            # JSON export
            if st.button("Export JSON"):
                try:
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'results': df.to_dict('records'),
                        'metadata': getattr(results, 'execution_metadata', {})
                    }
                    json_data = json.dumps(export_data, indent=2, default=str)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "Download JSON",
                        json_data,
                        f"ligandforge_results_{timestamp}.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"JSON export failed: {e}")
        
        with col3:
            # SDF export
            if st.button("Export SDF") and RDKIT_AVAILABLE:
                try:
                    molecules = results.generation_results.molecules
                    results_data = df.to_dict('records')
                    sdf_data = create_sdf_from_molecules(molecules, results_data)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "Download SDF",
                        sdf_data,
                        f"ligandforge_structures_{timestamp}.sdf",
                        "chemical/x-mdl-sdfile"
                    )
                except Exception as e:
                    st.error(f"SDF export failed: {e}")
        
        with col4:
            # Configuration export
            if st.button("Export Config"):
                try:
                    config_data = {
                        'config': st.session_state.config.__dict__.copy(),
                        'opt_params': st.session_state.opt_params.copy(),
                        'display_options': st.session_state.display_options.copy(),
                        'timestamp': datetime.now().isoformat()
                    }
                    config_json = json.dumps(config_data, indent=2, default=str)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "Download Config",
                        config_json,
                        f"ligandforge_config_{timestamp}.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Config export failed: {e}")
    
    def sidebar_info(self):
        """Display sidebar information"""
        st.subheader("Session Info")
        
        # Session duration
        session_duration = time.time() - st.session_state.session_start
        st.write(f"**Session Duration:** {session_duration/60:.1f} minutes")
        
        # Performance info if available
        if PSUTIL_AVAILABLE and st.session_state.performance_metrics['memory_usage']:
            latest_memory = st.session_state.performance_metrics['memory_usage'][-1]['usage']
            st.write(f"**Memory Usage:** {latest_memory:.1f}%")
        
        # Pipeline status
        if st.session_state.pipeline:
            st.success("Pipeline Ready")
        else:
            st.info("Pipeline Not Initialized")
        
        # Results status
        if st.session_state.results:
            molecules = st.session_state.results.generation_results.molecules
            st.success(f"Results Available ({len(molecules)} molecules)")
            
            # Display options status
            display_opts = st.session_state.display_options
            if display_opts['show_structures'] and RDKIT_AVAILABLE:
                st.info("🧬 2D Structure Display: Enabled")
            else:
                st.info("2D Structure Display: Disabled")
        else:
            st.info("No Results Yet")
        
        # Help section
        with st.expander("Help"):
            st.markdown("""
            **Quick Start:**
            1. Upload a PDB file or use sample structure
            2. Define binding site center
            3. Select target interactions
            4. Configure parameters in sidebar
            5. Run LigandForge
            
            **New Features:**
            - 2D molecular structure visualization
            - Interactive structure grid
            - Enhanced results display
            - Multiple export formats (CSV, JSON, SDF)
            
            **Tips:**
            - Enable 2D structures in Display Options
            - Use structure grid for visual comparison
            - Export SDF for use in other software
            - Adjust structure size for better viewing
            """)


def main():
    """Main application entry point"""
    try:
        app = LigandForgeApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your installation and dependencies")


if __name__ == "__main__":
    main()