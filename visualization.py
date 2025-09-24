"""
Enhanced Visualization Module
Plotting and visualization functions for LigandForge results with 2D structure display
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import warnings
import base64
import io
from PIL import Image

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski, QED, Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def _coerce_numeric(series):
    """Safely coerce a pandas Series to numeric, dropping NaNs."""
    return pd.to_numeric(series, errors='coerce').dropna()


def generate_2d_structure_image(smiles: str, size: Tuple[int, int] = (200, 200), 
                               return_format: str = 'base64') -> Optional[str]:
    """
    Generate 2D molecular structure image from SMILES
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
        return_format: 'base64' for base64 encoded string, 'pil' for PIL Image
    
    Returns:
        Base64 encoded image string or PIL Image, or None if failed
    """
    if not RDKIT_AVAILABLE or not smiles:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates if not present
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        
        # Set drawing options for better appearance
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().bondLineWidth = 2
        drawer.drawOptions().highlightBondWidthMultiplier = 2
        
        # Draw molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get image data
        img_data = drawer.GetDrawingText()
        
        if return_format == 'base64':
            # Convert to base64 for HTML display
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        elif return_format == 'pil':
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            return img
        else:
            return img_data
            
    except Exception as e:
        warnings.warn(f"Failed to generate 2D structure for SMILES {smiles}: {e}")
        return None


def generate_2d_structure_svg(smiles: str, size: Tuple[int, int] = (200, 200)) -> Optional[str]:
    """
    Generate SVG 2D molecular structure from SMILES
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
    
    Returns:
        SVG string or None if failed
    """
    if not RDKIT_AVAILABLE or not smiles:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates if not present
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        
        # Create SVG drawer
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        
        # Set drawing options
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().bondLineWidth = 2
        
        # Draw molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get SVG text
        svg = drawer.GetDrawingText()
        return svg
        
    except Exception as e:
        warnings.warn(f"Failed to generate SVG structure for SMILES {smiles}: {e}")
        return None


def create_molecular_grid_image(molecules: List[Chem.Mol], 
                               legends: Optional[List[str]] = None,
                               mols_per_row: int = 4, 
                               sub_img_size: Tuple[int, int] = (200, 200)) -> Optional[Image.Image]:
    """
    Create a grid image of molecular structures
    
    Args:
        molecules: List of RDKit molecule objects
        legends: Optional list of labels for each molecule
        mols_per_row: Number of molecules per row
        sub_img_size: Size of each individual molecule image
    
    Returns:
        PIL Image of the grid or None if failed
    """
    if not RDKIT_AVAILABLE or not molecules:
        return None
    
    try:
        # Filter valid molecules
        valid_mols = [mol for mol in molecules if mol is not None]
        if not valid_mols:
            return None
        
        # Use RDKit's built-in grid functionality
        img = Draw.MolsToGridImage(
            valid_mols[:20],  # Limit to first 20 molecules for performance
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=legends[:20] if legends else None,
            useSVG=False
        )
        
        return img
        
    except Exception as e:
        warnings.warn(f"Failed to create molecular grid: {e}")
        return None


def add_structure_images_to_dataframe(df: pd.DataFrame, molecules: List[Chem.Mol] = None) -> pd.DataFrame:
    """
    Add 2D structure images to results dataframe
    
    Args:
        df: Results dataframe with SMILES column
        molecules: Optional list of RDKit molecules (if available)
    
    Returns:
        Enhanced dataframe with structure images
    """
    if df is None or df.empty:
        return df
    
    df_enhanced = df.copy()
    
    # Add structure column
    if 'SMILES' in df.columns:
        structure_images = []
        
        for idx, row in df.iterrows():
            smiles = row.get('SMILES', '')
            if smiles and RDKIT_AVAILABLE:
                # Generate 2D structure image
                img_data = generate_2d_structure_image(smiles, size=(150, 150))
                if img_data:
                    structure_images.append(img_data)
                else:
                    structure_images.append(None)
            else:
                structure_images.append(None)
        
        # Insert structure column after ID column
        if 'ID' in df_enhanced.columns:
            id_idx = df_enhanced.columns.get_loc('ID')
            df_enhanced.insert(id_idx + 1, 'Structure', structure_images)
        else:
            df_enhanced.insert(0, 'Structure', structure_images)
    
    return df_enhanced


def create_enhanced_results_display(df: pd.DataFrame, molecules: List[Chem.Mol] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Create enhanced results display with 2D structures and interactive features
    
    Args:
        df: Results dataframe
        molecules: List of RDKit molecules
    
    Returns:
        Tuple of (enhanced_dataframe, display_options)
    """
    # Add structure images
    df_enhanced = add_structure_images_to_dataframe(df, molecules)
    
    # Prepare display options
    display_options = {
        'show_structures': True,
        'structure_size': (150, 150),
        'max_displayed_rows': 20,
        'sortable_columns': [col for col in df_enhanced.columns if col not in ['Structure']],
        'filterable_columns': [col for col in df_enhanced.columns if 'Score' in col or col in ['MW', 'LogP', 'QED']]
    }
    
    return df_enhanced, display_options


def create_score_distribution_plot(df: pd.DataFrame):
    """Create robust score distribution plot"""
    if df is None or df.empty:
        return None

    score_columns = ['Total_Score', 'Pharmacophore_Score', 'Drug_Likeness_Score', 'Synthetic_Score']
    present_cols = [c for c in score_columns if c in df.columns]
    if not present_cols:
        return None

    # Coerce columns to numeric
    df_plot = df.copy()
    for c in present_cols:
        df_plot[c] = _coerce_numeric(df_plot[c])

    if PLOTLY_AVAILABLE:
        try:
            # Create subplot grid
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=tuple(present_cols + [""] * (4 - len(present_cols))),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            colors = ['blue', 'red', 'green', 'orange']
            
            for i, (col, pos) in enumerate(zip(present_cols, positions)):
                values = df_plot[col].dropna()
                if len(values) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=values, 
                            name=col, 
                            nbinsx=20, 
                            showlegend=False,
                            marker_color=colors[i % len(colors)],
                            opacity=0.7
                        ),
                        row=pos[0], col=pos[1]
                    )

            fig.update_layout(
                height=600, 
                title_text="Score Distributions",
                title_x=0.5,
                showlegend=False
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Score", row=2, col=1)
            fig.update_xaxes(title_text="Score", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            return fig

        except Exception as e:
            warnings.warn(f"Plotly plotting failed: {e}")

    # Fallback to matplotlib
    if MATPLOTLIB_AVAILABLE:
        try:
            n = len(present_cols)
            if n == 0:
                return None

            rows, cols = 2, 2
            fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
            axes = axes.flatten()

            for i, col in enumerate(present_cols):
                values = df_plot[col].dropna().values
                if len(values) > 0:
                    axes[i].hist(values, bins=20, alpha=0.7, color=f'C{i}')
                    axes[i].set_title(col)
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Count')
                    axes[i].grid(True, alpha=0.3)

            # Hide unused subplots
            for j in range(i + 1, rows * cols):
                fig.delaxes(axes[j])

            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"Matplotlib plotting failed: {e}")

    return None


def create_property_space_plot(df: pd.DataFrame):
    """Create property space visualization (MW vs LogP, colored by Total_Score, size by QED)"""
    if df is None or df.empty:
        return None

    needed = ['MW', 'LogP', 'QED', 'Total_Score']
    if not all(c in df.columns for c in needed):
        return None

    df_plot = df.copy()
    df_plot['MW_num'] = _coerce_numeric(df_plot['MW'])
    df_plot['LogP_num'] = _coerce_numeric(df_plot['LogP'])
    df_plot['QED_num'] = _coerce_numeric(df_plot['QED'])
    df_plot['Score_num'] = _coerce_numeric(df_plot['Total_Score'])

    # Drop rows with any NaNs in required numeric fields
    df_plot = df_plot.dropna(subset=['MW_num', 'LogP_num', 'QED_num', 'Score_num'])
    if df_plot.empty:
        return None

    if PLOTLY_AVAILABLE:
        try:
            # Add ID column if it exists for hover data
            hover_data = ['ID'] if 'ID' in df_plot.columns else None
            
            fig = px.scatter(
                df_plot, 
                x='MW_num', 
                y='LogP_num',
                color='Score_num',
                size='QED_num',
                hover_data=hover_data,
                title='Molecular Property Space',
                labels={
                    'MW_num': 'Molecular Weight (Da)', 
                    'LogP_num': 'LogP', 
                    'Score_num': 'Total Score',
                    'QED_num': 'QED'
                },
                color_continuous_scale='viridis'
            )
            
            # Add Lipinski space boundaries
            fig.add_hline(y=5, line_dash="dash", line_color="red", 
                         annotation_text="LogP = 5 (Lipinski limit)")
            fig.add_vline(x=500, line_dash="dash", line_color="red", 
                         annotation_text="MW = 500 Da (Lipinski limit)")
            
            # Update layout
            fig.update_layout(
                height=600,
                title_x=0.5,
                xaxis_title="Molecular Weight (Da)",
                yaxis_title="LogP"
            )
            
            return fig

        except Exception as e:
            warnings.warn(f"Plotly plotting failed: {e}")

    # Fallback to matplotlib
    if MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create scatter plot
            scatter = ax.scatter(df_plot['MW_num'], df_plot['LogP_num'], 
                               c=df_plot['Score_num'], s=df_plot['QED_num']*100, 
                               cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Molecular Weight (Da)')
            ax.set_ylabel('LogP')
            ax.set_title('Molecular Property Space')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Total Score')
            
            # Add Lipinski boundaries
            ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='LogP = 5')
            ax.axvline(x=500, color='red', linestyle='--', alpha=0.7, label='MW = 500')
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig

        except Exception as e:
            warnings.warn(f"Matplotlib plotting failed: {e}")

    return None


def create_radar_plot(scores_dict: Dict[str, float], title: str = "Molecular Scores"):
    """Create radar plot for molecular scores"""
    if not scores_dict or not PLOTLY_AVAILABLE:
        return None
    
    try:
        categories = list(scores_dict.keys())
        values = list(scores_dict.values())
        
        # Ensure we close the radar plot
        categories += [categories[0]]
        values += [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Scores',
            line_color='rgb(32, 146, 230)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            title_x=0.5
        )
        
        return fig
        
    except Exception as e:
        warnings.warn(f"Radar plot creation failed: {e}")
        return None


def create_convergence_plot(optimization_history: Dict):
    """Create convergence plot for optimization"""
    if not optimization_history:
        return None
    
    # Handle different optimization methods
    if optimization_history.get('method') == 'hybrid':
        return create_hybrid_convergence_plot(optimization_history)
    elif 'best_fitness_history' in optimization_history:
        return create_ga_convergence_plot(optimization_history)
    elif 'average_reward' in optimization_history:
        return create_rl_convergence_plot(optimization_history)
    
    return None


def create_ga_convergence_plot(ga_stats: Dict):
    """Create convergence plot for genetic algorithm"""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        best_fitness = ga_stats.get('best_fitness_history', [])
        avg_fitness = ga_stats.get('avg_fitness_history', [])
        diversity = ga_stats.get('diversity_history', [])
        
        if not best_fitness:
            return None
        
        generations = list(range(len(best_fitness)))
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Fitness Evolution', 'Population Diversity'),
            vertical_spacing=0.15
        )
        
        # Fitness plot
        fig.add_trace(
            go.Scatter(x=generations, y=best_fitness, name='Best Fitness', 
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        if avg_fitness:
            fig.add_trace(
                go.Scatter(x=generations, y=avg_fitness, name='Average Fitness',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Diversity plot
        if diversity:
            fig.add_trace(
                go.Scatter(x=generations, y=diversity, name='Diversity',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title_text="Genetic Algorithm Convergence",
            title_x=0.5
        )
        
        fig.update_xaxes(title_text="Generation", row=2, col=1)
        fig.update_yaxes(title_text="Fitness", row=1, col=1)
        fig.update_yaxes(title_text="Diversity", row=2, col=1)
        
        return fig
        
    except Exception as e:
        warnings.warn(f"GA convergence plot failed: {e}")
        return None


def create_rl_convergence_plot(rl_stats: Dict):
    """Create convergence plot for reinforcement learning"""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        # RL stats might not have detailed history, so create a simple plot
        avg_reward = rl_stats.get('average_reward', 0)
        best_reward = rl_stats.get('best_reward', 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Average Reward', 'Best Reward'],
            y=[avg_reward, best_reward],
            marker_color=['blue', 'red']
        ))
        
        fig.update_layout(
            title="RL Performance Summary",
            title_x=0.5,
            yaxis_title="Reward",
            height=400
        )
        
        return fig
        
    except Exception as e:
        warnings.warn(f"RL convergence plot failed: {e}")
        return None


def create_hybrid_convergence_plot(hybrid_stats: Dict):
    """Create convergence plot for hybrid optimization"""
    if not PLOTLY_AVAILABLE:
        return None
    
    try:
        rl_stats = hybrid_stats.get('rl_stats', {})
        ga_stats = hybrid_stats.get('ga_stats', {})
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('RL Phase', 'GA Phase'),
            horizontal_spacing=0.15
        )
        
        # RL phase
        if rl_stats:
            avg_reward = rl_stats.get('average_reward', 0)
            best_reward = rl_stats.get('best_reward', 0)
            
            fig.add_trace(
                go.Bar(x=['Avg', 'Best'], y=[avg_reward, best_reward],
                      name='RL Rewards', marker_color='blue'),
                row=1, col=1
            )
        
        # GA phase
        if ga_stats and ga_stats.get('best_fitness_history'):
            generations = list(range(len(ga_stats['best_fitness_history'])))
            
            fig.add_trace(
                go.Scatter(x=generations, y=ga_stats['best_fitness_history'],
                          name='GA Best Fitness', line=dict(color='red')),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            title_text="Hybrid Optimization Performance",
            title_x=0.5,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        warnings.warn(f"Hybrid convergence plot failed: {e}")
        return None


def create_diversity_analysis_plot(molecules: List[Chem.Mol]):
    """Create diversity analysis plot"""
    if not molecules or not PLOTLY_AVAILABLE:
        return None
    
    try:
        from rdkit.Chem import DataStructs, AllChem
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Calculate fingerprints
        fps = []
        valid_molecules = []
        
        for mol in molecules:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
                valid_molecules.append(mol)
            except:
                continue
        
        if len(fps) < 2:
            return None
        
        # Convert to numpy array for dimensionality reduction
        fp_array = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_array.append(arr)
        
        fp_matrix = np.array(fp_array)
        
        # Perform dimensionality reduction
        if len(fps) > 50:
            # Use t-SNE for larger datasets
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fps)//3))
        else:
            # Use PCA for smaller datasets
            reducer = PCA(n_components=2, random_state=42)
        
        coords_2d = reducer.fit_transform(fp_matrix)
        
        # Calculate molecular properties for coloring
        mw_values = []
        logp_values = []
        
        for mol in valid_molecules:
            try:
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                logp = Crippen.MolLogP(mol)
                mw_values.append(mw)
                logp_values.append(logp)
            except:
                mw_values.append(0)
                logp_values.append(0)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=mw_values,
                colorscale='viridis',
                colorbar=dict(title="Molecular Weight"),
                line=dict(width=1, color='black')
            ),
            text=[f"MW: {mw:.1f}, LogP: {logp:.2f}" for mw, logp in zip(mw_values, logp_values)],
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))
        
        method_name = "t-SNE" if len(fps) > 50 else "PCA"
        fig.update_layout(
            title=f"Chemical Space Diversity ({method_name})",
            title_x=0.5,
            xaxis_title=f"{method_name} 1",
            yaxis_title=f"{method_name} 2",
            height=600,
            width=600
        )
        
        return fig
        
    except Exception as e:
        warnings.warn(f"Diversity analysis plot failed: {e}")
        return None


def create_sdf_from_molecules(molecules: List[Chem.Mol], results_data: List[Dict]) -> str:
    """Create SDF format string from molecules"""
    from io import StringIO
    
    sdf_buffer = StringIO()
    
    for mol, data in zip(molecules, results_data):
        try:
            # Ensure we have a 3D conformer for proper SDF blocks
            if mol.GetNumConformers() == 0:
                try:
                    m3d = Chem.AddHs(Chem.Mol(mol))
                    AllChem.EmbedMolecule(m3d, randomSeed=42)
                    AllChem.UFFOptimizeMolecule(m3d)
                    mol = m3d
                except Exception:
                    pass  # Fall back to 2D block

            # Add properties to molecule
            for key in ["ID", "Total_Score", "MW", "LogP", "QED"]:
                if key in data and data[key] is not None:
                    mol.SetProp(key, str(data[key]))

            # Write mol block
            mol_block = Chem.MolToMolBlock(mol)
            sdf_buffer.write(mol_block)

            # Write additional properties
            for key, value in data.items():
                if key != 'SMILES' and value is not None:
                    sdf_buffer.write(f"> <{key}>\n{value}\n\n")
            
            sdf_buffer.write("$$\n")
            
        except Exception as e:
            warnings.warn(f"Error processing molecule {data.get('ID', 'unknown')}: {e}")
            continue
    
    return sdf_buffer.getvalue()


def create_property_correlation_matrix(df: pd.DataFrame):
    """Create correlation matrix of molecular properties"""
    if df is None or df.empty:
        return None
    
    # Select numeric columns
    numeric_cols = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'QED', 'Total_Score', 
                   'Pharmacophore_Score', 'Drug_Likeness_Score', 'Synthetic_Score']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    if len(available_cols) < 2:
        return None
    
    # Convert to numeric and calculate correlation
    df_numeric = df[available_cols].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df_numeric.corr()
    
    if PLOTLY_AVAILABLE:
        try:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Property Correlation Matrix",
                title_x=0.5,
                height=600,
                width=600
            )
            
            return fig
            
        except Exception as e:
            warnings.warn(f"Plotly correlation matrix failed: {e}")
    
    if MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            im = ax.imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.columns)
            
            # Add text annotations
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black")
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            ax.set_title("Property Correlation Matrix")
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            warnings.warn(f"Matplotlib correlation matrix failed: {e}")
    
    return None


def create_score_comparison_plot(molecules_data: List[Dict]):
    """Create comparison plot of different score components"""
    if not molecules_data or not PLOTLY_AVAILABLE:
        return None
    
    try:
        df = pd.DataFrame(molecules_data)
        
        score_columns = [col for col in df.columns if 'Score' in col and col != 'Total_Score']
        if not score_columns:
            return None
        
        fig = go.Figure()
        
        x_pos = list(range(len(df)))
        
        for score_col in score_columns:
            if score_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=x_pos,
                    y=df[score_col],
                    mode='lines+markers',
                    name=score_col.replace('_Score', ''),
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Score Component Comparison",
            title_x=0.5,
            xaxis_title="Molecule Index",
            yaxis_title="Score",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        warnings.warn(f"Score comparison plot failed: {e}")
        return None


def create_molecular_weight_distribution(df: pd.DataFrame):
    """Create molecular weight distribution with drug-like ranges"""
    if df is None or df.empty or 'MW' not in df.columns:
        return None
    
    mw_values = _coerce_numeric(df['MW'])
    if mw_values.empty:
        return None
    
    if PLOTLY_AVAILABLE:
        try:
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=mw_values,
                nbinsx=30,
                name='Molecular Weight',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add drug-like ranges
            fig.add_vline(x=150, line_dash="dash", line_color="green", 
                         annotation_text="Min drug-like")
            fig.add_vline(x=500, line_dash="dash", line_color="red", 
                         annotation_text="Lipinski limit")
            fig.add_vline(x=800, line_dash="dash", line_color="orange", 
                         annotation_text="Max oral drug")
            
            fig.update_layout(
                title="Molecular Weight Distribution",
                title_x=0.5,
                xaxis_title="Molecular Weight (Da)",
                yaxis_title="Count",
                height=400
            )
            
            return fig
            
        except Exception as e:
            warnings.warn(f"MW distribution plot failed: {e}")
    
    return None


def create_lipinski_compliance_plot(df: pd.DataFrame):
    """Create Lipinski rule compliance visualization"""
    if df is None or df.empty:
        return None
    
    required_cols = ['MW', 'LogP', 'HBD', 'HBA']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Calculate violations
    violations = []
    for _, row in df.iterrows():
        violation_count = 0
        if pd.notna(row['MW']) and row['MW'] > 500:
            violation_count += 1
        if pd.notna(row['LogP']) and row['LogP'] > 5:
            violation_count += 1
        if pd.notna(row['HBD']) and row['HBD'] > 5:
            violation_count += 1
        if pd.notna(row['HBA']) and row['HBA'] > 10:
            violation_count += 1
        violations.append(violation_count)
    
    if PLOTLY_AVAILABLE:
        try:
            # Count violations
            violation_counts = pd.Series(violations).value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=violation_counts.index,
                    y=violation_counts.values,
                    marker_color=['green' if x == 0 else 'orange' if x <= 1 else 'red' 
                                for x in violation_counts.index]
                )
            ])
            
            fig.update_layout(
                title="Lipinski Rule Compliance",
                title_x=0.5,
                xaxis_title="Number of Violations",
                yaxis_title="Number of Molecules",
                height=400
            )
            
            return fig
            
        except Exception as e:
            warnings.warn(f"Lipinski compliance plot failed: {e}")
    
    return None


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for molecular properties"""
    if df is None or df.empty:
        return {}
    
    stats = {}
    
    # Numeric columns to analyze
    numeric_cols = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'QED', 'Total_Score']
    
    for col in numeric_cols:
        if col in df.columns:
            values = _coerce_numeric(df[col])
            if not values.empty:
                stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75))
                }
    
    # Count-based statistics
    if 'Total_Score' in df.columns:
        scores = _coerce_numeric(df['Total_Score'])
        if not scores.empty:
            stats['score_distribution'] = {
                'high_quality': int(sum(scores > 0.7)),
                'medium_quality': int(sum((scores >= 0.4) & (scores <= 0.7))),
                'low_quality': int(sum(scores < 0.4))
            }
    
    # Lipinski compliance
    if all(col in df.columns for col in ['MW', 'LogP', 'HBD', 'HBA']):
        compliant = 0
        for _, row in df.iterrows():
            violations = 0
            if pd.notna(row['MW']) and row['MW'] > 500:
                violations += 1
            if pd.notna(row['LogP']) and row['LogP'] > 5:
                violations += 1
            if pd.notna(row['HBD']) and row['HBD'] > 5:
                violations += 1
            if pd.notna(row['HBA']) and row['HBA'] > 10:
                violations += 1
            
            if violations <= 1:  # Allow 1 violation
                compliant += 1
        
        stats['lipinski_compliance'] = {
            'compliant_molecules': compliant,
            'total_molecules': len(df),
            'compliance_rate': compliant / len(df) if len(df) > 0 else 0
        }
    
    return stats