# Add these imports to the top of app.py:

try:
    from retrosynthesis import RetrosyntheticAnalyzer, RetrosyntheticRoute
    from retrosynthetic_visualization import (
        display_retrosynthetic_route, 
        create_retrosynthetic_comparison,
        export_routes_to_excel
    )
    RETROSYNTHESIS_AVAILABLE = True
except ImportError:
    RETROSYNTHESIS_AVAILABLE = False
    st.warning("Retrosynthesis module not available")


# Update the display_results method in LigandForgeApp class:

def display_results(self):
    """Display pipeline results with enhanced 2D structure visualization and retrosynthetic routes"""
    if not st.session_state.results:
        return
    
    results = st.session_state.results
    molecules = results.generation_results.molecules
    scores = results.generation_results.scores
    
    if not molecules or not scores:
        st.error("No valid results to display")
        return
    
    # Initialize retrosynthetic analyzer if available
    if RETROSYNTHESIS_AVAILABLE:
        if 'retro_analyzer' not in st.session_state:
            st.session_state.retro_analyzer = RetrosyntheticAnalyzer()
        
        # Analyze routes if not already done
        if 'retro_routes' not in st.session_state:
            with st.spinner("Analyzing retrosynthetic routes..."):
                st.session_state.retro_routes = []
                for i, mol in enumerate(molecules):
                    try:
                        # Get assembly metadata if available
                        metadata = None
                        if hasattr(results.generation_results, 'assembly_metadata'):
                            metadata = results.generation_results.assembly_metadata.get(i)
                        
                        route = st.session_state.retro_analyzer.analyze_molecule(mol, metadata)
                        st.session_state.retro_routes.append(route)
                    except Exception as e:
                        st.warning(f"Could not analyze route for molecule {i+1}: {e}")
                        st.session_state.retro_routes.append(None)
    
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
                
                # Add retrosynthetic data if available
                if RETROSYNTHESIS_AVAILABLE and i < len(st.session_state.retro_routes):
                    route = st.session_state.retro_routes[i]
                    if route:
                        row['Synth_Steps'] = route.total_steps
                        row['Synth_Yield_%'] = route.total_yield_estimate * 100
                        row['Synth_Difficulty'] = route.overall_difficulty
                        row['Route_Feasibility'] = route.route_feasibility
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
    
    # Results tabs - ADD RETROSYNTHETIC TAB
    tabs = [
        "Results Overview", 
        "Top Molecules", 
        "Structure Grid", 
        "Visualizations", 
        "Analysis Report"
    ]
    
    if RETROSYNTHESIS_AVAILABLE:
        tabs.append("🧪 Retrosynthetic Routes")
    
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:
        self.display_results_overview(results_df, results, molecules)
    
    with tab_objects[1]:
        self.display_top_molecules(results_df, molecules, scores)
    
    with tab_objects[2]:
        self.display_structure_grid(molecules, scores)
    
    with tab_objects[3]:
        self.display_visualizations(results_df, molecules, results)
    
    with tab_objects[4]:
        self.display_analysis_report(results)
    
    # NEW: Retrosynthetic Routes Tab
    if RETROSYNTHESIS_AVAILABLE:
        with tab_objects[5]:
            self.display_retrosynthetic_routes_tab(molecules, results_df)


# Add this new method to LigandForgeApp class:

def display_retrosynthetic_routes_tab(self, molecules, results_df):
    """Display retrosynthetic routes analysis tab"""
    st.header("🧪 Retrosynthetic Routes")
    
    if not RETROSYNTHESIS_AVAILABLE:
        st.error("Retrosynthesis analysis not available. Please install required modules.")
        return
    
    if 'retro_routes' not in st.session_state or not st.session_state.retro_routes:
        st.warning("No retrosynthetic routes available. Routes are analyzed during results generation.")
        return
    
    routes = st.session_state.retro_routes
    
    # Summary statistics
    st.subheader("📊 Route Statistics Overview")
    
    valid_routes = [r for r in routes if r and r.steps]
    
    if valid_routes:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_steps = np.mean([r.total_steps for r in valid_routes])
            st.metric("Avg Steps", f"{avg_steps:.1f}")
        
        with col2:
            avg_yield = np.mean([r.total_yield_estimate for r in valid_routes]) * 100
            st.metric("Avg Overall Yield", f"{avg_yield:.1f}%")
        
        with col3:
            avg_difficulty = np.mean([r.overall_difficulty for r in valid_routes])
            difficulty_text = "Easy" if avg_difficulty < 0.3 else "Moderate" if avg_difficulty < 0.4 else "Hard"
            st.metric("Avg Difficulty", difficulty_text)
        
        with col4:
            avg_feasibility = np.mean([r.route_feasibility for r in valid_routes])
            st.metric("Avg Feasibility", f"{avg_feasibility:.2f}")
    
    # Route comparison table
    st.subheader("📋 Route Comparison")
    
    if 'Synth_Steps' in results_df.columns:
        # Display sortable comparison table
        comparison_cols = ['ID', 'Total_Score', 'Synth_Steps', 'Synth_Yield_%', 
                          'Synth_Difficulty', 'Route_Feasibility']
        available_cols = [col for col in comparison_cols if col in results_df.columns]
        
        if available_cols:
            comparison_df = results_df[available_cols].copy()
            
            # Color-code the table
            st.dataframe(
                comparison_df.style.background_gradient(
                    cmap='RdYlGn', 
                    subset=['Synth_Yield_%', 'Route_Feasibility']
                ).background_gradient(
                    cmap='RdYlGn_r', 
                    subset=['Synth_Difficulty']
                ),
                use_container_width=True,
                height=400
            )
    
    # Individual route selection
    st.subheader("🔬 Detailed Route Analysis")
    
    # Molecule selector
    mol_options = [f"{row['ID']} - Score: {row['Total_Score']:.3f}" 
                  for _, row in results_df.head(20).iterrows()]
    
    selected_mol = st.selectbox(
        "Select molecule to view retrosynthetic route:",
        mol_options,
        help="Choose a molecule to see its detailed synthetic route"
    )
    
    if selected_mol:
        mol_idx = int(selected_mol.split()[0].split('_')[1]) - 1
        
        if mol_idx < len(routes):
            route = routes[mol_idx]
            
            if route and route.steps:
                # Display the route using the visualization module
                display_retrosynthetic_route(route, mol_idx)
            else:
                st.warning(f"No valid retrosynthetic route available for molecule {mol_idx + 1}")
    
    # Route comparison visualization
    if len(valid_routes) > 1:
        st.subheader("📊 Multi-Molecule Route Comparison")
        
        # Select molecules to compare
        compare_options = st.multiselect(
            "Select molecules to compare (max 5):",
            [f"MOL_{i+1:03d}" for i in range(len(routes)) if routes[i] and routes[i].steps],
            default=[f"MOL_{i+1:03d}" for i in range(min(3, len(routes))) if routes[i] and routes[i].steps]
        )
        
        if compare_options:
            compare_indices = [int(opt.split('_')[1]) - 1 for opt in compare_options]
            compare_routes = [routes[i] for i in compare_indices]
            
            create_retrosynthetic_comparison(compare_routes, [i+1 for i in compare_indices])
    
    # Export options
    st.subheader("📥 Export Retrosynthetic Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export All Routes (TXT)"):
            try:
                export_text = []
                for i, route in enumerate(routes):
                    if route and route.steps:
                        export_text.append(f"\n{'='*80}\n")
                        export_text.append(f"MOLECULE {i+1:03d}\n")
                        export_text.append(f"{'='*80}\n")
                        export_text.append(st.session_state.retro_analyzer.generate_route_summary(route))
                        export_text.append("\n")
                
                full_export = "\n".join(export_text)
                
                st.download_button(
                    "Download TXT",
                    full_export,
                    f"retrosynthetic_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col2:
        if st.button("Export Routes (Excel)"):
            try:
                molecule_ids = list(range(1, len(routes) + 1))
                excel_data = export_routes_to_excel(
                    [r for r in routes if r and r.steps], 
                    molecule_ids
                )
                
                if excel_data:
                    st.download_button(
                        "Download Excel",
                        excel_data,
                        f"retrosynthetic_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Excel export failed: {e}")
    
    with col3:
        if st.button("Export Comparison (CSV)"):
            try:
                if 'Synth_Steps' in results_df.columns:
                    comparison_cols = ['ID', 'SMILES', 'Total_Score', 'Synth_Steps', 
                                     'Synth_Yield_%', 'Synth_Difficulty', 'Route_Feasibility']
                    available_cols = [col for col in comparison_cols if col in results_df.columns]
                    export_df = results_df[available_cols]
                    
                    csv_data = export_df.to_csv(index=False)
                    
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        f"route_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"CSV export failed: {e}")
    
    # Help section
    with st.expander("ℹ️ Understanding Retrosynthetic Routes"):
        st.markdown("""
        ### Retrosynthetic Analysis
        
        Retrosynthetic analysis works backward from the target molecule to identify 
        possible synthetic routes using available starting materials and known reactions.
        
        **Key Metrics:**
        
        - **Steps**: Number of synthetic transformations required
        - **Overall Yield**: Cumulative yield across all steps
        - **Difficulty**: Complexity of reactions (0=easy, 1=very difficult)
        - **Feasibility**: Overall practical feasibility of the route
        - **Complexity**: Molecular complexity score
        
        **Route Quality:**
        - ✅ **Good Route**: ≤5 steps, >50% yield, low difficulty
        - ⚠️ **Acceptable Route**: 6-8 steps, 30-50% yield, moderate difficulty
        - ❌ **Challenging Route**: >8 steps, <30% yield, high difficulty
        
        **Color Coding:**
        - 🟢 Green: Easy/High yield/High feasibility
        - 🟡 Yellow: Moderate
        - 🔴 Red/Orange: Difficult/Low yield/Low feasibility
        
        ### Interpreting the Route Tree
        
        The reaction tree shows:
        1. **Starting Materials** (top): Commercially available compounds
        2. **Synthetic Steps** (middle): Individual reactions with conditions
        3. **Target Molecule** (bottom): Your generated ligand
        
        Each step includes:
        - Reaction type and conditions
        - Required reagents
        - Estimated yield
        - Difficulty rating
        
        ### Critical Steps
        
        Steps marked as "critical" are:
        - High difficulty (>0.3)
        - Low yield (<65%)
        - May require optimization or alternative routes
        """)
