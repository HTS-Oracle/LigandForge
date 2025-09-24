"""
Pipeline Module
Main pipeline orchestrator for LigandForge
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
import warnings

from rdkit import Chem

from .config import LigandForgeConfig, ConfigPresets
from .data_structures import PipelineConfiguration, PipelineResults, GenerationResult, validate_position_array
from .pdb_parser import PDBParser
from .voxel_analyzer import VoxelBasedAnalyzer
from .pocket_analyzer import LigandForgeEnhancedAnalyzer
from .molecular_assembly import StructureGuidedAssembly
from .diversity_manager import EnhancedDiversityManager
from .scoring import MultiObjectiveScorer
from .optimization.rl_optimizer import RLOptimizer
from .optimization.genetic_algorithm import GeneticAlgorithm
from .fragment_library7 import FragmentLibrary


class LigandForgePipeline:
    """Main pipeline orchestrator for molecular design"""
    
    def __init__(self, config: LigandForgeConfig = None):
        """Initialize pipeline with configuration"""
        self.config = config or LigandForgeConfig()
        self._initialize_components()
        
        # Pipeline state
        self.current_run_id = None
        self.execution_metadata = {}
        
        # Results storage
        self.last_results = None
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing LigandForge pipeline components...")
        
        # Core analyzers
        self.pdb_parser = PDBParser(strict_parsing=False, ignore_alt_locs=True)
        self.voxel_analyzer = VoxelBasedAnalyzer(self.config)
        self.pocket_analyzer = LigandForgeEnhancedAnalyzer(self.config)
        
        # Fragment library and diversity management
        self.fragment_library = FragmentLibrary(self.config)
        self.diversity_manager = EnhancedDiversityManager(self.config)
        
        print("Pipeline components initialized successfully.")
    
    def run_full_pipeline(self, 
                         pdb_text: str, 
                         center: np.ndarray, 
                         target_interactions: List[str],
                         optimization_method: str = "hybrid",
                         **kwargs) -> PipelineResults:
        """
        Run complete ligand design pipeline
        
        Args:
            pdb_text: PDB structure as text
            center: Binding site center coordinates
            target_interactions: List of target interaction types
            optimization_method: 'rl', 'ga', or 'hybrid'
            **kwargs: Additional parameters
        
        Returns:
            PipelineResults object with comprehensive results
        """
        
        start_time = time.time()
        self.current_run_id = f"run_{int(start_time)}"
        
        print(f"Starting LigandForge pipeline run: {self.current_run_id}")
        print(f"Optimization method: {optimization_method}")
        print(f"Target interactions: {', '.join(target_interactions)}")
        
        try:
            # Create pipeline configuration
            pipeline_config = PipelineConfiguration(
                config=self.config,
                target_interactions=target_interactions,
                optimization_method=optimization_method,
                binding_site_center=validate_position_array(center),
                binding_site_radius=kwargs.get('radius', 10.0),
                reference_molecules=kwargs.get('reference_molecules', []),
                custom_constraints=kwargs.get('custom_constraints', {})
            )
            
            # Step 1: Parse and validate structure
            print("\n=== Step 1: Structure Analysis ===")
            structure = self._parse_and_validate_structure(pdb_text)
            
            # Step 2: Comprehensive binding site analysis
            print("\n=== Step 2: Binding Site Analysis ===")
            pocket_analysis = self._analyze_binding_site(
                pdb_text, center, kwargs.get('radius', 10.0)
            )
            
            # Step 3: Optional voxel-based analysis
            voxel_analysis = None
            if format.lower() == 'sdf':
                self._export_sdf(molecules, scores, filepath)
            elif format.lower() == 'csv':
                self._export_csv(molecules, scores, filepath)
            elif format.lower() == 'json':
                self._export_json(molecules, scores, filepath)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def _export_sdf(self, molecules: List[Chem.Mol], scores: List, filepath: str):
        """Export to SDF format"""
        writer = Chem.SDWriter(filepath)
        for i, (mol, score) in enumerate(zip(molecules, scores)):
            # Add properties to molecule
            mol.SetProp("Molecule_ID", f"LF_{i+1:04d}")
            mol.SetProp("Total_Score", f"{score.total_score:.4f}")
            mol.SetProp("Pharmacophore_Score", f"{score.pharmacophore_score:.4f}")
            mol.SetProp("Drug_Likeness_Score", f"{score.drug_likeness_score:.4f}")
            mol.SetProp("Synthetic_Score", f"{score.synthetic_score:.4f}")
            mol.SetProp("Novelty_Score", f"{score.novelty_score:.4f}")
            mol.SetProp("SMILES", Chem.MolToSmiles(mol))
            
            # Add calculated properties
            try:
                mol.SetProp("MW", f"{Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}")
                mol.SetProp("LogP", f"{Chem.Crippen.MolLogP(mol):.2f}")
                mol.SetProp("HBD", str(Chem.Lipinski.NumHDonors(mol)))
                mol.SetProp("HBA", str(Chem.Lipinski.NumHAcceptors(mol)))
                mol.SetProp("TPSA", f"{Chem.rdMolDescriptors.CalcTPSA(mol):.2f}")
            except:
                pass
            
            if score.violations:
                mol.SetProp("Violations", "; ".join(score.violations))
            
            writer.write(mol)
        
        writer.close()
    
    def _export_csv(self, molecules: List[Chem.Mol], scores: List, filepath: str):
        """Export to CSV format"""
        import pandas as pd
        
        data = []
        for i, (mol, score) in enumerate(zip(molecules, scores)):
            row = {
                'Molecule_ID': f"LF_{i+1:04d}",
                'SMILES': Chem.MolToSmiles(mol),
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
            
            # Add molecular properties
            try:
                row.update({
                    'MW': Chem.rdMolDescriptors.CalcExactMolWt(mol),
                    'LogP': Chem.Crippen.MolLogP(mol),
                    'HBD': Chem.Lipinski.NumHDonors(mol),
                    'HBA': Chem.Lipinski.NumHAcceptors(mol),
                    'TPSA': Chem.rdMolDescriptors.CalcTPSA(mol),
                    'Heavy_Atoms': mol.GetNumHeavyAtoms(),
                    'Rings': Chem.rdMolDescriptors.CalcNumRings(mol),
                    'Rotatable_Bonds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                })
            except:
                pass
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _export_json(self, molecules: List[Chem.Mol], scores: List, filepath: str):
        """Export to JSON format"""
        import json
        
        data = {
            'metadata': self.execution_metadata,
            'molecules': []
        }
        
        for i, (mol, score) in enumerate(zip(molecules, scores)):
            mol_data = {
                'id': f"LF_{i+1:04d}",
                'smiles': Chem.MolToSmiles(mol),
                'scores': {
                    'total': score.total_score,
                    'pharmacophore': score.pharmacophore_score,
                    'drug_likeness': score.drug_likeness_score,
                    'synthetic': score.synthetic_score,
                    'novelty': score.novelty_score,
                    'selectivity': score.selectivity_score,
                    'water_displacement': score.water_displacement_score
                },
                'violations': score.violations,
                'confidence': score.confidence
            }
            
            # Add properties if available
            if hasattr(score, 'property_values') and score.property_values:
                mol_data['properties'] = score.property_values
            
            data['molecules'].append(mol_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_report(self) -> str:
        """Create a comprehensive text report"""
        if not self.last_results:
            return "No results available for report generation."
        
        report = []
        report.append("=" * 60)
        report.append("LIGANDFORGE MOLECULAR DESIGN REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Execution metadata
        metadata = self.execution_metadata
        report.append("EXECUTION SUMMARY")
        report.append("-" * 20)
        report.append(f"Run ID: {metadata.get('run_id', 'N/A')}")
        report.append(f"Execution Time: {metadata.get('execution_time', 0):.2f} seconds")
        report.append(f"Optimization Method: {metadata.get('optimization_method', 'N/A')}")
        report.append(f"Target Interactions: {', '.join(metadata.get('target_interactions', []))}")
        report.append("")
        
        # Pocket analysis
        pocket = self.last_results.pocket_analysis
        report.append("BINDING SITE ANALYSIS")
        report.append("-" * 20)
        report.append(f"Druggability Score: {pocket.druggability_score:.3f}")
        report.append(f"Pocket Volume: {pocket.volume:.1f} Å³")
        report.append(f"Surface Area: {pocket.surface_area:.1f} Å²")
        report.append(f"Interaction Hotspots: {len(pocket.hotspots)}")
        report.append(f"Water Sites: {len(pocket.water_sites)}")
        
        # Interaction summary
        interaction_summary = pocket.get_interaction_summary()
        if interaction_summary:
            report.append("\nInteraction Types:")
            for interaction, count in interaction_summary.items():
                report.append(f"  {interaction}: {count}")
        report.append("")
        
        # Generation results
        gen_results = self.last_results.generation_results
        report.append("GENERATION RESULTS")
        report.append("-" * 20)
        report.append(f"Total Molecules Generated: {len(gen_results.molecules)}")
        
        if gen_results.scores:
            scores = [s.total_score for s in gen_results.scores]
            report.append(f"Best Score: {max(scores):.3f}")
            report.append(f"Average Score: {np.mean(scores):.3f}")
            report.append(f"Score Range: {min(scores):.3f} - {max(scores):.3f}")
        
        # Diversity statistics
        div_stats = gen_results.diversity_statistics
        if div_stats:
            report.append(f"Unique Scaffolds: {div_stats.get('unique_scaffolds', 'N/A')}")
            report.append(f"Diversity Rate: {div_stats.get('diversity_rate', 0):.3f}")
        report.append("")
        
        # Top molecules
        report.append("TOP 10 MOLECULES")
        report.append("-" * 20)
        for i, (mol, score) in enumerate(zip(gen_results.molecules[:10], gen_results.scores[:10])):
            report.append(f"{i+1:2d}. Score: {score.total_score:.3f} | SMILES: {Chem.MolToSmiles(mol)}")
        report.append("")
        
        # Score breakdown for top molecule
        if gen_results.scores:
            top_score = gen_results.scores[0]
            report.append("TOP MOLECULE SCORE BREAKDOWN")
            report.append("-" * 30)
            report.append(f"Pharmacophore:     {top_score.pharmacophore_score:.3f}")
            report.append(f"Drug-likeness:     {top_score.drug_likeness_score:.3f}")
            report.append(f"Synthetic:         {top_score.synthetic_score:.3f}")
            report.append(f"Novelty:           {top_score.novelty_score:.3f}")
            report.append(f"Selectivity:       {top_score.selectivity_score:.3f}")
            report.append(f"Water Displacement: {top_score.water_displacement_score:.3f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def optimize_config_for_target(self, target_type: str) -> LigandForgeConfig:
        """Get optimized configuration for specific target types"""
        target_configs = {
            'kinase': ConfigPresets.kinase_focused(),
            'gpcr': ConfigPresets.gpcr_focused(),
            'fragment': ConfigPresets.fragment_based(),
            'lead_opt': ConfigPresets.lead_optimization(),
            'fast': ConfigPresets.fast_screening(),
            'high_quality': ConfigPresets.high_quality()
        }
        
        if target_type.lower() in target_configs:
            optimized_config = target_configs[target_type.lower()]
            self.config = optimized_config
            self._initialize_components()  # Reinitialize with new config
            return optimized_config
        else:
            available = ', '.join(target_configs.keys())
            raise ValueError(f"Unknown target type: {target_type}. Available: {available}")
    
    def benchmark_performance(self, test_molecules: List[Chem.Mol], 
                            expected_scores: Optional[List[float]] = None) -> Dict:
        """Benchmark pipeline performance against test molecules"""
        if not self.last_results:
            raise ValueError("No pipeline results available for benchmarking")
        
        benchmark_results = {
            'test_molecules': len(test_molecules),
            'processing_times': [],
            'score_accuracy': None,
            'coverage_analysis': {}
        }
        
        # Time individual molecule processing
        scorer = MultiObjectiveScorer(self.config, self.last_results.pocket_analysis, self.diversity_manager)
        
        computed_scores = []
        for mol in test_molecules:
            start_time = time.time()
            try:
                score = scorer.calculate_comprehensive_score(mol)
                computed_scores.append(score.total_score)
                processing_time = time.time() - start_time
                benchmark_results['processing_times'].append(processing_time)
            except Exception as e:
                benchmark_results['processing_times'].append(None)
                computed_scores.append(0.0)
        
        benchmark_results['avg_processing_time'] = np.mean([t for t in benchmark_results['processing_times'] if t is not None])
        
        # Compare with expected scores if provided
        if expected_scores and len(expected_scores) == len(computed_scores):
            correlation = np.corrcoef(expected_scores, computed_scores)[0, 1]
            rmse = np.sqrt(np.mean((np.array(expected_scores) - np.array(computed_scores)) ** 2))
            benchmark_results['score_accuracy'] = {
                'correlation': correlation,
                'rmse': rmse,
                'mean_expected': np.mean(expected_scores),
                'mean_computed': np.mean(computed_scores)
            }
        
        return benchmark_results
    
    def get_pipeline_statistics(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        if not self.last_results:
            return {}
        
        stats = {
            'execution_metadata': self.execution_metadata,
            'pocket_analysis': {
                'druggability_score': self.last_results.pocket_analysis.druggability_score,
                'num_hotspots': len(self.last_results.pocket_analysis.hotspots),
                'num_water_sites': len(self.last_results.pocket_analysis.water_sites),
                'volume': self.last_results.pocket_analysis.volume,
                'surface_area': self.last_results.pocket_analysis.surface_area
            },
            'generation_statistics': self.last_results.generation_results.generation_statistics,
            'diversity_statistics': self.last_results.generation_results.diversity_statistics,
            'optimization_history': self.last_results.optimization_history
        }
        
        return stats
    
    def reset_pipeline(self):
        """Reset pipeline state"""
        self.diversity_manager.reset_tracking()
        self.current_run_id = None
        self.execution_metadata = {}
        self.last_results = None
        
        # Clear any cached data
        if hasattr(self, 'scorer'):
            self.scorer.clear_cache()
    
    def save_pipeline_state(self, filepath: str):
        """Save current pipeline state"""
        import pickle
        
        state = {
            'config': self.config,
            'execution_metadata': self.execution_metadata,
            'diversity_manager_stats': self.diversity_manager.get_statistics(),
            'last_results': self.last_results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_pipeline_state(self, filepath: str):
        """Load pipeline state"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.execution_metadata = state['execution_metadata']
        self.last_results = state['last_results']
        
        # Reinitialize components with loaded config
        self._initialize_components() kwargs.get('include_voxel_analysis', False):
                print("\n=== Step 3: Voxel Field Analysis ===")
                voxel_analysis = self._perform_voxel_analysis(
                    pdb_text, center, kwargs.get('radius', 10.0)
                )
            
            # Step 4: Initialize molecular generation components
            print("\n=== Step 4: Initializing Generation Components ===")
            assembly, scorer = self._initialize_generation_components(
                pocket_analysis, pipeline_config.reference_molecules
            )
            
            # Step 5: Generate and optimize molecules
            print("\n=== Step 5: Molecular Generation and Optimization ===")
            generation_results, optimization_history = self._run_optimization(
                assembly, scorer, target_interactions, optimization_method, **kwargs
            )
            
            # Step 6: Post-processing and analysis
            print("\n=== Step 6: Post-processing ===")
            final_results = self._post_process_results(
                generation_results, pocket_analysis, voxel_analysis
            )
            
            # Create comprehensive results object
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.execution_metadata = {
                'run_id': self.current_run_id,
                'execution_time': execution_time,
                'start_time': start_time,
                'end_time': end_time,
                'optimization_method': optimization_method,
                'target_interactions': target_interactions,
                'config_snapshot': self.config.__dict__.copy(),
                'structure_info': {
                    'num_atoms': len(structure.atoms),
                    'num_chains': len(structure.chains),
                    'resolution': structure.resolution
                }
            }
            
            pipeline_results = PipelineResults(
                configuration=pipeline_config,
                pocket_analysis=pocket_analysis,
                voxel_analysis=voxel_analysis,
                generation_results=final_results,
                optimization_history=optimization_history,
                execution_metadata=self.execution_metadata
            )
            
            self.last_results = pipeline_results
            
            print(f"\n=== Pipeline Complete ===")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Generated {len(final_results.molecules)} molecules")
            print(f"Best score: {max(s.total_score for s in final_results.scores):.3f}")
            
            return pipeline_results
            
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            raise
    
    def _parse_and_validate_structure(self, pdb_text: str):
        """Parse and validate PDB structure"""
        print("Parsing PDB structure...")
        
        structure = self.pdb_parser.parse_pdb_structure(pdb_text)
        
        # Validate structure
        is_valid, issues = self.pdb_parser.validate_structure(structure)
        if not is_valid:
            warnings.warn(f"Structure validation issues: {'; '.join(issues)}")
        
        print(f"Parsed structure: {len(structure.atoms)} atoms, {len(structure.chains)} chains")
        if structure.resolution:
            print(f"Resolution: {structure.resolution} Å")
        
        return structure
    
    def _analyze_binding_site(self, pdb_text: str, center: np.ndarray, radius: float):
        """Comprehensive binding site analysis"""
        print(f"Analyzing binding site (center: {center}, radius: {radius} Å)...")
        
        comprehensive_results = self.pocket_analyzer.analyze_binding_site_comprehensive(
            pdb_text, center, radius
        )
        
        pocket_analysis = comprehensive_results['compatible_format']
        
        print(f"Identified {len(pocket_analysis.hotspots)} interaction hotspots")
        print(f"Found {len(pocket_analysis.water_sites)} water sites")
        print(f"Druggability score: {pocket_analysis.druggability_score:.3f}")
        
        # Print interaction summary
        interaction_summary = pocket_analysis.get_interaction_summary()
        if interaction_summary:
            print("Interaction types found:")
            for interaction, count in interaction_summary.items():
                print(f"  {interaction}: {count}")
        
        return pocket_analysis
    
    def _perform_voxel_analysis(self, pdb_text: str, center: np.ndarray, radius: float):
        """Perform detailed voxel-based field analysis"""
        print("Performing voxel-based field analysis...")
        
        voxel_results = self.voxel_analyzer.analyze_binding_site_voxels(
            pdb_text, center, radius
        )
        
        print(f"Voxel analysis complete:")
        print(f"  Cavity volume: {voxel_results.cavity_volume:.1f} Å³")
        print(f"  Surface area: {voxel_results.surface_area:.1f} Å²")
        print(f"  Field complexity: {voxel_results.field_complexity_score:.3f}")
        print(f"  Hotspot voxels: {len(voxel_results.hotspot_voxels)}")
        
        return voxel_results
    
    def _initialize_generation_components(self, pocket_analysis, reference_molecules: List[Chem.Mol]):
        """Initialize molecular generation components"""
        print("Setting up molecular generation components...")
        
        # Initialize molecular assembly
        assembly = StructureGuidedAssembly(
            self.fragment_library, pocket_analysis, self.config
        )
        
        # Add reference molecules to diversity manager
        if reference_molecules:
            self.diversity_manager.add_reference_molecules(reference_molecules)
            print(f"Added {len(reference_molecules)} reference molecules")
        
        # Initialize scorer
        scorer = MultiObjectiveScorer(
            self.config, pocket_analysis, self.diversity_manager
        )
        
        if reference_molecules:
            scorer.add_reference_molecules(reference_molecules)
        
        print("Generation components ready.")
        
        return assembly, scorer
    
    def _run_optimization(self, assembly: StructureGuidedAssembly, scorer: MultiObjectiveScorer,
                         target_interactions: List[str], optimization_method: str, 
                         **kwargs) -> Tuple[GenerationResult, Any]:
        """Run molecular optimization"""
        
        # Generate initial population
        n_initial = kwargs.get('n_initial', 200)
        print(f"Generating {n_initial} initial molecules...")
        
        seed_molecules = assembly.generate_structure_guided(
            target_interactions, n_molecules=n_initial
        )
        
        # Filter valid and diverse molecules
        valid_seeds = []
        for mol in seed_molecules:
            if assembly._validate_molecule(mol) and self.diversity_manager.is_diverse(mol):
                valid_seeds.append(mol)
        
        print(f"Generated {len(valid_seeds)} valid diverse seed molecules")
        
        if not valid_seeds:
            raise ValueError("No valid seed molecules generated")
        
        # Run optimization based on method
        optimized_molecules = []
        optimization_history = None
        
        if optimization_method.lower() == "rl":
            optimizer = RLOptimizer(self.config, scorer, assembly)
            optimized_molecules = optimizer.optimize_generation(
                valid_seeds, n_iterations=kwargs.get('rl_iterations', 20)
            )
            optimization_history = optimizer.get_optimization_statistics()
            
        elif optimization_method.lower() == "ga":
            optimizer = GeneticAlgorithm(self.config, scorer, assembly, self.diversity_manager)
            optimized_molecules = optimizer.run(
                target_interactions,
                population_size=kwargs.get('population_size', 150),
                generations=kwargs.get('generations', 30),
                seed_population=valid_seeds
            )
            optimization_history = optimizer.get_statistics()
            
        elif optimization_method.lower() == "hybrid":
            # RL then GA
            print("Running RL phase...")
            rl_optimizer = RLOptimizer(self.config, scorer, assembly)
            rl_results = rl_optimizer.optimize_generation(
                valid_seeds, n_iterations=kwargs.get('rl_iterations', 15)
            )
            
            print("Running GA phase...")
            ga_optimizer = GeneticAlgorithm(self.config, scorer, assembly, self.diversity_manager)
            optimized_molecules = ga_optimizer.run(
                target_interactions,
                population_size=kwargs.get('population_size', 150),
                generations=kwargs.get('generations', 15),
                seed_population=rl_results if rl_results else valid_seeds
            )
            
            optimization_history = {
                'rl_stats': rl_optimizer.get_optimization_statistics(),
                'ga_stats': ga_optimizer.get_statistics(),
                'method': 'hybrid'
            }
        
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        print(f"Optimization complete: {len(optimized_molecules)} molecules")
        
        # Score all final molecules
        scored_molecules = []
        for mol in optimized_molecules:
            try:
                score = scorer.calculate_comprehensive_score(mol)
                scored_molecules.append((mol, score))
            except Exception as e:
                warnings.warn(f"Error scoring molecule: {e}")
                continue
        
        # Sort by total score
        scored_molecules.sort(key=lambda x: x[1].total_score, reverse=True)
        
        final_molecules = [mol for mol, _ in scored_molecules]
        final_scores = [score for _, score in scored_molecules]
        
        # Create generation result
        generation_stats = {
            'initial_molecules': len(seed_molecules),
            'valid_seeds': len(valid_seeds),
            'final_molecules': len(final_molecules),
            'optimization_method': optimization_method,
            'diversity_stats': self.diversity_manager.get_statistics(),
            'assembly_stats': assembly.get_generation_statistics()
        }
        
        generation_result = GenerationResult(
            molecules=final_molecules,
            scores=final_scores,
            generation_statistics=generation_stats,
            diversity_statistics=self.diversity_manager.get_statistics(),
            total_time=0.0  # Will be updated later
        )
        
        return generation_result, optimization_history
    
    def _post_process_results(self, generation_results: GenerationResult, 
                            pocket_analysis, voxel_analysis) -> GenerationResult:
        """Post-process and enhance results"""
        print("Post-processing results...")
        
        # Apply final diversity filtering
        if len(generation_results.molecules) > 50:
            print("Applying final diversity filtering...")
            diverse_molecules = self.diversity_manager.cluster_and_select(
                generation_results.molecules[:100], n_clusters=50
            )
            
            # Re-score diverse molecules
            diverse_scores = []
            for mol in diverse_molecules:
                # Find corresponding score
                mol_smiles = Chem.MolToSmiles(mol, canonical=True)
                for orig_mol, score in zip(generation_results.molecules, generation_results.scores):
                    orig_smiles = Chem.MolToSmiles(orig_mol, canonical=True)
                    if mol_smiles == orig_smiles:
                        diverse_scores.append(score)
                        break
            
            generation_results.molecules = diverse_molecules
            generation_results.scores = diverse_scores
        
        # Update statistics
        generation_results.successful_generations = len(generation_results.molecules)
        generation_results.diversity_statistics = self.diversity_manager.get_statistics()
        
        print(f"Final results: {len(generation_results.molecules)} diverse molecules")
        
        return generation_results
    
    def get_molecule_analysis(self, molecule_idx: int) -> Dict[str, Any]:
        """Get detailed analysis of a specific molecule"""
        if not self.last_results or molecule_idx >= len(self.last_results.generation_results.molecules):
            return {}
        
        mol = self.last_results.generation_results.molecules[molecule_idx]
        score = self.last_results.generation_results.scores[molecule_idx]
        
        analysis = {
            'molecule_index': molecule_idx,
            'smiles': Chem.MolToSmiles(mol),
            'scores': {
                'total': score.total_score,
                'pharmacophore': score.pharmacophore_score,
                'synthetic': score.synthetic_score,
                'drug_likeness': score.drug_likeness_score,
                'novelty': score.novelty_score,
                'selectivity': score.selectivity_score,
                'water_displacement': score.water_displacement_score
            },
            'properties': score.property_values,
            'violations': score.violations,
            'confidence': score.confidence
        }
        
        # Add chemical analysis
        try:
            analysis['molecular_weight'] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            analysis['heavy_atoms'] = mol.GetNumHeavyAtoms()
            analysis['rings'] = Chem.rdMolDescriptors.CalcNumRings(mol)
            analysis['aromatic_rings'] = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
            analysis['rotatable_bonds'] = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
            
            # Scaffold analysis
            from rdkit.Chem.Scaffolds import MurckoScaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            analysis['scaffold_smiles'] = Chem.MolToSmiles(scaffold)
            
        except Exception as e:
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def export_results(self, filepath: str, format: str = 'sdf') -> None:
        """Export results to file"""
        if not self.last_results:
            raise ValueError("No results to export")
        
        molecules = self.last_results.generation_results.molecules
        scores = self.last_results.generation_results.scores
        
        if
