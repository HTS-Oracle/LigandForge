"""
Genetic Algorithm Optimizer Module
Genetic Algorithm for molecular optimization with BRICS-based crossover
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import warnings

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, DataStructs

from config import LigandForgeConfig
from data_structures import OptimizationState, OptimizationHistory
from scoring import MultiObjectiveScorer
from molecular_assembly import StructureGuidedAssembly
from diversity_manager import EnhancedDiversityManager


class GeneticAlgorithm:
    """Genetic Algorithm for molecular optimization"""
    
    def __init__(self,
                 config: LigandForgeConfig,
                 scorer: MultiObjectiveScorer,
                 assembly: StructureGuidedAssembly,
                 diversity: EnhancedDiversityManager):
        self.config = config
        self.scorer = scorer
        self.assembly = assembly
        self.diversity = diversity
        self.rng = random.Random(config.random_seed)
        
        # GA parameters
        self.ga_config = self.config.get_optimization_config("ga")
        
        # Statistics tracking
        self.ga_stats = {
            'total_generations': 0,
            'successful_crossovers': 0,
            'successful_mutations': 0,
            'crossover_attempts': 0,
            'mutation_attempts': 0,
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': []
        }
        
        # Population tracking
        self.population_history = []
        self.fitness_history = []

    def run(self,
            target_interactions: List[str],
            population_size: int = None,
            generations: int = None,
            crossover_rate: float = None,
            mutation_rate: float = None,
            elitism: int = None,
            tournament_size: int = None,
            seed_population: Optional[List[Chem.Mol]] = None) -> List[Chem.Mol]:
        """Execute GA and return top molecules"""

        # Use provided parameters or defaults from config
        pop_size = population_size or self.ga_config['population_size']
        max_gens = generations or self.ga_config['generations']
        crossover_prob = crossover_rate or self.ga_config['crossover_rate']
        mutation_prob = mutation_rate or self.ga_config['mutation_rate']
        elite_count = elitism or self.ga_config['elitism']
        tournament_k = tournament_size or self.ga_config['tournament_size']
        
        print(f"Starting GA with {pop_size} individuals for {max_gens} generations")
        
        # Initialize population
        if seed_population and len(seed_population) > 0:
            population = self._pad_or_trim_population(seed_population, pop_size, target_interactions)
        else:
            population = self._initialize_population(pop_size, target_interactions)

        if not population:
            warnings.warn("Failed to initialize population")
            return []

        # Create optimization history
        optimization_history = OptimizationHistory(
            method="GA",
            states=[],
            start_time=self._get_current_time()
        )

        # Evaluate initial population
        fitness_scores = self._evaluate_population(population, gen_round=0)
        self._update_statistics(population, fitness_scores, 0)

        # Evolution loop
        for generation in range(1, max_gens + 1):
            print(f"Generation {generation}/{max_gens}")
            
            # Selection and reproduction
            new_population = []
            
            # Elitism - keep best individuals
            if elite_count > 0:
                elites = self._select_elites(population, fitness_scores, elite_count)
                new_population.extend(elites)
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores, k=tournament_k)
                parent2 = self._tournament_selection(population, fitness_scores, k=tournament_k)
                
                # Crossover
                child = None
                if self.rng.random() < crossover_prob:
                    child = self._crossover(parent1, parent2)
                    self.ga_stats['crossover_attempts'] += 1
                    if child is not None:
                        self.ga_stats['successful_crossovers'] += 1
                
                # If crossover failed, copy better parent
                if child is None:
                    child = parent1 if self._score_single(parent1) >= self._score_single(parent2) else parent2
                
                # Mutation
                if self.rng.random() < mutation_prob:
                    mutated = self._mutate(child, target_interactions)
                    self.ga_stats['mutation_attempts'] += 1
                    if mutated is not None:
                        child = mutated
                        self.ga_stats['successful_mutations'] += 1
                
                # Validate and check diversity
                if self.assembly._validate_molecule(child) and self.diversity.is_diverse(child):
                    new_population.append(child)
                elif len(new_population) < pop_size:
                    # If diversity check fails, add anyway to maintain population size
                    if self.assembly._validate_molecule(child):
                        new_population.append(child)

            # Ensure we have exactly the right population size
            population = new_population[:pop_size]
            
            # Evaluate new population
            fitness_scores = self._evaluate_population(population, gen_round=generation)
            
            # Update statistics and history
            self._update_statistics(population, fitness_scores, generation)
            
            # Create optimization state
            best_idx = np.argmax(fitness_scores)
            best_molecule = population[best_idx]
            best_score = fitness_scores[best_idx]
            avg_score = np.mean(fitness_scores)
            diversity = self.diversity.calculate_population_diversity(population)
            
            state = OptimizationState(
                iteration=generation,
                current_population=population.copy(),
                current_scores=fitness_scores.copy(),
                best_molecule=best_molecule,
                best_score=best_score,
                population_diversity=diversity,
                convergence_metric=self._calculate_convergence_metric(generation)
            )
            
            optimization_history.states.append(state)
            
            print(f"Best fitness: {best_score:.3f}, Avg fitness: {avg_score:.3f}, Diversity: {diversity:.3f}")
            
            # Check for convergence
            if self._check_convergence(generation):
                print(f"Converged at generation {generation}")
                break

        optimization_history.end_time = self._get_current_time()
        self.ga_stats['total_generations'] = generation

        # Return top-ranked molecules
        final_fitness = self._evaluate_population(population, gen_round=generation)
        ranked = sorted(zip(population, final_fitness), key=lambda x: x[1], reverse=True)
        
        # Return diverse set of top performers
        top_molecules = [mol for mol, _ in ranked[:max(20, elite_count * 2)]]
        diverse_top = self.diversity.cluster_and_select(top_molecules, n_clusters=min(15, len(top_molecules)))
        
        return diverse_top

    def _initialize_population(self, size: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Initialize population with diverse molecules"""
        molecules = []
        max_attempts = size * 10
        attempts = 0
        
        while len(molecules) < size and attempts < max_attempts:
            attempts += 1
            
            # Generate molecule using assembly strategies
            mol = self.assembly.generate_structure_guided(target_interactions, n_molecules=1)
            
            if mol and len(mol) > 0:
                candidate = mol[0]
                if (self.assembly._validate_molecule(candidate) and 
                    self.diversity.is_diverse(candidate)):
                    molecules.append(candidate)
        
        # If we don't have enough molecules, pad with simpler structures
        if len(molecules) < size:
            simple_molecules = self._generate_simple_molecules(size - len(molecules), target_interactions)
            molecules.extend(simple_molecules)
        
        return molecules[:size]

    def _pad_or_trim_population(self, seed: List[Chem.Mol], size: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Pad or trim seed population to desired size"""
        valid_seed = [m for m in seed if self.assembly._validate_molecule(m)]
        
        # Ensure diversity bookkeeping includes seeds
        for mol in valid_seed:
            self.diversity.is_diverse(mol)
        
        if len(valid_seed) >= size:
            # Rank by fitness and take top performers
            fitness_scores = [self._score_single(m) for m in valid_seed]
            ranked = sorted(zip(valid_seed, fitness_scores), key=lambda x: x[1], reverse=True)
            return [m for m, _ in ranked[:size]]
        
        # Need to grow population
        population = valid_seed.copy()
        needed = size - len(population)
        
        # Generate additional molecules
        additional = self._initialize_population(needed, target_interactions)
        population.extend(additional)
        
        return population[:size]

    def _evaluate_population(self, population: List[Chem.Mol], gen_round: int) -> List[float]:
        """Evaluate fitness for entire population"""
        fitness_scores = []
        
        for mol in population:
            try:
                score_result = self.scorer.calculate_comprehensive_score(mol, generation_round=gen_round)
                fitness = score_result.total_score
                
                # Apply diversity pressure if configured
                if self.ga_config.get('diversity_pressure', 0) > 0:
                    diversity_bonus = self._calculate_diversity_bonus(mol, population)
                    fitness += self.ga_config['diversity_pressure'] * diversity_bonus
                
                fitness_scores.append(fitness)
                
            except Exception as e:
                warnings.warn(f"Error evaluating molecule: {e}")
                fitness_scores.append(0.0)
        
        return fitness_scores

    def _score_single(self, mol: Chem.Mol) -> float:
        """Score a single molecule"""
        try:
            return self.scorer.calculate_comprehensive_score(mol).total_score
        except Exception:
            return 0.0

    def _select_elites(self, population: List[Chem.Mol], fitness: List[float], k: int) -> List[Chem.Mol]:
        """Select top k individuals as elites"""
        if k <= 0:
            return []
        
        elite_indices = np.argsort(fitness)[::-1][:min(k, len(population))]
        return [population[i] for i in elite_indices]

    def _tournament_selection(self, population: List[Chem.Mol], fitness: List[float], k: int = 3) -> Chem.Mol:
        """Tournament selection"""
        k = max(2, min(k, len(population)))
        tournament_indices = self.rng.sample(range(len(population)), k)
        best_idx = max(tournament_indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _crossover(self, parent1: Chem.Mol, parent2: Chem.Mol) -> Optional[Chem.Mol]:
        """BRICS-based crossover"""
        try:
            # Decompose parents into BRICS fragments
            frags1 = list(BRICS.BRICSDecompose(parent1, minFragmentSize=2))
            frags2 = list(BRICS.BRICSDecompose(parent2, minFragmentSize=2))
            
            if not frags1 or not frags2:
                return None

            # Mix fragments from both parents
            # Take random subset from each parent
            n1 = max(1, min(len(frags1), self.rng.randint(1, min(4, len(frags1)))))
            n2 = max(1, min(len(frags2), self.rng.randint(1, min(4, len(frags2)))))
            
            selected1 = self.rng.sample(frags1, n1)
            selected2 = self.rng.sample(frags2, n2)
            
            # Combine fragment pools
            combined_frags = set(selected1 + selected2)

            # Attempt to build new molecule from mixed fragments
            for child in BRICS.BRICSBuild(combined_frags, minFragmentSize=2):
                try:
                    Chem.SanitizeMol(child)
                    if self.assembly._validate_molecule(child):
                        return child
                except:
                    continue
            
            return None
            
        except Exception:
            return None

    def _mutate(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Apply mutation operators"""
        mutation_operators = [
            self._mutate_bioisostere,
            self._mutate_add_substituent,
            self._mutate_modify_functional_group,
            self._mutate_ring_system
        ]
        
        # Shuffle operators for random application
        self.rng.shuffle(mutation_operators)
        
        for operator in mutation_operators:
            try:
                mutated = operator(mol, target_interactions)
                if mutated is not None and self.assembly._validate_molecule(mutated):
                    return mutated
            except Exception:
                continue
        
        return None

    def _mutate_bioisostere(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Bioisosteric replacement mutation"""
        bioisosteres = self.assembly.fragment_lib.bioisosteres
        mol_smiles = Chem.MolToSmiles(mol)
        
        available_replacements = [key for key in bioisosteres.keys() if key in mol_smiles]
        if not available_replacements:
            return None
        
        original = self.rng.choice(available_replacements)
        replacements = bioisosteres[original]
        
        if not replacements:
            return None
        
        replacement = self.rng.choice(replacements)
        new_smiles = mol_smiles.replace(original, replacement, 1)
        
        try:
            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol:
                Chem.SanitizeMol(new_mol)
                return new_mol
        except:
            pass
        
        return None

    def _mutate_add_substituent(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Add substituent mutation"""
        substituents = self.assembly.fragment_lib.fragments.get('substituents', [])
        if not substituents:
            return None
        
        # Filter by target interactions
        suitable_subs = []
        for interaction in target_interactions:
            suitable_subs.extend([s for s in substituents if interaction in s.interaction_types])
        
        if not suitable_subs:
            suitable_subs = substituents
        
        substituent = self.rng.choice(suitable_subs)
        return self.assembly._add_substituent(mol, substituent)

    def _mutate_modify_functional_group(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Modify functional group mutation"""
        # Simple functional group modifications
        modifications = {
            'CC(=O)O': 'CC(=O)N',  # Carboxylic acid to amide
            'C#N': 'C(=O)N',       # Nitrile to amide
            'c1ccccc1': 'c1ccncc1', # Benzene to pyridine
            'C(F)(F)F': 'C#N',     # Trifluoromethyl to nitrile
        }
        
        mol_smiles = Chem.MolToSmiles(mol)
        
        for original, replacement in modifications.items():
            if original in mol_smiles:
                new_smiles = mol_smiles.replace(original, replacement, 1)
                try:
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if new_mol:
                        Chem.SanitizeMol(new_mol)
                        return new_mol
                except:
                    continue
        
        return None

    def _mutate_ring_system(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Ring system mutation"""
        # Simple ring modifications
        ring_modifications = {
            'c1ccccc1': ['c1ccncc1', 'c1cnccc1', 'c1ncccc1'],  # Benzene to pyridines
            'c1ccncc1': ['c1ccccc1', 'c1cncnc1'],              # Pyridine variants
        }
        
        mol_smiles = Chem.MolToSmiles(mol)
        
        for original, replacements in ring_modifications.items():
            if original in mol_smiles:
                replacement = self.rng.choice(replacements)
                new_smiles = mol_smiles.replace(original, replacement, 1)
                try:
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if new_mol:
                        Chem.SanitizeMol(new_mol)
                        return new_mol
                except:
                    continue
        
        return None

    def _generate_simple_molecules(self, count: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Generate simple molecules as fallback"""
        simple_molecules = []
        
        # Simple SMILES patterns
        simple_patterns = [
            'c1ccccc1',           # Benzene
            'c1ccncc1',           # Pyridine
            'CCO',                # Ethanol
            'CC(=O)O',            # Acetic acid
            'c1ccc(O)cc1',        # Phenol
            'c1ccc(N)cc1',        # Aniline
            'c1ccc(C)cc1',        # Toluene
            'CC(C)C',             # Isobutane
            'c1ccc2ccccc2c1',     # Naphthalene
            'c1cnccn1',           # Pyrimidine
        ]
        
        for pattern in simple_patterns[:count]:
            try:
                mol = Chem.MolFromSmiles(pattern)
                if mol and self.assembly._validate_molecule(mol):
                    simple_molecules.append(mol)
            except:
                continue
        
        return simple_molecules

    def _calculate_diversity_bonus(self, mol: Chem.Mol, population: List[Chem.Mol]) -> float:
        """Calculate diversity bonus for fitness"""
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            
            similarities = []
            for other_mol in population:
                if other_mol == mol:
                    continue
                try:
                    other_fp = AllChem.GetMorganFingerprintAsBitVect(other_mol, radius=2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(fp, other_fp)
                    similarities.append(similarity)
                except:
                    continue
            
            if similarities:
                avg_similarity = np.mean(similarities)
                return 1.0 - avg_similarity  # Higher bonus for more diverse molecules
            else:
                return 0.0
                
        except:
            return 0.0

    def _update_statistics(self, population: List[Chem.Mol], fitness: List[float], generation: int):
        """Update GA statistics"""
        if fitness:
            best_fitness = max(fitness)
            avg_fitness = np.mean(fitness)
            
            self.ga_stats['best_fitness_history'].append(best_fitness)
            self.ga_stats['avg_fitness_history'].append(avg_fitness)
            
            # Calculate population diversity
            diversity = self.diversity.calculate_population_diversity(population)
            self.ga_stats['diversity_history'].append(diversity)
        
        self.population_history.append(population.copy())
        self.fitness_history.append(fitness.copy())

    def _calculate_convergence_metric(self, generation: int) -> float:
        """Calculate convergence metric"""
        if len(self.ga_stats['best_fitness_history']) < 5:
            return 1.0
        
        # Look at improvement in best fitness over last 5 generations
        recent_best = self.ga_stats['best_fitness_history'][-5:]
        improvement = recent_best[-1] - recent_best[0]
        
        return abs(improvement)  # Smaller values indicate convergence

    def _check_convergence(self, generation: int) -> bool:
        """Check if algorithm has converged"""
        if generation < 5:
            return False
        
        # Check if best fitness hasn't improved significantly
        convergence_metric = self._calculate_convergence_metric(generation)
        
        if convergence_metric < 0.01:  # Very small improvement
            return True
        
        # Check if diversity has dropped too low
        if self.ga_stats['diversity_history']:
            current_diversity = self.ga_stats['diversity_history'][-1]
            if current_diversity < 0.1:  # Very low diversity
                return True
        
        return False

    def _get_current_time(self) -> float:
        """Get current time"""
        import time
        return time.time()

    def get_statistics(self) -> Dict:
        """Get comprehensive GA statistics"""
        stats = self.ga_stats.copy()
        
        # Calculate success rates
        if stats['crossover_attempts'] > 0:
            stats['crossover_success_rate'] = stats['successful_crossovers'] / stats['crossover_attempts']
        else:
            stats['crossover_success_rate'] = 0.0
        
        if stats['mutation_attempts'] > 0:
            stats['mutation_success_rate'] = stats['successful_mutations'] / stats['mutation_attempts']
        else:
            stats['mutation_success_rate'] = 0.0
        
        # Current state
        if self.ga_stats['best_fitness_history']:
            stats['current_best_fitness'] = self.ga_stats['best_fitness_history'][-1]
            stats['current_avg_fitness'] = self.ga_stats['avg_fitness_history'][-1]
        
        if self.ga_stats['diversity_history']:
            stats['current_diversity'] = self.ga_stats['diversity_history'][-1]
        
        # Performance metrics
        if len(self.ga_stats['best_fitness_history']) > 1:
            stats['fitness_improvement'] = (self.ga_stats['best_fitness_history'][-1] - 
                                           self.ga_stats['best_fitness_history'][0])
        
        return stats

    def get_convergence_data(self) -> Dict:
        """Get data for plotting convergence"""
        return {
            'generations': list(range(len(self.ga_stats['best_fitness_history']))),
            'best_fitness': self.ga_stats['best_fitness_history'].copy(),
            'avg_fitness': self.ga_stats['avg_fitness_history'].copy(),
            'diversity': self.ga_stats['diversity_history'].copy()
        }

    def reset_algorithm(self):
        """Reset algorithm state"""
        self.ga_stats = {
            'total_generations': 0,
            'successful_crossovers': 0,
            'successful_mutations': 0,
            'crossover_attempts': 0,
            'mutation_attempts': 0,
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': []
        }
        
        self.population_history = []
        self.fitness_history = []

    def save_population(self, filepath: str, generation: int = -1):
        """Save population to SDF file"""
        if not self.population_history:
            return
        
        population = self.population_history[generation]
        fitness = self.fitness_history[generation]
        
        try:
            from rdkit import Chem
            
            writer = Chem.SDWriter(filepath)
            
            for i, (mol, fit) in enumerate(zip(population, fitness)):
                # Add properties to molecule
                mol.SetProp("Generation", str(abs(generation)))
                mol.SetProp("Individual_ID", str(i))
                mol.SetProp("Fitness", str(fit))
                mol.SetProp("SMILES", Chem.MolToSmiles(mol))
                
                # Add calculated properties
                try:
                    mol.SetProp("MW", str(round(Chem.rdMolDescriptors.CalcExactMolWt(mol), 2)))
                    mol.SetProp("LogP", str(round(Chem.Crippen.MolLogP(mol), 2)))
                    mol.SetProp("HBD", str(Chem.Lipinski.NumHDonors(mol)))
                    mol.SetProp("HBA", str(Chem.Lipinski.NumHAcceptors(mol)))
                except:
                    pass
                
                writer.write(mol)
            
            writer.close()
            
        except Exception as e:
            warnings.warn(f"Failed to save population: {e}")

    def load_population(self, filepath: str) -> List[Chem.Mol]:
        """Load population from SDF file"""
        molecules = []
        
        try:
            from rdkit import Chem
            
            supplier = Chem.SDMolSupplier(filepath)
            
            for mol in supplier:
                if mol is not None:
                    molecules.append(mol)
            
        except Exception as e:
            warnings.warn(f"Failed to load population: {e}")
        
        return molecules

    def analyze_population_chemistry(self, generation: int = -1) -> Dict:
        """Analyze chemical space of population"""
        if not self.population_history:
            return {}
        
        population = self.population_history[generation]
        
        analysis = {
            'molecular_weights': [],
            'logp_values': [],
            'ring_counts': [],
            'functional_groups': defaultdict(int),
            'scaffolds': defaultdict(int)
        }
        
        for mol in population:
            try:
                # Basic properties
                mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
                logp = Chem.Crippen.MolLogP(mol)
                rings = Chem.rdMolDescriptors.CalcNumRings(mol)
                
                analysis['molecular_weights'].append(mw)
                analysis['logp_values'].append(logp)
                analysis['ring_counts'].append(rings)
                
                # Scaffold analysis
                from rdkit.Chem.Scaffolds import MurckoScaffold
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                analysis['scaffolds'][scaffold_smiles] += 1
                
                # Functional group analysis (simplified)
                smiles = Chem.MolToSmiles(mol)
                if 'C(=O)O' in smiles:
                    analysis['functional_groups']['carboxylic_acid'] += 1
                if 'C(=O)N' in smiles:
                    analysis['functional_groups']['amide'] += 1
                if 'C#N' in smiles:
                    analysis['functional_groups']['nitrile'] += 1
                if 'S(=O)(=O)' in smiles:
                    analysis['functional_groups']['sulfonyl'] += 1
                
            except Exception:
                continue
        
        # Calculate statistics
        if analysis['molecular_weights']:
            analysis['mw_mean'] = np.mean(analysis['molecular_weights'])
            analysis['mw_std'] = np.std(analysis['molecular_weights'])
        
        if analysis['logp_values']:
            analysis['logp_mean'] = np.mean(analysis['logp_values'])
            analysis['logp_std'] = np.std(analysis['logp_values'])
        
        return analysis
