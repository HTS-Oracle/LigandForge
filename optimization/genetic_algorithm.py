"""
Improved Genetic Algorithm Optimizer Module
Enhanced Genetic Algorithm for molecular optimization with advanced strategies
"""

import numpy as np
import random
import time
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, deque
import warnings
import math
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, DataStructs, rdMolDescriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import LigandForgeConfig
from data_structures import OptimizationState, OptimizationHistory
from scoring import MultiObjectiveScorer
from molecular_assembly import StructureGuidedAssembly
from diversity_manager import EnhancedDiversityManager


class GeneticAlgorithm:
    """Enhanced Genetic Algorithm for molecular optimization"""
    
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
        
        # Enhanced statistics tracking
        self.ga_stats = {
            'total_generations': 0,
            'successful_crossovers': 0,
            'successful_mutations': 0,
            'crossover_attempts': 0,
            'mutation_attempts': 0,
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': [],
            'operator_success_rates': defaultdict(lambda: {'attempts': 0, 'successes': 0}),
            'fitness_stagnation_count': 0,
            'early_stopping_triggered': False
        }
        
        # Population tracking with enhanced memory
        self.population_history = []
        self.fitness_history = []
        self.elite_archive = []  # Archive of best molecules across generations
        self.fitness_cache = {}  # Cache for expensive fitness evaluations
        
        # Adaptive parameters
        self.adaptive_rates = {
            'crossover_rate': self.ga_config.get('crossover_rate', 0.8),
            'mutation_rate': self.ga_config.get('mutation_rate', 0.2),
            'tournament_size': self.ga_config.get('tournament_size', 3)
        }
        
        # Enhanced mutation operators with weights
        self.mutation_operators = [
            (self._mutate_bioisostere, 0.3),
            (self._mutate_add_substituent, 0.25),
            (self._mutate_modify_functional_group, 0.25),
            (self._mutate_ring_system, 0.2)
        ]
        
        # Convergence tracking
        self.convergence_window = deque(maxlen=10)
        self.stagnation_threshold = 5

    def _cached_fingerprint(self, smiles: str):
        """Cache molecular fingerprints for efficiency"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        except Exception:
            pass
        return None

    def run(self,
            target_interactions: List[str],
            population_size: int = None,
            generations: int = None,
            crossover_rate: float = None,
            mutation_rate: float = None,
            elitism: int = None,
            tournament_size: int = None,
            seed_population: Optional[List[Chem.Mol]] = None) -> List[Chem.Mol]:
        """Execute enhanced GA with adaptive strategies"""

        # Use provided parameters or defaults from config
        pop_size = population_size or self.ga_config['population_size']
        max_gens = generations or self.ga_config['generations']
        crossover_prob = crossover_rate or self.adaptive_rates['crossover_rate']
        mutation_prob = mutation_rate or self.adaptive_rates['mutation_rate']
        elite_count = elitism or self.ga_config['elitism']
        tournament_k = tournament_size or self.adaptive_rates['tournament_size']
        
        print(f"Starting Enhanced GA with {pop_size} individuals for {max_gens} generations")
        
        # Initialize population with improved diversity
        if seed_population and len(seed_population) > 0:
            population = self._initialize_diverse_population(seed_population, pop_size, target_interactions)
        else:
            population = self._initialize_population_multi_strategy(pop_size, target_interactions)

        if not population:
            warnings.warn("Failed to initialize population")
            return []

        # Create optimization history
        optimization_history = OptimizationHistory(
            method="Enhanced_GA",
            states=[],
            start_time=self._get_current_time()
        )

        # Evaluate initial population with caching
        fitness_scores = self._evaluate_population_cached(population, gen_round=0)
        self._update_statistics(population, fitness_scores, 0)
        self._update_elite_archive(population, fitness_scores)

        # Evolution loop with adaptive strategies
        for generation in range(1, max_gens + 1):
            print(f"Generation {generation}/{max_gens}")
            
            # Adaptive parameter adjustment
            self._adapt_parameters(generation, max_gens)
            
            # Enhanced selection and reproduction
            new_population = []
            
            # Dynamic elitism based on convergence
            dynamic_elite_count = self._calculate_dynamic_elitism(elite_count, generation)
            
            if dynamic_elite_count > 0:
                elites = self._select_elites_diverse(population, fitness_scores, dynamic_elite_count)
                new_population.extend(elites)
            
            # Generate offspring with improved strategies
            offspring_count = 0
            max_offspring_attempts = pop_size * 3  # Prevent infinite loops
            
            while len(new_population) < pop_size and offspring_count < max_offspring_attempts:
                offspring_count += 1
                
                # Enhanced parent selection with mating restrictions
                parent1, parent2 = self._enhanced_parent_selection(
                    population, fitness_scores, tournament_k
                )
                
                # Multi-point crossover with validation
                child = None
                if self.rng.random() < self.adaptive_rates['crossover_rate']:
                    child = self._enhanced_crossover(parent1, parent2, target_interactions)
                    self.ga_stats['crossover_attempts'] += 1
                    if child is not None:
                        self.ga_stats['successful_crossovers'] += 1
                
                # Fallback to better parent if crossover fails
                if child is None:
                    child = parent1 if self._get_cached_fitness(parent1) >= self._get_cached_fitness(parent2) else parent2
                
                # Enhanced mutation with operator selection
                if self.rng.random() < self.adaptive_rates['mutation_rate']:
                    mutated = self._adaptive_mutation(child, target_interactions)
                    self.ga_stats['mutation_attempts'] += 1
                    if mutated is not None:
                        child = mutated
                        self.ga_stats['successful_mutations'] += 1
                
                # Validation with relaxed diversity requirements
                if self._validate_and_accept_child(child, new_population, generation):
                    new_population.append(child)

            # Ensure exact population size
            if len(new_population) < pop_size:
                # Fill remaining slots with archive members or random generation
                remaining = pop_size - len(new_population)
                fill_population = self._fill_population_smartly(remaining, target_interactions, new_population)
                new_population.extend(fill_population)
            
            population = new_population[:pop_size]
            
            # Evaluate with caching and update archives
            fitness_scores = self._evaluate_population_cached(population, gen_round=generation)
            self._update_statistics(population, fitness_scores, generation)
            self._update_elite_archive(population, fitness_scores)
            
            # Enhanced convergence tracking
            best_idx = np.argmax(fitness_scores)
            best_molecule = population[best_idx]
            best_score = fitness_scores[best_idx]
            avg_score = np.mean(fitness_scores)
            diversity = self.diversity.calculate_population_diversity(population)
            
            self.convergence_window.append(best_score)
            
            state = OptimizationState(
                iteration=generation,
                current_population=population.copy(),
                current_scores=fitness_scores.copy(),
                best_molecule=best_molecule,
                best_score=best_score,
                population_diversity=diversity,
                convergence_metric=self._enhanced_convergence_metric()
            )
            
            optimization_history.states.append(state)
            
            print(f"Best: {best_score:.3f}, Avg: {avg_score:.3f}, Diversity: {diversity:.3f}, "
                  f"CX Rate: {self.adaptive_rates['crossover_rate']:.2f}, "
                  f"Mut Rate: {self.adaptive_rates['mutation_rate']:.2f}")
            
            # Enhanced early stopping
            if self._check_enhanced_convergence(generation):
                print(f"Enhanced convergence detected at generation {generation}")
                self.ga_stats['early_stopping_triggered'] = True
                break

        optimization_history.end_time = self._get_current_time()
        self.ga_stats['total_generations'] = generation

        # Return enhanced final selection
        return self._select_final_diverse_molecules(population, fitness_scores, elite_count)

    def _initialize_diverse_population(self, seed: List[Chem.Mol], size: int, 
                                     target_interactions: List[str]) -> List[Chem.Mol]:
        """Initialize population with enhanced diversity from seed"""
        valid_seed = [m for m in seed if self.assembly._validate_molecule(m)]
        
        # Cluster seed molecules for diversity
        if len(valid_seed) > size:
            diverse_seed = self.diversity.cluster_and_select(valid_seed, n_clusters=size)
            return diverse_seed
        
        # Ensure diversity bookkeeping
        for mol in valid_seed:
            self.diversity.is_diverse(mol)
        
        population = valid_seed.copy()
        needed = size - len(population)
        
        if needed > 0:
            additional = self._initialize_population_multi_strategy(needed, target_interactions)
            population.extend(additional)
        
        return population[:size]

    def _initialize_population_multi_strategy(self, size: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Initialize population using multiple strategies for diversity"""
        molecules = []
        strategies = [
            (self._generate_structure_guided, 0.4),
            (self._generate_fragment_based, 0.3),
            (self._generate_scaffold_based, 0.2),
            (self._generate_simple_diverse, 0.1)
        ]
        
        for strategy, fraction in strategies:
            target_count = int(size * fraction)
            strategy_molecules = strategy(target_count, target_interactions)
            molecules.extend(strategy_molecules)
            
            if len(molecules) >= size:
                break
        
        # Fill remaining if needed
        while len(molecules) < size:
            mol = self._generate_fallback_molecule(target_interactions)
            if mol:
                molecules.append(mol)
            else:
                break
        
        return molecules[:size]

    def _generate_structure_guided(self, count: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Generate molecules using structure-guided assembly"""
        molecules = []
        attempts = 0
        max_attempts = count * 5
        
        while len(molecules) < count and attempts < max_attempts:
            attempts += 1
            mol_list = self.assembly.generate_structure_guided(target_interactions, n_molecules=1)
            
            if mol_list and len(mol_list) > 0:
                candidate = mol_list[0]
                if (self.assembly._validate_molecule(candidate) and 
                    self.diversity.is_diverse(candidate)):
                    molecules.append(candidate)
        
        return molecules

    def _generate_fragment_based(self, count: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Generate molecules by combining fragments"""
        molecules = []
        
        # Check if fragment library exists
        if not (hasattr(self.assembly, 'fragment_lib') and 
                hasattr(self.assembly.fragment_lib, 'fragments')):
            return molecules
        
        fragment_lib = self.assembly.fragment_lib
        
        for _ in range(count * 3):  # More attempts
            try:
                # Select 2-4 compatible fragments
                n_frags = self.rng.randint(2, 4)
                fragments = []
                
                for interaction in target_interactions[:n_frags]:
                    if interaction in fragment_lib.fragments:
                        frag_list = fragment_lib.fragments[interaction]
                        if frag_list:
                            fragments.append(self.rng.choice(frag_list))
                
                if len(fragments) >= 2:
                    # Try to combine fragments
                    combined_mol = self._combine_fragments_intelligently(fragments)
                    if combined_mol and self.diversity.is_diverse(combined_mol):
                        molecules.append(combined_mol)
                        
                        if len(molecules) >= count:
                            break
            
            except Exception:
                continue
        
        return molecules

    def _generate_scaffold_based(self, count: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Generate molecules based on common scaffolds"""
        molecules = []
        common_scaffolds = [
            'c1ccccc1',           # Benzene
            'c1ccncc1',           # Pyridine
            'c1ccc2ccccc2c1',     # Naphthalene
            'c1cnccn1',           # Pyrimidine
            'c1csc2ccccc12',      # Benzothiophene
            'c1ccc2[nH]ccc2c1',   # Indole
        ]
        
        for scaffold_smiles in common_scaffolds:
            if len(molecules) >= count:
                break
                
            try:
                scaffold = Chem.MolFromSmiles(scaffold_smiles)
                if scaffold:
                    # Add substituents based on target interactions
                    decorated = self._decorate_scaffold(scaffold, target_interactions)
                    if decorated and self.diversity.is_diverse(decorated):
                        molecules.append(decorated)
            except Exception:
                continue
        
        return molecules

    def _generate_simple_diverse(self, count: int, target_interactions: List[str]) -> List[Chem.Mol]:
        """Generate simple but diverse molecules"""
        molecules = []
        simple_patterns = [
            'CCO', 'CC(=O)O', 'c1ccc(O)cc1', 'c1ccc(N)cc1', 'c1ccc(C)cc1',
            'CC(C)C', 'c1coc2ccccc12', 'c1cnc2ccccc2c1', 'c1ccc(S)cc1'
        ]
        
        for pattern in simple_patterns[:count]:
            try:
                mol = Chem.MolFromSmiles(pattern)
                if mol and self.assembly._validate_molecule(mol):
                    molecules.append(mol)
            except Exception:
                continue
        
        return molecules

    def _evaluate_population_cached(self, population: List[Chem.Mol], gen_round: int) -> List[float]:
        """Evaluate population with intelligent caching"""
        fitness_scores = []
        
        for mol in population:
            smiles = Chem.MolToSmiles(mol)
            
            # Check cache first
            if smiles in self.fitness_cache:
                fitness = self.fitness_cache[smiles]
            else:
                try:
                    score_result = self.scorer.calculate_comprehensive_score(mol, generation_round=gen_round)
                    fitness = score_result.total_score
                    
                    # Apply enhanced diversity pressure
                    if self.ga_config.get('diversity_pressure', 0) > 0:
                        diversity_bonus = self._enhanced_diversity_bonus(mol, population)
                        fitness += self.ga_config['diversity_pressure'] * diversity_bonus
                    
                    # Cache the result with memory management
                    self.fitness_cache[smiles] = fitness
                    
                    # Limit cache size to prevent memory issues
                    if len(self.fitness_cache) > 10000:
                        # Remove oldest 20% of entries (simple FIFO)
                        items_to_remove = list(self.fitness_cache.keys())[:2000]
                        for key in items_to_remove:
                            del self.fitness_cache[key]
                
                except Exception as e:
                    warnings.warn(f"Error evaluating molecule: {e}")
                    fitness = 0.0
                    self.fitness_cache[smiles] = fitness
            
            fitness_scores.append(fitness)
        
        return fitness_scores

    def _get_cached_fitness(self, mol: Chem.Mol) -> float:
        """Get cached fitness or calculate if needed"""
        smiles = Chem.MolToSmiles(mol)
        if smiles in self.fitness_cache:
            return self.fitness_cache[smiles]
        
        try:
            fitness = self.scorer.calculate_comprehensive_score(mol).total_score
            self.fitness_cache[smiles] = fitness
            return fitness
        except Exception:
            return 0.0

    def _adapt_parameters(self, generation: int, max_generations: int):
        """Adapt GA parameters based on progress and performance"""
        progress = generation / max_generations
        
        # Adaptive crossover rate - decrease over time for exploitation
        base_cx_rate = self.ga_config.get('crossover_rate', 0.8)
        self.adaptive_rates['crossover_rate'] = base_cx_rate * (1.0 - 0.3 * progress)
        
        # Adaptive mutation rate - increase if stagnating
        base_mut_rate = self.ga_config.get('mutation_rate', 0.2)
        stagnation_bonus = 0.1 if self.ga_stats['fitness_stagnation_count'] > 3 else 0.0
        self.adaptive_rates['mutation_rate'] = min(0.5, base_mut_rate + stagnation_bonus)
        
        # Adaptive tournament size - increase for more selection pressure
        base_tournament = self.ga_config.get('tournament_size', 3)
        self.adaptive_rates['tournament_size'] = min(7, int(base_tournament + progress * 2))

    def _calculate_dynamic_elitism(self, base_elite_count: int, generation: int) -> int:
        """Calculate dynamic elitism based on convergence state"""
        if self.ga_stats['fitness_stagnation_count'] > 3:
            # Reduce elitism when stagnating to allow more exploration
            return max(1, base_elite_count // 2)
        else:
            return base_elite_count

    def _enhanced_parent_selection(self, population: List[Chem.Mol], fitness: List[float], 
                                 tournament_k: int) -> Tuple[Chem.Mol, Chem.Mol]:
        """Enhanced parent selection with mating restrictions"""
        parent1 = self._tournament_selection(population, fitness, k=tournament_k)
        
        # Ensure some diversity in parent selection
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            parent2 = self._tournament_selection(population, fitness, k=tournament_k)
            
            # Check if parents are too similar
            if not self._are_parents_too_similar(parent1, parent2):
                return parent1, parent2
            
            attempts += 1
        
        # Fallback to any second parent if similarity check fails
        parent2 = self._tournament_selection(population, fitness, k=tournament_k)
        return parent1, parent2

    def _are_parents_too_similar(self, parent1: Chem.Mol, parent2: Chem.Mol, 
                                threshold: float = 0.9) -> bool:
        """Check if parents are too similar for effective crossover"""
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(parent1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(parent2, radius=2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity > threshold
        except Exception:
            return False

    def _enhanced_crossover(self, parent1: Chem.Mol, parent2: Chem.Mol, 
                          target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Enhanced crossover with multiple strategies"""
        crossover_strategies = [
            self._brics_crossover,
            self._scaffold_crossover,
            self._fragment_swap_crossover
        ]
        
        # Try strategies in order of preference
        for strategy in crossover_strategies:
            try:
                child = strategy(parent1, parent2, target_interactions)
                if child is not None and self.assembly._validate_molecule(child):
                    return child
            except Exception:
                continue
        
        return None

    def _brics_crossover(self, parent1: Chem.Mol, parent2: Chem.Mol, 
                        target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Improved BRICS-based crossover"""
        try:
            frags1 = list(BRICS.BRICSDecompose(parent1, minFragmentSize=2))
            frags2 = list(BRICS.BRICSDecompose(parent2, minFragmentSize=2))
            
            if not frags1 or not frags2:
                return None

            # Intelligent fragment selection based on target interactions
            selected_frags = self._select_fragments_by_interaction(
                frags1, frags2, target_interactions
            )

            # Build molecules with size constraints
            for child in BRICS.BRICSBuild(selected_frags, minFragmentSize=2):
                try:
                    Chem.SanitizeMol(child)
                    
                    # Check molecular properties
                    if self._is_reasonable_molecule(child):
                        return child
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None

    def _scaffold_crossover(self, parent1: Chem.Mol, parent2: Chem.Mol, 
                          target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Scaffold-based crossover"""
        try:
            scaffold1 = MurckoScaffold.GetScaffoldForMol(parent1)
            scaffold2 = MurckoScaffold.GetScaffoldForMol(parent2)
            
            # Choose one scaffold and decorations from the other
            chosen_scaffold = self.rng.choice([scaffold1, scaffold2])
            other_parent = parent2 if chosen_scaffold == scaffold1 else parent1
            
            # Attempt to transfer substituents
            decorated = self._transfer_substituents(chosen_scaffold, other_parent)
            
            if decorated and self.assembly._validate_molecule(decorated):
                return decorated
                
        except Exception:
            pass
        
        return None

    def _fragment_swap_crossover(self, parent1: Chem.Mol, parent2: Chem.Mol, 
                                target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Fragment swapping crossover"""
        try:
            # Simple SMILES-based fragment swapping
            smiles1 = Chem.MolToSmiles(parent1)
            smiles2 = Chem.MolToSmiles(parent2)
            
            # Find common substructures and swap them
            if len(smiles1) > 10 and len(smiles2) > 10:
                # Simple substring swapping
                mid1 = len(smiles1) // 2
                mid2 = len(smiles2) // 2
                
                hybrid_smiles = smiles1[:mid1] + smiles2[mid2:]
                child = Chem.MolFromSmiles(hybrid_smiles)
                
                if child:
                    Chem.SanitizeMol(child)
                    return child
                    
        except Exception:
            pass
        
        return None

    def _adaptive_mutation(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Adaptive mutation with operator selection based on success rates"""
        # Calculate operator weights based on success rates
        operator_weights = []
        for operator, base_weight in self.mutation_operators:
            op_name = operator.__name__
            stats = self.ga_stats['operator_success_rates'][op_name]
            
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']
                # Bias towards successful operators
                weight = base_weight * (1.0 + success_rate)
            else:
                weight = base_weight
            
            operator_weights.append(weight)
        
        # Weighted selection of mutation operator
        total_weight = sum(operator_weights)
        if total_weight == 0:
            return None
        
        r = self.rng.random() * total_weight
        cumulative = 0
        
        for (operator, _), weight in zip(self.mutation_operators, operator_weights):
            cumulative += weight
            if r <= cumulative:
                op_name = operator.__name__
                self.ga_stats['operator_success_rates'][op_name]['attempts'] += 1
                
                try:
                    mutated = operator(mol, target_interactions)
                    if mutated is not None:
                        self.ga_stats['operator_success_rates'][op_name]['successes'] += 1
                        return mutated
                except Exception:
                    pass
                break
        
        return None

    def _validate_and_accept_child(self, child: Chem.Mol, current_population: List[Chem.Mol], 
                                  generation: int) -> bool:
        """Enhanced child validation with relaxed diversity in later generations"""
        if not self.assembly._validate_molecule(child):
            return False
        
        # Relaxed diversity requirements in later generations
        if generation > 20:  # After 20 generations, be less strict about diversity
            return True
        
        return self.diversity.is_diverse(child)

    def _fill_population_smartly(self, count: int, target_interactions: List[str], 
                               existing_population: List[Chem.Mol]) -> List[Chem.Mol]:
        """Smart population filling when generation fails"""
        molecules = []
        
        # First try elite archive
        for elite_mol in self.elite_archive[-count:]:
            if len(molecules) < count:
                # Mutate elite molecules slightly
                mutated = self._light_mutation(elite_mol, target_interactions)
                if mutated:
                    molecules.append(mutated)
                else:
                    molecules.append(elite_mol)
        
        # Fill remaining with new generation
        remaining = count - len(molecules)
        if remaining > 0:
            new_molecules = self._generate_simple_diverse(remaining, target_interactions)
            molecules.extend(new_molecules)
        
        return molecules

    def _light_mutation(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Light mutation for filling population"""
        try:
            # Simple substituent addition/modification
            return self._mutate_add_substituent(mol, target_interactions)
        except Exception:
            return None

    def _update_elite_archive(self, population: List[Chem.Mol], fitness_scores: List[float]):
        """Update elite archive with best molecules"""
        elite_pairs = list(zip(population, fitness_scores))
        elite_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Add top performers to archive
        for mol, fitness in elite_pairs[:5]:  # Top 5
            self.elite_archive.append(mol)
        
        # Limit archive size and ensure diversity
        if len(self.elite_archive) > 50:
            # Keep diverse elites
            diverse_elites = self.diversity.cluster_and_select(self.elite_archive, n_clusters=30)
            self.elite_archive = diverse_elites

    def _enhanced_diversity_bonus(self, mol: Chem.Mol, population: List[Chem.Mol]) -> float:
        """Enhanced diversity bonus calculation"""
        try:
            smiles = Chem.MolToSmiles(mol)
            fp = self._cached_fingerprint(smiles)
            
            if fp is None:
                return 0.0
            
            similarities = []
            for other_mol in population:
                if other_mol == mol:
                    continue
                
                try:
                    other_smiles = Chem.MolToSmiles(other_mol)
                    other_fp = self._cached_fingerprint(other_smiles)
                    
                    if other_fp:
                        similarity = DataStructs.TanimotoSimilarity(fp, other_fp)
                        similarities.append(similarity)
                except Exception:
                    continue
            
            if similarities:
                # Use both average and minimum similarity for bonus
                avg_similarity = np.mean(similarities)
                min_similarity = min(similarities)
                return (1.0 - avg_similarity) * (1.0 - min_similarity)
            else:
                return 1.0
                
        except Exception:
            return 0.0

    def _enhanced_convergence_metric(self) -> float:
        """Enhanced convergence detection"""
        if len(self.convergence_window) < 5:
            return 1.0
        
        recent_scores = list(self.convergence_window)
        
        # Check for stagnation
        if len(recent_scores) >= 5:
            improvement = recent_scores[-1] - recent_scores[-5]
            if abs(improvement) < 0.001:
                self.ga_stats['fitness_stagnation_count'] += 1
            else:
                self.ga_stats['fitness_stagnation_count'] = 0
        
        return abs(recent_scores[-1] - recent_scores[0]) if len(recent_scores) > 1 else 1.0

    def _check_enhanced_convergence(self, generation: int) -> bool:
        """Enhanced convergence checking"""
        if generation < 10:
            return False
        
        # Check fitness stagnation
        if self.ga_stats['fitness_stagnation_count'] >= self.stagnation_threshold:
            return True
        
        # Check diversity collapse
        if self.ga_stats['diversity_history']:
            current_diversity = self.ga_stats['diversity_history'][-1]
            if current_diversity < 0.05:  # Very low diversity threshold
                return True
        
        # Check convergence window variance
        if len(self.convergence_window) >= 8:
            variance = np.var(list(self.convergence_window))
            if variance < 1e-6:  # Very small variance
                return True
        
        return False

    def _select_final_diverse_molecules(self, population: List[Chem.Mol], 
                                      fitness_scores: List[float], 
                                      target_count: int) -> List[Chem.Mol]:
        """Select final diverse set of top molecules"""
        # Combine current population with elite archive
        all_molecules = population + self.elite_archive
        all_fitness = fitness_scores + [self._get_cached_fitness(mol) for mol in self.elite_archive]
        
        # Remove duplicates while preserving fitness
        unique_molecules = []
        unique_fitness = []
        seen_smiles = set()
        
        for mol, fit in zip(all_molecules, all_fitness):
            smiles = Chem.MolToSmiles(mol)
            if smiles not in seen_smiles:
                unique_molecules.append(mol)
                unique_fitness.append(fit)
                seen_smiles.add(smiles)
        
        # Sort by fitness
        ranked = sorted(zip(unique_molecules, unique_fitness), key=lambda x: x[1], reverse=True)
        
        # Select top performers with diversity
        top_candidates = [mol for mol, _ in ranked[:min(50, len(ranked))]]
        
        # Use diversity manager for final selection
        final_count = min(target_count * 3, len(top_candidates))
        return self.diversity.cluster_and_select(top_candidates, n_clusters=final_count)

    # Enhanced helper methods
    
    def _select_fragments_by_interaction(self, frags1: List[str], frags2: List[str], 
                                       target_interactions: List[str]) -> Set[str]:
        """Select fragments based on target interactions"""
        combined_frags = set(frags1 + frags2)
        
        # If we have interaction information, prefer relevant fragments
        if (target_interactions and 
            hasattr(self.assembly, 'fragment_lib') and
            hasattr(self.assembly.fragment_lib, 'fragments')):
            
            relevant_frags = set()
            for interaction in target_interactions:
                if interaction in self.assembly.fragment_lib.fragments:
                    interaction_frags = self.assembly.fragment_lib.fragments[interaction]
                    for frag in interaction_frags:
                        if hasattr(frag, 'smiles') and frag.smiles in combined_frags:
                            relevant_frags.add(frag.smiles)
            
            if relevant_frags:
                # Mix relevant and random fragments
                n_relevant = min(len(relevant_frags), 3)
                n_random = min(len(combined_frags) - len(relevant_frags), 2)
                
                selected = set(list(relevant_frags)[:n_relevant])
                remaining = combined_frags - relevant_frags
                selected.update(list(remaining)[:n_random])
                return selected
        
        # Fallback to mixed selection
        n1 = max(1, min(len(frags1), self.rng.randint(1, 3)))
        n2 = max(1, min(len(frags2), self.rng.randint(1, 3)))
        
        selected1 = set(self.rng.sample(frags1, n1))
        selected2 = set(self.rng.sample(frags2, n2))
        
        return selected1.union(selected2)

    def _is_reasonable_molecule(self, mol: Chem.Mol) -> bool:
        """Check if molecule has reasonable properties"""
        try:
            # Basic property checks
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            if mw > 800 or mw < 100:  # Reasonable MW range
                return False
            
            # Check for reasonable number of atoms
            num_atoms = mol.GetNumAtoms()
            if num_atoms > 60 or num_atoms < 5:
                return False
            
            # Check rotatable bonds
            rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rot_bonds > 15:
                return False
            
            # Check ring count
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            if ring_count > 6:
                return False
            
            return True
            
        except Exception:
            return False

    def _combine_fragments_intelligently(self, fragments: List) -> Optional[Chem.Mol]:
        """Intelligently combine fragments"""
        try:
            if len(fragments) < 2:
                return None
            
            # Convert fragments to SMILES if needed
            fragment_smiles = []
            for frag in fragments:
                if hasattr(frag, 'smiles'):
                    fragment_smiles.append(frag.smiles)
                elif isinstance(frag, str):
                    fragment_smiles.append(frag)
                else:
                    continue
            
            if len(fragment_smiles) < 2:
                return None
            
            # Try BRICS assembly
            for mol in BRICS.BRICSBuild(fragment_smiles, minFragmentSize=2):
                try:
                    Chem.SanitizeMol(mol)
                    if self._is_reasonable_molecule(mol):
                        return mol
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None

    def _decorate_scaffold(self, scaffold: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Decorate scaffold with relevant substituents"""
        try:
            # Find attachment points on scaffold
            scaffold_copy = Chem.RWMol(scaffold)
            
            # Simple decoration: add methyl groups to available positions
            for atom in scaffold_copy.GetAtoms():
                if atom.GetTotalNumHs() > 0 and self.rng.random() < 0.3:
                    # Add a simple substituent
                    methyl_idx = scaffold_copy.AddAtom(Chem.Atom(6))  # Carbon
                    scaffold_copy.AddBond(atom.GetIdx(), methyl_idx, Chem.BondType.SINGLE)
            
            # Sanitize and return
            try:
                Chem.SanitizeMol(scaffold_copy)
                return scaffold_copy.GetMol()
            except Exception:
                return scaffold
                
        except Exception:
            return None

    def _transfer_substituents(self, scaffold: Chem.Mol, donor_mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Transfer substituents from donor to scaffold"""
        try:
            # This is a simplified implementation
            # In practice, this would require more sophisticated substructure matching
            scaffold_copy = Chem.RWMol(scaffold)
            
            # Simple approach: if donor has more atoms, try to add some functionality
            if donor_mol.GetNumAtoms() > scaffold.GetNumAtoms():
                # Add a functional group
                for atom in scaffold_copy.GetAtoms():
                    if atom.GetTotalNumHs() > 0 and self.rng.random() < 0.2:
                        # Add oxygen (for OH group)
                        o_idx = scaffold_copy.AddAtom(Chem.Atom(8))  # Oxygen
                        scaffold_copy.AddBond(atom.GetIdx(), o_idx, Chem.BondType.SINGLE)
                        break
            
            try:
                Chem.SanitizeMol(scaffold_copy)
                return scaffold_copy.GetMol()
            except Exception:
                return scaffold
                
        except Exception:
            return None

    def _generate_fallback_molecule(self, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Generate fallback molecule when other methods fail"""
        fallback_smiles = [
            'c1ccccc1',           # Benzene
            'c1ccncc1',           # Pyridine
            'CCO',                # Ethanol
            'CC(=O)O',            # Acetic acid
            'c1ccc(O)cc1',        # Phenol
        ]
        
        smiles = self.rng.choice(fallback_smiles)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol and self.assembly._validate_molecule(mol):
                return mol
        except Exception:
            pass
        
        return None

    def _select_elites_diverse(self, population: List[Chem.Mol], fitness: List[float], k: int) -> List[Chem.Mol]:
        """Select diverse elites instead of just top fitness"""
        if k <= 0:
            return []
        
        # Get top performers (more than needed)
        top_indices = np.argsort(fitness)[::-1][:min(k * 3, len(population))]
        top_molecules = [population[i] for i in top_indices]
        
        # Use diversity manager to select diverse subset
        if len(top_molecules) <= k:
            return top_molecules
        else:
            return self.diversity.cluster_and_select(top_molecules, n_clusters=k)

    # Enhanced mutation operators with better error handling
    
    def _mutate_bioisostere(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Enhanced bioisosteric replacement mutation"""
        try:
            bioisosteres = {}
            if (hasattr(self.assembly, 'fragment_lib') and 
                hasattr(self.assembly.fragment_lib, 'bioisosteres')):
                bioisosteres = getattr(self.assembly.fragment_lib, 'bioisosteres', {})
            
            if not bioisosteres:
                # Fallback bioisosteres
                bioisosteres = {
                    'c1ccccc1': ['c1ccncc1', 'c1cnccc1'],  # Benzene to pyridines
                    'C(=O)O': ['C(=O)N', 'C#N'],           # Carboxylic acid replacements
                    'C#N': ['C(=O)N', 'C(=O)O'],           # Nitrile replacements
                }
            
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
            
            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol:
                Chem.SanitizeMol(new_mol)
                if self._is_reasonable_molecule(new_mol):
                    return new_mol
        
        except Exception:
            pass
        
        return None

    def _mutate_add_substituent(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Enhanced substituent addition mutation"""
        try:
            # Get substituents from fragment library
            substituents = []
            if hasattr(self.assembly.fragment_lib, 'fragments'):
                substituents = self.assembly.fragment_lib.fragments.get('substituents', [])
            
            # Fallback substituents
            if not substituents:
                fallback_substituents = [
                    'C',      # Methyl
                    'O',      # Hydroxyl
                    'N',      # Amino
                    'F',      # Fluoro
                    'CC',     # Ethyl
                    'C(=O)O', # Carboxyl
                ]
                substituents = [type('Substituent', (), {'smiles': s, 'interaction_types': target_interactions})() 
                              for s in fallback_substituents]
            
            if not substituents:
                return None
            
            # Filter by target interactions if possible
            suitable_subs = []
            for interaction in target_interactions:
                for sub in substituents:
                    if hasattr(sub, 'interaction_types') and interaction in sub.interaction_types:
                        suitable_subs.append(sub)
            
            if not suitable_subs:
                suitable_subs = substituents
            
            substituent = self.rng.choice(suitable_subs)
            
            # Try to add substituent using assembly method
            if hasattr(self.assembly, '_add_substituent'):
                return self.assembly._add_substituent(mol, substituent)
            else:
                # Fallback: simple functional group addition
                return self._simple_substituent_addition(mol, substituent)
        
        except Exception:
            pass
        
        return None

    def _simple_substituent_addition(self, mol: Chem.Mol, substituent) -> Optional[Chem.Mol]:
        """Simple substituent addition as fallback"""
        try:
            mol_copy = Chem.RWMol(mol)
            
            # Find an atom with available hydrogens
            for atom in mol_copy.GetAtoms():
                if atom.GetTotalNumHs() > 0 and self.rng.random() < 0.5:
                    # Add a simple group (e.g., methyl)
                    new_atom_idx = mol_copy.AddAtom(Chem.Atom(6))  # Carbon
                    mol_copy.AddBond(atom.GetIdx(), new_atom_idx, Chem.BondType.SINGLE)
                    break
            
            Chem.SanitizeMol(mol_copy)
            return mol_copy.GetMol()
            
        except Exception:
            return None

    def _mutate_modify_functional_group(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Enhanced functional group modification"""
        try:
            # Enhanced functional group modifications
            modifications = {
                'C(=O)O': ['C(=O)N', 'C(=O)OC', 'C#N'],           # Carboxylic acid variants
                'C#N': ['C(=O)N', 'C(=O)O', 'N'],                 # Nitrile variants
                'c1ccccc1': ['c1ccncc1', 'c1cnccc1', 'c1ncccc1'],  # Benzene to pyridines
                'C(F)(F)F': ['C#N', 'C(=O)N', 'C'],               # Trifluoromethyl replacements
                'O': ['N', 'S', 'C(=O)O'],                        # Hydroxyl replacements
                'N': ['O', 'C(=O)N', 'C#N'],                      # Amino replacements
            }
            
            mol_smiles = Chem.MolToSmiles(mol)
            
            # Try multiple modifications
            for _ in range(3):  # Multiple attempts
                available_mods = [(orig, repl) for orig, repls in modifications.items() 
                                for repl in repls if orig in mol_smiles]
                
                if not available_mods:
                    break
                
                original, replacement = self.rng.choice(available_mods)
                new_smiles = mol_smiles.replace(original, replacement, 1)
                
                try:
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if new_mol:
                        Chem.SanitizeMol(new_mol)
                        if self._is_reasonable_molecule(new_mol):
                            return new_mol
                except Exception:
                    continue
        
        except Exception:
            pass
        
        return None

    def _mutate_ring_system(self, mol: Chem.Mol, target_interactions: List[str]) -> Optional[Chem.Mol]:
        """Enhanced ring system mutation"""
        try:
            # Enhanced ring modifications
            ring_modifications = {
                'c1ccccc1': ['c1ccncc1', 'c1cnccc1', 'c1ncccc1', 'c1ccc2ccccc2c1'],  # Benzene variants
                'c1ccncc1': ['c1ccccc1', 'c1cncnc1', 'c1cnccn1'],                    # Pyridine variants
                'c1cnccc1': ['c1ccccc1', 'c1ccncc1', 'c1ncccc1'],                    # Other pyridines
                'c1cnccn1': ['c1ccncc1', 'c1cncnc1'],                                # Pyrimidine variants
            }
            
            mol_smiles = Chem.MolToSmiles(mol)
            
            # Try multiple ring modifications
            for _ in range(3):
                available_rings = [(orig, repls) for orig, repls in ring_modifications.items() 
                                 if orig in mol_smiles]
                
                if not available_rings:
                    break
                
                original, replacements = self.rng.choice(available_rings)
                replacement = self.rng.choice(replacements)
                new_smiles = mol_smiles.replace(original, replacement, 1)
                
                try:
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if new_mol:
                        Chem.SanitizeMol(new_mol)
                        if self._is_reasonable_molecule(new_mol):
                            return new_mol
                except Exception:
                    continue
        
        except Exception:
            pass
        
        return None

    # Keep all existing methods unchanged
    def _tournament_selection(self, population: List[Chem.Mol], fitness: List[float], k: int = 3) -> Chem.Mol:
        """Tournament selection (unchanged)"""
        k = max(2, min(k, len(population)))
        tournament_indices = self.rng.sample(range(len(population)), k)
        best_idx = max(tournament_indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _update_statistics(self, population: List[Chem.Mol], fitness: List[float], generation: int):
        """Update GA statistics (unchanged)"""
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

    def _get_current_time(self) -> float:
        """Get current time"""
        return time.time()

    def get_statistics(self) -> Dict:
        """Get comprehensive GA statistics with enhancements"""
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
        
        # Enhanced statistics
        stats['cache_size'] = len(self.fitness_cache)
        stats['elite_archive_size'] = len(self.elite_archive)
        stats['adaptive_crossover_rate'] = self.adaptive_rates['crossover_rate']
        stats['adaptive_mutation_rate'] = self.adaptive_rates['mutation_rate']
        stats['fitness_stagnation_count'] = self.ga_stats['fitness_stagnation_count']
        
        # Operator performance
        stats['operator_performance'] = {}
        for op_name, op_stats in self.ga_stats['operator_success_rates'].items():
            if op_stats['attempts'] > 0:
                stats['operator_performance'][op_name] = op_stats['successes'] / op_stats['attempts']
            else:
                stats['operator_performance'][op_name] = 0.0
        
        return stats

    # Keep remaining methods unchanged from original implementation
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
            'diversity_history': [],
            'operator_success_rates': defaultdict(lambda: {'attempts': 0, 'successes': 0}),
            'fitness_stagnation_count': 0,
            'early_stopping_triggered': False
        }
        
        self.population_history = []
        self.fitness_history = []
        self.elite_archive = []
        self.fitness_cache = {}
        
        # Reset adaptive parameters
        self.adaptive_rates = {
            'crossover_rate': self.ga_config.get('crossover_rate', 0.8),
            'mutation_rate': self.ga_config.get('mutation_rate', 0.2),
            'tournament_size': self.ga_config.get('tournament_size', 3)
        }
        
        self.convergence_window.clear()

    def save_population(self, filepath: str, generation: int = -1):
        """Save population to SDF file (unchanged but enhanced)"""
        if not self.population_history:
            return
        
        population = self.population_history[generation]
        fitness = self.fitness_history[generation]
        
        try:
            writer = Chem.SDWriter(filepath)
            
            for i, (mol, fit) in enumerate(zip(population, fitness)):
                # Add properties to molecule
                mol.SetProp("Generation", str(abs(generation)))
                mol.SetProp("Individual_ID", str(i))
                mol.SetProp("Fitness", str(fit))
                mol.SetProp("SMILES", Chem.MolToSmiles(mol))
                
                # Add calculated properties
                try:
                    mol.SetProp("MW", str(round(rdMolDescriptors.CalcExactMolWt(mol), 2)))
                    mol.SetProp("LogP", str(round(Chem.Crippen.MolLogP(mol), 2)))
                    mol.SetProp("HBD", str(Lipinski.NumHDonors(mol)))
                    mol.SetProp("HBA", str(Lipinski.NumHAcceptors(mol)))
                    mol.SetProp("TPSA", str(round(rdMolDescriptors.CalcTPSA(mol), 2)))
                    mol.SetProp("NumRings", str(rdMolDescriptors.CalcNumRings(mol)))
                except Exception:
                    pass
                
                writer.write(mol)
            
            writer.close()
            
        except Exception as e:
            warnings.warn(f"Failed to save population: {e}")

    def load_population(self, filepath: str) -> List[Chem.Mol]:
        """Load population from SDF file (unchanged)"""
        molecules = []
        
        try:
            supplier = Chem.SDMolSupplier(filepath)
            
            for mol in supplier:
                if mol is not None:
                    molecules.append(mol)
            
        except Exception as e:
            warnings.warn(f"Failed to load population: {e}")
        
        return molecules

    def analyze_population_chemistry(self, generation: int = -1) -> Dict:
        """Analyze chemical space of population (enhanced)"""
        if not self.population_history:
            return {}
        
        population = self.population_history[generation]
        
        analysis = {
            'molecular_weights': [],
            'logp_values': [],
            'ring_counts': [],
            'tpsa_values': [],
            'hbd_counts': [],
            'hba_counts': [],
            'functional_groups': defaultdict(int),
            'scaffolds': defaultdict(int),
            'complexity_scores': []
        }
        
        for mol in population:
            try:
                # Basic properties
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                logp = Chem.Crippen.MolLogP(mol)
                rings = rdMolDescriptors.CalcNumRings(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                
                analysis['molecular_weights'].append(mw)
                analysis['logp_values'].append(logp)
                analysis['ring_counts'].append(rings)
                analysis['tpsa_values'].append(tpsa)
                analysis['hbd_counts'].append(hbd)
                analysis['hba_counts'].append(hba)
                
                # Complexity
                complexity = rdMolDescriptors.BertzCT(mol)
                analysis['complexity_scores'].append(complexity)
                
                # Scaffold analysis
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                analysis['scaffolds'][scaffold_smiles] += 1
                
                # Enhanced functional group analysis
                smiles = Chem.MolToSmiles(mol)
                functional_groups = {
                    'carboxylic_acid': 'C(=O)O',
                    'amide': 'C(=O)N',
                    'nitrile': 'C#N',
                    'sulfonyl': 'S(=O)(=O)',
                    'hydroxyl': 'O',
                    'amino': 'N',
                    'ester': 'C(=O)OC',
                    'ether': 'COC',
                    'halogen': ['F', 'Cl', 'Br', 'I']
                }
                
                for fg_name, patterns in functional_groups.items():
                    if isinstance(patterns, list):
                        for pattern in patterns:
                            if pattern in smiles:
                                analysis['functional_groups'][fg_name] += 1
                                break
                    else:
                        if patterns in smiles:
                            analysis['functional_groups'][fg_name] += 1
                
            except Exception:
                continue
        
        # Calculate enhanced statistics
        for prop in ['molecular_weights', 'logp_values', 'tpsa_values', 'complexity_scores']:
            if analysis[prop]:
                analysis[f'{prop}_mean'] = np.mean(analysis[prop])
                analysis[f'{prop}_std'] = np.std(analysis[prop])
                analysis[f'{prop}_median'] = np.median(analysis[prop])
        
        # Drug-likeness analysis
        analysis['drug_like_count'] = 0
        analysis['lipinski_violations'] = []
        
        for mol in population:
            try:
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                logp = Chem.Crippen.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                
                violations = 0
                if mw > 650: violations += 1
                if logp > 5: violations += 1
                if hbd > 5: violations += 1
                if hba > 10: violations += 1
                
                analysis['lipinski_violations'].append(violations)
                if violations <= 1:  # Allow 1 violation
                    analysis['drug_like_count'] += 1
                    
            except Exception:
                continue
        
        if analysis['lipinski_violations']:
            analysis['avg_lipinski_violations'] = np.mean(analysis['lipinski_violations'])
            analysis['drug_like_percentage'] = (analysis['drug_like_count'] / len(population)) * 100
        
        return analysis
