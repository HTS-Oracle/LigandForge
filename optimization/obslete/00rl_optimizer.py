"""
Reinforcement Learning Optimizer Module
RL-based molecular optimization using policy gradient methods
"""

import numpy as np
import random
import math
from typing import List, Dict, Optional, Tuple, Any
from collections import deque, defaultdict
import warnings

from rdkit import Chem

from config import LigandForgeConfig
from data_structures import OptimizationState, OptimizationHistory
from scoring import MultiObjectiveScorer
from molecular_assembly import StructureGuidedAssembly


class RLOptimizer:
    """Reinforcement learning-based molecular optimization"""
    
    def __init__(self, config: LigandForgeConfig, scorer: MultiObjectiveScorer, 
                 assembly: StructureGuidedAssembly):
        self.config = config
        self.scorer = scorer
        self.assembly = assembly
        
        # RL state and history
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.state_history = []
        
        # Policy parameters - Enhanced with adaptive learning
        self.action_weights = self._initialize_action_weights()
        self.exploration_rate = self.config.exploration_factor
        self.learning_rate = getattr(self.config, 'learning_rate', 0.01)
        
        # Experience replay buffer with prioritization
        self.experience_buffer = deque(maxlen=self.config.rl_memory_size)
        self.priority_buffer = deque(maxlen=self.config.rl_memory_size)
        
        # Performance tracking with enhanced metrics
        self.optimization_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'average_reward': 0.0,
            'best_reward': 0.0,
            'policy_updates': 0,
            'convergence_history': [],
            'action_performance': defaultdict(list),
            'exploration_efficiency': 0.0
        }
        
        # Adaptive thresholds
        self.quality_threshold = 0.7
        self.convergence_patience = 5
        self.stagnation_counter = 0
        
        # Action success tracking for intelligent exploration
        self.action_success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
        
        # Initialize _last_population_scores
        self._last_population_scores = []
    
    def optimize_generation(self, initial_molecules: List[Chem.Mol], 
                          n_iterations: int = 20) -> List[Chem.Mol]:
        """Optimize molecule generation using RL with enhanced strategy"""
        
        if not initial_molecules:
            warnings.warn("No initial molecules provided for RL optimization")
            return []
        
        current_population = initial_molecules.copy()
        best_molecules = []
        best_scores_history = []
        
        optimization_history = OptimizationHistory(
            method="RL",
            states=[],
            start_time=self._get_current_time()
        )
        
        # Dynamic population sizing
        base_population_size = len(initial_molecules)
        
        for iteration in range(n_iterations):
            print(f"RL Iteration {iteration + 1}/{n_iterations}")
            
            # Adaptive episode size based on performance
            episode_size = self._calculate_adaptive_episode_size(iteration, base_population_size)
            
            # Generate new molecules with enhanced strategy
            new_molecules = self._generate_episode(current_population, iteration, episode_size)
            
            # Evaluate and select with diversity consideration
            all_molecules, all_scores = self._evaluate_population_with_diversity(
                current_population + new_molecules, iteration
            )
            
            if not all_molecules:
                continue
            
            # Enhanced selection strategy
            current_population, current_scores = self._select_next_population(
                all_molecules, all_scores, base_population_size, iteration
            )
            
            # Track high-quality molecules with deduplication
            self._update_best_molecules(best_molecules, all_molecules, all_scores)
            
            # Enhanced state tracking
            best_score = max(current_scores) if current_scores else 0.0
            avg_score = np.mean(current_scores) if current_scores else 0.0
            diversity = self._calculate_population_diversity(current_population)
            
            best_scores_history.append(best_score)
            
            state = OptimizationState(
                iteration=iteration,
                current_population=current_population,
                current_scores=current_scores,
                best_molecule=current_population[0] if current_population else None,
                best_score=best_score,
                population_diversity=diversity,
                convergence_metric=self._calculate_enhanced_convergence_metric(best_scores_history)
            )
            
            optimization_history.states.append(state)
            self.state_history.append(state)
            
            # Enhanced policy updates with performance tracking
            if iteration > 0 and iteration % max(1, self.config.policy_update_frequency) == 0:
                self._update_policy_enhanced()
            
            # Adaptive exploration with performance feedback
            self._update_exploration_rate_adaptive(iteration, n_iterations, best_scores_history)
            
            # Convergence detection and adaptive strategies
            self._check_convergence_and_adapt(best_scores_history)
            
            print(f"Best: {best_score:.3f}, Avg: {avg_score:.3f}, Diversity: {diversity:.3f}, "
                  f"Exploration: {self.exploration_rate:.3f}")
        
        optimization_history.end_time = self._get_current_time()
        
        # Enhanced final selection with scoring
        final_molecules = self._select_final_molecules(best_molecules)
        
        return final_molecules
    
    def _calculate_adaptive_episode_size(self, iteration: int, base_size: int) -> int:
        """Calculate adaptive episode size based on performance"""
        min_size = max(5, base_size // 4)
        max_size = max(20, base_size)
        
        # Start with larger episodes, reduce as we learn
        decay_factor = math.exp(-iteration / 10.0)
        adaptive_size = int(min_size + (max_size - min_size) * decay_factor)
        
        # Boost size if we're not finding good molecules
        if len(self.reward_history) > 5:
            recent_avg = np.mean(list(self.reward_history)[-5:])
            if recent_avg < 0.3:
                adaptive_size = int(adaptive_size * 1.5)
        
        return min(max_size, max(min_size, adaptive_size))
    
    def _generate_episode(self, current_population: List[Chem.Mol], 
                         iteration: int, episode_size: int = None) -> List[Chem.Mol]:
        """Generate new molecules with enhanced strategy"""
        if episode_size is None:
            episode_size = max(10, len(current_population) // 2)
        
        new_molecules = []
        failed_attempts = 0
        max_failures = episode_size * 2
        
        while len(new_molecules) < episode_size and failed_attempts < max_failures:
            # Enhanced action selection
            action = self._select_action_enhanced(iteration)
            
            # Execute action with better error handling
            mol = self._execute_action_robust(action, current_population)
            
            if mol and self._validate_molecule_enhanced(mol):
                reward = self._calculate_reward_enhanced(mol, iteration, action)
                
                # Enhanced experience storage
                experience = {
                    'action': action,
                    'molecule': mol,
                    'reward': reward,
                    'iteration': iteration,
                    'base_population_size': len(current_population),
                    'success': True
                }
                
                # Prioritized experience replay
                priority = abs(reward - np.mean(self.reward_history)) if self.reward_history else 1.0
                self.experience_buffer.append(experience)
                self.priority_buffer.append(priority)
                
                # Track action performance
                self._update_action_success_rates(action, True, reward)
                
                self.action_history.append(action)
                self.reward_history.append(reward)
                new_molecules.append(mol)
                self.optimization_stats['successful_episodes'] += 1
            else:
                self._update_action_success_rates(action, False, 0.0)
                failed_attempts += 1
            
            self.optimization_stats['total_episodes'] += 1
        
        return new_molecules
    
    def _select_action_enhanced(self, iteration: int) -> Dict[str, Any]:
        """Enhanced action selection with performance-based weighting"""
        
        # Multi-armed bandit approach with UCB
        if iteration > 10 and random.random() > self.exploration_rate:
            action_type = self._select_action_ucb()
        elif random.random() < self.exploration_rate:
            # Intelligent exploration based on action success rates
            action_type = self._select_action_intelligent_exploration()
        else:
            action_type = self._get_best_action_type()
        
        # Enhanced parameter selection
        action = {
            'type': action_type,
            'target_interaction': self._select_target_interaction_intelligent(),
            'intensity': self._select_action_intensity_adaptive(action_type),
            'strategy': self._select_strategy_adaptive(iteration),
            'fragment_type': self._select_fragment_type(action_type),
            'confidence': self._calculate_action_confidence(action_type)
        }
        
        return action
    
    def _select_action_ucb(self) -> str:
        """Upper Confidence Bound action selection"""
        actions = list(self.action_success_rates.keys())
        base_actions = ['modify', 'extend', 'substitute', 'bioisostere', 'scaffold_hop', 'ring_modification']
        
        # Include base actions even if not in success rates yet
        all_actions = list(set(actions + base_actions))
        
        if not all_actions:
            return random.choice(base_actions)
        
        total_trials = sum(self.action_success_rates[a]['total'] for a in all_actions)
        if total_trials == 0:
            return random.choice(all_actions)
        
        ucb_values = {}
        for action in all_actions:
            stats = self.action_success_rates.get(action, {'success': 0, 'total': 0})
            if stats['total'] == 0:
                ucb_values[action] = float('inf')
            else:
                avg_reward = stats['success'] / stats['total']
                confidence = math.sqrt(2 * math.log(total_trials) / stats['total'])
                ucb_values[action] = avg_reward + confidence
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def _select_action_intelligent_exploration(self) -> str:
        """Intelligent exploration focusing on under-explored promising actions"""
        actions = ['modify', 'extend', 'substitute', 'bioisostere', 'scaffold_hop', 'ring_modification']
        
        # Prefer actions with few trials but potential
        exploration_scores = {}
        for action in actions:
            stats = self.action_success_rates.get(action, {'success': 0, 'total': 0})
            
            # Reward under-explored actions
            exploration_bonus = 1.0 / max(1, stats['total']) 
            
            # Consider success rate if we have data
            success_rate = stats['success'] / max(1, stats['total'])
            
            exploration_scores[action] = exploration_bonus + success_rate * 0.5
        
        # Weighted random selection
        total_score = sum(exploration_scores.values())
        if total_score > 0:
            probs = [exploration_scores[a] / total_score for a in actions]
            return np.random.choice(actions, p=probs)
        
        return random.choice(actions)
    
    def _execute_action_robust(self, action: Dict[str, Any], 
                              population: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """Execute action with enhanced robustness and fallback strategies"""
        if not population:
            return None
        
        action_type = action['type']
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Select base molecule with strategy
                base_mol = self._select_base_molecule_strategic(population, action)
                
                result = None
                if action_type == 'modify':
                    result = self._modify_molecule_enhanced(base_mol, action)
                elif action_type == 'extend':
                    result = self._extend_molecule_enhanced(base_mol, action)
                elif action_type == 'substitute':
                    result = self._substitute_group_enhanced(base_mol, action)
                elif action_type == 'bioisostere':
                    result = self._apply_bioisostere_enhanced(base_mol, action)
                elif action_type == 'scaffold_hop':
                    result = self._scaffold_hop_enhanced(base_mol, action)
                elif action_type == 'ring_modification':
                    result = self._modify_ring_system_enhanced(base_mol, action)
                
                if result:
                    return result
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    warnings.warn(f"Action {action_type} failed after {max_attempts} attempts: {e}")
                continue
        
        return None
    
    def _select_base_molecule_strategic(self, population: List[Chem.Mol], 
                                       action: Dict[str, Any]) -> Chem.Mol:
        """Strategically select base molecule based on action and confidence"""
        if len(population) == 1:
            return population[0]
        
        strategy = action.get('strategy', 'random')
        confidence = action.get('confidence', 0.5)
        
        if strategy == 'conservative' or confidence < 0.3:
            # Pick from best performers if we have scores
            if hasattr(self, '_last_population_scores') and self._last_population_scores:
                if len(self._last_population_scores) == len(population):
                    scored_mols = list(zip(population, self._last_population_scores))
                    scored_mols.sort(key=lambda x: x[1], reverse=True)
                    top_n = max(1, len(population) // 3)
                    return random.choice(scored_mols[:top_n])[0]
        
        return random.choice(population)
    
    def _modify_molecule_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced molecule modification with better fragment selection"""
        target_interaction = action['target_interaction']
        intensity = action.get('intensity', 0.5)
        
        # Get fragments with quality filtering
        suitable_fragments = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
        
        if not suitable_fragments:
            # Fallback to any suitable fragments
            suitable_fragments = getattr(self.assembly.fragment_lib, 'all_fragments', [])
        
        if not suitable_fragments:
            return None
        
        # Select fragment based on intensity and diversity
        if intensity > 0.7:  # High intensity - try novel fragments
            fragment = random.choice(suitable_fragments)
        else:  # Conservative - use proven fragments
            # Sort by some quality metric if available
            fragment = suitable_fragments[0] if suitable_fragments else None
        
        if fragment:
            return self.assembly._add_substituent(mol, fragment)
        
        return None
    
    def _extend_molecule_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced molecule extension with smart site selection"""
        growth_sites = self.assembly._find_growth_sites(mol)
        if not growth_sites:
            return None
        
        # Select growth site strategically
        confidence = action.get('confidence', 0.5)
        if confidence > 0.6 and len(growth_sites) > 1:
            # Use site scoring if available
            site = self._select_best_growth_site(growth_sites, mol)
        else:
            site = random.choice(growth_sites)
        
        target_interaction = action['target_interaction']
        suitable_fragments = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
        
        if suitable_fragments:
            fragment = random.choice(suitable_fragments)
            return self.assembly._grow_at_site(mol, fragment, site)
        
        return None
    
    def _substitute_group_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced group substitution with better site selection"""
        try:
            editable = Chem.RWMol(mol)
            
            # Enhanced site finding
            substitution_sites = self._find_substitution_sites_enhanced(editable)
            
            if not substitution_sites:
                return None
            
            # Strategic site selection
            intensity = action.get('intensity', 0.5)
            if intensity > 0.6 and len(substitution_sites) > 1:
                site = self._score_substitution_sites(substitution_sites, editable)
            else:
                site = random.choice(substitution_sites)
            
            target_interaction = action['target_interaction']
            
            # Enhanced attachment
            star = Chem.Atom(0)
            star.SetAtomMapNum(1)
            star_idx = editable.AddAtom(star)
            editable.AddBond(site, star_idx, Chem.BondType.SINGLE)
            
            temp_mol = editable.GetMol()
            Chem.SanitizeMol(temp_mol)
            
            suitable_fragments = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
            if suitable_fragments:
                fragment = self._select_fragment_by_intensity(suitable_fragments, action)
                return self.assembly._attach_fragment(temp_mol, fragment, 1)
            
        except Exception:
            return None
        
        return None
    
    def _apply_bioisostere_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced bioisosteric replacement with better selection logic"""
        bioisosteres = self.assembly.fragment_lib.bioisosteres
        mol_smiles = Chem.MolToSmiles(mol)
        
        intensity = action.get('intensity', 0.5)
        strategy = action.get('strategy', 'moderate')
        
        # Prioritize replacements by frequency and success
        available_replacements = list(bioisosteres.keys())
        
        # Sort by frequency in molecule (more frequent = more conservative)
        replacement_scores = []
        for original in available_replacements:
            if original in mol_smiles:
                frequency = mol_smiles.count(original)
                replacements = bioisosteres[original]
                
                if replacements:
                    # Score based on strategy
                    if strategy == 'conservative':
                        score = frequency  # Prefer common patterns
                    else:
                        score = 1.0 / (frequency + 1)  # Prefer rare patterns
                    
                    replacement_scores.append((original, score, replacements))
        
        # Sort by score
        replacement_scores.sort(key=lambda x: x[1], reverse=True)
        
        for original, _, replacements in replacement_scores:
            # Select replacement based on intensity
            if intensity > 0.8:
                replacement = random.choice(replacements)
            elif intensity > 0.5:
                replacement = replacements[len(replacements)//2] if len(replacements) > 1 else replacements[0]
            else:
                replacement = replacements[0]
            
            new_smiles = mol_smiles.replace(original, replacement, 1)
            try:
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol:
                    Chem.SanitizeMol(new_mol)
                    return new_mol
            except:
                continue
        
        return None
    
    def _scaffold_hop_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced scaffold hopping with preservation strategy"""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            target_interaction = action['target_interaction']
            
            suitable_cores = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
            cores = [c for c in suitable_cores if getattr(c, 'scaffold_type', None) == 'core']
            
            if not cores:
                # Fallback to any core-like fragments
                all_fragments = getattr(self.assembly.fragment_lib, 'all_fragments', [])
                cores = [f for f in all_fragments if 'core' in getattr(f, 'tags', [])]
            
            if not cores:
                return None
            
            # Select new core based on strategy
            strategy = action.get('strategy', 'moderate')
            if strategy == 'conservative':
                # Choose structurally similar cores
                new_core = self._find_similar_core(scaffold, cores)
            else:
                new_core = random.choice(cores)
            
            if new_core:
                new_scaffold = Chem.MolFromSmiles(new_core.smiles)
                if new_scaffold:
                    # In a full implementation, would preserve side chains
                    return new_scaffold
                    
        except Exception:
            return None
        
        return None
    
    def _modify_ring_system_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced ring system modification"""
        try:
            strategy = action.get('strategy', 'add')
            intensity = action.get('intensity', 0.5)
            
            if strategy == 'add' or (strategy == 'moderate' and intensity > 0.5):
                return self._add_ring_enhanced(mol, action)
            elif strategy == 'expand':
                return self._expand_ring_enhanced(mol, action)
            else:
                return mol
                
        except Exception:
            return None
    
    def _add_ring_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced ring addition with better atom selection"""
        try:
            editable = Chem.RWMol(mol)
            intensity = action.get('intensity', 0.5)
            
            # Find suitable atoms for ring formation
            candidate_atoms = []
            for atom in editable.GetAtoms():
                if (atom.GetDegree() < 3 and 
                    atom.GetSymbol() in ['C', 'N'] and 
                    atom.GetTotalNumHs() > 0):
                    candidate_atoms.append(atom)
            
            if len(candidate_atoms) < 2:
                return None
            
            # Select atoms strategically
            if intensity > 0.7:
                # High intensity - try larger rings
                ring_size = random.choice([6, 7])
                atom1, atom2 = random.sample(candidate_atoms, 2)
            else:
                # Conservative - use 5-6 membered rings
                ring_size = random.choice([5, 6])
                # Select closer atoms if possible
                atom1, atom2 = random.sample(candidate_atoms[:min(4, len(candidate_atoms))], 2)
            
            # Add carbons to form ring
            new_atoms = []
            for i in range(ring_size - 2):
                carbon = Chem.Atom(6)
                new_atom_idx = editable.AddAtom(carbon)
                new_atoms.append(new_atom_idx)
            
            # Connect atoms to form ring
            if new_atoms:
                editable.AddBond(atom1.GetIdx(), new_atoms[0], Chem.BondType.SINGLE)
                for i in range(len(new_atoms) - 1):
                    editable.AddBond(new_atoms[i], new_atoms[i+1], Chem.BondType.SINGLE)
                editable.AddBond(new_atoms[-1], atom2.GetIdx(), Chem.BondType.SINGLE)
            
            result = editable.GetMol()
            Chem.SanitizeMol(result)
            return result
                
        except Exception:
            pass
        
        return None
    
    def _expand_ring_enhanced(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Enhanced ring expansion"""
        try:
            ring_info = mol.GetRingInfo()
            if ring_info.NumRings() == 0:
                return None
            
            # For now, return original (would implement expansion logic)
            return mol
            
        except Exception:
            return None
    
    def _calculate_reward_enhanced(self, mol: Chem.Mol, iteration: int, action: Dict[str, Any]) -> float:
        """Enhanced reward calculation with action-specific bonuses"""
        try:
            score_result = self.scorer.calculate_comprehensive_score(mol, iteration)
            base_reward = score_result.total_score
            
            # Action-specific bonuses
            action_bonus = 0.0
            action_type = action['type']
            
            if action_type == 'bioisostere' and score_result.drug_likeness_score > 0.7:
                action_bonus += 0.05
            elif action_type == 'extend' and score_result.pharmacophore_score > 0.6:
                action_bonus += 0.03
            elif action_type == 'scaffold_hop' and score_result.novelty_score > 0.8:
                action_bonus += 0.08
            
            # Strategy alignment bonus
            strategy = action.get('strategy', 'moderate')
            if strategy == 'conservative' and len(score_result.violations) == 0:
                action_bonus += 0.02
            elif strategy == 'aggressive' and score_result.novelty_score > 0.7:
                action_bonus += 0.03
            
            # Confidence-based scaling
            confidence = action.get('confidence', 0.5)
            if confidence > 0.7 and base_reward > 0.6:
                action_bonus *= 1.2
            
            total_reward = base_reward + action_bonus
            
            # Enhanced penalty system
            penalty = 0.0
            if score_result.violations:
                penalty = 0.03 * len(score_result.violations)
            
            # Molecular complexity penalty for overly complex molecules
            if hasattr(score_result, 'complexity_score') and score_result.complexity_score > 0.8:
                penalty += 0.02
            
            final_reward = max(0.0, min(1.0, total_reward - penalty))
            
            return final_reward
            
        except Exception:
            return 0.0
    
    def _evaluate_population_with_diversity(self, molecules: List[Chem.Mol], 
                                          iteration: int) -> Tuple[List[Chem.Mol], List[float]]:
        """Enhanced population evaluation with diversity consideration"""
        if not molecules:
            return [], []
        
        # Calculate base scores
        evaluated = []
        for mol in molecules:
            try:
                score_result = self.scorer.calculate_comprehensive_score(mol, iteration)
                base_score = score_result.total_score
                
                # Add diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(mol, [m for m, _ in evaluated])
                
                final_score = base_score + diversity_bonus * 0.1  # 10% weight for diversity
                evaluated.append((mol, final_score))
                
            except Exception:
                continue
        
        if not evaluated:
            return [], []
        
        molecules, scores = zip(*evaluated)
        self._last_population_scores = list(scores)  # Store for strategic selection
        
        return list(molecules), list(scores)
    
    def _select_next_population(self, molecules: List[Chem.Mol], scores: List[float], 
                               target_size: int, iteration: int) -> Tuple[List[Chem.Mol], List[float]]:
        """Enhanced population selection with diversity preservation"""
        if not molecules:
            return [], []
        
        ranked_molecules = list(zip(molecules, scores))
        ranked_molecules.sort(key=lambda x: x[1], reverse=True)
        
        # Dynamic selection strategy
        if iteration < 5:  # Early iterations - focus on quality
            selected = ranked_molecules[:target_size]
        else:  # Later iterations - balance quality and diversity
            # Select top performers
            top_fraction = max(1, target_size // 2)
            top_molecules = ranked_molecules[:top_fraction]
            
            # Select diverse molecules from remaining
            remaining = ranked_molecules[top_fraction:]
            diverse_molecules = self._select_diverse_molecules(remaining, target_size - top_fraction)
            
            selected = top_molecules + diverse_molecules
        
        if not selected:
            return [], []
        
        molecules, scores = zip(*selected)
        return list(molecules), list(scores)
    
    def _update_best_molecules(self, best_molecules: List[Chem.Mol], 
                              all_molecules: List[Chem.Mol], all_scores: List[float]):
        """Update best molecules list with quality filtering"""
        threshold = max(self.quality_threshold, np.mean(all_scores) + np.std(all_scores))
        
        for mol, score in zip(all_molecules, all_scores):
            if score > threshold:
                # Check for duplicates before adding
                smiles = Chem.MolToSmiles(mol, canonical=True)
                existing_smiles = {Chem.MolToSmiles(m, canonical=True) for m in best_molecules}
                
                if smiles not in existing_smiles:
                    best_molecules.append(mol)
    
    def _update_policy_enhanced(self):
        """Enhanced policy update with prioritized experience replay"""
        if len(self.reward_history) < max(5, self.config.rl_batch_size // 2):
            return
        
        batch_size = min(self.config.rl_batch_size, len(self.experience_buffer))
        
        # Prioritized sampling
        if len(self.priority_buffer) >= batch_size:
            priorities = list(self.priority_buffer)[-batch_size:]
            probabilities = np.array(priorities) / sum(priorities) if sum(priorities) > 0 else None
            
            if probabilities is not None:
                indices = np.random.choice(batch_size, size=batch_size, p=probabilities)
                experiences = [list(self.experience_buffer)[-batch_size:][i] for i in indices]
            else:
                experiences = list(self.experience_buffer)[-batch_size:]
        else:
            experiences = list(self.experience_buffer)[-batch_size:]
        
        # Enhanced advantage calculation
        rewards = [exp['reward'] for exp in experiences]
        baseline = np.mean(rewards)
        advantages = [r - baseline for r in rewards]
        
        # Update action weights with momentum and regularization
        momentum = 0.9
        regularization = 0.01
        
        action_updates = defaultdict(float)
        action_counts = defaultdict(int)
        
        for exp, advantage in zip(experiences, advantages):
            action_type = exp['action']['type']
            action_updates[action_type] += advantage
            action_counts[action_type] += 1
        
        # Apply updates with momentum
        for action_type in action_updates:
            if action_counts[action_type] > 0:
                avg_advantage = action_updates[action_type] / action_counts[action_type]
                
                # Momentum update
                old_weight = self.action_weights.get(action_type, 1.0 / len(self.action_weights))
                update = self.learning_rate * avg_advantage
                
                # Apply momentum and regularization
                new_weight = momentum * old_weight + (1 - momentum) * (old_weight + update)
                new_weight = max(0.01, new_weight - regularization * old_weight)  # L2 regularization
                
                self.action_weights[action_type] = new_weight
        
        # Normalize weights
        total_weight = sum(self.action_weights.values())
        if total_weight > 0:
            self.action_weights = {k: v / total_weight for k, v in self.action_weights.items()}
        
        # Update learning rate (adaptive)
        recent_variance = np.var(rewards) if len(rewards) > 1 else 0.5
        if recent_variance < 0.01:  # Low variance - increase learning rate
            self.learning_rate = min(0.1, self.learning_rate * 1.05)
        elif recent_variance > 0.1:  # High variance - decrease learning rate
            self.learning_rate = max(0.001, self.learning_rate * 0.95)
        
        self.optimization_stats['policy_updates'] += 1
        self.optimization_stats['convergence_history'].append(recent_variance)
    
    def _update_exploration_rate_adaptive(self, iteration: int, total_iterations: int, 
                                         best_scores_history: List[float]):
        """Adaptive exploration rate based on performance trends"""
        base_decay_rate = 3.0 / total_iterations
        base_exploration = self.config.exploration_factor * math.exp(-base_decay_rate * iteration)
        
        # Performance-based adjustment
        if len(best_scores_history) >= 3:
            recent_trend = np.mean(best_scores_history[-3:]) - np.mean(best_scores_history[-6:-3]) if len(best_scores_history) >= 6 else 0
            
            if recent_trend < -0.01:  # Performance declining - increase exploration
                exploration_boost = 1.5
            elif recent_trend > 0.02:  # Performance improving - can reduce exploration
                exploration_boost = 0.8
            else:  # Stable performance
                exploration_boost = 1.0
            
            self.exploration_rate = base_exploration * exploration_boost
        else:
            self.exploration_rate = base_exploration
        
        # Bounds
        self.exploration_rate = max(0.02, min(0.8, self.exploration_rate))
        
        # Update efficiency tracking
        if self.optimization_stats['total_episodes'] > 0:
            self.optimization_stats['exploration_efficiency'] = (
                self.optimization_stats['successful_episodes'] / self.optimization_stats['total_episodes']
            )
    
    def _check_convergence_and_adapt(self, best_scores_history: List[float]):
        """Check for convergence and adapt strategy accordingly"""
        if len(best_scores_history) < self.convergence_patience:
            return
        
        # Check for stagnation
        recent_scores = best_scores_history[-self.convergence_patience:]
        score_variance = np.var(recent_scores)
        
        if score_variance < 0.001:  # Very low variance indicates stagnation
            self.stagnation_counter += 1
            
            if self.stagnation_counter >= 3:
                # Implement anti-stagnation measures
                self._apply_anti_stagnation_measures()
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0
    
    def _apply_anti_stagnation_measures(self):
        """Apply measures to escape local optima"""
        print("Applying anti-stagnation measures...")
        
        # Temporarily increase exploration
        self.exploration_rate = min(0.6, self.exploration_rate * 2.0)
        
        # Reset some action weights to encourage exploration
        num_actions = len(self.action_weights)
        for action in random.sample(list(self.action_weights.keys()), 
                                   max(1, num_actions // 2)):
            self.action_weights[action] = 1.0 / num_actions
        
        # Normalize
        total_weight = sum(self.action_weights.values())
        if total_weight > 0:
            self.action_weights = {k: v / total_weight for k, v in self.action_weights.items()}
    
    def _update_action_success_rates(self, action: Dict[str, Any], success: bool, reward: float):
        """Update action success rates for intelligent exploration"""
        action_type = action['type']
        self.action_success_rates[action_type]['total'] += 1
        
        if success:
            # Weight success by reward quality
            success_weight = 1.0 if reward > 0.5 else 0.5
            self.action_success_rates[action_type]['success'] += success_weight
        
        # Track performance by action parameters
        strategy = action.get('strategy', 'moderate')
        strategy_key = f"{action_type}_{strategy}"
        
        if strategy_key not in self.optimization_stats['action_performance']:
            self.optimization_stats['action_performance'][strategy_key] = []
        
        self.optimization_stats['action_performance'][strategy_key].append(reward)
    
    def _calculate_enhanced_convergence_metric(self, best_scores_history: List[float]) -> float:
        """Enhanced convergence metric considering trends and variance"""
        if len(best_scores_history) < 5:
            return 1.0
        
        # Combine variance and trend information
        recent_scores = best_scores_history[-5:]
        variance_component = float(np.var(recent_scores))
        
        # Trend component
        if len(best_scores_history) >= 10:
            early_avg = np.mean(best_scores_history[:5])
            recent_avg = np.mean(recent_scores)
            trend_component = max(0, 1.0 - (recent_avg - early_avg))
        else:
            trend_component = 0.5
        
        return 0.7 * variance_component + 0.3 * trend_component
    
    def _calculate_diversity_bonus(self, mol: Chem.Mol, existing_molecules: List[Chem.Mol]) -> float:
        """Calculate diversity bonus for population selection"""
        if not existing_molecules:
            return 0.1  # Base diversity bonus for first molecule
        
        try:
            from rdkit import DataStructs
            from rdkit.Chem import rdMolDescriptors
            
            mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
            
            similarities = []
            for existing_mol in existing_molecules:
                try:
                    existing_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(existing_mol, 2)
                    similarity = DataStructs.TanimotoSimilarity(mol_fp, existing_fp)
                    similarities.append(similarity)
                except:
                    continue
            
            if similarities:
                max_similarity = max(similarities)
                diversity_bonus = 1.0 - max_similarity  # Higher bonus for more diverse molecules
                return diversity_bonus
            
        except ImportError:
            # Fallback if rdkit modules not available
            pass
        except Exception:
            pass
        
        return 0.05  # Default small bonus
    
    def _select_diverse_molecules(self, molecules_scores: List[Tuple], target_count: int) -> List[Tuple]:
        """Select diverse molecules from candidates"""
        if not molecules_scores or target_count <= 0:
            return []
        
        if len(molecules_scores) <= target_count:
            return molecules_scores
        
        selected = []
        remaining = molecules_scores.copy()
        
        # Start with highest scoring molecule
        selected.append(remaining.pop(0))
        
        # Iteratively select most diverse molecules
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_diversity_score = -1
            best_idx = -1
            
            for idx, (mol, score) in enumerate(remaining):
                selected_mols = [m for m, _ in selected]
                diversity_bonus = self._calculate_diversity_bonus(mol, selected_mols)
                diversity_score = 0.7 * score + 0.3 * diversity_bonus  # Weight score and diversity
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = (mol, score)
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    def _validate_molecule_enhanced(self, mol: Chem.Mol) -> bool:
        """Enhanced molecule validation with additional checks"""
        try:
            # Basic validation
            if not self.assembly._validate_molecule(mol):
                return False
            
            # Additional checks
            
            # Size constraints
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 5 or num_atoms > 100:  # Reasonable size limits
                return False
            
            # Complexity check
            num_rings = mol.GetRingInfo().NumRings()
            if num_rings > 8:  # Too complex
                return False
            
            # Basic drug-like properties
            mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            if mw > 1000:  # Too heavy
                return False
            
            # Check for problematic substructures (basic)
            smiles = Chem.MolToSmiles(mol)
            problematic_patterns = ['[N+](=O)[O-]', '[S+2]', '[P+]']  # Basic reactive patterns
            
            for pattern in problematic_patterns:
                if pattern in smiles:
                    return False
            
            return True
            
        except:
            return False
    
    def _select_target_interaction_intelligent(self) -> str:
        """Intelligent target interaction selection based on pocket analysis and success history"""
        interactions = ['hbd', 'hba', 'hydrophobic', 'aromatic', 'electrostatic']
        
        # Prioritize based on pocket hotspots
        if hasattr(self.assembly, 'pocket') and self.assembly.pocket.hotspots:
            pocket_interactions = [h.interaction_type for h in self.assembly.pocket.hotspots]
            
            # Weight by hotspot strength if available
            weighted_interactions = []
            for hotspot in self.assembly.pocket.hotspots:
                strength = getattr(hotspot, 'strength', 1.0)
                for _ in range(int(strength * 3)):  # Weight by strength
                    weighted_interactions.append(hotspot.interaction_type)
            
            if weighted_interactions:
                return random.choice(weighted_interactions)
        
        # Fallback to performance-based selection
        interaction_performance = {}
        for interaction in interactions:
            # Look for performance data
            performance_key = f"interaction_{interaction}"
            if performance_key in self.optimization_stats['action_performance']:
                avg_performance = np.mean(self.optimization_stats['action_performance'][performance_key])
                interaction_performance[interaction] = avg_performance
        
        if interaction_performance:
            # Weighted selection based on performance
            interactions_list = list(interaction_performance.keys())
            weights = list(interaction_performance.values())
            total_weight = sum(weights)
            
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                return np.random.choice(interactions_list, p=probabilities)
        
        return random.choice(interactions)
    
    def _select_action_intensity_adaptive(self, action_type: str) -> float:
        """Adaptive action intensity based on action type and current performance"""
        base_intensity = random.uniform(0.2, 0.9)
        
        # Adjust based on action type performance
        action_performance = self.optimization_stats['action_performance'].get(action_type, [])
        
        if action_performance:
            recent_performance = np.mean(action_performance[-5:]) if len(action_performance) >= 5 else np.mean(action_performance)
            
            if recent_performance > 0.7:  # Good performance - can be more aggressive
                intensity_modifier = 1.2
            elif recent_performance < 0.3:  # Poor performance - be more conservative
                intensity_modifier = 0.7
            else:
                intensity_modifier = 1.0
            
            base_intensity *= intensity_modifier
        
        return max(0.1, min(1.0, base_intensity))
    
    def _select_strategy_adaptive(self, iteration: int) -> str:
        """Adaptive strategy selection based on iteration and performance"""
        strategies = ['conservative', 'moderate', 'aggressive']
        
        # Early iterations - be more conservative
        if iteration < 5:
            return random.choice(['conservative', 'moderate'])
        
        # Later iterations - adapt based on performance
        if len(self.reward_history) >= 10:
            recent_performance = np.mean(list(self.reward_history)[-10:])
            
            if recent_performance > 0.6:  # Good performance - can be aggressive
                weights = [0.2, 0.3, 0.5]
            elif recent_performance < 0.3:  # Poor performance - be conservative
                weights = [0.5, 0.4, 0.1]
            else:  # Moderate performance
                weights = [0.3, 0.4, 0.3]
            
            return np.random.choice(strategies, p=weights)
        
        return random.choice(strategies)
    
    def _calculate_action_confidence(self, action_type: str) -> float:
        """Calculate confidence in action based on historical performance"""
        action_stats = self.action_success_rates.get(action_type, {'success': 0, 'total': 0})
        
        if action_stats['total'] == 0:
            return 0.5  # Neutral confidence for untested actions
        
        success_rate = action_stats['success'] / action_stats['total']
        
        # Adjust confidence based on sample size
        sample_size_factor = min(1.0, action_stats['total'] / 20.0)  # Full confidence after 20 trials
        
        confidence = success_rate * sample_size_factor + 0.5 * (1 - sample_size_factor)
        
        return confidence
    
    def _find_substitution_sites_enhanced(self, mol: Chem.RWMol) -> List[int]:
        """Enhanced substitution site finding with scoring"""
        sites = []
        
        for atom in mol.GetAtoms():
            if (atom.GetTotalNumHs() > 0 and 
                atom.GetSymbol() in ['C', 'N', 'O', 'S'] and 
                atom.GetDegree() < 4):
                
                # Additional filters
                if atom.IsInRing() and atom.GetDegree() >= 2:
                    continue  # Avoid ring substitution that might break rings
                
                sites.append(atom.GetIdx())
        
        return sites
    
    def _score_substitution_sites(self, sites: List[int], mol: Chem.RWMol) -> int:
        """Score substitution sites and return best one"""
        if not sites:
            return 0
            
        if len(sites) == 1:
            return sites[0]
        
        site_scores = {}
        
        for site_idx in sites:
            atom = mol.GetAtomWithIdx(site_idx)
            score = 0.0
            
            # Prefer carbon atoms
            if atom.GetSymbol() == 'C':
                score += 1.0
            
            # Prefer atoms with fewer neighbors (easier substitution)
            score += (4 - atom.GetDegree()) * 0.3
            
            # Prefer atoms not in rings
            if not atom.IsInRing():
                score += 0.5
            
            site_scores[site_idx] = score
        
        # Return site with highest score
        return max(site_scores.items(), key=lambda x: x[1])[0]
    
    def _select_fragment_by_intensity(self, fragments: List, action: Dict[str, Any]):
        """Select fragment based on action intensity"""
        if not fragments:
            return None
        
        intensity = action.get('intensity', 0.5)
        
        if intensity > 0.8:  # High intensity - try diverse/novel fragments
            return random.choice(fragments)
        elif intensity > 0.4:  # Medium intensity - balanced selection
            mid_point = len(fragments) // 2
            return fragments[mid_point] if mid_point < len(fragments) else fragments[0]
        else:  # Low intensity - conservative selection
            return fragments[0]
    
    def _find_similar_core(self, scaffold: Chem.Mol, cores: List) -> Optional:
        """Find structurally similar core for conservative scaffold hopping"""
        if not cores:
            return None
        
        try:
            from rdkit import DataStructs
            from rdkit.Chem import rdMolDescriptors
            
            scaffold_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(scaffold, 2)
            
            similarities = []
            for core in cores:
                try:
                    core_mol = Chem.MolFromSmiles(core.smiles)
                    if core_mol:
                        core_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(core_mol, 2)
                        similarity = DataStructs.TanimotoSimilarity(scaffold_fp, core_fp)
                        similarities.append((core, similarity))
                except:
                    continue
            
            if similarities:
                # Return most similar core
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[0][0]
            
        except ImportError:
            # Fallback if rdkit modules not available
            pass
        except Exception:
            pass
        
        return random.choice(cores) if cores else None
    
    def _select_best_growth_site(self, sites: List, mol: Chem.Mol):
        """Select best growth site based on molecular context"""
        if len(sites) <= 1:
            return sites[0] if sites else None
        
        # Simple scoring - prefer sites away from other functional groups
        site_scores = {}
        
        for site in sites:
            score = 1.0
            
            # For simple implementations, just add some randomness
            # In practice, would analyze chemical environment
            site_scores[site] = score + random.random() * 0.1
        
        return max(site_scores.items(), key=lambda x: x[1])[0] if site_scores else sites[0]
    
    def _select_final_molecules(self, best_molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        """Enhanced final molecule selection with scoring and diversity"""
        if not best_molecules:
            return []
        
        # Remove duplicates
        unique_molecules = self._remove_duplicate_molecules(best_molecules)
        
        if len(unique_molecules) <= 50:
            return unique_molecules
        
        # Score all molecules for final ranking
        scored_molecules = []
        for mol in unique_molecules:
            try:
                score_result = self.scorer.calculate_comprehensive_score(mol, -1)  # Final scoring
                final_score = score_result.total_score
                
                # Add diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(mol, [m for m, _ in scored_molecules])
                final_score += diversity_bonus * 0.05
                
                scored_molecules.append((mol, final_score))
            except:
                continue
        
        # Sort by score and select top 50
        scored_molecules.sort(key=lambda x: x[1], reverse=True)
        
        return [mol for mol, _ in scored_molecules[:50]]