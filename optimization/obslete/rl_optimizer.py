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
        
        # Policy parameters
        self.action_weights = self._initialize_action_weights()
        self.exploration_rate = self.config.exploration_factor
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=self.config.rl_memory_size)
        
        # Performance tracking
        self.optimization_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'average_reward': 0.0,
            'best_reward': 0.0,
            'policy_updates': 0
        }
    
    def optimize_generation(self, initial_molecules: List[Chem.Mol], 
                          n_iterations: int = 20) -> List[Chem.Mol]:
        """Optimize molecule generation using RL"""
        
        if not initial_molecules:
            warnings.warn("No initial molecules provided for RL optimization")
            return []
        
        current_population = initial_molecules.copy()
        best_molecules = []
        optimization_history = OptimizationHistory(
            method="RL",
            states=[],
            start_time=self._get_current_time()
        )
        
        for iteration in range(n_iterations):
            print(f"RL Iteration {iteration + 1}/{n_iterations}")
            
            # Generate new molecules based on current policy
            new_molecules = self._generate_episode(current_population, iteration)
            
            # Evaluate and select best molecules
            all_molecules, all_scores = self._evaluate_population(
                current_population + new_molecules, iteration
            )
            
            # Update current population (keep top performers)
            population_size = len(initial_molecules)
            ranked_molecules = list(zip(all_molecules, all_scores))
            ranked_molecules.sort(key=lambda x: x[1], reverse=True)
            
            current_population = [mol for mol, _ in ranked_molecules[:population_size]]
            current_scores = [score for _, score in ranked_molecules[:population_size]]
            
            # Track best molecules
            for mol, score in ranked_molecules:
                if score > 0.7:  # High-quality threshold
                    best_molecules.append(mol)
            
            # Create optimization state
            best_score = max(current_scores) if current_scores else 0.0
            avg_score = np.mean(current_scores) if current_scores else 0.0
            diversity = self._calculate_population_diversity(current_population)
            
            state = OptimizationState(
                iteration=iteration,
                current_population=current_population,
                current_scores=current_scores,
                best_molecule=ranked_molecules[0][0] if ranked_molecules else None,
                best_score=best_score,
                population_diversity=diversity,
                convergence_metric=self._calculate_convergence_metric(iteration)
            )
            
            optimization_history.states.append(state)
            self.state_history.append(state)
            
            # Update policy periodically
            if iteration % self.config.policy_update_frequency == 0:
                self._update_policy()
            
            # Adaptive exploration rate
            self._update_exploration_rate(iteration, n_iterations)
            
            print(f"Best score: {best_score:.3f}, Avg score: {avg_score:.3f}, Diversity: {diversity:.3f}")
        
        optimization_history.end_time = self._get_current_time()
        
        # Remove duplicates from best molecules
        unique_best = self._remove_duplicate_molecules(best_molecules)
        
        return unique_best[:50]  # Return top 50 unique molecules
    
    def _generate_episode(self, current_population: List[Chem.Mol], 
                         iteration: int) -> List[Chem.Mol]:
        """Generate new molecules for one episode"""
        new_molecules = []
        episode_size = max(10, len(current_population) // 2)
        
        for _ in range(episode_size):
            # Select action based on current policy
            action = self._select_action(iteration)
            
            # Execute action to generate molecule
            mol = self._execute_action(action, current_population)
            
            if mol and self.assembly._validate_molecule(mol):
                reward = self._calculate_reward(mol, iteration)
                
                # Store experience
                experience = {
                    'action': action,
                    'molecule': mol,
                    'reward': reward,
                    'iteration': iteration
                }
                self.experience_buffer.append(experience)
                
                # Track for policy update
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                new_molecules.append(mol)
                self.optimization_stats['successful_episodes'] += 1
            
            self.optimization_stats['total_episodes'] += 1
        
        return new_molecules
    
    def _select_action(self, iteration: int) -> Dict[str, Any]:
        """Select action based on current policy with exploration"""
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Explore: random action
            action_type = random.choice([
                'modify', 'extend', 'substitute', 'bioisostere', 
                'scaffold_hop', 'ring_modification'
            ])
        else:
            # Exploit: use learned policy
            action_type = self._get_best_action_type()
        
        # Select specific parameters for the action
        action = {
            'type': action_type,
            'target_interaction': self._select_target_interaction(),
            'intensity': self._select_action_intensity(),
            'strategy': self._select_strategy(),
            'fragment_type': self._select_fragment_type(action_type)
        }
        
        return action
    
    def _execute_action(self, action: Dict[str, Any], 
                       population: List[Chem.Mol]) -> Optional[Chem.Mol]:
        """Execute the selected action to generate a molecule"""
        if not population:
            return None
        
        base_mol = random.choice(population)
        action_type = action['type']
        target_interaction = action['target_interaction']
        
        try:
            if action_type == 'modify':
                return self._modify_molecule(base_mol, action)
            elif action_type == 'extend':
                return self._extend_molecule(base_mol, action)
            elif action_type == 'substitute':
                return self._substitute_group(base_mol, action)
            elif action_type == 'bioisostere':
                return self._apply_bioisostere(base_mol, action)
            elif action_type == 'scaffold_hop':
                return self._scaffold_hop(base_mol, action)
            elif action_type == 'ring_modification':
                return self._modify_ring_system(base_mol, action)
            
        except Exception as e:
            warnings.warn(f"Action execution failed: {e}")
            return None
        
        return None
    
    def _modify_molecule(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Modify existing molecule by adding functional groups"""
        target_interaction = action['target_interaction']
        suitable_fragments = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
        
        if not suitable_fragments:
            return None
        
        fragment = random.choice(suitable_fragments)
        return self.assembly._add_substituent(mol, fragment)
    
    def _extend_molecule(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Extend molecule with new fragment"""
        growth_sites = self.assembly._find_growth_sites(mol)
        if not growth_sites:
            return None
        
        site = random.choice(growth_sites)
        target_interaction = action['target_interaction']
        suitable_fragments = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
        
        if suitable_fragments:
            fragment = random.choice(suitable_fragments)
            return self.assembly._grow_at_site(mol, fragment, site)
        
        return None
    
    def _substitute_group(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Substitute existing group with new functionality"""
        try:
            editable = Chem.RWMol(mol)
            
            # Find suitable substitution sites
            substitution_sites = []
            for atom in editable.GetAtoms():
                if (atom.GetTotalNumHs() > 0 and 
                    atom.GetSymbol() in ['C', 'N'] and 
                    atom.GetDegree() < 4):
                    substitution_sites.append(atom.GetIdx())
            
            if not substitution_sites:
                return None
            
            site = random.choice(substitution_sites)
            target_interaction = action['target_interaction']
            
            # Add attachment point
            star = Chem.Atom(0)
            star.SetAtomMapNum(1)
            star_idx = editable.AddAtom(star)
            editable.AddBond(site, star_idx, Chem.BondType.SINGLE)
            
            # Attach suitable fragment
            temp_mol = editable.GetMol()
            Chem.SanitizeMol(temp_mol)
            
            suitable_fragments = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
            if suitable_fragments:
                fragment = random.choice(suitable_fragments)
                return self.assembly._attach_fragment(temp_mol, fragment, 1)
            
        except Exception:
            return None
        
        return None
    
    def _apply_bioisostere(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Apply bioisosteric replacement"""
        bioisosteres = self.assembly.fragment_lib.bioisosteres
        
        mol_smiles = Chem.MolToSmiles(mol)
        
        # Select bioisostere replacement based on action intensity
        intensity = action.get('intensity', 0.5)
        
        available_replacements = list(bioisosteres.keys())
        random.shuffle(available_replacements)
        
        for original in available_replacements:
            if original in mol_smiles:
                replacements = bioisosteres[original]
                if replacements:
                    # Choose replacement based on intensity (conservative vs aggressive)
                    if intensity > 0.7:
                        replacement = random.choice(replacements)
                    else:
                        # Choose more conservative replacements (shorter list)
                        replacement = replacements[0] if replacements else original
                    
                    new_smiles = mol_smiles.replace(original, replacement, 1)
                    try:
                        new_mol = Chem.MolFromSmiles(new_smiles)
                        if new_mol:
                            Chem.SanitizeMol(new_mol)
                            return new_mol
                    except:
                        continue
        
        return None
    
    def _scaffold_hop(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Perform scaffold hopping"""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            # Get current scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            
            # Find alternative scaffolds
            target_interaction = action['target_interaction']
            suitable_cores = self.assembly.fragment_lib.get_fragments_for_interaction(target_interaction)
            cores = [c for c in suitable_cores if c.scaffold_type == 'core']
            
            if not cores:
                return None
            
            # Choose new core
            new_core = random.choice(cores)
            new_scaffold = Chem.MolFromSmiles(new_core.smiles)
            
            if new_scaffold is None:
                return None
            
            # Try to preserve side chains (simplified approach)
            # In practice, this would be more sophisticated
            return new_scaffold
            
        except Exception:
            return None
    
    def _modify_ring_system(self, mol: Chem.Mol, action: Dict[str, Any]) -> Optional[Chem.Mol]:
        """Modify ring system"""
        try:
            # Simple ring modification: add or remove rings
            strategy = action.get('strategy', 'add')
            
            if strategy == 'add':
                # Add a ring by cyclization
                return self._add_ring(mol)
            elif strategy == 'expand':
                # Expand existing ring
                return self._expand_ring(mol)
            else:
                # Ring contraction or other modifications
                return mol
                
        except Exception:
            return None
    
    def _add_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a ring to the molecule"""
        # Simplified ring addition
        try:
            editable = Chem.RWMol(mol)
            
            # Find two atoms that could be connected to form a ring
            atoms = [atom for atom in editable.GetAtoms() 
                    if atom.GetDegree() < 3 and atom.GetSymbol() == 'C']
            
            if len(atoms) >= 2:
                atom1, atom2 = random.sample(atoms, 2)
                
                # Add carbons to form a 5 or 6-membered ring
                ring_size = random.choice([5, 6])
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
    
    def _expand_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Expand an existing ring"""
        # Simplified ring expansion
        try:
            ring_info = mol.GetRingInfo()
            if ring_info.NumRings() == 0:
                return None
            
            # For simplicity, just return the original molecule
            # Real implementation would expand rings
            return mol
            
        except Exception:
            return None
    
    def _calculate_reward(self, mol: Chem.Mol, iteration: int) -> float:
        """Calculate reward for generated molecule"""
        try:
            score_result = self.scorer.calculate_comprehensive_score(mol, iteration)
            
            # Base reward is the total score
            reward = score_result.total_score
            
            # Bonus for meeting specific criteria
            if score_result.pharmacophore_score > 0.7:
                reward += 0.1
            
            if score_result.drug_likeness_score > 0.8:
                reward += 0.1
            
            if score_result.novelty_score > 0.6:
                reward += 0.05
            
            # Penalty for violations
            if score_result.violations:
                reward -= 0.05 * len(score_result.violations)
            
            return max(0.0, min(1.0, reward))
            
        except Exception:
            return 0.0
    
    def _evaluate_population(self, molecules: List[Chem.Mol], 
                           iteration: int) -> Tuple[List[Chem.Mol], List[float]]:
        """Evaluate population and return molecules with scores"""
        evaluated = []
        
        for mol in molecules:
            try:
                score_result = self.scorer.calculate_comprehensive_score(mol, iteration)
                evaluated.append((mol, score_result.total_score))
            except Exception:
                continue
        
        if not evaluated:
            return [], []
        
        molecules, scores = zip(*evaluated)
        return list(molecules), list(scores)
    
    def _update_policy(self):
        """Update policy based on recent experience"""
        if len(self.reward_history) < self.config.rl_batch_size:
            return
        
        # Simple policy gradient update
        recent_actions = list(self.action_history)[-self.config.rl_batch_size:]
        recent_rewards = list(self.reward_history)[-self.config.rl_batch_size:]
        
        # Calculate advantage (reward - baseline)
        baseline = np.mean(recent_rewards)
        advantages = [r - baseline for r in recent_rewards]
        
        # Update action weights based on advantages
        for action, advantage in zip(recent_actions, advantages):
            action_type = action['type']
            if advantage > 0:
                self.action_weights[action_type] *= (1 + self.config.learning_rate * advantage)
            else:
                self.action_weights[action_type] *= (1 + self.config.learning_rate * advantage * 0.5)
        
        # Normalize weights
        total_weight = sum(self.action_weights.values())
        if total_weight > 0:
            self.action_weights = {k: v / total_weight for k, v in self.action_weights.items()}
        
        self.optimization_stats['policy_updates'] += 1
    
    def _get_best_action_type(self) -> str:
        """Get best action type based on current policy"""
        if not self.action_weights:
            return random.choice(['modify', 'extend', 'substitute', 'bioisostere'])
        
        # Weighted random selection
        actions = list(self.action_weights.keys())
        weights = list(self.action_weights.values())
        
        return np.random.choice(actions, p=weights)
    
    def _select_target_interaction(self) -> str:
        """Select target interaction type"""
        interactions = ['hbd', 'hba', 'hydrophobic', 'aromatic', 'electrostatic']
        
        # Bias based on pocket analysis
        if hasattr(self.assembly, 'pocket') and self.assembly.pocket.hotspots:
            pocket_interactions = [h.interaction_type for h in self.assembly.pocket.hotspots]
            if pocket_interactions:
                return random.choice(pocket_interactions)
        
        return random.choice(interactions)
    
    def _select_action_intensity(self) -> float:
        """Select action intensity (0.1 to 1.0)"""
        return random.uniform(0.1, 1.0)
    
    def _select_strategy(self) -> str:
        """Select strategy for action"""
        strategies = ['conservative', 'moderate', 'aggressive']
        return random.choice(strategies)
    
    def _select_fragment_type(self, action_type: str) -> str:
        """Select fragment type based on action"""
        if action_type in ['modify', 'extend']:
            return random.choice(['substituent', 'linker'])
        elif action_type == 'substitute':
            return 'substituent'
        else:
            return random.choice(['core', 'substituent', 'linker'])
    
    def _initialize_action_weights(self) -> Dict[str, float]:
        """Initialize action weights for policy"""
        actions = ['modify', 'extend', 'substitute', 'bioisostere', 'scaffold_hop', 'ring_modification']
        return {action: 1.0 / len(actions) for action in actions}
    
    def _update_exploration_rate(self, iteration: int, total_iterations: int):
        """Update exploration rate based on progress"""
        # Exponential decay
        decay_rate = 3.0 / total_iterations
        self.exploration_rate = self.config.exploration_factor * math.exp(-decay_rate * iteration)
        self.exploration_rate = max(0.05, self.exploration_rate)  # Minimum exploration
    
    def _calculate_convergence_metric(self, iteration: int) -> float:
        """Calculate convergence metric"""
        if len(self.reward_history) < 10:
            return 1.0
        
        # Look at variance in recent rewards
        recent_rewards = list(self.reward_history)[-10:]
        return float(np.var(recent_rewards))
    
    def _calculate_population_diversity(self, molecules: List[Chem.Mol]) -> float:
        """Calculate population diversity"""
        try:
            return self.assembly.diversity.calculate_population_diversity(molecules)
        except:
            return 0.5
    
    def _remove_duplicate_molecules(self, molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        """Remove duplicate molecules"""
        unique_molecules = []
        seen_smiles = set()
        
        for mol in molecules:
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if smiles not in seen_smiles:
                    unique_molecules.append(mol)
                    seen_smiles.add(smiles)
            except:
                continue
        
        return unique_molecules
    
    def _get_current_time(self) -> float:
        """Get current time for timing"""
        import time
        return time.time()
    
    def get_optimization_statistics(self) -> Dict:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()
        
        if self.reward_history:
            stats['average_reward'] = np.mean(self.reward_history)
            stats['best_reward'] = max(self.reward_history)
            stats['reward_variance'] = np.var(self.reward_history)
        
        if self.optimization_stats['total_episodes'] > 0:
            stats['success_rate'] = (self.optimization_stats['successful_episodes'] / 
                                   self.optimization_stats['total_episodes'])
        
        stats['current_exploration_rate'] = self.exploration_rate
        stats['action_weights'] = self.action_weights.copy()
        
        return stats
    
    def reset_optimizer(self):
        """Reset optimizer state"""
        self.action_history.clear()
        self.reward_history.clear()
        self.state_history.clear()
        self.experience_buffer.clear()
        
        self.action_weights = self._initialize_action_weights()
        self.exploration_rate = self.config.exploration_factor
        
        self.optimization_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'average_reward': 0.0,
            'best_reward': 0.0,
            'policy_updates': 0
        }
