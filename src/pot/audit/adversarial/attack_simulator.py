"""
Attack Simulator

Simulates various adversarial attack strategies against the PoT system.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
import hashlib
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AttackStrategy(Enum):
    """Types of attack strategies"""
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    MIMICRY = "mimicry"
    BACKDOOR = "backdoor"
    EXTRACTION = "extraction"


@dataclass
class AttackScenario:
    """Configuration for an attack scenario"""
    name: str
    strategy: AttackStrategy
    target: str  # What to attack (model, verifier, audit)
    objective: str  # Goal of attack
    constraints: Dict[str, Any]
    parameters: Dict[str, Any]


@dataclass
class AttackOutcome:
    """Result of an attack simulation"""
    scenario: AttackScenario
    success: bool
    iterations: int
    queries_used: int
    detection_triggered: bool
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]


class AttackSimulator:
    """
    Simulates adversarial attacks against PoT verification systems.
    
    Features:
    - Multiple attack strategies
    - Adaptive attack planning
    - Query-efficient attacks
    - Detection evasion
    - Attack success measurement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the attack simulator.
        
        Args:
            config: Configuration for attack simulation
        """
        self.config = config or {}
        self.max_queries = self.config.get('max_queries', 1000)
        self.detection_budget = self.config.get('detection_budget', 0.1)
        self.strategies = self._initialize_strategies()
        self.query_count = 0
        self.detection_count = 0
        
    def _initialize_strategies(self) -> Dict[AttackStrategy, Callable]:
        """Initialize attack strategy implementations"""
        return {
            AttackStrategy.RANDOM: self._random_attack,
            AttackStrategy.ADAPTIVE: self._adaptive_attack,
            AttackStrategy.GRADIENT_BASED: self._gradient_attack,
            AttackStrategy.EVOLUTIONARY: self._evolutionary_attack,
            AttackStrategy.MIMICRY: self._mimicry_attack,
            AttackStrategy.BACKDOOR: self._backdoor_attack,
            AttackStrategy.EXTRACTION: self._extraction_attack
        }
    
    def simulate_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> AttackOutcome:
        """
        Simulate an attack scenario.
        
        Args:
            scenario: Attack scenario to simulate
            target_system: Target system to attack
            initial_state: Initial state for attack
            
        Returns:
            AttackOutcome with results
        """
        logger.info(f"Starting attack simulation: {scenario.name}")
        
        # Reset counters
        self.query_count = 0
        self.detection_count = 0
        
        # Get strategy implementation
        strategy_fn = self.strategies.get(scenario.strategy)
        if not strategy_fn:
            raise ValueError(f"Unknown strategy: {scenario.strategy}")
        
        # Execute attack
        start_time = time.time()
        success, artifacts = strategy_fn(scenario, target_system, initial_state)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'execution_time': elapsed_time,
            'queries_per_second': self.query_count / elapsed_time if elapsed_time > 0 else 0,
            'detection_rate': self.detection_count / self.query_count if self.query_count > 0 else 0,
            'efficiency': 1.0 / self.query_count if self.query_count > 0 and success else 0
        }
        
        return AttackOutcome(
            scenario=scenario,
            success=success,
            iterations=self.query_count,
            queries_used=self.query_count,
            detection_triggered=self.detection_count > 0,
            artifacts=artifacts,
            metrics=metrics
        )
    
    def _random_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Random attack strategy - tries random perturbations.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'attempts': []}
        success = False
        
        for i in range(min(self.max_queries, 100)):
            # Generate random perturbation
            perturbation = self._generate_random_perturbation(scenario)
            
            # Apply and test
            result = self._test_perturbation(target_system, perturbation)
            self.query_count += 1
            
            artifacts['attempts'].append({
                'iteration': i,
                'perturbation': perturbation,
                'result': result
            })
            
            if result.get('success', False):
                success = True
                artifacts['successful_perturbation'] = perturbation
                break
            
            if result.get('detected', False):
                self.detection_count += 1
                if self.detection_count / self.query_count > self.detection_budget:
                    break
        
        return success, artifacts
    
    def _adaptive_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Adaptive attack that learns from previous attempts.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'attempts': [], 'learning_curve': []}
        success = False
        
        # Initialize adaptive parameters
        exploration_rate = 1.0
        exploitation_rate = 0.0
        best_perturbation = None
        best_score = -float('inf')
        
        for i in range(min(self.max_queries, 200)):
            # Adaptive strategy selection
            if np.random.random() < exploration_rate:
                # Explore new perturbations
                perturbation = self._generate_random_perturbation(scenario)
            else:
                # Exploit best known perturbation with variation
                perturbation = self._mutate_perturbation(best_perturbation, scenario)
            
            # Test perturbation
            result = self._test_perturbation(target_system, perturbation)
            self.query_count += 1
            
            # Calculate score
            score = self._calculate_attack_score(result, scenario)
            
            # Update best
            if score > best_score:
                best_score = score
                best_perturbation = perturbation
            
            # Adapt exploration rate
            exploration_rate = max(0.1, exploration_rate * 0.99)
            exploitation_rate = 1.0 - exploration_rate
            
            artifacts['attempts'].append({
                'iteration': i,
                'score': score,
                'exploration_rate': exploration_rate
            })
            
            artifacts['learning_curve'].append(best_score)
            
            if result.get('success', False):
                success = True
                artifacts['successful_perturbation'] = perturbation
                break
            
            if result.get('detected', False):
                self.detection_count += 1
                # Increase exploration after detection
                exploration_rate = min(1.0, exploration_rate * 1.5)
        
        return success, artifacts
    
    def _gradient_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Gradient-based attack using numerical gradients.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'gradients': [], 'trajectory': []}
        success = False
        
        # Initialize perturbation
        current_perturbation = np.zeros(scenario.parameters.get('dimension', 10))
        learning_rate = scenario.parameters.get('learning_rate', 0.01)
        
        for i in range(min(self.max_queries // 2, 50)):  # Divide by 2 for gradient estimation
            # Estimate gradient
            gradient = self._estimate_gradient(
                target_system,
                current_perturbation,
                scenario
            )
            
            # Update perturbation
            current_perturbation += learning_rate * gradient
            
            # Clip to constraints
            current_perturbation = self._clip_to_constraints(
                current_perturbation,
                scenario.constraints
            )
            
            # Test current perturbation
            result = self._test_perturbation(
                target_system,
                current_perturbation.tolist()
            )
            self.query_count += 1
            
            artifacts['gradients'].append(gradient.tolist())
            artifacts['trajectory'].append(current_perturbation.tolist())
            
            if result.get('success', False):
                success = True
                artifacts['successful_perturbation'] = current_perturbation.tolist()
                break
            
            # Adaptive learning rate
            if i > 0 and i % 10 == 0:
                learning_rate *= 0.9
        
        return success, artifacts
    
    def _evolutionary_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evolutionary algorithm-based attack.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'generations': [], 'best_fitness': []}
        success = False
        
        # Initialize population
        population_size = scenario.parameters.get('population_size', 20)
        population = [
            self._generate_random_perturbation(scenario)
            for _ in range(population_size)
        ]
        
        generations = min(self.max_queries // population_size, 20)
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                result = self._test_perturbation(target_system, individual)
                self.query_count += 1
                
                score = self._calculate_attack_score(result, scenario)
                fitness_scores.append(score)
                
                if result.get('success', False):
                    success = True
                    artifacts['successful_perturbation'] = individual
                    break
            
            if success:
                break
            
            # Selection
            selected = self._tournament_selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover(selected[i], selected[i+1])
                    new_population.extend([
                        self._mutate_perturbation(child1, scenario),
                        self._mutate_perturbation(child2, scenario)
                    ])
            
            population = new_population[:population_size]
            
            artifacts['generations'].append({
                'generation': gen,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores)
            })
            artifacts['best_fitness'].append(max(fitness_scores))
        
        return success, artifacts
    
    def _mimicry_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Mimicry attack that tries to imitate legitimate behavior.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'mimicry_attempts': []}
        success = False
        
        # Learn legitimate patterns
        legitimate_samples = scenario.parameters.get('legitimate_samples', [])
        if not legitimate_samples:
            # Generate synthetic legitimate samples
            legitimate_samples = [
                self._generate_legitimate_sample(scenario)
                for _ in range(10)
            ]
        
        # Extract patterns
        patterns = self._extract_patterns(legitimate_samples)
        
        for i in range(min(self.max_queries, 100)):
            # Generate mimicry perturbation
            perturbation = self._generate_mimicry_perturbation(patterns, scenario)
            
            # Test perturbation
            result = self._test_perturbation(target_system, perturbation)
            self.query_count += 1
            
            artifacts['mimicry_attempts'].append({
                'iteration': i,
                'similarity_score': self._calculate_similarity(perturbation, patterns),
                'detected': result.get('detected', False)
            })
            
            if result.get('success', False):
                success = True
                artifacts['successful_perturbation'] = perturbation
                break
        
        return success, artifacts
    
    def _backdoor_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Backdoor attack that plants triggers.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'triggers': [], 'activation_attempts': []}
        success = False
        
        # Generate backdoor triggers
        num_triggers = scenario.parameters.get('num_triggers', 5)
        triggers = []
        
        for i in range(num_triggers):
            trigger = self._generate_backdoor_trigger(scenario)
            triggers.append(trigger)
            
            # Plant trigger
            plant_result = self._plant_backdoor(target_system, trigger)
            self.query_count += 1
            
            artifacts['triggers'].append({
                'trigger': trigger,
                'planted': plant_result.get('success', False)
            })
        
        # Test trigger activation
        for trigger in triggers:
            activation_input = self._create_trigger_input(trigger, scenario)
            result = self._test_perturbation(target_system, activation_input)
            self.query_count += 1
            
            artifacts['activation_attempts'].append({
                'trigger': trigger,
                'activated': result.get('backdoor_activated', False)
            })
            
            if result.get('backdoor_activated', False):
                success = True
                artifacts['successful_trigger'] = trigger
                break
        
        return success, artifacts
    
    def _extraction_attack(
        self,
        scenario: AttackScenario,
        target_system: Any,
        initial_state: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Model extraction attack through queries.
        
        Args:
            scenario: Attack scenario
            target_system: Target system
            initial_state: Initial state
            
        Returns:
            Tuple of (success, artifacts)
        """
        artifacts = {'extracted_info': {}, 'query_responses': []}
        
        # Generate extraction queries
        num_queries = min(self.max_queries, scenario.parameters.get('extraction_queries', 500))
        extraction_queries = self._generate_extraction_queries(num_queries, scenario)
        
        responses = []
        for query in extraction_queries:
            response = self._query_target(target_system, query)
            self.query_count += 1
            responses.append((query, response))
            
            artifacts['query_responses'].append({
                'query_hash': hashlib.md5(str(query).encode()).hexdigest()[:8],
                'response_type': type(response).__name__
            })
        
        # Analyze responses for information extraction
        extracted_model = self._reconstruct_model(responses, scenario)
        artifacts['extracted_info'] = extracted_model
        
        # Test extraction success
        if extracted_model:
            fidelity = self._calculate_extraction_fidelity(
                target_system,
                extracted_model,
                scenario
            )
            artifacts['extraction_fidelity'] = fidelity
            success = fidelity > scenario.parameters.get('fidelity_threshold', 0.8)
        else:
            success = False
        
        return success, artifacts
    
    # Helper methods
    
    def _generate_random_perturbation(self, scenario: AttackScenario) -> Any:
        """Generate random perturbation based on scenario"""
        dim = scenario.parameters.get('dimension', 10)
        magnitude = scenario.constraints.get('max_perturbation', 1.0)
        
        if scenario.parameters.get('type') == 'discrete':
            return np.random.choice([-1, 0, 1], size=dim).tolist()
        else:
            return (np.random.randn(dim) * magnitude).tolist()
    
    def _test_perturbation(self, target_system: Any, perturbation: Any) -> Dict[str, Any]:
        """Test a perturbation against the target system"""
        try:
            response = target_system.process(perturbation)
            return {
                'success': response.get('attack_successful', False),
                'detected': response.get('anomaly_detected', False),
                'score': response.get('confidence', 0.5)
            }
        except Exception as e:
            logger.debug(f"Perturbation test failed: {e}")
            return {'success': False, 'detected': True, 'score': 0.0}
    
    def _calculate_attack_score(self, result: Dict[str, Any], scenario: AttackScenario) -> float:
        """Calculate attack score based on result and scenario objectives"""
        score = 0.0
        
        # Success bonus
        if result.get('success', False):
            score += 10.0
        
        # Detection penalty
        if result.get('detected', False):
            score -= 5.0
        
        # Confidence-based score
        if scenario.objective == 'high_confidence':
            score += result.get('score', 0.0) * 2.0
        elif scenario.objective == 'low_confidence':
            score += (1.0 - result.get('score', 0.5)) * 2.0
        
        return score
    
    def _mutate_perturbation(self, perturbation: Any, scenario: AttackScenario) -> Any:
        """Mutate a perturbation for evolutionary algorithms"""
        if perturbation is None:
            return self._generate_random_perturbation(scenario)
        
        mutation_rate = scenario.parameters.get('mutation_rate', 0.1)
        
        if isinstance(perturbation, list):
            mutated = perturbation.copy()
            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    mutated[i] += np.random.randn() * 0.1
            return mutated
        else:
            return perturbation
    
    def _estimate_gradient(
        self,
        target_system: Any,
        current: np.ndarray,
        scenario: AttackScenario
    ) -> np.ndarray:
        """Estimate gradient using finite differences"""
        epsilon = 0.001
        gradient = np.zeros_like(current)
        
        base_result = self._test_perturbation(target_system, current.tolist())
        base_score = self._calculate_attack_score(base_result, scenario)
        self.query_count += 1
        
        for i in range(len(current)):
            perturbed = current.copy()
            perturbed[i] += epsilon
            
            result = self._test_perturbation(target_system, perturbed.tolist())
            score = self._calculate_attack_score(result, scenario)
            self.query_count += 1
            
            gradient[i] = (score - base_score) / epsilon
        
        return gradient
    
    def _clip_to_constraints(
        self,
        perturbation: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Clip perturbation to satisfy constraints"""
        max_norm = constraints.get('max_norm', float('inf'))
        max_value = constraints.get('max_value', float('inf'))
        
        # Clip values
        perturbation = np.clip(perturbation, -max_value, max_value)
        
        # Clip norm
        norm = np.linalg.norm(perturbation)
        if norm > max_norm:
            perturbation = perturbation * (max_norm / norm)
        
        return perturbation
    
    def _tournament_selection(
        self,
        population: List[Any],
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> List[Any]:
        """Tournament selection for evolutionary algorithm"""
        selected = []
        
        for _ in range(len(population)):
            tournament_indices = np.random.choice(
                len(population),
                size=tournament_size,
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def _crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Crossover operation for evolutionary algorithm"""
        if isinstance(parent1, list) and isinstance(parent2, list):
            crossover_point = np.random.randint(1, len(parent1))
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        else:
            return parent1, parent2
    
    def _generate_legitimate_sample(self, scenario: AttackScenario) -> Any:
        """Generate synthetic legitimate sample"""
        dim = scenario.parameters.get('dimension', 10)
        return np.random.randn(dim).tolist()
    
    def _extract_patterns(self, samples: List[Any]) -> Dict[str, Any]:
        """Extract patterns from legitimate samples"""
        if not samples:
            return {}
        
        patterns = {
            'mean': np.mean(samples, axis=0).tolist() if isinstance(samples[0], list) else None,
            'std': np.std(samples, axis=0).tolist() if isinstance(samples[0], list) else None,
            'samples': samples[:5]  # Keep some examples
        }
        return patterns
    
    def _generate_mimicry_perturbation(
        self,
        patterns: Dict[str, Any],
        scenario: AttackScenario
    ) -> Any:
        """Generate perturbation that mimics legitimate patterns"""
        if patterns.get('mean') and patterns.get('std'):
            mean = np.array(patterns['mean'])
            std = np.array(patterns['std'])
            perturbation = np.random.normal(mean, std)
            
            # Add small malicious component
            malicious = self._generate_random_perturbation(scenario)
            if isinstance(malicious, list):
                malicious = np.array(malicious) * 0.1  # Small magnitude
                perturbation = (perturbation * 0.9 + malicious).tolist()
            
            return perturbation
        else:
            return self._generate_random_perturbation(scenario)
    
    def _calculate_similarity(self, sample: Any, patterns: Dict[str, Any]) -> float:
        """Calculate similarity between sample and patterns"""
        if not patterns.get('mean'):
            return 0.0
        
        if isinstance(sample, list):
            sample_array = np.array(sample)
            mean_array = np.array(patterns['mean'])
            distance = np.linalg.norm(sample_array - mean_array)
            similarity = np.exp(-distance)
            return float(similarity)
        
        return 0.0
    
    def _generate_backdoor_trigger(self, scenario: AttackScenario) -> Dict[str, Any]:
        """Generate a backdoor trigger"""
        trigger_type = scenario.parameters.get('trigger_type', 'pattern')
        
        if trigger_type == 'pattern':
            return {
                'type': 'pattern',
                'pattern': np.random.randint(0, 256, size=10).tolist(),
                'location': 'fixed'
            }
        elif trigger_type == 'semantic':
            return {
                'type': 'semantic',
                'keyword': f"trigger_{np.random.randint(1000, 9999)}",
                'context': 'any'
            }
        else:
            return {'type': 'unknown'}
    
    def _plant_backdoor(self, target_system: Any, trigger: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to plant a backdoor trigger"""
        try:
            # This would interact with the actual system
            # For simulation, we return a mock result
            return {'success': np.random.random() > 0.7}
        except Exception:
            return {'success': False}
    
    def _create_trigger_input(self, trigger: Dict[str, Any], scenario: AttackScenario) -> Any:
        """Create input that activates a trigger"""
        base_input = self._generate_legitimate_sample(scenario)
        
        if trigger['type'] == 'pattern' and isinstance(base_input, list):
            # Embed pattern in input
            pattern = trigger['pattern']
            for i, val in enumerate(pattern[:len(base_input)]):
                base_input[i] = val / 255.0  # Normalize
        
        return base_input
    
    def _generate_extraction_queries(self, num_queries: int, scenario: AttackScenario) -> List[Any]:
        """Generate queries for model extraction"""
        queries = []
        
        # Diverse query strategies
        strategies = ['random', 'boundary', 'adversarial', 'synthetic']
        
        for i in range(num_queries):
            strategy = strategies[i % len(strategies)]
            
            if strategy == 'random':
                query = self._generate_random_perturbation(scenario)
            elif strategy == 'boundary':
                # Generate near decision boundary
                query = np.random.randn(scenario.parameters.get('dimension', 10)) * 0.1
                query = query.tolist()
            elif strategy == 'adversarial':
                # Use adversarial examples
                base = self._generate_random_perturbation(scenario)
                if isinstance(base, list):
                    noise = np.random.randn(len(base)) * 0.01
                    query = (np.array(base) + noise).tolist()
                else:
                    query = base
            else:  # synthetic
                query = self._generate_legitimate_sample(scenario)
            
            queries.append(query)
        
        return queries
    
    def _query_target(self, target_system: Any, query: Any) -> Any:
        """Query the target system"""
        try:
            return target_system.predict(query)
        except Exception:
            try:
                return target_system.process(query)
            except Exception:
                return None
    
    def _reconstruct_model(
        self,
        query_responses: List[Tuple[Any, Any]],
        scenario: AttackScenario
    ) -> Dict[str, Any]:
        """Attempt to reconstruct model from query responses"""
        if not query_responses:
            return {}
        
        # Simple reconstruction - in practice would use more sophisticated methods
        extracted = {
            'num_queries': len(query_responses),
            'response_types': list(set(type(r[1]).__name__ for r in query_responses if r[1])),
            'estimated_complexity': 'unknown'
        }
        
        # Analyze response patterns
        numeric_responses = [r[1] for r in query_responses if isinstance(r[1], (int, float))]
        if numeric_responses:
            extracted['response_statistics'] = {
                'mean': float(np.mean(numeric_responses)),
                'std': float(np.std(numeric_responses)),
                'min': float(np.min(numeric_responses)),
                'max': float(np.max(numeric_responses))
            }
        
        return extracted
    
    def _calculate_extraction_fidelity(
        self,
        target_system: Any,
        extracted_model: Dict[str, Any],
        scenario: AttackScenario
    ) -> float:
        """Calculate fidelity of extracted model"""
        if not extracted_model:
            return 0.0
        
        # Simple fidelity metric - would be more sophisticated in practice
        fidelity = 0.0
        
        if 'response_statistics' in extracted_model:
            fidelity += 0.5
        
        if extracted_model.get('num_queries', 0) > 100:
            fidelity += 0.3
        
        if len(extracted_model.get('response_types', [])) > 0:
            fidelity += 0.2
        
        return min(fidelity, 1.0)