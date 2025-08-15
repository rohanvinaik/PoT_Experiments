"""
Coverage-Separation Trade-off for Challenge Design
Based on paper Section 4: Challenge Design Principles
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import itertools
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import hashlib


@dataclass 
class CoverageMetrics:
    """Metrics for coverage-separation trade-off"""
    coverage_score: float      # How well challenges cover input space
    separation_score: float     # How well challenges separate models
    trade_off_score: float      # Combined metric
    intra_distance: float       # Average distance within challenges
    inter_distance: float       # Average distance between challenge types
    diversity_index: float      # Shannon entropy of challenge distribution


class CoverageSeparationOptimizer:
    """
    Optimize challenge design for coverage-separation trade-off
    From paper Section 4.1: "Challenges should maximize coverage of 
    the input space while maintaining separation between genuine and adversarial models"
    """
    
    def __init__(self, input_dim: int, n_challenges: int, seed: int = 42):
        """
        Initialize optimizer
        
        Args:
            input_dim: Dimensionality of input space
            n_challenges: Number of challenges to generate
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.n_challenges = n_challenges
        self.rng = np.random.default_rng(seed)
        
    def compute_coverage(self, challenges: np.ndarray, 
                        space_bounds: Tuple[float, float] = (-1, 1)) -> float:
        """
        Compute coverage score: how well challenges cover the input space
        
        Uses Voronoi cell volumes as proxy for coverage
        """
        if len(challenges) < 2:
            return 0.0
        
        # Normalize challenges to [0, 1]
        min_val, max_val = space_bounds
        normalized = (challenges - min_val) / (max_val - min_val)
        
        # Compute pairwise distances
        distances = pdist(normalized)
        
        # Coverage score: inverse of average minimum distance to nearest neighbor
        dist_matrix = squareform(distances)
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        
        # Higher score = better coverage (challenges are spread out)
        coverage = 1.0 / (1.0 + np.mean(min_distances))
        
        return coverage
    
    def compute_separation(self, challenges: np.ndarray, 
                          model_responses: Dict[str, np.ndarray]) -> float:
        """
        Compute separation score: how well challenges distinguish between models
        
        Args:
            challenges: Challenge vectors (n_challenges x input_dim)
            model_responses: Dict mapping model_id to response vectors
            
        Returns:
            Separation score in [0, 1]
        """
        if len(model_responses) < 2:
            return 0.0
        
        # Compute inter-model distances for each challenge
        separations = []
        
        for i in range(len(challenges)):
            responses = [resp[i] for resp in model_responses.values()]
            if len(responses) >= 2:
                # Variance in responses = good separation
                separation = np.var(responses)
                separations.append(separation)
        
        if not separations:
            return 0.0
        
        # Normalize to [0, 1]
        sep_score = np.mean(separations)
        return min(1.0, sep_score)  # Cap at 1.0
    
    def optimize_challenges(self, 
                          coverage_weight: float = 0.5,
                          separation_weight: float = 0.5,
                          n_iterations: int = 100) -> np.ndarray:
        """
        Optimize challenge distribution for coverage-separation trade-off
        
        Uses simulated annealing to find optimal challenge positions
        """
        assert coverage_weight + separation_weight == 1.0
        
        # Initialize challenges randomly
        best_challenges = self.rng.uniform(-1, 1, (self.n_challenges, self.input_dim))
        best_score = 0.0
        
        # Simulated annealing
        temperature = 1.0
        cooling_rate = 0.95
        
        for iteration in range(n_iterations):
            # Perturb challenges
            new_challenges = best_challenges + self.rng.normal(0, temperature, 
                                                              best_challenges.shape)
            new_challenges = np.clip(new_challenges, -1, 1)
            
            # Compute combined score
            coverage = self.compute_coverage(new_challenges)
            
            # For separation, we need model responses (simulate for now)
            # In practice, this would use actual model evaluations
            simulated_responses = self._simulate_model_responses(new_challenges)
            separation = self.compute_separation(new_challenges, simulated_responses)
            
            score = coverage_weight * coverage + separation_weight * separation
            
            # Accept or reject
            if score > best_score or self.rng.random() < np.exp((score - best_score) / temperature):
                best_challenges = new_challenges
                best_score = score
            
            # Cool down
            temperature *= cooling_rate
        
        return best_challenges
    
    def _simulate_model_responses(self, challenges: np.ndarray) -> Dict[str, np.ndarray]:
        """Simulate model responses for optimization (placeholder)"""
        # In practice, this would evaluate actual models
        # Here we simulate 3 different "models"
        responses = {}
        
        # Model 1: Linear response
        responses["model1"] = np.sum(challenges, axis=1)
        
        # Model 2: Nonlinear response  
        responses["model2"] = np.sum(challenges**2, axis=1)
        
        # Model 3: Mixed response
        responses["model3"] = np.sum(challenges * np.sin(challenges), axis=1)
        
        return responses
    
    def stratified_sampling(self, n_strata: int = 5) -> np.ndarray:
        """
        Generate challenges using stratified sampling for better coverage
        From paper: "Stratified sampling ensures coverage of diverse input regions"
        """
        challenges = []
        challenges_per_stratum = self.n_challenges // n_strata
        remainder = self.n_challenges % n_strata
        
        for stratum in range(n_strata):
            # Define stratum bounds
            lower = -1 + (2 * stratum / n_strata)
            upper = -1 + (2 * (stratum + 1) / n_strata)
            
            # Sample within stratum
            n_samples = challenges_per_stratum + (1 if stratum < remainder else 0)
            stratum_challenges = self.rng.uniform(lower, upper, 
                                                 (n_samples, self.input_dim))
            challenges.append(stratum_challenges)
        
        return np.vstack(challenges)
    
    def latin_hypercube_sampling(self) -> np.ndarray:
        """
        Generate challenges using Latin Hypercube Sampling
        Ensures each dimension is evenly sampled
        """
        challenges = np.zeros((self.n_challenges, self.input_dim))
        
        for dim in range(self.input_dim):
            # Create evenly spaced intervals
            intervals = np.linspace(-1, 1, self.n_challenges + 1)
            
            # Sample one point from each interval
            for i in range(self.n_challenges):
                challenges[i, dim] = self.rng.uniform(intervals[i], intervals[i + 1])
            
            # Shuffle to avoid correlation between dimensions
            self.rng.shuffle(challenges[:, dim])
        
        return challenges
    
    def diversity_maximization(self, candidates: np.ndarray, 
                              n_select: int) -> np.ndarray:
        """
        Select diverse subset of challenges from candidates
        Uses max-min diversity selection
        """
        if len(candidates) <= n_select:
            return candidates
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Start with random point
        first_idx = self.rng.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select points that maximize minimum distance to selected set
        while len(selected_indices) < n_select and remaining_indices:
            max_min_dist = -1
            best_idx = None
            
            for idx in remaining_indices:
                # Compute minimum distance to selected points
                min_dist = float('inf')
                for sel_idx in selected_indices:
                    dist = np.linalg.norm(candidates[idx] - candidates[sel_idx])
                    min_dist = min(min_dist, dist)
                
                # Track point with maximum minimum distance
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return candidates[selected_indices]
    
    def evaluate_challenge_set(self, challenges: np.ndarray) -> CoverageMetrics:
        """
        Comprehensively evaluate a challenge set
        
        Returns:
            CoverageMetrics with various quality measures
        """
        # Coverage score
        coverage = self.compute_coverage(challenges)
        
        # Separation score (simulated)
        model_responses = self._simulate_model_responses(challenges)
        separation = self.compute_separation(challenges, model_responses)
        
        # Intra-challenge distances
        if len(challenges) > 1:
            distances = pdist(challenges)
            intra_distance = np.mean(distances)
        else:
            intra_distance = 0.0
        
        # Inter-cluster distances (using k-means with k=3)
        if len(challenges) >= 3:
            kmeans = KMeans(n_clusters=min(3, len(challenges)), random_state=42)
            labels = kmeans.fit_predict(challenges)
            
            cluster_centers = kmeans.cluster_centers_
            if len(cluster_centers) > 1:
                inter_distance = np.mean(pdist(cluster_centers))
            else:
                inter_distance = 0.0
        else:
            inter_distance = 0.0
        
        # Diversity index (Shannon entropy of spatial distribution)
        # Discretize space into bins
        n_bins = 10
        hist, _ = np.histogramdd(challenges, bins=n_bins)
        hist_flat = hist.flatten()
        hist_norm = hist_flat / hist_flat.sum()
        
        # Shannon entropy
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        max_entropy = np.log(len(hist_flat))
        diversity_index = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combined trade-off score
        trade_off = (coverage + separation) / 2
        
        return CoverageMetrics(
            coverage_score=coverage,
            separation_score=separation,
            trade_off_score=trade_off,
            intra_distance=intra_distance,
            inter_distance=inter_distance,
            diversity_index=diversity_index
        )


class AdaptiveChallengeGenerator:
    """
    Adaptive challenge generation based on model behavior
    Implements ideas from paper Section 4.2 on adaptive protocols
    """
    
    def __init__(self, base_challenges: np.ndarray, adaptation_rate: float = 0.1):
        """
        Initialize adaptive generator
        
        Args:
            base_challenges: Initial challenge set
            adaptation_rate: Rate of adaptation based on feedback
        """
        self.base_challenges = base_challenges
        self.adaptation_rate = adaptation_rate
        self.history = []
        
    def adapt_challenges(self, model_responses: np.ndarray, 
                        success_mask: np.ndarray) -> np.ndarray:
        """
        Adapt challenges based on model performance
        
        Args:
            model_responses: Model outputs for current challenges
            success_mask: Boolean mask of successful verifications
            
        Returns:
            Adapted challenge set
        """
        n_challenges = len(self.base_challenges)
        adapted = self.base_challenges.copy()
        
        # Focus on challenges where model struggles (low success rate)
        failure_indices = np.where(~success_mask)[0]
        
        if len(failure_indices) > 0:
            # Increase difficulty around failure points
            for idx in failure_indices:
                # Add noise to create variations
                noise = np.random.normal(0, self.adaptation_rate, 
                                       self.base_challenges[idx].shape)
                adapted[idx] += noise
        
        # Also explore new regions if success rate is too high
        success_rate = np.mean(success_mask)
        if success_rate > 0.9:
            # Replace some easy challenges with harder ones
            n_replace = int(n_challenges * 0.1)
            replace_indices = np.random.choice(np.where(success_mask)[0], 
                                             n_replace, replace=False)
            
            # Generate new challenging points
            for idx in replace_indices:
                # Move further from successful regions
                adapted[idx] *= (1 + self.adaptation_rate)
        
        # Record history
        self.history.append({
            'challenges': adapted.copy(),
            'success_rate': success_rate,
            'n_failures': len(failure_indices)
        })
        
        return adapted
    
    def get_hardest_challenges(self, top_k: int = 10) -> np.ndarray:
        """
        Return the k hardest challenges based on historical performance
        """
        if not self.history:
            return self.base_challenges[:top_k]
        
        # Aggregate failure counts across history
        all_challenges = []
        all_failures = []
        
        for record in self.history:
            all_challenges.extend(record['challenges'])
            # Approximate failure likelihood
            failure_prob = 1 - record['success_rate']
            all_failures.extend([failure_prob] * len(record['challenges']))
        
        # Sort by failure probability
        sorted_indices = np.argsort(all_failures)[::-1]
        hardest_indices = sorted_indices[:top_k]
        
        return np.array(all_challenges)[hardest_indices]


def generate_optimal_challenges(challenge_type: str,
                               n_challenges: int,
                               optimization_method: str = 'latin_hypercube',
                               **kwargs) -> Dict[str, Any]:
    """
    Generate optimized challenges for verification
    
    Args:
        challenge_type: Type of challenges ('vision', 'lm', 'multimodal')
        n_challenges: Number of challenges to generate
        optimization_method: Method for optimization
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with challenges and metadata
    """
    # Determine input dimensionality based on type
    dim_map = {
        'vision': 100,      # e.g., 10x10 patch
        'lm': 768,          # e.g., BERT embedding dim
        'multimodal': 1024  # Combined representation
    }
    
    input_dim = dim_map.get(challenge_type, 100)
    
    # Initialize optimizer
    optimizer = CoverageSeparationOptimizer(input_dim, n_challenges)
    
    # Generate challenges based on method
    if optimization_method == 'latin_hypercube':
        challenges = optimizer.latin_hypercube_sampling()
    elif optimization_method == 'stratified':
        challenges = optimizer.stratified_sampling(kwargs.get('n_strata', 5))
    elif optimization_method == 'optimized':
        challenges = optimizer.optimize_challenges(
            coverage_weight=kwargs.get('coverage_weight', 0.5),
            separation_weight=kwargs.get('separation_weight', 0.5)
        )
    else:
        # Default: random sampling
        challenges = np.random.uniform(-1, 1, (n_challenges, input_dim))
    
    # Evaluate challenge quality
    metrics = optimizer.evaluate_challenge_set(challenges)
    
    # Create unique IDs for challenges
    challenge_ids = []
    for i, challenge in enumerate(challenges):
        challenge_bytes = challenge.tobytes()
        challenge_id = hashlib.sha256(challenge_bytes).hexdigest()[:16]
        challenge_ids.append(challenge_id)
    
    return {
        'challenges': challenges,
        'challenge_ids': challenge_ids,
        'type': challenge_type,
        'method': optimization_method,
        'metrics': metrics,
        'metadata': {
            'n_challenges': n_challenges,
            'input_dim': input_dim,
            'timestamp': np.datetime64('now').astype(str)
        }
    }