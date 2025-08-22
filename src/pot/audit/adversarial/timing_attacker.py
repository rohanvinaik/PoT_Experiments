"""
Timing Channel Attacker

Implements timing-based side channel attacks.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TimingProfile:
    """Profile of timing measurements"""
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    samples: List[float]
    correlations: Dict[str, float]


@dataclass
class TimingAttackResult:
    """Result of timing attack"""
    information_leaked: float
    timing_profiles: Dict[str, TimingProfile]
    successful_inference: bool
    inferred_properties: Dict[str, Any]


class TimingChannelAttacker:
    """
    Exploits timing side channels to extract information.
    
    Features:
    - High-precision timing measurements
    - Statistical timing analysis
    - Cache timing attacks
    - Network timing attacks
    - Timing pattern correlation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the timing attacker"""
        self.config = config or {}
        self.precision = self.config.get('precision', 1e-6)  # microsecond precision
        self.samples_per_measurement = self.config.get('samples', 100)
        self.timing_data = {}
        
    def measure_timing(
        self,
        target_function: Callable,
        input_data: Any,
        iterations: int = 100
    ) -> TimingProfile:
        """
        Measure timing characteristics of target function.
        
        Args:
            target_function: Function to measure
            input_data: Input to the function
            iterations: Number of measurements
            
        Returns:
            TimingProfile
        """
        timings = []
        
        for _ in range(iterations):
            # Warm up
            target_function(input_data)
            
            # Actual measurement
            start = time.perf_counter()
            result = target_function(input_data)
            end = time.perf_counter()
            
            elapsed = end - start
            timings.append(elapsed)
        
        # Remove outliers (>3 std from mean)
        mean = np.mean(timings)
        std = np.std(timings)
        filtered_timings = [t for t in timings if abs(t - mean) <= 3 * std]
        
        return TimingProfile(
            mean_time=np.mean(filtered_timings),
            std_time=np.std(filtered_timings),
            min_time=np.min(filtered_timings),
            max_time=np.max(filtered_timings),
            samples=filtered_timings,
            correlations={}
        )
    
    def timing_attack(
        self,
        target_system: Any,
        attack_vectors: List[Any],
        reference_vectors: Optional[List[Any]] = None
    ) -> TimingAttackResult:
        """
        Perform timing attack on target system.
        
        Args:
            target_system: System to attack
            attack_vectors: Attack input vectors
            reference_vectors: Reference vectors for comparison
            
        Returns:
            TimingAttackResult
        """
        timing_profiles = {}
        
        # Measure timing for attack vectors
        for i, vector in enumerate(attack_vectors):
            profile = self.measure_timing(
                lambda x: target_system.process(x),
                vector,
                self.samples_per_measurement
            )
            timing_profiles[f'attack_{i}'] = profile
        
        # Measure timing for reference vectors if provided
        if reference_vectors:
            for i, vector in enumerate(reference_vectors):
                profile = self.measure_timing(
                    lambda x: target_system.process(x),
                    vector,
                    self.samples_per_measurement
                )
                timing_profiles[f'reference_{i}'] = profile
        
        # Analyze timing patterns
        information_leaked = self._analyze_timing_leakage(timing_profiles)
        inferred_properties = self._infer_properties(timing_profiles)
        
        # Check if attack was successful
        successful = information_leaked > 0.1 or len(inferred_properties) > 0
        
        return TimingAttackResult(
            information_leaked=information_leaked,
            timing_profiles=timing_profiles,
            successful_inference=successful,
            inferred_properties=inferred_properties
        )
    
    def cache_timing_attack(
        self,
        target_system: Any,
        memory_addresses: List[int]
    ) -> Dict[str, Any]:
        """
        Perform cache timing attack.
        
        Args:
            target_system: Target system
            memory_addresses: Memory addresses to probe
            
        Returns:
            Attack results
        """
        cache_hits = {}
        cache_misses = {}
        
        for address in memory_addresses:
            # Prime cache
            self._prime_cache(target_system, address)
            
            # Measure access time
            access_time = self._measure_cache_access(target_system, address)
            
            # Classify as hit or miss based on timing
            if access_time < self.config.get('cache_hit_threshold', 1e-7):
                cache_hits[address] = access_time
            else:
                cache_misses[address] = access_time
        
        return {
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': len(cache_hits) / len(memory_addresses) if memory_addresses else 0,
            'inferred_access_pattern': self._infer_access_pattern(cache_hits, cache_misses)
        }
    
    def network_timing_attack(
        self,
        target_url: str,
        payloads: List[str]
    ) -> Dict[str, Any]:
        """
        Perform network timing attack.
        
        Args:
            target_url: Target URL
            payloads: Payloads to test
            
        Returns:
            Attack results
        """
        import requests
        
        timing_results = {}
        
        for payload in payloads:
            timings = []
            
            for _ in range(10):  # Multiple measurements
                start = time.perf_counter()
                try:
                    response = requests.post(target_url, data=payload, timeout=5)
                    elapsed = time.perf_counter() - start
                    timings.append(elapsed)
                except Exception as e:
                    logger.debug(f"Network timing failed: {e}")
                    timings.append(float('inf'))
            
            timing_results[payload[:20]] = {  # Truncate payload for key
                'mean': np.mean([t for t in timings if t != float('inf')]),
                'std': np.std([t for t in timings if t != float('inf')]),
                'timeouts': sum(1 for t in timings if t == float('inf'))
            }
        
        # Analyze for timing correlations
        correlations = self._find_timing_correlations(timing_results)
        
        return {
            'timing_results': timing_results,
            'correlations': correlations,
            'vulnerable': any(c > 0.7 for c in correlations.values())
        }
    
    def differential_timing_analysis(
        self,
        target_system: Any,
        input_pairs: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """
        Perform differential timing analysis.
        
        Args:
            target_system: Target system
            input_pairs: Pairs of inputs to compare
            
        Returns:
            Analysis results
        """
        differential_timings = []
        
        for input1, input2 in input_pairs:
            # Measure timing for both inputs
            profile1 = self.measure_timing(
                lambda x: target_system.process(x),
                input1,
                50
            )
            profile2 = self.measure_timing(
                lambda x: target_system.process(x),
                input2,
                50
            )
            
            # Calculate differential
            diff = abs(profile1.mean_time - profile2.mean_time)
            differential_timings.append({
                'pair': (str(input1)[:20], str(input2)[:20]),
                'time_diff': diff,
                'significant': diff > 3 * max(profile1.std_time, profile2.std_time)
            })
        
        # Statistical analysis
        time_diffs = [d['time_diff'] for d in differential_timings]
        
        return {
            'differential_timings': differential_timings,
            'mean_difference': np.mean(time_diffs),
            'max_difference': np.max(time_diffs),
            'significant_pairs': sum(1 for d in differential_timings if d['significant']),
            'information_leakage': self._estimate_information_leakage(time_diffs)
        }
    
    def _analyze_timing_leakage(self, profiles: Dict[str, TimingProfile]) -> float:
        """Analyze timing profiles for information leakage"""
        if len(profiles) < 2:
            return 0.0
        
        # Calculate variance in timing across different inputs
        mean_times = [p.mean_time for p in profiles.values()]
        
        # Normalize by average time
        avg_time = np.mean(mean_times)
        if avg_time == 0:
            return 0.0
        
        normalized_variance = np.var(mean_times) / (avg_time ** 2)
        
        # High variance indicates potential leakage
        leakage_score = min(normalized_variance * 100, 1.0)
        
        return leakage_score
    
    def _infer_properties(self, profiles: Dict[str, TimingProfile]) -> Dict[str, Any]:
        """Infer system properties from timing profiles"""
        inferred = {}
        
        # Group profiles by timing characteristics
        fast_group = []
        slow_group = []
        
        mean_time = np.mean([p.mean_time for p in profiles.values()])
        
        for name, profile in profiles.items():
            if profile.mean_time < mean_time * 0.8:
                fast_group.append(name)
            elif profile.mean_time > mean_time * 1.2:
                slow_group.append(name)
        
        if fast_group:
            inferred['fast_path_inputs'] = fast_group
        if slow_group:
            inferred['slow_path_inputs'] = slow_group
        
        # Check for timing correlations
        if len(profiles) > 5:
            times = [p.mean_time for p in profiles.values()]
            # Simple autocorrelation check
            if len(times) > 1:
                autocorr = np.corrcoef(times[:-1], times[1:])[0, 1]
                if abs(autocorr) > 0.5:
                    inferred['temporal_correlation'] = autocorr
        
        # Detect discrete timing classes
        times = [p.mean_time for p in profiles.values()]
        if len(set(times)) < len(times) / 2:
            inferred['discrete_timing_classes'] = len(set(times))
        
        return inferred
    
    def _prime_cache(self, target_system: Any, address: int) -> None:
        """Prime cache for cache timing attack"""
        # Simplified cache priming
        try:
            # Access memory to evict target from cache
            for offset in range(0, 64 * 1024, 64):  # Assume 64-byte cache lines
                dummy_address = (address & ~0xFFFF) | offset
                target_system.access_memory(dummy_address)
        except Exception:
            pass
    
    def _measure_cache_access(self, target_system: Any, address: int) -> float:
        """Measure cache access time"""
        try:
            start = time.perf_counter()
            target_system.access_memory(address)
            return time.perf_counter() - start
        except Exception:
            return float('inf')
    
    def _infer_access_pattern(
        self,
        cache_hits: Dict[int, float],
        cache_misses: Dict[int, float]
    ) -> str:
        """Infer memory access pattern from cache behavior"""
        if not cache_hits and not cache_misses:
            return "unknown"
        
        hit_rate = len(cache_hits) / (len(cache_hits) + len(cache_misses))
        
        if hit_rate > 0.8:
            return "sequential"
        elif hit_rate < 0.2:
            return "random"
        else:
            return "mixed"
    
    def _find_timing_correlations(self, timing_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Find correlations in timing results"""
        correlations = {}
        
        if len(timing_results) < 2:
            return correlations
        
        # Extract timing values
        keys = list(timing_results.keys())
        times = [timing_results[k]['mean'] for k in keys]
        
        # Check for linear correlation with input properties
        for i, key1 in enumerate(keys[:-1]):
            for j, key2 in enumerate(keys[i+1:], i+1):
                if times[i] != float('inf') and times[j] != float('inf'):
                    # Simple correlation based on timing similarity
                    similarity = 1.0 - abs(times[i] - times[j]) / max(times[i], times[j])
                    correlations[f"{key1}_vs_{key2}"] = similarity
        
        return correlations
    
    def _estimate_information_leakage(self, time_differences: List[float]) -> float:
        """Estimate information leakage from timing differences"""
        if not time_differences:
            return 0.0
        
        # Calculate entropy of timing differences
        # Discretize into bins
        num_bins = min(10, len(time_differences))
        hist, _ = np.histogram(time_differences, bins=num_bins)
        
        # Calculate entropy
        probs = hist / sum(hist)
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize to [0, 1]
        max_entropy = np.log2(num_bins)
        if max_entropy > 0:
            return entropy / max_entropy
        
        return 0.0