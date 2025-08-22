"""
Adaptive Challenge Generation for models with different vocabulary sizes.

This module implements intelligent challenge generation that adapts to vocabulary
differences between models, focusing on shared token spaces for fair comparison.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
from .challenge import Challenge, ChallengeConfig, generate_challenges
from .vocabulary_compatibility import (
    VocabularyCompatibilityAnalyzer,
    VocabularyMismatchBehavior,
    VocabularyCompatibilityReport
)
from .prf import prf_integers, prf_choice, prf_floats

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveChallengeConfig(ChallengeConfig):
    """Extended challenge configuration with vocabulary adaptation settings."""
    
    # Vocabulary sizes for adaptation
    vocab_size_a: Optional[int] = None
    vocab_size_b: Optional[int] = None
    
    # Model names for better diagnostics
    model_name_a: Optional[str] = None
    model_name_b: Optional[str] = None
    
    # Adaptation strategy
    adaptation_strategy: str = "shared_core"  # shared_core, common_tokens, frequency_weighted
    
    # Quality thresholds
    min_token_coverage: float = 0.8  # Minimum fraction of shared tokens to use
    min_challenges: int = 5  # Minimum number of challenges to generate


class AdaptiveChallengeGenerator:
    """
    Generates challenges that adapt to vocabulary differences between models.
    
    Key features:
    - Detects vocabulary size differences
    - Focuses challenges on shared token space
    - Provides fallback strategies for low overlap
    - Includes quality metrics for adapted challenges
    """
    
    def __init__(
        self,
        min_overlap_ratio: float = 0.95,
        core_vocab_size: int = 30000,
        enable_frequency_weighting: bool = False
    ):
        """
        Initialize adaptive challenge generator.
        
        Args:
            min_overlap_ratio: Minimum vocabulary overlap for standard generation
            core_vocab_size: Size of core vocabulary (common tokens across models)
            enable_frequency_weighting: Whether to weight tokens by frequency
        """
        self.vocab_analyzer = VocabularyCompatibilityAnalyzer(
            min_overlap_ratio=min_overlap_ratio,
            mismatch_behavior=VocabularyMismatchBehavior.ADAPT
        )
        self.core_vocab_size = core_vocab_size
        self.enable_frequency_weighting = enable_frequency_weighting
        
        # Statistics tracking
        self.adaptation_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "fallback_used": 0,
            "average_token_coverage": 0.0
        }
    
    def generate_adaptive_challenges(
        self,
        config: AdaptiveChallengeConfig,
        token_frequencies: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate challenges adapted to vocabulary differences.
        
        Args:
            config: Adaptive challenge configuration
            token_frequencies: Optional token frequency distribution
        
        Returns:
            Dictionary with adapted challenges and metadata
        """
        # Check if adaptation is needed
        if config.vocab_size_a is None or config.vocab_size_b is None:
            # No vocabulary info, use standard generation
            logger.info("No vocabulary information provided, using standard generation")
            return generate_challenges(config)
        
        # Analyze vocabulary compatibility
        compat_report = self.vocab_analyzer.analyze_vocabulary_overlap(
            config.vocab_size_a,
            config.vocab_size_b,
            config.model_name_a,
            config.model_name_b
        )
        
        # Get verification strategy
        strategy = self.vocab_analyzer.suggest_verification_strategy(compat_report)
        
        if not strategy["can_proceed"]:
            logger.error(f"Cannot generate challenges: {strategy['notes']}")
            return {
                "challenge_id": None,
                "family": config.family,
                "items": [],
                "challenges": [],
                "salt": "",
                "error": "Vocabulary incompatibility",
                "details": strategy["notes"]
            }
        
        # Adapt challenges based on strategy
        if config.family.startswith("lm:"):
            # Language model challenges need token adaptation
            adapted_result = self._adapt_language_challenges(
                config, compat_report, token_frequencies
            )
        else:
            # Vision/other challenges don't need vocabulary adaptation
            adapted_result = generate_challenges(config)
            adapted_result["vocabulary_adapted"] = False
        
        # Add adaptation metadata
        adapted_result["vocabulary_analysis"] = {
            "overlap_ratio": compat_report.overlap_ratio,
            "shared_tokens": compat_report.shared_tokens,
            "adaptation_method": strategy["method"],
            "confidence_adjustment": strategy["confidence_adjustment"]
        }
        
        # Update statistics
        self.adaptation_stats["total_adaptations"] += 1
        if adapted_result.get("vocabulary_adapted", False):
            self.adaptation_stats["successful_adaptations"] += 1
        
        return adapted_result
    
    def _adapt_language_challenges(
        self,
        config: AdaptiveChallengeConfig,
        compat_report: VocabularyCompatibilityReport,
        token_frequencies: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Adapt language model challenges to vocabulary differences.
        
        Args:
            config: Challenge configuration
            compat_report: Vocabulary compatibility analysis
            token_frequencies: Optional token frequency distribution
        
        Returns:
            Adapted challenges dictionary
        """
        # Determine shared token space
        shared_start, shared_end = self.vocab_analyzer.determine_shared_token_space(
            config.vocab_size_a, config.vocab_size_b
        )
        
        logger.info(f"Adapting challenges to shared token space: [{shared_start}, {shared_end})")
        
        # Calculate token coverage
        total_tokens = max(config.vocab_size_a, config.vocab_size_b)
        shared_tokens = shared_end - shared_start
        token_coverage = shared_tokens / total_tokens if total_tokens > 0 else 0
        
        if token_coverage < config.min_token_coverage:
            # Use fallback strategy for low overlap
            logger.warning(f"Low token coverage ({token_coverage:.1%}), using fallback strategy")
            return self._generate_fallback_challenges(config, compat_report)
        
        # Select adaptation strategy
        if config.adaptation_strategy == "shared_core":
            adapted_challenges = self._generate_shared_core_challenges(
                config, shared_start, shared_end, token_frequencies
            )
        elif config.adaptation_strategy == "common_tokens":
            adapted_challenges = self._generate_common_token_challenges(
                config, shared_start, shared_end
            )
        elif config.adaptation_strategy == "frequency_weighted":
            adapted_challenges = self._generate_frequency_weighted_challenges(
                config, shared_start, shared_end, token_frequencies
            )
        else:
            # Default to shared core strategy
            adapted_challenges = self._generate_shared_core_challenges(
                config, shared_start, shared_end, token_frequencies
            )
        
        # Add adaptation metadata
        adapted_challenges["vocabulary_adapted"] = True
        adapted_challenges["adaptation_strategy"] = config.adaptation_strategy
        adapted_challenges["token_coverage"] = token_coverage
        adapted_challenges["shared_token_range"] = [shared_start, shared_end]
        
        # Calculate quality metrics
        quality_metrics = self._calculate_challenge_quality(
            adapted_challenges["challenges"],
            shared_start,
            shared_end,
            token_frequencies
        )
        adapted_challenges["quality_metrics"] = quality_metrics
        
        return adapted_challenges
    
    def _generate_shared_core_challenges(
        self,
        config: AdaptiveChallengeConfig,
        shared_start: int,
        shared_end: int,
        token_frequencies: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate challenges focusing on core shared vocabulary.
        
        This strategy focuses on the first N tokens (typically 30,000) which
        are most likely to be shared across model families.
        """
        import hashlib
        from .prf import prf_derive_seed, prf_bytes
        import xxhash
        
        # Limit to core vocabulary size
        core_end = min(shared_end, shared_start + self.core_vocab_size)
        
        logger.info(f"Generating challenges for core vocabulary [{shared_start}, {core_end})")
        
        # Generate base challenges
        base_result = generate_challenges(config)
        
        # Adapt challenges to use only core tokens
        adapted_challenges = []
        
        master_key = bytes.fromhex(config.master_key_hex)
        nonce = bytes.fromhex(config.session_nonce_hex)
        
        # Create seed for adaptation
        if config.model_id:
            params_with_model = {**config.params, "model_id": config.model_id}
            seed = prf_derive_seed(master_key, config.family, params_with_model, nonce)
        else:
            seed = prf_derive_seed(master_key, config.family, config.params, nonce)
        
        for i, base_challenge in enumerate(base_result["challenges"]):
            # Create adapted challenge
            adapted_params = base_challenge.parameters.copy()
            
            # Add token constraints for language model challenges
            if config.family == "lm:templates":
                # Constrain token generation to core vocabulary
                adapted_params["token_constraints"] = {
                    "min_token_id": shared_start,
                    "max_token_id": core_end,
                    "strategy": "core_vocabulary"
                }
                
                # If we have frequency data, prefer high-frequency tokens
                if token_frequencies and self.enable_frequency_weighting:
                    # Select high-frequency tokens in the core range
                    core_tokens = [
                        tid for tid in range(shared_start, core_end)
                        if tid in token_frequencies
                    ]
                    if core_tokens:
                        # Sort by frequency and use top tokens
                        sorted_tokens = sorted(
                            core_tokens,
                            key=lambda x: token_frequencies.get(x, 0),
                            reverse=True
                        )
                        # Use top 80% most frequent tokens
                        cutoff = int(len(sorted_tokens) * 0.8)
                        adapted_params["preferred_tokens"] = sorted_tokens[:cutoff]
            
            elif config.family == "lm:masks":
                # For mask-based challenges, adapt mask positions
                adapted_params["mask_positions"] = self._adapt_mask_positions(
                    adapted_params.get("mask_positions", []),
                    shared_start,
                    core_end
                )
            
            # Generate new challenge ID for adapted challenge
            challenge_info = f"adapted_{i}_{shared_start}_{core_end}".encode()
            challenge_bytes = prf_bytes(seed, challenge_info, 16)
            challenge_id = xxhash.xxh3_64_hexdigest(challenge_bytes)
            
            adapted_challenge = Challenge(
                challenge_id=challenge_id,
                index=i,
                family=config.family,
                parameters=adapted_params
            )
            adapted_challenges.append(adapted_challenge)
        
        # Build result dictionary
        result = base_result.copy()
        result["challenges"] = adapted_challenges
        result["items"] = [c.parameters for c in adapted_challenges]
        
        # Regenerate challenge set ID
        result["challenge_id"] = xxhash.xxh3_128_hexdigest(
            repr(result["items"]).encode() + bytes.fromhex(result["salt"])
        )
        
        return result
    
    def _generate_common_token_challenges(
        self,
        config: AdaptiveChallengeConfig,
        shared_start: int,
        shared_end: int
    ) -> Dict[str, Any]:
        """
        Generate challenges using only tokens common to both vocabularies.
        
        This strategy is more restrictive but ensures exact compatibility.
        """
        logger.info(f"Generating challenges with common tokens only [{shared_start}, {shared_end})")
        
        # Modify config to restrict token space
        modified_config = ChallengeConfig(
            master_key_hex=config.master_key_hex,
            session_nonce_hex=config.session_nonce_hex,
            n=config.n,
            family=config.family,
            params={
                **config.params,
                "token_range": [shared_start, shared_end],
                "strict_mode": True
            },
            model_id=config.model_id
        )
        
        # Generate with modified config
        result = generate_challenges(modified_config)
        
        # Mark as adapted
        result["adaptation_note"] = f"Using only common tokens [{shared_start}, {shared_end}]"
        
        return result
    
    def _generate_frequency_weighted_challenges(
        self,
        config: AdaptiveChallengeConfig,
        shared_start: int,
        shared_end: int,
        token_frequencies: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate challenges weighted by token frequency.
        
        This strategy prefers high-frequency tokens that are more likely
        to be meaningful across different models.
        """
        if not token_frequencies:
            logger.warning("No frequency data provided, falling back to shared core strategy")
            return self._generate_shared_core_challenges(
                config, shared_start, shared_end, None
            )
        
        logger.info("Generating frequency-weighted challenges")
        
        # Get frequencies for shared tokens
        shared_frequencies = {
            tid: freq
            for tid, freq in token_frequencies.items()
            if shared_start <= tid < shared_end
        }
        
        if not shared_frequencies:
            logger.warning("No frequency data for shared tokens, using uniform distribution")
            return self._generate_common_token_challenges(config, shared_start, shared_end)
        
        # Sort tokens by frequency
        sorted_tokens = sorted(
            shared_frequencies.keys(),
            key=lambda x: shared_frequencies[x],
            reverse=True
        )
        
        # Use top tokens that cover 95% of frequency mass
        cumulative_freq = 0.0
        total_freq = sum(shared_frequencies.values())
        selected_tokens = []
        
        for token_id in sorted_tokens:
            selected_tokens.append(token_id)
            cumulative_freq += shared_frequencies[token_id]
            if cumulative_freq / total_freq >= 0.95:
                break
        
        logger.info(f"Selected {len(selected_tokens)} high-frequency tokens")
        
        # Modify config to use selected tokens
        modified_config = ChallengeConfig(
            master_key_hex=config.master_key_hex,
            session_nonce_hex=config.session_nonce_hex,
            n=config.n,
            family=config.family,
            params={
                **config.params,
                "allowed_tokens": selected_tokens,
                "weighting": "frequency"
            },
            model_id=config.model_id
        )
        
        result = generate_challenges(modified_config)
        result["frequency_weighted"] = True
        result["num_selected_tokens"] = len(selected_tokens)
        
        return result
    
    def _generate_fallback_challenges(
        self,
        config: AdaptiveChallengeConfig,
        compat_report: VocabularyCompatibilityReport
    ) -> Dict[str, Any]:
        """
        Generate fallback challenges for low vocabulary overlap.
        
        This strategy uses:
        1. Very basic/common tokens (first 1000)
        2. Simple patterns that don't rely on specific vocabulary
        3. Reduced number of challenges
        """
        logger.warning("Using fallback strategy due to low vocabulary overlap")
        
        self.adaptation_stats["fallback_used"] += 1
        
        # Use only the most basic tokens (typically punctuation, numbers, common words)
        basic_token_limit = min(1000, compat_report.shared_tokens)
        
        # Reduce number of challenges
        reduced_n = max(config.min_challenges, config.n // 2)
        
        # Create fallback config
        fallback_config = ChallengeConfig(
            master_key_hex=config.master_key_hex,
            session_nonce_hex=config.session_nonce_hex,
            n=reduced_n,
            family=config.family,
            params={
                **config.params,
                "token_range": [0, basic_token_limit],
                "fallback_mode": True,
                "simplified": True
            },
            model_id=config.model_id
        )
        
        result = generate_challenges(fallback_config)
        
        # Add fallback metadata
        result["fallback_used"] = True
        result["fallback_reason"] = f"Low overlap: {compat_report.overlap_ratio:.1%}"
        result["reduced_challenges"] = reduced_n
        result["basic_token_limit"] = basic_token_limit
        
        # Add warning to challenges
        for challenge in result.get("challenges", []):
            challenge.parameters["warning"] = "Generated with fallback strategy due to vocabulary mismatch"
        
        return result
    
    def _adapt_mask_positions(
        self,
        original_positions: List[int],
        min_token: int,
        max_token: int
    ) -> List[int]:
        """
        Adapt mask positions to stay within shared token range.
        
        Args:
            original_positions: Original mask positions
            min_token: Minimum valid token ID
            max_token: Maximum valid token ID
        
        Returns:
            Adapted mask positions within valid range
        """
        adapted = []
        token_range = max_token - min_token
        
        for pos in original_positions:
            if min_token <= pos < max_token:
                # Position is already valid
                adapted.append(pos)
            else:
                # Map position to valid range
                mapped_pos = min_token + (pos % token_range)
                adapted.append(mapped_pos)
        
        return adapted
    
    def _calculate_challenge_quality(
        self,
        challenges: List[Challenge],
        shared_start: int,
        shared_end: int,
        token_frequencies: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for adapted challenges.
        
        Metrics include:
        - Token coverage: Fraction of shared tokens used
        - Diversity: Entropy of token distribution
        - Frequency alignment: How well challenges match token frequencies
        
        Args:
            challenges: List of adapted challenges
            shared_start: Start of shared token range
            shared_end: End of shared token range
            token_frequencies: Optional token frequency distribution
        
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "token_coverage": 0.0,
            "diversity_score": 0.0,
            "frequency_alignment": 0.0,
            "challenge_validity": 1.0
        }
        
        if not challenges:
            return metrics
        
        # Track which tokens are used
        used_tokens = set()
        token_counts = {}
        
        for challenge in challenges:
            params = challenge.parameters
            
            # Extract token information based on challenge type
            if "preferred_tokens" in params:
                for token_id in params["preferred_tokens"]:
                    used_tokens.add(token_id)
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            if "token_constraints" in params:
                constraints = params["token_constraints"]
                # Sample some tokens from the range for metrics
                for i in range(min(10, constraints["max_token_id"] - constraints["min_token_id"])):
                    token_id = constraints["min_token_id"] + i
                    used_tokens.add(token_id)
        
        # Calculate token coverage
        total_shared = shared_end - shared_start
        if total_shared > 0:
            metrics["token_coverage"] = len(used_tokens) / total_shared
        
        # Calculate diversity (entropy)
        if token_counts:
            total_count = sum(token_counts.values())
            entropy = 0.0
            for count in token_counts.values():
                p = count / total_count
                if p > 0:
                    entropy -= p * np.log2(p)
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(token_counts))
            if max_entropy > 0:
                metrics["diversity_score"] = entropy / max_entropy
        
        # Calculate frequency alignment if we have frequency data
        if token_frequencies and used_tokens:
            # Calculate correlation between usage and frequency
            used_freqs = [
                token_frequencies.get(tid, 0.0)
                for tid in used_tokens
            ]
            if used_freqs:
                # Higher frequency tokens should be used more
                avg_used_freq = np.mean(used_freqs)
                all_shared_freqs = [
                    token_frequencies.get(tid, 0.0)
                    for tid in range(shared_start, shared_end)
                    if tid in token_frequencies
                ]
                if all_shared_freqs:
                    avg_all_freq = np.mean(all_shared_freqs)
                    if avg_all_freq > 0:
                        # Alignment score: ratio of average frequencies
                        metrics["frequency_alignment"] = min(1.0, avg_used_freq / avg_all_freq)
        
        return metrics
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about challenge adaptations.
        
        Returns:
            Dictionary of adaptation statistics
        """
        stats = self.adaptation_stats.copy()
        
        # Calculate success rate
        if stats["total_adaptations"] > 0:
            stats["success_rate"] = stats["successful_adaptations"] / stats["total_adaptations"]
            stats["fallback_rate"] = stats["fallback_used"] / stats["total_adaptations"]
        else:
            stats["success_rate"] = 0.0
            stats["fallback_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset adaptation statistics."""
        self.adaptation_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "fallback_used": 0,
            "average_token_coverage": 0.0
        }


def create_adaptive_challenge_generator(
    config_path: Optional[str] = None
) -> AdaptiveChallengeGenerator:
    """
    Factory function to create an adaptive challenge generator.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        Configured AdaptiveChallengeGenerator instance
    """
    # Load configuration if provided
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vocab_config = config.get("vocabulary_compatibility", {})
        
        generator = AdaptiveChallengeGenerator(
            min_overlap_ratio=vocab_config.get("min_overlap_ratio", 0.95),
            core_vocab_size=vocab_config.get("core_vocab_size", 30000),
            enable_frequency_weighting=vocab_config.get("enable_frequency_weighting", False)
        )
    else:
        # Use defaults
        generator = AdaptiveChallengeGenerator()
    
    return generator