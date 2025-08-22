"""
Shard Scheduler for Optimal Loading Order

Determines optimal shard loading order to minimize I/O and maximize cache hits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque
import heapq

logger = logging.getLogger(__name__)


@dataclass
class ShardDependency:
    """Dependency between shards"""
    source_shard: int
    target_shard: int
    strength: float  # 0.0 to 1.0


@dataclass
class ScheduleEntry:
    """Entry in shard schedule"""
    shard_id: int
    priority: float
    estimated_load_time: float
    dependencies_met: bool
    prefetch_candidates: List[int]


class ShardScheduler:
    """
    Schedules shard loading for optimal performance.
    
    Features:
    - Dependency-aware scheduling
    - I/O optimization
    - Cache-aware ordering
    - Parallel loading support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize shard scheduler.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or {}
        self.enable_parallel = self.config.get('enable_parallel', False)
        self.lookahead_window = self.config.get('lookahead_window', 3)
        self.io_bandwidth_mbps = self.config.get('io_bandwidth_mbps', 100)
        
        self.dependencies: List[ShardDependency] = []
        self.shard_sizes: Dict[int, int] = {}
        self.load_history: List[int] = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def build_dependency_graph(
        self,
        shard_infos: List[Any],
        model_architecture: Optional[str] = None
    ) -> None:
        """
        Build dependency graph between shards.
        
        Args:
            shard_infos: Information about shards
            model_architecture: Model architecture type
        """
        self.dependencies.clear()
        
        # Build dependencies based on layer connectivity
        for i, shard in enumerate(shard_infos):
            # Sequential dependencies (transformer layers)
            if i > 0:
                self.dependencies.append(ShardDependency(
                    source_shard=i - 1,
                    target_shard=i,
                    strength=0.9  # Strong sequential dependency
                ))
            
            # Attention dependencies (for attention layers)
            if model_architecture == 'transformer' and 'attention' in str(shard):
                # Attention layers may need access to earlier layers
                for j in range(max(0, i - 3), i):
                    self.dependencies.append(ShardDependency(
                        source_shard=j,
                        target_shard=i,
                        strength=0.3
                    ))
            
            # Store shard sizes
            self.shard_sizes[i] = shard.size_bytes if hasattr(shard, 'size_bytes') else 1024 * 1024
    
    def create_schedule(
        self,
        num_shards: int,
        memory_limit_mb: int,
        processing_order: Optional[List[int]] = None
    ) -> List[ScheduleEntry]:
        """
        Create optimal shard loading schedule.
        
        Args:
            num_shards: Total number of shards
            memory_limit_mb: Available memory limit
            processing_order: Required processing order (if any)
            
        Returns:
            Ordered list of ScheduleEntry
        """
        if processing_order:
            # Use specified order but optimize prefetching
            return self._create_fixed_schedule(processing_order, memory_limit_mb)
        else:
            # Create optimal schedule
            return self._create_optimal_schedule(num_shards, memory_limit_mb)
    
    def _create_optimal_schedule(
        self,
        num_shards: int,
        memory_limit_mb: int
    ) -> List[ScheduleEntry]:
        """Create optimal schedule based on dependencies"""
        schedule = []
        processed = set()
        ready_queue = []
        
        # Initialize with shards that have no dependencies
        for shard_id in range(num_shards):
            if not any(d.target_shard == shard_id for d in self.dependencies):
                priority = self._calculate_priority(shard_id, processed)
                heapq.heappush(ready_queue, (-priority, shard_id))
        
        # If no independent shards, start with first
        if not ready_queue:
            heapq.heappush(ready_queue, (0, 0))
        
        while ready_queue:
            _, shard_id = heapq.heappop(ready_queue)
            
            if shard_id in processed:
                continue
            
            # Check if dependencies are met
            deps_met = all(
                d.source_shard in processed
                for d in self.dependencies
                if d.target_shard == shard_id
            )
            
            # Calculate estimated load time
            shard_size_mb = self.shard_sizes.get(shard_id, 100 * 1024 * 1024) / 1024 / 1024
            load_time = shard_size_mb / self.io_bandwidth_mbps
            
            # Determine prefetch candidates
            prefetch = self._get_prefetch_candidates(
                shard_id,
                num_shards,
                processed,
                memory_limit_mb
            )
            
            entry = ScheduleEntry(
                shard_id=shard_id,
                priority=-len(schedule),  # Processing order
                estimated_load_time=load_time,
                dependencies_met=deps_met,
                prefetch_candidates=prefetch
            )
            
            schedule.append(entry)
            processed.add(shard_id)
            
            # Add newly ready shards to queue
            for dep in self.dependencies:
                if dep.source_shard == shard_id and dep.target_shard not in processed:
                    target = dep.target_shard
                    if all(d.source_shard in processed for d in self.dependencies if d.target_shard == target):
                        priority = self._calculate_priority(target, processed)
                        heapq.heappush(ready_queue, (-priority, target))
        
        # Add any remaining shards
        for shard_id in range(num_shards):
            if shard_id not in processed:
                entry = ScheduleEntry(
                    shard_id=shard_id,
                    priority=-len(schedule),
                    estimated_load_time=1.0,
                    dependencies_met=False,
                    prefetch_candidates=[]
                )
                schedule.append(entry)
        
        return schedule
    
    def _create_fixed_schedule(
        self,
        processing_order: List[int],
        memory_limit_mb: int
    ) -> List[ScheduleEntry]:
        """Create schedule with fixed processing order"""
        schedule = []
        processed = set()
        
        for i, shard_id in enumerate(processing_order):
            # Calculate load time
            shard_size_mb = self.shard_sizes.get(shard_id, 100 * 1024 * 1024) / 1024 / 1024
            load_time = shard_size_mb / self.io_bandwidth_mbps
            
            # Get prefetch candidates
            prefetch = []
            if i + 1 < len(processing_order):
                # Prefetch next shards if memory allows
                remaining_memory = memory_limit_mb - shard_size_mb
                for next_id in processing_order[i + 1:i + 1 + self.lookahead_window]:
                    next_size_mb = self.shard_sizes.get(next_id, 100 * 1024 * 1024) / 1024 / 1024
                    if remaining_memory >= next_size_mb:
                        prefetch.append(next_id)
                        remaining_memory -= next_size_mb
            
            entry = ScheduleEntry(
                shard_id=shard_id,
                priority=i,
                estimated_load_time=load_time,
                dependencies_met=True,
                prefetch_candidates=prefetch
            )
            
            schedule.append(entry)
            processed.add(shard_id)
        
        return schedule
    
    def _calculate_priority(self, shard_id: int, processed: set) -> float:
        """Calculate priority score for shard"""
        priority = 0.0
        
        # Prioritize shards with many dependents
        num_dependents = sum(
            1 for d in self.dependencies
            if d.source_shard == shard_id and d.target_shard not in processed
        )
        priority += num_dependents * 10
        
        # Consider dependency strength
        total_strength = sum(
            d.strength for d in self.dependencies
            if d.target_shard == shard_id and d.source_shard in processed
        )
        priority += total_strength * 5
        
        # Prefer smaller shards (faster to load)
        shard_size_mb = self.shard_sizes.get(shard_id, 100 * 1024 * 1024) / 1024 / 1024
        priority -= shard_size_mb * 0.01
        
        return priority
    
    def _get_prefetch_candidates(
        self,
        current_shard: int,
        num_shards: int,
        processed: set,
        memory_limit_mb: int
    ) -> List[int]:
        """Get shards to prefetch"""
        candidates = []
        remaining_memory = memory_limit_mb
        
        # Current shard size
        current_size_mb = self.shard_sizes.get(current_shard, 100 * 1024 * 1024) / 1024 / 1024
        remaining_memory -= current_size_mb
        
        # Find dependent shards
        for dep in self.dependencies:
            if dep.source_shard == current_shard and dep.target_shard not in processed:
                target = dep.target_shard
                target_size_mb = self.shard_sizes.get(target, 100 * 1024 * 1024) / 1024 / 1024
                
                if remaining_memory >= target_size_mb:
                    candidates.append(target)
                    remaining_memory -= target_size_mb
        
        # Add sequential shards if space remains
        for offset in range(1, self.lookahead_window + 1):
            next_shard = current_shard + offset
            if next_shard < num_shards and next_shard not in processed and next_shard not in candidates:
                next_size_mb = self.shard_sizes.get(next_shard, 100 * 1024 * 1024) / 1024 / 1024
                
                if remaining_memory >= next_size_mb:
                    candidates.append(next_shard)
                    remaining_memory -= next_size_mb
        
        return candidates
    
    def record_load(self, shard_id: int, hit_cache: bool) -> None:
        """
        Record shard load for statistics.
        
        Args:
            shard_id: Loaded shard ID
            hit_cache: Whether it was a cache hit
        """
        self.load_history.append(shard_id)
        
        if hit_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        total_loads = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_loads if total_loads > 0 else 0
        
        # Analyze load patterns
        sequential_loads = 0
        if len(self.load_history) > 1:
            for i in range(1, len(self.load_history)):
                if self.load_history[i] == self.load_history[i-1] + 1:
                    sequential_loads += 1
        
        return {
            'total_loads': total_loads,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'sequential_loads': sequential_loads,
            'load_history_length': len(self.load_history)
        }
    
    def optimize_for_pattern(self, access_pattern: str) -> Dict[str, Any]:
        """
        Optimize scheduling for specific access pattern.
        
        Args:
            access_pattern: 'sequential', 'random', 'strided'
            
        Returns:
            Optimized configuration
        """
        if access_pattern == 'sequential':
            return {
                'lookahead_window': 5,
                'enable_parallel': True,
                'prefetch_aggressive': True
            }
        elif access_pattern == 'random':
            return {
                'lookahead_window': 1,
                'enable_parallel': False,
                'prefetch_aggressive': False
            }
        elif access_pattern == 'strided':
            return {
                'lookahead_window': 3,
                'enable_parallel': True,
                'prefetch_aggressive': False
            }
        else:
            return self.config