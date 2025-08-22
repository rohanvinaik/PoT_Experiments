"""
Memory Manager for Sharded Verification

Intelligent memory management for large model verification with automatic 
garbage collection, memory pressure handling, and OOM prevention.
"""

import gc
import os
import psutil
import resource
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
from collections import deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MemoryState:
    """Current memory state snapshot"""
    timestamp: float
    rss_bytes: int
    vms_bytes: int
    available_bytes: int
    percent_used: float
    swap_used_bytes: int
    page_faults: int
    gc_stats: Dict[str, int]


@dataclass
class MemoryPolicy:
    """Memory management policy"""
    max_memory_percent: float  # Maximum memory usage percentage
    gc_threshold_percent: float  # Trigger GC at this usage
    emergency_threshold_percent: float  # Emergency cleanup threshold
    swap_limit_mb: int  # Maximum swap usage
    oom_safety_margin_mb: int  # Safety margin before OOM


class MemoryManager:
    """
    Manages memory for sharded model verification.
    
    Features:
    - Real-time memory monitoring
    - Automatic garbage collection
    - Memory pressure detection
    - OOM prevention
    - Swap usage monitoring
    """
    
    def __init__(self, policy: Optional[MemoryPolicy] = None):
        """
        Initialize memory manager.
        
        Args:
            policy: Memory management policy
        """
        self.policy = policy or MemoryPolicy(
            max_memory_percent=80.0,
            gc_threshold_percent=70.0,
            emergency_threshold_percent=90.0,
            swap_limit_mb=1024,
            oom_safety_margin_mb=512
        )
        
        self.memory_history = deque(maxlen=1000)
        self.gc_history = []
        self.oom_prevented_count = 0
        self.emergency_callbacks: List[Callable] = []
        self._baseline_memory = None
        self._monitor_active = False
        
        # Set up resource limits
        self._setup_resource_limits()
        
        # Establish baseline
        self.establish_baseline()
    
    def establish_baseline(self) -> MemoryState:
        """
        Establish baseline memory usage.
        
        Returns:
            Baseline MemoryState
        """
        # Force garbage collection
        gc.collect()
        
        # Get current state
        state = self.get_current_state()
        self._baseline_memory = state.rss_bytes
        
        logger.info(f"Baseline memory established: {state.rss_bytes / 1024 / 1024:.1f} MB")
        return state
    
    def get_current_state(self) -> MemoryState:
        """
        Get current memory state.
        
        Returns:
            Current MemoryState
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        vm_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        # Get page faults if available
        try:
            page_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        except:
            page_faults = 0
        
        # Get GC stats
        gc_stats = {
            f"gen{i}": gc.get_count()[i] 
            for i in range(len(gc.get_count()))
        }
        
        state = MemoryState(
            timestamp=time.time(),
            rss_bytes=mem_info.rss,
            vms_bytes=mem_info.vms,
            available_bytes=vm_info.available,
            percent_used=vm_info.percent,
            swap_used_bytes=swap_info.used,
            page_faults=page_faults,
            gc_stats=gc_stats
        )
        
        # Record history
        self.memory_history.append(state)
        
        return state
    
    def check_memory_available(self, required_bytes: int) -> bool:
        """
        Check if required memory is available.
        
        Args:
            required_bytes: Bytes required
            
        Returns:
            True if memory available, False otherwise
        """
        state = self.get_current_state()
        
        # Check against policy limits
        total_memory = psutil.virtual_memory().total
        max_allowed = total_memory * (self.policy.max_memory_percent / 100)
        
        projected_usage = state.rss_bytes + required_bytes
        
        if projected_usage > max_allowed:
            logger.warning(f"Memory request would exceed limit: "
                          f"{projected_usage / 1024 / 1024:.1f} MB > "
                          f"{max_allowed / 1024 / 1024:.1f} MB")
            return False
        
        # Check available memory with safety margin
        safety_margin = self.policy.oom_safety_margin_mb * 1024 * 1024
        if state.available_bytes - required_bytes < safety_margin:
            logger.warning(f"Insufficient memory: {state.available_bytes / 1024 / 1024:.1f} MB available, "
                          f"{required_bytes / 1024 / 1024:.1f} MB required")
            return False
        
        return True
    
    def allocate_memory(self, size_bytes: int, name: str = "unnamed") -> bool:
        """
        Allocate memory with safety checks.
        
        Args:
            size_bytes: Bytes to allocate
            name: Name for logging
            
        Returns:
            True if allocation successful
        """
        logger.info(f"Allocating {size_bytes / 1024 / 1024:.1f} MB for {name}")
        
        # Check if memory available
        if not self.check_memory_available(size_bytes):
            # Try to free memory
            freed = self.free_memory(size_bytes)
            if not freed:
                logger.error(f"Failed to allocate memory for {name}")
                return False
        
        # Check memory pressure
        self._check_memory_pressure()
        
        return True
    
    def free_memory(self, target_bytes: int) -> bool:
        """
        Free up memory to meet target.
        
        Args:
            target_bytes: Bytes to free
            
        Returns:
            True if target met
        """
        logger.info(f"Attempting to free {target_bytes / 1024 / 1024:.1f} MB")
        
        initial_state = self.get_current_state()
        
        # Level 1: Standard garbage collection
        gc.collect()
        
        current_state = self.get_current_state()
        freed = initial_state.rss_bytes - current_state.rss_bytes
        
        if freed >= target_bytes:
            logger.info(f"Freed {freed / 1024 / 1024:.1f} MB with GC")
            return True
        
        # Level 2: Full garbage collection
        gc.collect(2)
        
        current_state = self.get_current_state()
        freed = initial_state.rss_bytes - current_state.rss_bytes
        
        if freed >= target_bytes:
            logger.info(f"Freed {freed / 1024 / 1024:.1f} MB with full GC")
            return True
        
        # Level 3: Emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        gc.collect()
        current_state = self.get_current_state()
        freed = initial_state.rss_bytes - current_state.rss_bytes
        
        if freed >= target_bytes:
            logger.info(f"Freed {freed / 1024 / 1024:.1f} MB with emergency callbacks")
            return True
        
        logger.warning(f"Could only free {freed / 1024 / 1024:.1f} MB "
                      f"of requested {target_bytes / 1024 / 1024:.1f} MB")
        return False
    
    def register_emergency_callback(self, callback: Callable) -> None:
        """
        Register callback for emergency memory cleanup.
        
        Args:
            callback: Function to call during emergency cleanup
        """
        self.emergency_callbacks.append(callback)
    
    def monitor_memory_pressure(self) -> Dict[str, Any]:
        """
        Monitor and report memory pressure.
        
        Returns:
            Memory pressure metrics
        """
        state = self.get_current_state()
        
        # Calculate pressure indicators
        pressure = {
            'level': 'normal',
            'percent_used': state.percent_used,
            'swap_pressure': state.swap_used_bytes / (1024 * 1024),
            'page_fault_rate': 0.0,
            'gc_pressure': 0.0,
            'recommendations': []
        }
        
        # Determine pressure level
        if state.percent_used > self.policy.emergency_threshold_percent:
            pressure['level'] = 'critical'
            pressure['recommendations'].append("Emergency memory cleanup required")
        elif state.percent_used > self.policy.gc_threshold_percent:
            pressure['level'] = 'high'
            pressure['recommendations'].append("Consider reducing memory usage")
        elif state.percent_used > 60:
            pressure['level'] = 'moderate'
        
        # Check swap pressure
        if state.swap_used_bytes > self.policy.swap_limit_mb * 1024 * 1024:
            pressure['level'] = 'critical'
            pressure['recommendations'].append("Excessive swap usage detected")
        
        # Calculate page fault rate
        if len(self.memory_history) > 1:
            prev_state = self.memory_history[-2]
            time_delta = state.timestamp - prev_state.timestamp
            if time_delta > 0:
                fault_rate = (state.page_faults - prev_state.page_faults) / time_delta
                pressure['page_fault_rate'] = fault_rate
                
                if fault_rate > 100:  # More than 100 faults/second
                    pressure['recommendations'].append("High page fault rate detected")
        
        # Check GC pressure
        total_gc = sum(state.gc_stats.values())
        if len(self.gc_history) > 0:
            recent_gc_rate = (total_gc - self.gc_history[-1]) / max(1, len(self.gc_history))
            pressure['gc_pressure'] = recent_gc_rate
            
            if recent_gc_rate > 10:
                pressure['recommendations'].append("Frequent garbage collection detected")
        
        self.gc_history.append(total_gc)
        
        return pressure
    
    def _check_memory_pressure(self) -> None:
        """Check and handle memory pressure"""
        pressure = self.monitor_memory_pressure()
        
        if pressure['level'] == 'critical':
            logger.warning("Critical memory pressure detected")
            self._handle_critical_pressure()
        elif pressure['level'] == 'high':
            logger.info("High memory pressure detected")
            gc.collect()
    
    def _handle_critical_pressure(self) -> None:
        """Handle critical memory pressure"""
        logger.warning("Handling critical memory pressure")
        
        # Force full GC
        gc.collect(2)
        
        # Call emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        # Final GC
        gc.collect()
        
        # Check if OOM prevented
        state = self.get_current_state()
        if state.percent_used < self.policy.emergency_threshold_percent:
            self.oom_prevented_count += 1
            logger.info(f"OOM prevented (count: {self.oom_prevented_count})")
    
    def _setup_resource_limits(self) -> None:
        """Set up system resource limits"""
        try:
            # Get current limits
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            
            # Set memory limit if not already set
            if soft == resource.RLIM_INFINITY:
                total_memory = psutil.virtual_memory().total
                limit = int(total_memory * (self.policy.max_memory_percent / 100))
                
                # Don't set hard limit, only soft
                resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
                logger.info(f"Set memory limit to {limit / 1024 / 1024:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage summary.
        
        Returns:
            Memory usage statistics
        """
        state = self.get_current_state()
        
        # Calculate statistics from history
        if len(self.memory_history) > 0:
            rss_history = [s.rss_bytes for s in self.memory_history]
            avg_rss = np.mean(rss_history)
            max_rss = np.max(rss_history)
            min_rss = np.min(rss_history)
        else:
            avg_rss = max_rss = min_rss = state.rss_bytes
        
        # Calculate growth rate
        growth_rate = 0.0
        if self._baseline_memory and self._baseline_memory > 0:
            growth_rate = (state.rss_bytes - self._baseline_memory) / self._baseline_memory * 100
        
        return {
            'current': {
                'rss_mb': state.rss_bytes / 1024 / 1024,
                'vms_mb': state.vms_bytes / 1024 / 1024,
                'available_mb': state.available_bytes / 1024 / 1024,
                'percent_used': state.percent_used,
                'swap_mb': state.swap_used_bytes / 1024 / 1024
            },
            'statistics': {
                'avg_rss_mb': avg_rss / 1024 / 1024,
                'max_rss_mb': max_rss / 1024 / 1024,
                'min_rss_mb': min_rss / 1024 / 1024,
                'growth_rate_percent': growth_rate
            },
            'gc': {
                'total_collections': sum(state.gc_stats.values()),
                'per_generation': state.gc_stats
            },
            'page_faults': state.page_faults,
            'oom_prevented': self.oom_prevented_count
        }
    
    def optimize_gc_settings(self) -> None:
        """Optimize garbage collection settings for large models"""
        # Adjust GC thresholds for large objects
        gc.set_threshold(700, 10, 10)
        
        # Disable GC during critical sections (re-enable manually)
        logger.info("Optimized GC settings for large model verification")
    
    def create_memory_snapshot(self) -> Dict[str, Any]:
        """
        Create detailed memory snapshot for analysis.
        
        Returns:
            Detailed memory snapshot
        """
        import tracemalloc
        
        # Start tracing if not already
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]
        
        state = self.get_current_state()
        pressure = self.monitor_memory_pressure()
        
        return {
            'timestamp': state.timestamp,
            'memory_state': {
                'rss_mb': state.rss_bytes / 1024 / 1024,
                'available_mb': state.available_bytes / 1024 / 1024,
                'percent_used': state.percent_used
            },
            'pressure': pressure,
            'top_allocations': [
                {
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats
            ],
            'gc_stats': state.gc_stats
        }