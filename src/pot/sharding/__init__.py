"""
Sharded Verification Module for Large Models

Enables efficient verification of large language models (70B+ parameters) 
on memory-constrained systems through intelligent sharding and resource management.
"""

from .adaptive_sharding import AdaptiveShardManager
from .memory_manager import MemoryManager
from .shard_scheduler import ShardScheduler
from .checkpoint_manager import CheckpointManager

__all__ = [
    'AdaptiveShardManager',
    'MemoryManager',
    'ShardScheduler',
    'CheckpointManager'
]

# Version info
__version__ = '1.0.0'