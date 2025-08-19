#!/usr/bin/env python3
"""
Provenance auditor for PoT framework.
Provides Merkle tree based verification of training history.
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Tuple, Optional


class ProvenanceAuditor:
    """Auditor for training provenance using Merkle trees."""
    
    def __init__(self):
        self.events = []
        self.merkle_tree = []
        
    def add_training_event(self, event: Dict[str, Any]):
        """Add a training event to the audit log."""
        # Add timestamp if not present
        if 'timestamp' not in event:
            event['timestamp'] = time.time()
        
        # Compute hash of event
        event_str = json.dumps(event, sort_keys=True)
        event_hash = hashlib.sha256(event_str.encode()).hexdigest()
        
        self.events.append({
            'data': event,
            'hash': event_hash
        })
    
    def _build_merkle_tree(self) -> List[List[str]]:
        """Build Merkle tree from events."""
        if not self.events:
            return []
        
        # Start with leaf nodes (event hashes)
        tree = []
        current_level = [e['hash'] for e in self.events]
        tree.append(current_level)
        
        # Build tree levels
        while len(current_level) > 1:
            next_level = []
            
            # Pair up nodes
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    combined = current_level[i] + current_level[i + 1]
                else:
                    # Odd number - duplicate last node
                    combined = current_level[i] + current_level[i]
                
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)
            
            tree.append(next_level)
            current_level = next_level
        
        return tree
    
    def get_merkle_root(self) -> str:
        """Get the Merkle root of all events."""
        tree = self._build_merkle_tree()
        if not tree:
            return ""
        return tree[-1][0]  # Root is at top level
    
    def get_merkle_proof(self, event_index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for an event."""
        if event_index >= len(self.events):
            return []
        
        tree = self._build_merkle_tree()
        if not tree:
            return []
        
        proof = []
        current_index = event_index
        
        # Traverse tree from leaf to root
        for level in range(len(tree) - 1):
            current_level = tree[level]
            
            # Find sibling
            if current_index % 2 == 0:
                # Even index - sibling is on right
                sibling_index = current_index + 1
                direction = 'right'
            else:
                # Odd index - sibling is on left
                sibling_index = current_index - 1
                direction = 'left'
            
            # Add sibling to proof if it exists
            if sibling_index < len(current_level):
                proof.append((direction, current_level[sibling_index]))
            else:
                # No sibling - duplicate self
                proof.append((direction, current_level[current_index]))
            
            # Move to parent index
            current_index = current_index // 2
        
        return proof
    
    def verify_merkle_proof(self, event: Dict[str, Any], proof: List[Tuple[str, str]], root: str) -> bool:
        """Verify a Merkle proof for an event."""
        # Compute event hash
        event_str = json.dumps(event, sort_keys=True)
        current_hash = hashlib.sha256(event_str.encode()).hexdigest()
        
        # Apply proof
        for direction, sibling_hash in proof:
            if direction == 'left':
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        # Check if we reach the root
        return current_hash == root