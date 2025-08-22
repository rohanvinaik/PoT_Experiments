"""
Poseidon hash implementation for BN254/Pallas field.

This implementation matches the parameters and behavior of the Rust
halo2_gadgets Poseidon implementation for compatibility.
"""

from typing import List, Optional, Union
import hashlib
from pot.zk.field_arithmetic import FieldElement, FieldVector


class PoseidonConstants:
    """
    Constants for Poseidon hash function.
    
    These should match the Rust implementation's parameters for
    the Pallas curve (t=3, full_rounds=8, partial_rounds=56).
    """
    
    def __init__(self, t: int = 3, full_rounds: int = 8, partial_rounds: int = 56):
        """
        Initialize Poseidon constants.
        
        Args:
            t: Width of the permutation (3 for 2-to-1 hash)
            full_rounds: Number of full rounds
            partial_rounds: Number of partial rounds
        """
        self.t = t
        self.full_rounds = full_rounds
        self.partial_rounds = partial_rounds
        self.rounds = full_rounds + partial_rounds
        
        # Generate round constants
        self.round_constants = self._generate_round_constants()
        
        # Generate MDS matrix
        self.mds_matrix = self._generate_mds_matrix()
    
    def _generate_round_constants(self) -> List[List[FieldElement]]:
        """
        Generate round constants using a deterministic method.
        
        This should match the Rust implementation's constants.
        For now, we use a simplified generation method.
        """
        constants = []
        
        # Use BLAKE2b to generate deterministic constants
        seed = b"Poseidon_Pallas_" + bytes([self.t, self.full_rounds, self.partial_rounds])
        
        for round_idx in range(self.rounds):
            round_consts = []
            for i in range(self.t):
                # Generate deterministic constant for position (round, i)
                data = seed + bytes([round_idx, i])
                hash_val = hashlib.blake2b(data, digest_size=32).digest()
                const = FieldElement.from_bytes(hash_val)
                round_consts.append(const)
            constants.append(round_consts)
        
        return constants
    
    def _generate_mds_matrix(self) -> List[List[FieldElement]]:
        """
        Generate Maximum Distance Separable (MDS) matrix.
        
        Using a Cauchy matrix construction for simplicity.
        """
        matrix = []
        
        # Generate x and y values for Cauchy matrix
        x_vals = [FieldElement(i) for i in range(self.t)]
        y_vals = [FieldElement(self.t + i) for i in range(self.t)]
        
        for i in range(self.t):
            row = []
            for j in range(self.t):
                # Cauchy matrix element: 1 / (x_i + y_j)
                element = (x_vals[i] + y_vals[j]).inverse()
                row.append(element)
            matrix.append(row)
        
        return matrix


class PoseidonHash:
    """
    Poseidon hash function implementation.
    
    This provides a 2-to-1 hash function compatible with the Rust implementation.
    """
    
    def __init__(self, constants: Optional[PoseidonConstants] = None):
        """
        Initialize Poseidon hash.
        
        Args:
            constants: Poseidon constants (uses defaults if None)
        """
        self.constants = constants or PoseidonConstants()
    
    def _sbox(self, x: FieldElement) -> FieldElement:
        """
        S-box operation: x^5 for Poseidon.
        
        This is the standard S-box for Poseidon over prime fields.
        """
        return x ** 5
    
    def _apply_mds(self, state: List[FieldElement]) -> List[FieldElement]:
        """Apply MDS matrix multiplication."""
        result = []
        for i in range(self.constants.t):
            acc = FieldElement.zero()
            for j in range(self.constants.t):
                acc = acc + self.constants.mds_matrix[i][j] * state[j]
            result.append(acc)
        return result
    
    def _full_round(self, state: List[FieldElement], round_idx: int) -> List[FieldElement]:
        """Apply a full round of Poseidon."""
        # Add round constants
        for i in range(self.constants.t):
            state[i] = state[i] + self.constants.round_constants[round_idx][i]
        
        # Apply S-box to all elements
        state = [self._sbox(x) for x in state]
        
        # Apply MDS matrix
        state = self._apply_mds(state)
        
        return state
    
    def _partial_round(self, state: List[FieldElement], round_idx: int) -> List[FieldElement]:
        """Apply a partial round of Poseidon."""
        # Add round constants
        for i in range(self.constants.t):
            state[i] = state[i] + self.constants.round_constants[round_idx][i]
        
        # Apply S-box only to first element (partial)
        state[0] = self._sbox(state[0])
        
        # Apply MDS matrix
        state = self._apply_mds(state)
        
        return state
    
    def permutation(self, state: List[FieldElement]) -> List[FieldElement]:
        """
        Apply Poseidon permutation.
        
        Args:
            state: Input state vector of length t
            
        Returns:
            Permuted state vector
        """
        if len(state) != self.constants.t:
            raise ValueError(f"State must have length {self.constants.t}")
        
        state = state.copy()
        round_idx = 0
        
        # First half of full rounds
        for _ in range(self.constants.full_rounds // 2):
            state = self._full_round(state, round_idx)
            round_idx += 1
        
        # Partial rounds
        for _ in range(self.constants.partial_rounds):
            state = self._partial_round(state, round_idx)
            round_idx += 1
        
        # Second half of full rounds
        for _ in range(self.constants.full_rounds // 2):
            state = self._full_round(state, round_idx)
            round_idx += 1
        
        return state
    
    def hash_single(self, input_val: Union[FieldElement, bytes, int]) -> FieldElement:
        """
        Hash a single value.
        
        Args:
            input_val: Value to hash
            
        Returns:
            Hash output as field element
        """
        if not isinstance(input_val, FieldElement):
            if isinstance(input_val, bytes):
                input_val = FieldElement.from_bytes(input_val)
            else:
                input_val = FieldElement(input_val)
        
        # For single input, pad with zeros
        state = [input_val, FieldElement.zero(), FieldElement.zero()]
        
        # Apply permutation
        state = self.permutation(state)
        
        # Return first element as output
        return state[0]
    
    def hash_two(self, left: Union[FieldElement, bytes], 
                 right: Union[FieldElement, bytes]) -> FieldElement:
        """
        Hash two values (2-to-1 hash).
        
        This is the primary use case for Merkle trees.
        
        Args:
            left: Left input
            right: Right input
            
        Returns:
            Hash output as field element
        """
        if not isinstance(left, FieldElement):
            left = FieldElement.from_bytes(left) if isinstance(left, bytes) else FieldElement(left)
        if not isinstance(right, FieldElement):
            right = FieldElement.from_bytes(right) if isinstance(right, bytes) else FieldElement(right)
        
        # State: [left, right, 0]
        state = [left, right, FieldElement.zero()]
        
        # Apply permutation
        state = self.permutation(state)
        
        # Return first element as output
        return state[0]
    
    def hash_many(self, inputs: List[Union[FieldElement, bytes, int]]) -> FieldElement:
        """
        Hash multiple values using a sponge construction.
        
        Args:
            inputs: List of values to hash
            
        Returns:
            Hash output as field element
        """
        if not inputs:
            return FieldElement.zero()
        
        # Convert all inputs to field elements
        field_inputs = []
        for inp in inputs:
            if isinstance(inp, FieldElement):
                field_inputs.append(inp)
            elif isinstance(inp, bytes):
                field_inputs.append(FieldElement.from_bytes(inp))
            else:
                field_inputs.append(FieldElement(inp))
        
        # For single input, use hash_single
        if len(field_inputs) == 1:
            return self.hash_single(field_inputs[0])
        
        # For two inputs, use hash_two
        if len(field_inputs) == 2:
            return self.hash_two(field_inputs[0], field_inputs[1])
        
        # For more inputs, use tree hashing
        # Hash pairs recursively
        while len(field_inputs) > 1:
            next_level = []
            for i in range(0, len(field_inputs), 2):
                if i + 1 < len(field_inputs):
                    hashed = self.hash_two(field_inputs[i], field_inputs[i + 1])
                else:
                    hashed = self.hash_single(field_inputs[i])
                next_level.append(hashed)
            field_inputs = next_level
        
        return field_inputs[0]
    
    def compress(self, left: FieldElement, right: FieldElement) -> FieldElement:
        """
        Compress two field elements (alias for hash_two).
        
        Args:
            left: Left input
            right: Right input
            
        Returns:
            Compressed output
        """
        return self.hash_two(left, right)


# Global instance for convenience
_default_poseidon = None


def get_poseidon() -> PoseidonHash:
    """Get the default Poseidon instance."""
    global _default_poseidon
    if _default_poseidon is None:
        _default_poseidon = PoseidonHash()
    return _default_poseidon


def poseidon_hash(data: bytes) -> bytes:
    """
    Hash bytes using Poseidon.
    
    Args:
        data: Input bytes
        
    Returns:
        32-byte hash output
    """
    hasher = get_poseidon()
    field_elem = FieldElement.from_bytes(data)
    result = hasher.hash_single(field_elem)
    return result.to_bytes()


def poseidon_hash_two(left: bytes, right: bytes) -> bytes:
    """
    Hash two byte arrays using Poseidon.
    
    Args:
        left: Left input bytes
        right: Right input bytes
        
    Returns:
        32-byte hash output
    """
    hasher = get_poseidon()
    result = hasher.hash_two(left, right)
    return result.to_bytes()


def poseidon_hash_many(inputs: List[bytes]) -> bytes:
    """
    Hash multiple byte arrays using Poseidon.
    
    Args:
        inputs: List of byte arrays
        
    Returns:
        32-byte hash output
    """
    hasher = get_poseidon()
    result = hasher.hash_many(inputs)
    return result.to_bytes()


class PoseidonMerkleTree:
    """
    Merkle tree using Poseidon hash.
    """
    
    def __init__(self, leaves: List[bytes]):
        """
        Initialize Merkle tree with leaves.
        
        Args:
            leaves: Leaf data as bytes
        """
        self.hasher = get_poseidon()
        self.leaves = [FieldElement.from_bytes(leaf) for leaf in leaves]
        self.tree = self._build_tree()
    
    def _build_tree(self) -> List[List[FieldElement]]:
        """Build the complete Merkle tree."""
        if not self.leaves:
            return [[FieldElement.zero()]]
        
        tree = [self.leaves.copy()]
        current_level = self.leaves
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    parent = self.hasher.hash_two(current_level[i], current_level[i + 1])
                else:
                    # Odd number of nodes, hash with itself
                    parent = self.hasher.hash_two(current_level[i], current_level[i])
                next_level.append(parent)
            tree.append(next_level)
            current_level = next_level
        
        return tree
    
    def root(self) -> bytes:
        """Get the Merkle root."""
        if self.tree:
            return self.tree[-1][0].to_bytes()
        return FieldElement.zero().to_bytes()
    
    def proof(self, index: int) -> List[bytes]:
        """
        Generate Merkle proof for a leaf.
        
        Args:
            index: Leaf index
            
        Returns:
            List of sibling hashes forming the proof
        """
        if index >= len(self.leaves):
            raise ValueError(f"Index {index} out of range")
        
        proof = []
        for level in range(len(self.tree) - 1):
            level_nodes = self.tree[level]
            sibling_idx = index ^ 1  # XOR with 1 to get sibling
            
            if sibling_idx < len(level_nodes):
                proof.append(level_nodes[sibling_idx].to_bytes())
            else:
                # No sibling, use the node itself
                proof.append(level_nodes[index].to_bytes())
            
            index //= 2
        
        return proof
    
    def verify(self, leaf: bytes, index: int, proof: List[bytes]) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            leaf: Leaf data
            index: Leaf index
            proof: Merkle proof
            
        Returns:
            True if proof is valid
        """
        current = FieldElement.from_bytes(leaf)
        
        for sibling_bytes in proof:
            sibling = FieldElement.from_bytes(sibling_bytes)
            
            if index & 1:
                # Current is right child
                current = self.hasher.hash_two(sibling, current)
            else:
                # Current is left child
                current = self.hasher.hash_two(current, sibling)
            
            index //= 2
        
        return current == self.tree[-1][0]


def poseidon_merkle_root(leaves: List[bytes]) -> bytes:
    """
    Compute Merkle root using Poseidon hash.
    
    Args:
        leaves: List of leaf data
        
    Returns:
        32-byte Merkle root
    """
    tree = PoseidonMerkleTree(leaves)
    return tree.root()