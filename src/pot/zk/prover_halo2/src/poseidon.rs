use ff::{Field, PrimeField};

/// Simplified Poseidon implementation for compatibility with Python
/// This is a minimal implementation that provides the interface needed
/// for integration with the PoT framework.

/// Utility functions for Poseidon hashing outside of circuits
pub mod primitives {
    use super::*;
    use pasta_curves::pallas;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Hash two field elements using a simplified Poseidon-like function
    /// In production, this would use a real Poseidon implementation
    pub fn hash_two(left: pallas::Base, right: pallas::Base) -> pallas::Base {
        // Simplified hash using standard library hasher
        // In production, this would be a proper Poseidon hash
        let mut hasher = DefaultHasher::new();
        left.to_repr().hash(&mut hasher);
        right.to_repr().hash(&mut hasher);
        let hash_result = hasher.finish();
        
        // Convert to field element
        pallas::Base::from_u128(hash_result as u128)
    }

    /// Hash a slice of bytes by converting to field elements
    pub fn hash_bytes(data: &[u8]) -> pallas::Base {
        // Convert bytes to field elements (chunk into 31-byte segments to fit in field)
        let mut elements = Vec::new();
        for chunk in data.chunks(31) {
            let mut padded = [0u8; 32];
            padded[1..chunk.len() + 1].copy_from_slice(chunk);
            if let Some(element) = pallas::Base::from_repr(padded).into() {
                elements.push(element);
            }
        }

        // If we have an odd number of elements, pad with zero
        if elements.len() % 2 == 1 {
            elements.push(pallas::Base::ZERO);
        }

        // Hash pairs iteratively to build a Merkle tree
        while elements.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in elements.chunks(2) {
                if chunk.len() == 2 {
                    next_level.push(hash_two(chunk[0], chunk[1]));
                } else {
                    next_level.push(chunk[0]);
                }
            }
            elements = next_level;
        }

        elements.get(0).copied().unwrap_or(pallas::Base::ZERO)
    }

    /// Convert field element to bytes (little-endian)
    pub fn field_to_bytes(element: pallas::Base) -> [u8; 32] {
        element.to_repr()
    }

    /// Convert bytes to field element
    pub fn bytes_to_field(bytes: &[u8]) -> Option<pallas::Base> {
        if bytes.len() != 32 {
            return None;
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(bytes);
        pallas::Base::from_repr(array).into()
    }

    /// Compute Merkle root from leaf values
    pub fn compute_merkle_root(leaves: &[pallas::Base]) -> pallas::Base {
        if leaves.is_empty() {
            return pallas::Base::ZERO;
        }
        
        if leaves.len() == 1 {
            return leaves[0];
        }

        let mut current_level = leaves.to_vec();
        
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            // Pad to even number if needed
            if current_level.len() % 2 == 1 {
                current_level.push(current_level[current_level.len() - 1]);
            }
            
            // Hash pairs
            for chunk in current_level.chunks(2) {
                next_level.push(hash_two(chunk[0], chunk[1]));
            }
            
            current_level = next_level;
        }
        
        current_level[0]
    }

    /// Generate Merkle proof for a leaf at given index
    pub fn generate_merkle_proof(leaves: &[pallas::Base], index: usize) -> Vec<pallas::Base> {
        if index >= leaves.len() {
            return Vec::new();
        }
        
        let mut proof = Vec::new();
        let mut current_level = leaves.to_vec();
        let mut current_index = index;
        
        while current_level.len() > 1 {
            // Pad to even number if needed
            if current_level.len() % 2 == 1 {
                current_level.push(current_level[current_level.len() - 1]);
            }
            
            // Find sibling
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };
            
            if sibling_index < current_level.len() {
                proof.push(current_level[sibling_index]);
            }
            
            // Move to next level
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                next_level.push(hash_two(chunk[0], chunk[1]));
            }
            
            current_level = next_level;
            current_index /= 2;
        }
        
        proof
    }

    /// Verify Merkle proof
    pub fn verify_merkle_proof(
        leaf: pallas::Base,
        proof: &[pallas::Base],
        index: usize,
        root: pallas::Base,
    ) -> bool {
        let mut current_hash = leaf;
        let mut current_index = index;
        
        for &sibling in proof {
            if current_index % 2 == 0 {
                current_hash = hash_two(current_hash, sibling);
            } else {
                current_hash = hash_two(sibling, current_hash);
            }
            current_index /= 2;
        }
        
        current_hash == root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pasta_curves::pallas;

    #[test]
    fn test_poseidon_hash_consistency() {
        let left = pallas::Base::from_u128(123u128);
        let right = pallas::Base::from_u128(456u128);
        
        let hash1 = primitives::hash_two(left, right);
        let hash2 = primitives::hash_two(left, right);
        
        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }

    #[test]
    fn test_merkle_operations() {
        let leaves: Vec<pallas::Base> = (0..8).map(|i| pallas::Base::from_u128(i as u128)).collect();
        let root = primitives::compute_merkle_root(&leaves);
        
        // Test proof generation and verification
        for i in 0..leaves.len() {
            let proof = primitives::generate_merkle_proof(&leaves, i);
            let is_valid = primitives::verify_merkle_proof(leaves[i], &proof, i, root);
            assert!(is_valid, "Merkle proof should be valid for index {}", i);
        }
    }

    #[test]
    fn test_bytes_conversion() {
        let original = pallas::Base::from_u128(12345u128);
        let bytes = primitives::field_to_bytes(original);
        let recovered = primitives::bytes_to_field(&bytes).unwrap();
        assert_eq!(original, recovered, "Bytes conversion should be reversible");
    }
}