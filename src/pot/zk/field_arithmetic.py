"""
Field arithmetic for BN254 curve matching Rust implementation.

This module provides field operations compatible with the pasta curves
used in the Rust Halo2 implementation.
"""

from typing import Union, List, Tuple
import hashlib


class FieldElement:
    """
    Field element for BN254/Pallas base field.
    
    The Pallas base field has modulus:
    p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
    
    This is a 255-bit prime.
    """
    
    # Pallas base field modulus (same as used in Rust)
    MODULUS = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
    
    def __init__(self, value: Union[int, bytes, 'FieldElement']):
        """Initialize field element from int, bytes, or another field element."""
        if isinstance(value, FieldElement):
            self.value = value.value
        elif isinstance(value, bytes):
            # Convert bytes to integer (big-endian)
            self.value = int.from_bytes(value, byteorder='big') % self.MODULUS
        elif isinstance(value, int):
            self.value = value % self.MODULUS
        else:
            raise TypeError(f"Cannot create FieldElement from {type(value)}")
    
    def __add__(self, other: 'FieldElement') -> 'FieldElement':
        """Field addition."""
        if not isinstance(other, FieldElement):
            other = FieldElement(other)
        return FieldElement((self.value + other.value) % self.MODULUS)
    
    def __sub__(self, other: 'FieldElement') -> 'FieldElement':
        """Field subtraction."""
        if not isinstance(other, FieldElement):
            other = FieldElement(other)
        return FieldElement((self.value - other.value) % self.MODULUS)
    
    def __mul__(self, other: 'FieldElement') -> 'FieldElement':
        """Field multiplication."""
        if not isinstance(other, FieldElement):
            other = FieldElement(other)
        return FieldElement((self.value * other.value) % self.MODULUS)
    
    def __pow__(self, exponent: int) -> 'FieldElement':
        """Field exponentiation."""
        return FieldElement(pow(self.value, exponent, self.MODULUS))
    
    def __neg__(self) -> 'FieldElement':
        """Field negation."""
        return FieldElement((-self.value) % self.MODULUS)
    
    def inverse(self) -> 'FieldElement':
        """Multiplicative inverse using Fermat's little theorem."""
        # For prime p, a^(p-1) = 1 mod p, so a^(p-2) = a^(-1) mod p
        return self ** (self.MODULUS - 2)
    
    def __truediv__(self, other: 'FieldElement') -> 'FieldElement':
        """Field division."""
        if not isinstance(other, FieldElement):
            other = FieldElement(other)
        return self * other.inverse()
    
    def __eq__(self, other: Union['FieldElement', int]) -> bool:
        """Equality comparison."""
        if isinstance(other, int):
            other = FieldElement(other)
        return isinstance(other, FieldElement) and self.value == other.value
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FieldElement(0x{self.value:064x})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"0x{self.value:064x}"
    
    def to_bytes(self, length: int = 32) -> bytes:
        """Convert to bytes representation."""
        return self.value.to_bytes(length, byteorder='big')
    
    def to_int(self) -> int:
        """Convert to integer."""
        return self.value
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'FieldElement':
        """Create field element from bytes."""
        return cls(data)
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'FieldElement':
        """Create field element from hex string."""
        if hex_str.startswith('0x'):
            hex_str = hex_str[2:]
        return cls(int(hex_str, 16))
    
    @classmethod
    def zero(cls) -> 'FieldElement':
        """Return the zero element."""
        return cls(0)
    
    @classmethod
    def one(cls) -> 'FieldElement':
        """Return the multiplicative identity."""
        return cls(1)
    
    @classmethod
    def random(cls) -> 'FieldElement':
        """Generate a random field element."""
        import secrets
        return cls(secrets.randbits(256))


def bytes_to_field(data: bytes) -> FieldElement:
    """
    Convert arbitrary bytes to a field element.
    
    This matches the Rust implementation's approach of hashing
    the data first to ensure uniform distribution.
    """
    if len(data) <= 31:
        # For small data, pad and convert directly
        padded = data + b'\x00' * (32 - len(data))
        return FieldElement.from_bytes(padded)
    else:
        # For larger data, hash first
        hashed = hashlib.blake2b(data, digest_size=32).digest()
        return FieldElement.from_bytes(hashed)


def field_to_bytes(element: FieldElement) -> bytes:
    """Convert field element to bytes."""
    return element.to_bytes(32)


def int_to_field(value: int) -> FieldElement:
    """Convert integer to field element."""
    return FieldElement(value)


def field_to_int(element: FieldElement) -> int:
    """Convert field element to integer."""
    return element.to_int()


class FieldVector:
    """Vector of field elements for batch operations."""
    
    def __init__(self, elements: List[Union[FieldElement, int]]):
        """Initialize from list of elements."""
        self.elements = [
            e if isinstance(e, FieldElement) else FieldElement(e)
            for e in elements
        ]
    
    def __len__(self) -> int:
        """Length of vector."""
        return len(self.elements)
    
    def __getitem__(self, index: int) -> FieldElement:
        """Get element at index."""
        return self.elements[index]
    
    def __setitem__(self, index: int, value: Union[FieldElement, int]):
        """Set element at index."""
        if not isinstance(value, FieldElement):
            value = FieldElement(value)
        self.elements[index] = value
    
    def append(self, element: Union[FieldElement, int]):
        """Append element to vector."""
        if not isinstance(element, FieldElement):
            element = FieldElement(element)
        self.elements.append(element)
    
    def extend(self, other: Union['FieldVector', List]):
        """Extend vector with another vector or list."""
        if isinstance(other, FieldVector):
            self.elements.extend(other.elements)
        else:
            for e in other:
                self.append(e)
    
    def to_bytes(self) -> bytes:
        """Convert vector to bytes."""
        result = b''
        for e in self.elements:
            result += e.to_bytes()
        return result
    
    @classmethod
    def from_bytes(cls, data: bytes, element_size: int = 32) -> 'FieldVector':
        """Create vector from bytes."""
        elements = []
        for i in range(0, len(data), element_size):
            chunk = data[i:i+element_size]
            elements.append(FieldElement.from_bytes(chunk))
        return cls(elements)


def interpolate_polynomial(points: List[Tuple[FieldElement, FieldElement]]) -> List[FieldElement]:
    """
    Lagrange interpolation to find polynomial coefficients.
    
    Args:
        points: List of (x, y) pairs as field elements
        
    Returns:
        Polynomial coefficients [a0, a1, ..., an] where
        p(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    """
    n = len(points)
    coeffs = [FieldElement.zero() for _ in range(n)]
    
    for i in range(n):
        xi, yi = points[i]
        
        # Compute Lagrange basis polynomial li(x)
        basis_coeffs = [FieldElement.one()]
        
        for j in range(n):
            if i == j:
                continue
            xj, _ = points[j]
            
            # Multiply by (x - xj) / (xi - xj)
            scale = (xi - xj).inverse()
            new_coeffs = [FieldElement.zero()] * (len(basis_coeffs) + 1)
            
            for k, coeff in enumerate(basis_coeffs):
                new_coeffs[k] = new_coeffs[k] + coeff * (-xj) * scale
                new_coeffs[k + 1] = new_coeffs[k + 1] + coeff * scale
            
            basis_coeffs = new_coeffs
        
        # Add yi * li(x) to result
        for k, coeff in enumerate(basis_coeffs):
            coeffs[k] = coeffs[k] + yi * coeff
    
    return coeffs


def evaluate_polynomial(coeffs: List[FieldElement], x: FieldElement) -> FieldElement:
    """
    Evaluate polynomial at a point.
    
    Args:
        coeffs: Polynomial coefficients [a0, a1, ..., an]
        x: Point to evaluate at
        
    Returns:
        p(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    """
    result = FieldElement.zero()
    x_power = FieldElement.one()
    
    for coeff in coeffs:
        result = result + coeff * x_power
        x_power = x_power * x
    
    return result


# Compatibility functions for existing code
def hash_to_field(data: bytes) -> FieldElement:
    """Hash data to field element (for compatibility)."""
    return bytes_to_field(data)


def combine_field_elements(left: FieldElement, right: FieldElement) -> FieldElement:
    """Combine two field elements (for Merkle trees)."""
    # Simple combination: hash(left || right)
    combined = left.to_bytes() + right.to_bytes()
    return bytes_to_field(combined)