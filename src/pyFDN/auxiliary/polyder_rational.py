from __future__ import annotations
import numpy as np

def polyder_rational(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of rational polynomial using quotient rule."""
    # Remove leading zeros
    b = np.trim_zeros(b, 'f')
    a = np.trim_zeros(a, 'f')
    
    if len(b) == 0:
        b = np.array([0.0])
    if len(a) == 0:
        a = np.array([1.0])
    
    # Compute derivatives of numerator and denominator
    db = np.polyder(b) if len(b) > 1 else np.array([0.0])
    da = np.polyder(a) if len(a) > 1 else np.array([0.0])
    
    # Apply quotient rule: (b/a)' = (b'*a - b*a') / a^2
    if len(db) == 0:
        db = np.array([0.0])
    if len(da) == 0:
        da = np.array([0.0])
        
    num1 = np.convolve(db, a)
    num2 = np.convolve(b, da)
    
    # Pad to same length
    max_len = max(len(num1), len(num2))
    if len(num1) < max_len:
        num1 = np.pad(num1, (max_len - len(num1), 0))
    if len(num2) < max_len:
        num2 = np.pad(num2, (max_len - len(num2), 0))
    
    q = num1 - num2
    p = np.convolve(a, a)
    
    # Remove leading zeros from result
    q = np.trim_zeros(q, 'f')
    p = np.trim_zeros(p, 'f')
    
    if len(q) == 0:
        q = np.array([0.0])
    if len(p) == 0:
        p = np.array([1.0])
    
    return q, p
