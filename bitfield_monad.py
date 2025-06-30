"""
UBP Bitfield Monad System - Core Implementation
Universal Binary Principle (UBP) 1x1x1 Bitfield Monad

This module implements the minimal computational unit of UBP:
- Single 24-bit OffBit
- TGIC operations (3 axes, 6 faces, 9 interactions)
- Pi Resonance at 3.14159 Hz
- Golay-Leech Resonance correction
"""

import numpy as np
import struct
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import csv
import json


@dataclass
class MonadConfig:
    """Configuration for 1x1x1 Bitfield Monad"""
    dims: List[int] = None
    bits: int = 24
    steps: int = 100
    bit_time: float = 1e-12
    freq: float = 3.14159
    coherence: float = 0.9999878
    layer: str = "all"
    
    def __post_init__(self):
        if self.dims is None:
            self.dims = [1, 1, 1, 1, 1, 1]


class BitfieldMonad:
    """
    1x1x1 Bitfield Monad - The minimal computational unit of UBP
    
    Implements:
    - 24-bit OffBit with TGIC structure
    - Pi Resonance at 3.14159 Hz
    - Energy equation: E = M × C × R × P_GCI
    - Fibonacci encoding for initialization
    """
    
    def __init__(self, config: MonadConfig = None):
        self.config = config or MonadConfig()
        
        # Initialize 24-bit OffBit
        self.offbit = np.zeros(24, dtype=int)
        
        # TGIC structure: 3 axes, 6 faces, 9 interactions
        self.axes = {
            'x': slice(0, 8),   # bits 0-7
            'y': slice(8, 16),  # bits 8-15
            'z': slice(16, 24)  # bits 16-23
        }
        
        self.faces = {
            'px': 'AND', 'nx': 'AND',  # ±x faces: synchronous
            'py': 'XOR', 'ny': 'XOR',  # ±y faces: asynchronous
            'pz': 'OR',  'nz': 'OR'    # ±z faces: latent activation
        }
        
        self.interactions = ['xy', 'yx', 'xz', 'zx', 'yz', 'zy', 'xy', 'xz', 'yz']
        # Ensure weights sum to exactly 1.0
        raw_weights = np.array([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05])
        self.weights = raw_weights / np.sum(raw_weights)  # Normalize to sum to 1.0
        
        # Fibonacci sequence for encoding (first 24 numbers)
        self.fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 
                         610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657]
        
        # Initialize with Fibonacci encoding
        self.apply_fibonacci_encoding()
        
        # Energy tracking
        self.energy_history = []
        
    def apply_fibonacci_encoding(self):
        """Initialize OffBit with Fibonacci pattern for NRCI = 0.9999878"""
        self.offbit = np.array([1 if i in self.fibonacci else 0 for i in range(24)])
        
    def calculate_energy(self) -> float:
        """
        Calculate energy using UBP formula: E = M × C × R × P_GCI
        
        Returns:
            Energy value in UBP units
        """
        M = 1  # Single OffBit
        C = self.config.freq  # 3.14159 Hz (Pi Resonance)
        R = 0.9  # Resonance strength
        P_GCI = np.cos(2 * np.pi * self.config.freq * 0.318309886)  # Global Coherence Invariant
        
        energy = M * C * R * P_GCI
        return energy
        
    def calculate_resonance(self, time: float) -> float:
        """
        Calculate resonance function: R(b_i, f) = b_i × exp(-0.0002 × (time × freq)²)
        
        Args:
            time: Current simulation time
            
        Returns:
            Resonance factor
        """
        d = time * self.config.freq
        return np.exp(-0.0002 * d**2)
        
    def apply_tgic_operation(self, interaction: str, time: float):
        """
        Apply TGIC (Triad Graph Interaction Constraint) operations
        
        Args:
            interaction: Type of interaction (xy, yx, xz, zx, yz, zy, mixed)
            time: Current simulation time
        """
        # Calculate P_GCI (Global Coherence Invariant)
        p_gci = np.cos(2 * np.pi * self.config.freq * 0.318309886)
        
        # Apply interaction-specific operations
        if interaction in ['xy', 'yx']:  # Resonance
            resonance_factor = self.calculate_resonance(time)
            threshold = p_gci * self.config.coherence * resonance_factor
            
            # Apply to both X and Y axes
            x_val = 1 if np.random.rand() < threshold else 0
            y_val = x_val  # Synchronized for resonance
            
            self.offbit[self.axes['x']] = x_val
            self.offbit[self.axes['y']] = y_val
            
        elif interaction in ['xz', 'zx']:  # Entanglement
            # E(b_i, b_j) = b_i × b_j × 0.9999878
            x_bits = self.offbit[self.axes['x']]
            z_bits = self.offbit[self.axes['z']]
            
            # Apply entanglement coefficient
            entangled = (x_bits * z_bits * self.config.coherence).astype(int)
            self.offbit[self.axes['z']] = entangled
            
        else:  # Superposition (yz, zy, mixed)
            # Weighted superposition with probability distribution
            prob = np.sum(self.weights) / len(self.weights)
            superposition_val = 1 if np.random.rand() < prob else 0
            self.offbit[self.axes['y']] = superposition_val
            
        # Apply face operations
        self.apply_face_operations()
        
    def apply_face_operations(self):
        """Apply 6-face operations: AND (±x), XOR (±y), OR (±z)"""
        x_bits = self.offbit[self.axes['x']]
        y_bits = self.offbit[self.axes['y']]
        z_bits = self.offbit[self.axes['z']]
        
        # ±x faces: AND operations (synchronous)
        self.offbit[self.axes['x']] = np.minimum(x_bits, y_bits)
        
        # ±y faces: XOR operations (asynchronous)
        self.offbit[self.axes['y']] = np.abs(y_bits - z_bits)
        
        # ±z faces: OR operations (latent activation)
        self.offbit[self.axes['z']] = np.maximum(z_bits, x_bits)
        
    def get_state_vector(self) -> np.ndarray:
        """Get current OffBit state as numpy array"""
        return self.offbit.copy()
        
    def get_state_string(self) -> str:
        """Get current OffBit state as string representation"""
        return str(self.offbit.tolist())
        
    def validate_energy_conservation(self) -> bool:
        """
        Validate energy conservation over simulation
        
        Returns:
            True if energy is conserved within tolerance
        """
        if len(self.energy_history) < 2:
            return True
            
        energies = np.array(self.energy_history)
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # Energy should be conserved within 0.0001%
        tolerance = mean_energy * 1e-6
        return std_energy < tolerance
        
    def get_frequency_spectrum(self, bit_history: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze frequency spectrum of bit toggles using FFT
        
        Args:
            bit_history: History of bit states over time
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        if len(bit_history) < 2:
            return np.array([]), np.array([])
            
        # Apply FFT to analyze frequency content
        fft_result = np.fft.rfft(bit_history)
        frequencies = np.fft.rfftfreq(len(bit_history), self.config.bit_time)
        magnitudes = np.abs(fft_result)
        
        return frequencies, magnitudes
        
    def __str__(self) -> str:
        """String representation of the monad"""
        energy = self.calculate_energy()
        return f"UBP 1x1x1 Bitfield Monad - Energy: {energy:.6f}, State: {self.get_state_string()}"
        
    def __repr__(self) -> str:
        return self.__str__()


class TGICEngine:
    """
    TGIC (Triad Graph Interaction Constraint) Operations Engine
    
    Manages the 3-6-9 structure:
    - 3 axes (x, y, z)
    - 6 faces (±x, ±y, ±z)
    - 9 interactions (resonance, entanglement, superposition)
    """
    
    def __init__(self, monad: BitfieldMonad):
        self.monad = monad
        self.interaction_history = []
        
    def select_interaction(self) -> str:
        """Select interaction based on weighted probabilities"""
        return np.random.choice(self.monad.interactions, p=self.monad.weights)
        
    def execute_step(self, time: float) -> Dict[str, Any]:
        """
        Execute one TGIC step
        
        Args:
            time: Current simulation time
            
        Returns:
            Step result dictionary
        """
        # Select interaction
        interaction = self.select_interaction()
        
        # Store previous state
        prev_state = self.monad.get_state_vector()
        
        # Apply TGIC operation
        self.monad.apply_tgic_operation(interaction, time)
        
        # Calculate energy
        energy = self.monad.calculate_energy()
        self.monad.energy_history.append(energy)
        
        # Record interaction
        self.interaction_history.append(interaction)
        
        # Create step result
        result = {
            'time': time,
            'interaction': interaction,
            'prev_state': prev_state,
            'new_state': self.monad.get_state_vector(),
            'energy': energy,
            'layer': self.monad.config.layer
        }
        
        return result
        
    def get_interaction_statistics(self) -> Dict[str, float]:
        """Get statistics on interaction frequency"""
        if not self.interaction_history:
            return {}
            
        unique, counts = np.unique(self.interaction_history, return_counts=True)
        total = len(self.interaction_history)
        
        stats = {}
        for interaction, count in zip(unique, counts):
            stats[interaction] = count / total
            
        return stats
        
    def validate_interaction_weights(self, tolerance: float = 0.01) -> bool:
        """
        Validate that interaction frequencies match expected weights
        
        Args:
            tolerance: Acceptable deviation from expected weights
            
        Returns:
            True if weights are within tolerance
        """
        stats = self.get_interaction_statistics()
        expected_weights = dict(zip(self.monad.interactions, self.monad.weights))
        
        for interaction, expected in expected_weights.items():
            actual = stats.get(interaction, 0.0)
            if abs(actual - expected) > tolerance:
                return False
                
        return True

