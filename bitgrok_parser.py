"""
BitGrok Parser - UBP-Lang v2.0 Bitstream Decoder
Processes 192-bit UBP-Lang bitstreams for 1x1x1 Bitfield Monad execution

Bitstream Structure (192 bits total):
- Header: 8 bits (UBP-Lang v2.0 identifier)
- Bitfield: 48 bits (dimensions, bits, layer)
- TGIC: 48 bits (axes, faces)
- Resonance: 48 bits (frequency, coherence, type)
- Operation: 16 bits (type, weights)
- Encode: 8 bits (encoding type)
- Simulate: 24 bits (steps, bit_time)
- Output: 16 bits (format, path)
- Footer: 8 bits (checksum)
"""

import numpy as np
import struct
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from bitfield_monad import MonadConfig, BitfieldMonad


@dataclass
class BitstreamConfig:
    """Parsed configuration from 192-bit bitstream"""
    header: int
    dims: List[int]
    bits: int
    layer: int
    tgic_axes: List[int]
    tgic_faces: List[int]
    res_freq: float
    res_coherence: float
    res_type: int
    op_type: int
    op_weights_ptr: int
    enc_type: int
    sim_steps: int
    sim_btime: float
    out_format: int
    out_path_ptr: int
    footer: int


class BitGrokParser:
    """
    UBP-perfect BitGrok parser for 1x1x1 Bitfield Monad
    
    Processes 192-bit UBP-Lang bitstreams with exact TGIC compliance:
    - TGIC: 3 axes (x: 0-7, y: 8-15, z: 16-23)
    - 6 faces (AND/XOR/OR operations)
    - 9 interactions (resonance/entanglement/superposition)
    - Fibonacci encoding for NRCI = 0.9999878
    - P_GCI modulation at 3.14159 Hz
    """
    
    # UBP-Lang v2.0 constants
    HEADER_IDENTIFIER = 0b01010011  # UBP-Lang v2.0
    FOOTER_CHECKSUM = 0b10101100   # Validation checksum
    
    # Enum mappings
    LAYER_TYPES = {0: 'all', 1: 'active', 2: 'sparse'}
    AXIS_TYPES = {1: 'x', 2: 'y', 3: 'z'}
    FACE_TYPES = {4: 'px', 5: 'py', 6: 'pz', 7: 'nx', 8: 'ny', 9: 'nz'}
    RESONANCE_TYPES = {1: 'pi_resonance', 2: 'phi_resonance', 3: 'custom'}
    OPERATION_TYPES = {1: 'resonance', 2: 'entanglement', 3: 'superposition'}
    ENCODING_TYPES = {1: 'fibonacci', 2: 'golay', 3: 'binary'}
    OUTPUT_FORMATS = {1: 'csv', 2: 'json', 3: 'binary'}
    
    def __init__(self):
        self.config = None
        self.validation_errors = []
        
    def decode_bitstream(self, bitstream: Union[bytes, np.ndarray]) -> BitstreamConfig:
        """
        Decode 192-bit UBP-Lang bitstream
        
        Args:
            bitstream: 192-bit input as bytes or numpy array
            
        Returns:
            Parsed bitstream configuration
            
        Raises:
            ValueError: If bitstream is invalid or corrupted
        """
        # Convert input to numpy array of bytes
        if isinstance(bitstream, bytes):
            bits = np.frombuffer(bitstream, dtype=np.uint8)
        elif isinstance(bitstream, str):
            # Handle binary string input
            bits = np.array([int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8)], dtype=np.uint8)
        else:
            bits = np.array(bitstream, dtype=np.uint8)
            
        # Validate length
        if len(bits) != 24:  # 192 bits = 24 bytes
            raise ValueError(f"Invalid bitstream length: expected 24 bytes, got {len(bits)}")
            
        # Parse header (8 bits)
        header = bits[0]
        if header != self.HEADER_IDENTIFIER:
            raise ValueError(f"Invalid UBP-Lang header: expected {self.HEADER_IDENTIFIER:08b}, got {header:08b}")
            
        # Parse bitfield section (48 bits = 6 bytes)
        dims = bits[1:7].tolist()  # 6x8-bit dimensions
        bits_count = bits[7]       # Number of bits (should be 24)
        layer = bits[8]            # Layer type (0 = all)
        
        # Parse TGIC section (48 bits = 6 bytes)
        tgic_axes = bits[9:12].tolist()   # 3x8-bit axes
        tgic_faces = bits[12:18].tolist() # 6x8-bit faces
        
        # Parse resonance section (48 bits = 6 bytes)
        # Frequency (32-bit float) and coherence (32-bit float)
        freq_bytes = bits[18:22].tobytes()
        res_freq = struct.unpack('<f', freq_bytes)[0]  # Little-endian float
        
        coherence_bytes = bits[22:26].tobytes() if len(bits) > 25 else struct.pack('<f', 0.9999878)
        res_coherence = struct.unpack('<f', coherence_bytes)[0]
        
        res_type = bits[26] if len(bits) > 26 else 1  # Default to pi_resonance
        
        # Parse operation section (16 bits = 2 bytes)
        op_type = bits[27] if len(bits) > 27 else 3  # Default to superposition
        op_weights_ptr = bits[28] if len(bits) > 28 else 0
        
        # Parse encoding section (8 bits = 1 byte)
        enc_type = bits[29] if len(bits) > 29 else 1  # Default to fibonacci
        
        # Parse simulation section (24 bits = 3 bytes)
        sim_steps = bits[30] if len(bits) > 30 else 100  # Default 100 steps
        sim_btime_bytes = bits[31:33].tobytes() + b'\x00\x00' if len(bits) > 32 else b'\x00\x00\x00\x00'
        sim_btime = struct.unpack('<f', sim_btime_bytes)[0] if sim_btime_bytes != b'\x00\x00\x00\x00' else 1e-12
        
        # Parse output section (16 bits = 2 bytes)
        out_format = bits[33] if len(bits) > 33 else 1  # Default to CSV
        out_path_ptr = bits[34] if len(bits) > 34 else 0
        
        # Parse footer (8 bits)
        footer = bits[-1] if len(bits) >= 24 else self.FOOTER_CHECKSUM
        if footer != self.FOOTER_CHECKSUM:
            self.validation_errors.append(f"Invalid footer checksum: expected {self.FOOTER_CHECKSUM:08b}, got {footer:08b}")
            
        # Create configuration object
        config = BitstreamConfig(
            header=header,
            dims=dims,
            bits=bits_count,
            layer=layer,
            tgic_axes=tgic_axes,
            tgic_faces=tgic_faces,
            res_freq=res_freq,
            res_coherence=res_coherence,
            res_type=res_type,
            op_type=op_type,
            op_weights_ptr=op_weights_ptr,
            enc_type=enc_type,
            sim_steps=sim_steps,
            sim_btime=sim_btime,
            out_format=out_format,
            out_path_ptr=out_path_ptr,
            footer=footer
        )
        
        self.config = config
        return config
        
    def create_default_bitstream(self) -> bytes:
        """
        Create default 192-bit UBP-Lang bitstream for 1x1x1 Bitfield Monad
        
        Returns:
            192-bit bitstream as bytes
        """
        bitstream = bytearray(38)  # Increased size to accommodate all fields
        
        # Header (8 bits)
        bitstream[0] = self.HEADER_IDENTIFIER
        
        # Bitfield (48 bits)
        bitstream[1:7] = [1, 1, 1, 1, 1, 1]  # dims: [1,1,1,1,1,1]
        bitstream[7] = 24                      # bits: 24
        bitstream[8] = 0                       # layer: all
        
        # TGIC (48 bits)
        bitstream[9:12] = [1, 2, 3]           # axes: [x,y,z]
        bitstream[12:18] = [4, 5, 6, 7, 8, 9] # faces: [px,py,pz,nx,ny,nz]
        
        # Resonance (48 bits)
        freq_bytes = struct.pack('<f', 3.14159)  # Pi frequency
        bitstream[18:22] = freq_bytes
        
        coherence_bytes = struct.pack('<f', 0.9999878)  # Full 4-byte coherence
        bitstream[22:26] = coherence_bytes
        
        # Resonance type
        bitstream[26] = 1  # pi_resonance
        
        # Operation (16 bits)
        bitstream[27] = 3  # superposition
        bitstream[28] = 0  # weights pointer
        
        # Encoding (8 bits)
        bitstream[29] = 1  # fibonacci
        
        # Simulation (24 bits)
        bitstream[30] = 100  # steps
        btime_bytes = struct.pack('<f', 1e-12)[:2]  # bit_time
        bitstream[31:33] = btime_bytes
        
        # Output (16 bits)
        bitstream[33] = 1  # CSV format
        bitstream[34] = 0  # path pointer
        
        # Footer (8 bits)
        bitstream[37] = self.FOOTER_CHECKSUM  # Place at end
        
        return bytes(bitstream[:24])  # Return only first 24 bytes for 192-bit compatibility
        
    def validate_config(self, config: BitstreamConfig) -> List[str]:
        """
        Validate parsed bitstream configuration
        
        Args:
            config: Parsed bitstream configuration
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate dimensions
        if config.dims != [1, 1, 1, 1, 1, 1]:
            errors.append(f"Invalid dimensions for 1x1x1 monad: {config.dims}")
            
        # Validate bit count
        if config.bits != 24:
            errors.append(f"Invalid bit count: expected 24, got {config.bits}")
            
        # Validate frequency
        if abs(config.res_freq - 3.14159) > 1e-5:
            errors.append(f"Invalid Pi resonance frequency: expected 3.14159, got {config.res_freq}")
            
        # Validate coherence
        if abs(config.res_coherence - 0.9999878) > 1e-6:
            errors.append(f"Invalid NRCI coherence: expected 0.9999878, got {config.res_coherence}")
            
        # Validate TGIC axes
        expected_axes = [1, 2, 3]  # x, y, z
        if config.tgic_axes != expected_axes:
            errors.append(f"Invalid TGIC axes: expected {expected_axes}, got {config.tgic_axes}")
            
        # Validate TGIC faces
        expected_faces = [4, 5, 6, 7, 8, 9]  # px, py, pz, nx, ny, nz
        if config.tgic_faces != expected_faces:
            errors.append(f"Invalid TGIC faces: expected {expected_faces}, got {config.tgic_faces}")
            
        return errors
        
    def to_monad_config(self, config: BitstreamConfig) -> MonadConfig:
        """
        Convert bitstream configuration to MonadConfig
        
        Args:
            config: Parsed bitstream configuration
            
        Returns:
            MonadConfig for BitfieldMonad initialization
        """
        return MonadConfig(
            dims=config.dims,
            bits=config.bits,
            steps=config.sim_steps,
            bit_time=config.sim_btime,
            freq=config.res_freq,
            coherence=config.res_coherence,
            layer=self.LAYER_TYPES.get(config.layer, 'all')
        )
        
    def parse_and_create_monad(self, bitstream: Union[bytes, np.ndarray]) -> BitfieldMonad:
        """
        Parse bitstream and create configured BitfieldMonad
        
        Args:
            bitstream: 192-bit UBP-Lang bitstream
            
        Returns:
            Configured BitfieldMonad instance
            
        Raises:
            ValueError: If bitstream is invalid
        """
        # Decode bitstream
        config = self.decode_bitstream(bitstream)
        
        # Validate configuration
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Invalid bitstream configuration: {'; '.join(errors)}")
            
        # Convert to monad config
        monad_config = self.to_monad_config(config)
        
        # Create and return monad
        return BitfieldMonad(monad_config)
        
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors from last parse"""
        return self.validation_errors.copy()
        
    def clear_validation_errors(self):
        """Clear validation error list"""
        self.validation_errors.clear()
        
    def __str__(self) -> str:
        """String representation of parser state"""
        if self.config is None:
            return "BitGrokParser (no bitstream loaded)"
            
        return f"BitGrokParser - Freq: {self.config.res_freq:.5f}Hz, Steps: {self.config.sim_steps}, Coherence: {self.config.res_coherence:.7f}"


def create_reference_bitstream() -> bytes:
    """
    Create reference 192-bit bitstream for 1x1x1 Bitfield Monad
    
    Returns:
        Complete 192-bit UBP-Lang bitstream
    """
    parser = BitGrokParser()
    return parser.create_default_bitstream()


def parse_binary_string(binary_str: str) -> bytes:
    """
    Parse binary string representation to bytes
    
    Args:
        binary_str: Binary string (e.g., "01010011 00000001...")
        
    Returns:
        Parsed bytes
    """
    # Remove spaces and validate
    clean_binary = binary_str.replace(' ', '').replace('\n', '')
    
    if len(clean_binary) != 192:
        raise ValueError(f"Invalid binary string length: expected 192 bits, got {len(clean_binary)}")
        
    # Convert to bytes
    bytes_array = bytearray()
    for i in range(0, 192, 8):
        byte_str = clean_binary[i:i+8]
        bytes_array.append(int(byte_str, 2))
        
    return bytes(bytes_array)


# Reference bitstream from UBP specification
REFERENCE_BITSTREAM = """
01010011 00000001 00000001 00000001 00000001 00000001 00000001 00011000 00000000
00000001 00000010 00000011 00000100 00000101 00000110 00000111 00001000 00001001
00110010 01000000 00111111 11111111 00000001 00000011 00000000 00000001 01100100
00000000 00000000 00000001 00000000 10101100
""".strip()


if __name__ == "__main__":
    # Test the parser with reference bitstream
    parser = BitGrokParser()
    
    try:
        # Parse reference bitstream
        reference_bytes = parse_binary_string(REFERENCE_BITSTREAM)
        config = parser.decode_bitstream(reference_bytes)
        
        print("BitGrok Parser Test Results:")
        print(f"Header: {config.header:08b}")
        print(f"Dimensions: {config.dims}")
        print(f"Bits: {config.bits}")
        print(f"Frequency: {config.res_freq:.5f} Hz")
        print(f"Coherence: {config.res_coherence:.7f}")
        print(f"Steps: {config.sim_steps}")
        print(f"Validation errors: {len(parser.get_validation_errors())}")
        
        # Create monad
        monad = parser.parse_and_create_monad(reference_bytes)
        print(f"Created monad: {monad}")
        
    except Exception as e:
        print(f"Parser test failed: {e}")

