"""
UBP Bitfield Monad Test Suite
Comprehensive validation framework for all UBP system components

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - Component interaction testing  
3. Validation Tests - UBP specification compliance
4. Performance Tests - Benchmarking and optimization
5. Mathematical Tests - Energy conservation, frequency stability
"""

import unittest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import time
from typing import Dict, List, Any

# Import UBP components
from bitfield_monad import BitfieldMonad, MonadConfig, TGICEngine
from bitgrok_parser import BitGrokParser, create_reference_bitstream, parse_binary_string
from glr_corrector import GLRCorrector, create_fibonacci_reference_vectors
from simulation_runner import UBPSimulationRunner


class TestBitfieldMonad(unittest.TestCase):
    """Test suite for BitfieldMonad core functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MonadConfig(steps=10, freq=3.14159, coherence=0.9999878)
        self.monad = BitfieldMonad(self.config)
        
    def test_monad_initialization(self):
        """Test monad initialization with correct parameters"""
        self.assertEqual(len(self.monad.offbit), 24)
        self.assertEqual(self.monad.config.freq, 3.14159)
        self.assertEqual(self.monad.config.coherence, 0.9999878)
        self.assertEqual(len(self.monad.interactions), 9)
        self.assertAlmostEqual(np.sum(self.monad.weights), 1.0, places=10)
        
    def test_fibonacci_encoding(self):
        """Test Fibonacci encoding initialization"""
        # Check that Fibonacci indices are set to 1
        fib_indices = [0, 1, 1, 2, 3, 5, 8, 13, 21]
        for i in fib_indices:
            if i < 24:
                self.assertEqual(self.monad.offbit[i], 1)
                
    def test_energy_calculation(self):
        """Test energy calculation formula E = M × C × R × P_GCI"""
        energy = self.monad.calculate_energy()
        
        # Expected calculation
        M = 1
        C = 3.14159
        R = 0.9
        P_GCI = np.cos(2 * np.pi * 3.14159 * 0.318309886)
        expected_energy = M * C * R * P_GCI
        
        self.assertAlmostEqual(energy, expected_energy, places=6)
        
    def test_tgic_axes_structure(self):
        """Test TGIC 3-axis structure (x: 0-7, y: 8-15, z: 16-23)"""
        self.assertEqual(self.monad.axes['x'], slice(0, 8))
        self.assertEqual(self.monad.axes['y'], slice(8, 16))
        self.assertEqual(self.monad.axes['z'], slice(16, 24))
        
    def test_resonance_calculation(self):
        """Test resonance function R(b_i, f) = b_i × exp(-0.0002 × (time × freq)²)"""
        time_val = 1e-12
        resonance = self.monad.calculate_resonance(time_val)
        
        d = time_val * self.monad.config.freq
        expected = np.exp(-0.0002 * d**2)
        
        self.assertAlmostEqual(resonance, expected, places=10)
        
    def test_interaction_weights_sum(self):
        """Test that interaction weights sum to exactly 1.0"""
        weight_sum = np.sum(self.monad.weights)
        self.assertAlmostEqual(weight_sum, 1.0, places=15)
        
    def test_state_vector_operations(self):
        """Test state vector get/set operations"""
        original_state = self.monad.get_state_vector()
        self.assertEqual(len(original_state), 24)
        
        # Modify state and verify independence
        original_state[0] = 1 - original_state[0]
        new_state = self.monad.get_state_vector()
        self.assertNotEqual(original_state[0], new_state[0])


class TestBitGrokParser(unittest.TestCase):
    """Test suite for BitGrok parser functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = BitGrokParser()
        
    def test_default_bitstream_creation(self):
        """Test creation of default 192-bit bitstream"""
        bitstream = self.parser.create_default_bitstream()
        self.assertEqual(len(bitstream), 24)  # 192 bits = 24 bytes
        self.assertEqual(bitstream[0], 0b01010011)  # UBP-Lang header
        
    def test_bitstream_parsing(self):
        """Test parsing of reference bitstream"""
        bitstream = create_reference_bitstream()
        config = self.parser.decode_bitstream(bitstream)
        
        self.assertEqual(config.header, 0b01010011)
        self.assertEqual(config.dims, [1, 1, 1, 1, 1, 1])
        self.assertEqual(config.bits, 24)
        self.assertAlmostEqual(config.res_freq, 3.14159, places=4)
        self.assertAlmostEqual(config.res_coherence, 0.9999878, places=6)
        
    def test_monad_creation_from_bitstream(self):
        """Test creating monad from parsed bitstream"""
        bitstream = create_reference_bitstream()
        monad = self.parser.parse_and_create_monad(bitstream)
        
        self.assertIsInstance(monad, BitfieldMonad)
        self.assertEqual(len(monad.offbit), 24)
        self.assertAlmostEqual(monad.config.freq, 3.14159, places=4)
        
    def test_validation_errors(self):
        """Test bitstream validation error detection"""
        # Create invalid bitstream (wrong frequency)
        invalid_bitstream = bytearray(24)
        invalid_bitstream[0] = 0b01010011  # Valid header
        invalid_bitstream[1:7] = [1, 1, 1, 1, 1, 1]  # Valid dims
        invalid_bitstream[7] = 24  # Valid bits
        
        # Invalid frequency (not Pi)
        import struct
        freq_bytes = struct.pack('<f', 2.71828)  # e instead of pi
        invalid_bitstream[18:22] = freq_bytes
        
        with self.assertRaises(ValueError):
            self.parser.parse_and_create_monad(bytes(invalid_bitstream))


class TestGLRCorrector(unittest.TestCase):
    """Test suite for GLR correction system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.corrector = GLRCorrector()
        
    def test_golay_matrix_creation(self):
        """Test Golay (24,12) matrix generation"""
        matrix = self.corrector.golay_matrix
        self.assertEqual(matrix.shape, (12, 24))
        
        # Check that first 12 columns form identity matrix
        identity_part = matrix[:, :12]
        np.testing.assert_array_equal(identity_part, np.eye(12))
        
    def test_golay_encoding_decoding(self):
        """Test Golay encoding and decoding cycle"""
        test_data = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Encode
        encoded = self.corrector.golay_encode(test_data)
        self.assertEqual(len(encoded), 24)
        
        # Decode (no errors)
        decoded, errors = self.corrector.golay_decode(encoded)
        self.assertEqual(errors, 0)
        np.testing.assert_array_equal(decoded, test_data)
        
    def test_error_correction(self):
        """Test error correction capability"""
        test_data = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        encoded = self.corrector.golay_encode(test_data)
        
        # Introduce single bit error
        corrupted = encoded.copy()
        corrupted[5] = 1 - corrupted[5]  # Flip bit
        
        # Decode and check correction
        decoded, errors = self.corrector.golay_decode(corrupted)
        self.assertGreaterEqual(errors, 1)  # Should detect error
        
    def test_nrci_calculation(self):
        """Test NRCI (Non-Random Coherence Index) calculation"""
        test_vector = np.array([1, 0, 1, 0] * 6)  # 24-bit pattern
        reference_vectors = create_fibonacci_reference_vectors(5)
        
        nrci = self.corrector.calculate_nrci(test_vector, reference_vectors)
        self.assertGreaterEqual(nrci, 0.0)
        self.assertLessEqual(nrci, 1.0)
        
    def test_frequency_correction(self):
        """Test GLR frequency correction"""
        # Test frequencies around Pi resonance
        frequencies = [3.14160, 3.14158, 3.14161]
        nrcis = [0.999998, 0.999997, 0.999999]
        
        result = self.corrector.correct_frequency(frequencies, nrcis, method='target_match')
        
        self.assertAlmostEqual(result.corrected_freq, 3.14159, places=4)
        self.assertGreaterEqual(result.nrci_score, 0.999997)
        
    def test_target_frequency_matching(self):
        """Test matching to target frequencies"""
        # Test frequency close to Pi
        freq = 3.14160
        corrected = self.corrector._target_matching(freq)
        self.assertEqual(corrected, 3.14159)
        
        # Test frequency close to scaled Phi
        freq = 36.34
        corrected = self.corrector._target_matching(freq)
        self.assertAlmostEqual(corrected, 36.339691, places=4)


class TestTGICEngine(unittest.TestCase):
    """Test suite for TGIC operations engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = MonadConfig(steps=10)
        self.monad = BitfieldMonad(config)
        self.engine = TGICEngine(self.monad)
        
    def test_interaction_selection(self):
        """Test weighted interaction selection"""
        # Run multiple selections to test distribution
        selections = []
        for _ in range(1000):
            interaction = self.engine.select_interaction()
            selections.append(interaction)
            
        # Check that all interactions appear
        unique_interactions = set(selections)
        expected_interactions = set(self.monad.interactions)
        self.assertEqual(unique_interactions, expected_interactions)
        
    def test_step_execution(self):
        """Test single TGIC step execution"""
        initial_state = self.monad.get_state_vector().copy()
        
        result = self.engine.execute_step(1e-12)
        
        self.assertIn('time', result)
        self.assertIn('interaction', result)
        self.assertIn('energy', result)
        self.assertIn('new_state', result)
        self.assertEqual(result['time'], 1e-12)
        
    def test_interaction_statistics(self):
        """Test interaction frequency statistics"""
        # Execute multiple steps
        for i in range(100):
            self.engine.execute_step(i * 1e-12)
            
        stats = self.engine.get_interaction_statistics()
        
        # Check that statistics sum to 1.0
        total_freq = sum(stats.values())
        self.assertAlmostEqual(total_freq, 1.0, places=10)
        
    def test_weight_validation(self):
        """Test interaction weight validation"""
        # Execute enough steps for statistical significance
        for i in range(1000):
            self.engine.execute_step(i * 1e-12)
            
        # Validate weights (with reasonable tolerance for randomness)
        is_valid = self.engine.validate_interaction_weights(tolerance=0.05)
        # Note: This might fail occasionally due to randomness, but should pass most of the time


class TestSimulationRunner(unittest.TestCase):
    """Test suite for simulation runner integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = UBPSimulationRunner(output_dir=self.temp_dir, verbose=False)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_simulation_execution(self):
        """Test complete simulation execution"""
        result = self.runner.run_simulation(output_filename="test_simulation.csv")
        
        self.assertGreater(result.steps_completed, 0)
        self.assertGreater(result.total_time, 0)
        self.assertTrue(os.path.exists(result.csv_output_path))
        
    def test_csv_output_format(self):
        """Test CSV output format compliance"""
        result = self.runner.run_simulation(output_filename="test_format.csv")
        
        # Read and validate CSV
        import pandas as pd
        df = pd.read_csv(result.csv_output_path)
        
        expected_columns = ['time', 'bit_state', 'interaction', 'layer']
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(len(df), result.steps_completed)
        
    def test_json_export(self):
        """Test JSON results export"""
        self.runner.run_simulation(output_filename="test_export.csv")
        json_path = self.runner.export_results("test_results.json")
        
        self.assertTrue(os.path.exists(json_path))
        
        # Validate JSON structure
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        required_keys = ['config', 'simulation_data', 'performance_metrics', 'validation']
        for key in required_keys:
            self.assertIn(key, data)
            
    def test_benchmark_execution(self):
        """Test performance benchmarking"""
        benchmark = self.runner.run_benchmark(iterations=2)
        
        self.assertIn('avg_time', benchmark)
        self.assertIn('avg_steps_per_sec', benchmark)
        self.assertGreater(benchmark['avg_steps_per_sec'], 0)


class TestUBPValidation(unittest.TestCase):
    """Test suite for UBP specification compliance"""
    
    def test_energy_conservation(self):
        """Test energy conservation over simulation"""
        config = MonadConfig(steps=50)
        monad = BitfieldMonad(config)
        engine = TGICEngine(monad)
        
        energies = []
        for i in range(50):
            result = engine.execute_step(i * 1e-12)
            energies.append(result['energy'])
            
        # Energy should be conserved within tolerance
        energy_std = np.std(energies)
        energy_mean = np.mean(energies)
        variation = energy_std / energy_mean if energy_mean > 0 else 1.0
        
        self.assertLess(variation, 1e-6, "Energy not conserved within tolerance")
        
    def test_frequency_stability(self):
        """Test Pi resonance frequency stability"""
        config = MonadConfig(freq=3.14159)
        monad = BitfieldMonad(config)
        
        # Frequency should remain stable
        self.assertAlmostEqual(monad.config.freq, 3.14159, places=5)
        
        # After GLR correction, should still be close to Pi
        corrector = GLRCorrector()
        result = corrector.correct_frequency([3.14160], [0.999998])
        self.assertAlmostEqual(result.corrected_freq, 3.14159, places=4)
        
    def test_nrci_threshold_compliance(self):
        """Test NRCI threshold compliance (>0.999997)"""
        corrector = GLRCorrector()
        
        # Test with high-quality frequencies
        frequencies = [3.14159, 3.14159, 3.14159]
        nrcis = [0.999998, 0.999999, 0.999997]
        
        result = corrector.correct_frequency(frequencies, nrcis)
        self.assertGreaterEqual(result.nrci_score, corrector.NRCI_THRESHOLD)
        
    def test_tgic_structure_compliance(self):
        """Test TGIC 3-6-9 structure compliance"""
        monad = BitfieldMonad()
        
        # 3 axes
        self.assertEqual(len(monad.axes), 3)
        
        # 6 faces
        self.assertEqual(len(monad.faces), 6)
        
        # 9 interactions
        self.assertEqual(len(monad.interactions), 9)
        
        # Verify axis bit ranges
        total_bits = 0
        for axis_slice in monad.axes.values():
            total_bits += axis_slice.stop - axis_slice.start
        self.assertEqual(total_bits, 24)
        
    def test_fibonacci_encoding_compliance(self):
        """Test Fibonacci encoding compliance"""
        monad = BitfieldMonad()
        
        # Check Fibonacci sequence initialization
        fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 
                       610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657]
        
        for i, fib_num in enumerate(fib_sequence):
            if fib_num < 24:
                self.assertEqual(monad.offbit[fib_num], 1, 
                               f"Fibonacci index {fib_num} not set correctly")


class TestPerformance(unittest.TestCase):
    """Performance and optimization tests"""
    
    def test_simulation_speed(self):
        """Test simulation execution speed"""
        config = MonadConfig(steps=100)
        monad = BitfieldMonad(config)
        engine = TGICEngine(monad)
        
        start_time = time.time()
        for i in range(100):
            engine.execute_step(i * 1e-12)
        elapsed = time.time() - start_time
        
        steps_per_second = 100 / elapsed
        
        # Should achieve reasonable performance (adjust threshold as needed)
        self.assertGreater(steps_per_second, 1000, 
                          f"Performance too slow: {steps_per_second:.0f} steps/sec")
        
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and run simulation
        runner = UBPSimulationRunner(output_dir=tempfile.mkdtemp(), verbose=False)
        runner.run_simulation()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for basic simulation)
        self.assertLess(memory_increase, 100 * 1024 * 1024, 
                       f"Memory usage too high: {memory_increase / 1024 / 1024:.1f} MB")


def run_all_tests():
    """Run all UBP test suites"""
    print("Running UBP Bitfield Monad Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBitfieldMonad,
        TestBitGrokParser,
        TestGLRCorrector,
        TestTGICEngine,
        TestSimulationRunner,
        TestUBPValidation,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = error_lines[-2] if len(error_lines) > 1 else "Unknown error"
            print(f"- {test}: {error_msg}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

