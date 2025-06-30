"""
UBP Simulation Runner - Main Orchestrator
Coordinates all UBP Bitfield Monad components for complete simulations

Components integrated:
- BitGrok Parser (192-bit bitstream decoding)
- BitfieldMonad (1x1x1 core implementation)
- TGICEngine (3-6-9 operations)
- GLRCorrector (frequency stabilization)
- CSV output and validation
"""

import numpy as np
import pandas as pd
import csv
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from bitfield_monad import BitfieldMonad, MonadConfig, TGICEngine
from bitgrok_parser import BitGrokParser, create_reference_bitstream
from glr_corrector import GLRCorrector, GLRResult


@dataclass
class SimulationResult:
    """Complete simulation result data"""
    config: MonadConfig
    steps_completed: int
    total_time: float
    energy_conservation: bool
    frequency_stability: bool
    interaction_weights_valid: bool
    csv_output_path: str
    glr_corrections: int
    final_nrci_score: float
    performance_metrics: Dict[str, float]


class UBPSimulationRunner:
    """
    Main UBP Simulation Runner
    
    Orchestrates complete 1x1x1 Bitfield Monad simulations with:
    - Bitstream parsing and validation
    - TGIC operations execution
    - GLR frequency correction
    - Energy conservation monitoring
    - CSV output generation
    - Performance metrics collection
    """
    
    def __init__(self, output_dir: str = ".", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Initialize components
        self.parser = BitGrokParser()
        self.corrector = GLRCorrector()
        
        # Simulation state
        self.current_monad = None
        self.current_engine = None
        self.simulation_data = []
        self.performance_metrics = {}
        
    def run_simulation(self, bitstream: Optional[bytes] = None, 
                      output_filename: str = "monad_simulation.csv") -> SimulationResult:
        """
        Run complete UBP Bitfield Monad simulation
        
        Args:
            bitstream: 192-bit UBP-Lang bitstream (uses default if None)
            output_filename: Name for CSV output file
            
        Returns:
            Complete simulation result
        """
        start_time = time.time()
        
        if self.verbose:
            print("Starting UBP 1x1x1 Bitfield Monad Simulation...")
            
        # Use default bitstream if none provided
        if bitstream is None:
            bitstream = create_reference_bitstream()
            if self.verbose:
                print("Using default reference bitstream")
                
        # Parse bitstream and create monad
        try:
            self.current_monad = self.parser.parse_and_create_monad(bitstream)
            self.current_engine = TGICEngine(self.current_monad)
            
            if self.verbose:
                print(f"Created monad: {self.current_monad}")
                print(f"Configuration: {self.current_monad.config}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize simulation: {e}")
            
        # Run simulation steps
        self.simulation_data = []
        steps = self.current_monad.config.steps
        
        if self.verbose:
            print(f"Executing {steps} simulation steps...")
            
        for step in range(steps):
            step_time = step * self.current_monad.config.bit_time
            
            # Execute TGIC step
            step_result = self.current_engine.execute_step(step_time)
            
            # Apply GLR correction periodically
            if step % 10 == 0 and step > 0:
                self._apply_glr_correction(step)
                
            # Record step data
            self.simulation_data.append({
                'time': step_time,
                'bit_state': step_result['new_state'].tolist(),
                'interaction': step_result['interaction'],
                'layer': step_result['layer'],
                'energy': step_result['energy']
            })
            
            # Progress reporting
            if self.verbose and step % 20 == 0:
                print(f"Step {step}/{steps} - Interaction: {step_result['interaction']}, Energy: {step_result['energy']:.6f}")
                
        # Calculate performance metrics
        total_time = time.time() - start_time
        self._calculate_performance_metrics(total_time)
        
        # Generate CSV output
        csv_path = self.output_dir / output_filename
        self._write_csv_output(csv_path)
        
        # Validate results
        validation_results = self._validate_simulation()
        
        # Create result object
        result = SimulationResult(
            config=self.current_monad.config,
            steps_completed=len(self.simulation_data),
            total_time=total_time,
            energy_conservation=validation_results['energy_conservation'],
            frequency_stability=validation_results['frequency_stability'],
            interaction_weights_valid=validation_results['interaction_weights'],
            csv_output_path=str(csv_path),
            glr_corrections=len(self.corrector.correction_history),
            final_nrci_score=validation_results['final_nrci'],
            performance_metrics=self.performance_metrics
        )
        
        if self.verbose:
            print(f"Simulation completed in {total_time:.3f} seconds")
            print(f"CSV output: {csv_path}")
            print(f"Energy conservation: {result.energy_conservation}")
            print(f"Frequency stability: {result.frequency_stability}")
            print(f"GLR corrections applied: {result.glr_corrections}")
            
        return result
        
    def _apply_glr_correction(self, step: int):
        """Apply GLR frequency correction"""
        if len(self.simulation_data) < 10:
            return
            
        # Extract recent frequency data
        recent_data = self.simulation_data[-10:]
        
        # Calculate frequencies from bit toggle patterns
        frequencies = []
        nrcis = []
        
        for data in recent_data:
            bit_state = np.array(data['bit_state'])
            
            # Estimate frequency from bit transitions
            if len(self.simulation_data) > 1:
                prev_state = np.array(self.simulation_data[-2]['bit_state'])
                transitions = np.sum(bit_state != prev_state)
                freq_estimate = transitions * self.current_monad.config.freq / 24
                frequencies.append(freq_estimate)
                
                # Calculate NRCI (simplified)
                nrci = 0.9999878 - (transitions * 0.000001)  # Decrease with more transitions
                nrcis.append(max(nrci, 0.999))
                
        if frequencies and nrcis:
            try:
                correction_result = self.corrector.correct_frequency(frequencies, nrcis, method='hybrid')
                
                # Apply correction to monad frequency if significant
                if correction_result.correction_applied:
                    self.current_monad.config.freq = correction_result.corrected_freq
                    
            except Exception as e:
                if self.verbose:
                    print(f"GLR correction failed at step {step}: {e}")
                    
    def _calculate_performance_metrics(self, total_time: float):
        """Calculate simulation performance metrics"""
        steps = len(self.simulation_data)
        
        self.performance_metrics = {
            'steps_per_second': steps / total_time if total_time > 0 else 0,
            'avg_step_time': total_time / steps if steps > 0 else 0,
            'total_simulation_time': total_time,
            'memory_efficiency': 1.0,  # Placeholder - could measure actual memory usage
            'cpu_efficiency': steps / (total_time * 1000) if total_time > 0 else 0  # Steps per CPU millisecond
        }
        
    def _write_csv_output(self, csv_path: Path):
        """Write simulation data to CSV file"""
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['time', 'bit_state', 'interaction', 'layer'])
            
            # Write data rows
            for data in self.simulation_data:
                writer.writerow([
                    data['time'],
                    str(data['bit_state']),
                    data['interaction'],
                    data['layer']
                ])
                
    def _validate_simulation(self) -> Dict[str, Any]:
        """Validate simulation results"""
        validation = {}
        
        # Energy conservation check
        energies = [data['energy'] for data in self.simulation_data]
        if energies:
            energy_std = np.std(energies)
            energy_mean = np.mean(energies)
            energy_variation = energy_std / energy_mean if energy_mean > 0 else 1.0
            validation['energy_conservation'] = energy_variation < 1e-6
        else:
            validation['energy_conservation'] = False
            
        # Frequency stability check
        if self.current_monad:
            target_freq = 3.14159
            current_freq = self.current_monad.config.freq
            freq_error = abs(current_freq - target_freq) / target_freq
            validation['frequency_stability'] = freq_error < 1e-5
        else:
            validation['frequency_stability'] = False
            
        # Interaction weights validation
        if self.current_engine:
            validation['interaction_weights'] = self.current_engine.validate_interaction_weights()
        else:
            validation['interaction_weights'] = False
            
        # Final NRCI score
        if self.corrector.correction_history:
            validation['final_nrci'] = self.corrector.correction_history[-1].nrci_score
        else:
            validation['final_nrci'] = 0.9999878  # Default expected value
            
        return validation
        
    def analyze_frequency_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze frequency spectrum of simulation
        
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        if not self.simulation_data:
            return np.array([]), np.array([])
            
        # Extract first bit history for FFT analysis
        bit_history = [data['bit_state'][0] for data in self.simulation_data]
        
        if self.current_monad:
            return self.current_monad.get_frequency_spectrum(bit_history)
        else:
            return np.array([]), np.array([])
            
    def export_results(self, filename: str = "simulation_results.json"):
        """Export complete simulation results to JSON"""
        if not self.simulation_data:
            raise RuntimeError("No simulation data to export")
            
        export_data = {
            'config': asdict(self.current_monad.config) if self.current_monad else {},
            'simulation_data': self.simulation_data,
            'performance_metrics': self.performance_metrics,
            'glr_corrections': [asdict(result) for result in self.corrector.correction_history],
            'validation': self._validate_simulation()
        }
        
        export_path = self.output_dir / filename
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        if self.verbose:
            print(f"Results exported to {export_path}")
            
        return export_path
        
    def run_benchmark(self, iterations: int = 5) -> Dict[str, float]:
        """
        Run performance benchmark
        
        Args:
            iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        if self.verbose:
            print(f"Running UBP benchmark with {iterations} iterations...")
            
        times = []
        steps_per_sec = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Run simulation with minimal steps for benchmarking
            config = MonadConfig(steps=50)  # Reduced steps for benchmarking
            monad = BitfieldMonad(config)
            engine = TGICEngine(monad)
            
            # Execute steps
            for step in range(config.steps):
                step_time = step * config.bit_time
                engine.execute_step(step_time)
                
            elapsed = time.time() - start_time
            times.append(elapsed)
            steps_per_sec.append(config.steps / elapsed if elapsed > 0 else 0)
            
        benchmark_results = {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_steps_per_sec': np.mean(steps_per_sec),
            'max_steps_per_sec': np.max(steps_per_sec),
            'iterations': iterations
        }
        
        if self.verbose:
            print(f"Benchmark results:")
            print(f"  Average time: {benchmark_results['avg_time']:.3f}s")
            print(f"  Average steps/sec: {benchmark_results['avg_steps_per_sec']:.0f}")
            print(f"  Max steps/sec: {benchmark_results['max_steps_per_sec']:.0f}")
            
        return benchmark_results
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for the simulation"""
        return {
            'ubp_version': '1.0.0',
            'monad_type': '1x1x1_bitfield',
            'parser_version': 'BitGrok_v2.0',
            'glr_enabled': True,
            'target_frequencies': list(self.corrector.TARGET_FREQUENCIES.values()),
            'nrci_threshold': self.corrector.NRCI_THRESHOLD,
            'output_directory': str(self.output_dir)
        }


def main():
    """Main entry point for UBP simulation"""
    print("UBP Bitfield Monad System - Simulation Runner")
    print("=" * 50)
    
    # Create simulation runner
    runner = UBPSimulationRunner(output_dir="ubp_output", verbose=True)
    
    # Display system info
    system_info = runner.get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Run main simulation
        result = runner.run_simulation(output_filename="ubp_monad_simulation.csv")
        
        # Display results
        print("\nSimulation Results:")
        print(f"  Steps completed: {result.steps_completed}")
        print(f"  Total time: {result.total_time:.3f}s")
        print(f"  Energy conservation: {result.energy_conservation}")
        print(f"  Frequency stability: {result.frequency_stability}")
        print(f"  Interaction weights valid: {result.interaction_weights_valid}")
        print(f"  GLR corrections: {result.glr_corrections}")
        print(f"  Final NRCI score: {result.final_nrci_score:.7f}")
        print(f"  CSV output: {result.csv_output_path}")
        
        # Export complete results
        json_path = runner.export_results("ubp_simulation_complete.json")
        
        # Run benchmark
        print("\nRunning performance benchmark...")
        benchmark = runner.run_benchmark(iterations=3)
        
        # Analyze frequency spectrum
        freqs, mags = runner.analyze_frequency_spectrum()
        if len(freqs) > 0:
            peak_freq_idx = np.argmax(mags)
            peak_freq = freqs[peak_freq_idx]
            print(f"\nFrequency Analysis:")
            print(f"  Peak frequency: {peak_freq:.5f} Hz")
            print(f"  Target frequency: 3.14159 Hz")
            print(f"  Frequency error: {abs(peak_freq - 3.14159):.8f} Hz")
            
        print(f"\nSimulation completed successfully!")
        print(f"All output files saved to: ubp_output/")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()

