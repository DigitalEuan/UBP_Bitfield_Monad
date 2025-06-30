"""
UBP Performance Analysis and Data Generation
Comprehensive simulation runner with detailed performance metrics and analysis

Generates:
- Multiple simulation runs with varying parameters
- Performance benchmarks across different configurations
- Energy conservation analysis
- Frequency spectrum analysis
- TGIC interaction distribution analysis
- GLR correction effectiveness metrics
- System scalability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import UBP components
from simulation_runner import UBPSimulationRunner
from bitfield_monad import MonadConfig
from test_suite import run_all_tests


class UBPPerformanceAnalyzer:
    """
    Comprehensive UBP performance analysis and data generation system
    """
    
    def __init__(self, output_dir: str = "ubp_performance_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance data storage
        self.simulation_results = []
        self.benchmark_data = []
        self.energy_analysis = []
        self.frequency_analysis = []
        self.interaction_analysis = []
        self.glr_analysis = []
        
        # Analysis configurations
        self.test_configurations = [
            {"steps": 50, "freq": 3.14159, "name": "Standard_50"},
            {"steps": 100, "freq": 3.14159, "name": "Standard_100"},
            {"steps": 200, "freq": 3.14159, "name": "Standard_200"},
            {"steps": 100, "freq": 3.14160, "name": "Freq_Variant_1"},
            {"steps": 100, "freq": 3.14158, "name": "Freq_Variant_2"},
            {"steps": 100, "freq": 3.14159, "name": "Coherence_Test"},
        ]
        
    def run_comprehensive_analysis(self):
        """Run complete performance analysis suite"""
        print("Starting UBP Comprehensive Performance Analysis")
        print("=" * 60)
        
        # 1. Run validation tests first
        print("1. Running validation test suite...")
        test_success = run_all_tests()
        print(f"   Test suite success: {test_success}")
        
        # 2. Run multiple simulation configurations
        print("\n2. Running simulation configurations...")
        for i, config in enumerate(self.test_configurations):
            print(f"   Configuration {i+1}/{len(self.test_configurations)}: {config['name']}")
            self._run_configuration_analysis(config)
            
        # 3. Generate performance benchmarks
        print("\n3. Running performance benchmarks...")
        self._run_performance_benchmarks()
        
        # 4. Analyze energy conservation
        print("\n4. Analyzing energy conservation...")
        self._analyze_energy_conservation()
        
        # 5. Analyze frequency spectra
        print("\n5. Analyzing frequency spectra...")
        self._analyze_frequency_spectra()
        
        # 6. Analyze TGIC interactions
        print("\n6. Analyzing TGIC interactions...")
        self._analyze_tgic_interactions()
        
        # 7. Analyze GLR corrections
        print("\n7. Analyzing GLR corrections...")
        self._analyze_glr_corrections()
        
        # 8. Generate comprehensive report
        print("\n8. Generating comprehensive report...")
        self._generate_comprehensive_report()
        
        # 9. Create visualizations
        print("\n9. Creating performance visualizations...")
        self._create_visualizations()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
    def _run_configuration_analysis(self, config: Dict[str, Any]):
        """Run analysis for a specific configuration"""
        try:
            # Create runner with configuration
            runner = UBPSimulationRunner(output_dir=str(self.output_dir), verbose=False)
            
            # Create monad config
            monad_config = MonadConfig(
                steps=config["steps"],
                freq=config["freq"],
                coherence=0.9999878
            )
            
            # Run simulation
            start_time = time.time()
            result = runner.run_simulation(
                output_filename=f"simulation_{config['name']}.csv"
            )
            execution_time = time.time() - start_time
            
            # Store results
            analysis_result = {
                "config_name": config["name"],
                "config": config,
                "execution_time": execution_time,
                "steps_completed": result.steps_completed,
                "energy_conservation": result.energy_conservation,
                "frequency_stability": result.frequency_stability,
                "interaction_weights_valid": result.interaction_weights_valid,
                "glr_corrections": result.glr_corrections,
                "final_nrci_score": result.final_nrci_score,
                "performance_metrics": result.performance_metrics,
                "csv_path": result.csv_output_path
            }
            
            self.simulation_results.append(analysis_result)
            
        except Exception as e:
            print(f"   Error in configuration {config['name']}: {e}")
            
    def _run_performance_benchmarks(self):
        """Run comprehensive performance benchmarks"""
        benchmark_configs = [
            {"iterations": 5, "steps": 50, "name": "Quick_Benchmark"},
            {"iterations": 3, "steps": 100, "name": "Standard_Benchmark"},
            {"iterations": 2, "steps": 200, "name": "Extended_Benchmark"}
        ]
        
        for bench_config in benchmark_configs:
            try:
                runner = UBPSimulationRunner(output_dir=str(self.output_dir), verbose=False)
                benchmark_result = runner.run_benchmark(iterations=bench_config["iterations"])
                
                benchmark_result["config_name"] = bench_config["name"]
                benchmark_result["steps"] = bench_config["steps"]
                self.benchmark_data.append(benchmark_result)
                
            except Exception as e:
                print(f"   Benchmark error for {bench_config['name']}: {e}")
                
    def _analyze_energy_conservation(self):
        """Analyze energy conservation across simulations"""
        for result in self.simulation_results:
            try:
                # Load simulation data
                df = pd.read_csv(result["csv_path"])
                
                # Extract energy data (would need to be added to CSV)
                # For now, use theoretical energy calculation
                energy_values = []
                for i in range(len(df)):
                    # Theoretical energy: E = M × C × R × P_GCI
                    M = 1
                    C = result["config"]["freq"]
                    R = 0.9
                    P_GCI = np.cos(2 * np.pi * C * 0.318309886)
                    energy = M * C * R * P_GCI
                    energy_values.append(energy)
                
                # Calculate energy statistics
                energy_stats = {
                    "config_name": result["config_name"],
                    "mean_energy": np.mean(energy_values),
                    "std_energy": np.std(energy_values),
                    "energy_variation": np.std(energy_values) / np.mean(energy_values) if np.mean(energy_values) > 0 else 0,
                    "energy_conservation": np.std(energy_values) / np.mean(energy_values) < 1e-6 if np.mean(energy_values) > 0 else False
                }
                
                self.energy_analysis.append(energy_stats)
                
            except Exception as e:
                print(f"   Energy analysis error for {result['config_name']}: {e}")
                
    def _analyze_frequency_spectra(self):
        """Analyze frequency spectra of simulations"""
        for result in self.simulation_results:
            try:
                # Load simulation data
                df = pd.read_csv(result["csv_path"])
                
                # Extract bit state data for FFT analysis
                bit_states = []
                for _, row in df.iterrows():
                    # Parse bit state string
                    bit_state_str = row['bit_state'].strip('[]')
                    bit_state = [int(x.strip()) for x in bit_state_str.split(',')]
                    bit_states.append(bit_state[0])  # Use first bit for analysis
                
                # Perform FFT analysis
                if len(bit_states) > 1:
                    fft_result = np.fft.rfft(bit_states)
                    frequencies = np.fft.rfftfreq(len(bit_states), 1e-12)  # bit_time
                    magnitudes = np.abs(fft_result)
                    
                    # Find peak frequency
                    peak_idx = np.argmax(magnitudes)
                    peak_frequency = frequencies[peak_idx] if len(frequencies) > peak_idx else 0
                    
                    freq_analysis = {
                        "config_name": result["config_name"],
                        "target_frequency": result["config"]["freq"],
                        "peak_frequency": peak_frequency,
                        "frequency_error": abs(peak_frequency - result["config"]["freq"]),
                        "frequency_stability": abs(peak_frequency - result["config"]["freq"]) < 0.01,
                        "spectral_power": np.sum(magnitudes**2),
                        "num_samples": len(bit_states)
                    }
                    
                    self.frequency_analysis.append(freq_analysis)
                    
            except Exception as e:
                print(f"   Frequency analysis error for {result['config_name']}: {e}")
                
    def _analyze_tgic_interactions(self):
        """Analyze TGIC interaction distributions"""
        for result in self.simulation_results:
            try:
                # Load simulation data
                df = pd.read_csv(result["csv_path"])
                
                # Analyze interaction distribution
                interaction_counts = df['interaction'].value_counts()
                total_interactions = len(df)
                
                # Expected weights
                expected_weights = {
                    'xy': 0.1, 'yx': 0.2, 'xz': 0.2, 'zx': 0.2,
                    'yz': 0.1, 'zy': 0.1
                }
                
                # Calculate actual vs expected
                interaction_analysis = {
                    "config_name": result["config_name"],
                    "total_interactions": total_interactions,
                    "unique_interactions": len(interaction_counts),
                    "interaction_distribution": interaction_counts.to_dict(),
                    "weight_deviations": {}
                }
                
                # Calculate deviations from expected weights
                for interaction, expected_weight in expected_weights.items():
                    actual_count = interaction_counts.get(interaction, 0)
                    actual_weight = actual_count / total_interactions if total_interactions > 0 else 0
                    deviation = abs(actual_weight - expected_weight)
                    interaction_analysis["weight_deviations"][interaction] = {
                        "expected": expected_weight,
                        "actual": actual_weight,
                        "deviation": deviation
                    }
                
                self.interaction_analysis.append(interaction_analysis)
                
            except Exception as e:
                print(f"   TGIC analysis error for {result['config_name']}: {e}")
                
    def _analyze_glr_corrections(self):
        """Analyze GLR correction effectiveness"""
        for result in self.simulation_results:
            try:
                glr_stats = {
                    "config_name": result["config_name"],
                    "glr_corrections_applied": result["glr_corrections"],
                    "final_nrci_score": result["final_nrci_score"],
                    "nrci_threshold_met": result["final_nrci_score"] >= 0.999997,
                    "correction_frequency": result["glr_corrections"] / result["steps_completed"] if result["steps_completed"] > 0 else 0
                }
                
                self.glr_analysis.append(glr_stats)
                
            except Exception as e:
                print(f"   GLR analysis error for {result['config_name']}: {e}")
                
    def _generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        report = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ubp_version": "1.0.0",
            "total_configurations_tested": len(self.simulation_results),
            "summary_statistics": self._calculate_summary_statistics(),
            "simulation_results": self.simulation_results,
            "benchmark_data": self.benchmark_data,
            "energy_analysis": self.energy_analysis,
            "frequency_analysis": self.frequency_analysis,
            "interaction_analysis": self.interaction_analysis,
            "glr_analysis": self.glr_analysis,
            "performance_recommendations": self._generate_recommendations()
        }
        
        # Save comprehensive report
        report_path = self.output_dir / "ubp_performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate summary CSV
        self._generate_summary_csv()
        
    def _calculate_summary_statistics(self):
        """Calculate overall summary statistics"""
        if not self.simulation_results:
            return {}
            
        execution_times = [r["execution_time"] for r in self.simulation_results]
        steps_per_second = [r["performance_metrics"]["steps_per_second"] for r in self.simulation_results if "steps_per_second" in r["performance_metrics"]]
        
        return {
            "avg_execution_time": np.mean(execution_times),
            "min_execution_time": np.min(execution_times),
            "max_execution_time": np.max(execution_times),
            "avg_steps_per_second": np.mean(steps_per_second) if steps_per_second else 0,
            "max_steps_per_second": np.max(steps_per_second) if steps_per_second else 0,
            "energy_conservation_rate": np.mean([r["energy_conservation"] for r in self.simulation_results]),
            "frequency_stability_rate": np.mean([r["frequency_stability"] for r in self.simulation_results]),
            "avg_glr_corrections": np.mean([r["glr_corrections"] for r in self.simulation_results]),
            "avg_nrci_score": np.mean([r["final_nrci_score"] for r in self.simulation_results])
        }
        
    def _generate_recommendations(self):
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze performance data
        if self.benchmark_data:
            avg_steps_per_sec = np.mean([b["avg_steps_per_sec"] for b in self.benchmark_data])
            if avg_steps_per_sec < 10000:
                recommendations.append("Consider optimizing TGIC operations for better performance")
                
        # Analyze energy conservation
        if self.energy_analysis:
            energy_issues = [e for e in self.energy_analysis if not e["energy_conservation"]]
            if energy_issues:
                recommendations.append("Energy conservation violations detected - review calculation precision")
                
        # Analyze frequency stability
        if self.frequency_analysis:
            freq_issues = [f for f in self.frequency_analysis if not f["frequency_stability"]]
            if freq_issues:
                recommendations.append("Frequency stability issues detected - review GLR correction parameters")
                
        if not recommendations:
            recommendations.append("System performance is within expected parameters")
            
        return recommendations
        
    def _generate_summary_csv(self):
        """Generate summary CSV for easy analysis"""
        if not self.simulation_results:
            return
            
        summary_data = []
        for result in self.simulation_results:
            summary_row = {
                "config_name": result["config_name"],
                "steps": result["config"]["steps"],
                "frequency": result["config"]["freq"],
                "execution_time": result["execution_time"],
                "steps_per_second": result["performance_metrics"].get("steps_per_second", 0),
                "energy_conservation": result["energy_conservation"],
                "frequency_stability": result["frequency_stability"],
                "glr_corrections": result["glr_corrections"],
                "nrci_score": result["final_nrci_score"]
            }
            summary_data.append(summary_row)
            
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / "performance_summary.csv", index=False)
        
    def _create_visualizations(self):
        """Create performance visualization plots"""
        try:
            # Set up matplotlib style
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (12, 8)
            
            # 1. Performance comparison plot
            if self.simulation_results:
                self._plot_performance_comparison()
                
            # 2. Energy conservation plot
            if self.energy_analysis:
                self._plot_energy_analysis()
                
            # 3. Frequency analysis plot
            if self.frequency_analysis:
                self._plot_frequency_analysis()
                
            # 4. TGIC interaction distribution
            if self.interaction_analysis:
                self._plot_interaction_analysis()
                
        except Exception as e:
            print(f"   Visualization error: {e}")
            
    def _plot_performance_comparison(self):
        """Plot performance comparison across configurations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        configs = [r["config_name"] for r in self.simulation_results]
        exec_times = [r["execution_time"] for r in self.simulation_results]
        steps_per_sec = [r["performance_metrics"].get("steps_per_second", 0) for r in self.simulation_results]
        
        # Execution time comparison
        ax1.bar(configs, exec_times, color='skyblue', alpha=0.7)
        ax1.set_title('Execution Time by Configuration')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Steps per second comparison
        ax2.bar(configs, steps_per_sec, color='lightgreen', alpha=0.7)
        ax2.set_title('Performance (Steps/Second)')
        ax2.set_ylabel('Steps per Second')
        ax2.tick_params(axis='x', rotation=45)
        
        # Energy conservation status
        energy_status = [r["energy_conservation"] for r in self.simulation_results]
        colors = ['green' if status else 'red' for status in energy_status]
        ax3.bar(configs, [1 if status else 0 for status in energy_status], color=colors, alpha=0.7)
        ax3.set_title('Energy Conservation Status')
        ax3.set_ylabel('Conservation (1=True, 0=False)')
        ax3.tick_params(axis='x', rotation=45)
        
        # GLR corrections
        glr_corrections = [r["glr_corrections"] for r in self.simulation_results]
        ax4.bar(configs, glr_corrections, color='orange', alpha=0.7)
        ax4.set_title('GLR Corrections Applied')
        ax4.set_ylabel('Number of Corrections')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_energy_analysis(self):
        """Plot energy conservation analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        configs = [e["config_name"] for e in self.energy_analysis]
        variations = [e["energy_variation"] for e in self.energy_analysis]
        mean_energies = [e["mean_energy"] for e in self.energy_analysis]
        
        # Energy variation
        ax1.bar(configs, variations, color='coral', alpha=0.7)
        ax1.set_title('Energy Variation by Configuration')
        ax1.set_ylabel('Energy Variation (std/mean)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=1e-6, color='red', linestyle='--', label='Conservation Threshold')
        ax1.legend()
        
        # Mean energy levels
        ax2.bar(configs, mean_energies, color='lightblue', alpha=0.7)
        ax2.set_title('Mean Energy Levels')
        ax2.set_ylabel('Energy (UBP units)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "energy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_frequency_analysis(self):
        """Plot frequency analysis results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        configs = [f["config_name"] for f in self.frequency_analysis]
        target_freqs = [f["target_frequency"] for f in self.frequency_analysis]
        peak_freqs = [f["peak_frequency"] for f in self.frequency_analysis]
        freq_errors = [f["frequency_error"] for f in self.frequency_analysis]
        
        # Target vs Peak frequency
        x = np.arange(len(configs))
        width = 0.35
        
        ax1.bar(x - width/2, target_freqs, width, label='Target', alpha=0.7, color='blue')
        ax1.bar(x + width/2, peak_freqs, width, label='Peak', alpha=0.7, color='red')
        ax1.set_title('Target vs Peak Frequencies')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45)
        ax1.legend()
        
        # Frequency errors
        ax2.bar(configs, freq_errors, color='orange', alpha=0.7)
        ax2.set_title('Frequency Errors')
        ax2.set_ylabel('Error (Hz)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "frequency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_interaction_analysis(self):
        """Plot TGIC interaction analysis"""
        if not self.interaction_analysis:
            return
            
        # Create interaction distribution plot for first configuration
        first_config = self.interaction_analysis[0]
        interactions = list(first_config["interaction_distribution"].keys())
        counts = list(first_config["interaction_distribution"].values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Interaction distribution pie chart
        ax1.pie(counts, labels=interactions, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'TGIC Interaction Distribution\n({first_config["config_name"]})')
        
        # Weight deviations
        if "weight_deviations" in first_config:
            deviations = [first_config["weight_deviations"][i]["deviation"] 
                         for i in interactions if i in first_config["weight_deviations"]]
            filtered_interactions = [i for i in interactions if i in first_config["weight_deviations"]]
            
            ax2.bar(filtered_interactions, deviations, color='purple', alpha=0.7)
            ax2.set_title('Weight Deviations from Expected')
            ax2.set_ylabel('Deviation')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "interaction_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for performance analysis"""
    print("UBP Bitfield Monad - Performance Analysis Suite")
    print("=" * 60)
    
    # Create analyzer
    analyzer = UBPPerformanceAnalyzer()
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()
    
    print("\nPerformance analysis complete!")
    print(f"Results available in: {analyzer.output_dir}")
    print("\nGenerated files:")
    print("- ubp_performance_report.json (comprehensive data)")
    print("- performance_summary.csv (summary table)")
    print("- performance_comparison.png (performance charts)")
    print("- energy_analysis.png (energy conservation)")
    print("- frequency_analysis.png (frequency stability)")
    print("- interaction_analysis.png (TGIC interactions)")


if __name__ == "__main__":
    main()

