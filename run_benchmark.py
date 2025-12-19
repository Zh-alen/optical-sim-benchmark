import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import pandas as pd
from scipy.special import erfc
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from traditional_platform.level1_b2b import benchmark_numpy
    from jax_platform.level1_b2b import benchmark_jax
    print("✅ Modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all files are correctly created")
    exit(1)

def save_results_to_files(results, config, output_dir):
    """Save benchmark results to files in testlevel1 folder"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save raw results as JSON
    json_filename = f"benchmark_results_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    results_to_save = {
        "metadata": {
            "timestamp": timestamp,
            "config": config,
            "platform": "Windows",
            "python_version": sys.version
        },
        "numpy_results": results[0],
        "jax_results": results[1],
        "speedups": results[2],
        "theory_ber": results[3]
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    
    print(f"📁 JSON results saved to: {json_path}")
    
    # 2. Save as CSV for easy analysis
    csv_filename = f"benchmark_results_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Prepare data for CSV
    data_rows = []
    for i, (np_res, jax_res) in enumerate(zip(results[0], results[1])):
        data_rows.append({
            "test_id": f"test_{i+1}",
            "num_symbols": np_res['num_symbols'],
            "numpy_avg_time_ms": np_res['avg_time'] * 1000,
            "numpy_std_time_ms": np_res['std_time'] * 1000,
            "numpy_avg_ber": np_res['avg_ber'],
            "jax_avg_time_ms": jax_res['avg_time'] * 1000,
            "jax_std_time_ms": jax_res['std_time'] * 1000,
            "jax_avg_ber": jax_res['avg_ber'],
            "speedup": results[2][i],
            "theory_ber": results[3],
            "relative_error_numpy": abs(np_res['avg_ber'] - results[3]) / results[3] * 100,
            "relative_error_jax": abs(jax_res['avg_ber'] - results[3]) / results[3] * 100
        })
    
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"📊 CSV results saved to: {csv_path}")
    
    # 3. Save summary statistics
    summary_filename = f"benchmark_summary_{timestamp}.txt"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("OPTICAL SIMULATION BENCHMARK - TEST LEVEL 1 SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python Version: {sys.version}\n\n")
        
        f.write("TEST CONFIGURATION:\n")
        f.write(f"  Modulation: {config['modulation']}\n")
        f.write(f"  Scenario: {config['scenario']}\n")
        f.write(f"  SNR: {config['snr_db']} dB\n")
        f.write(f"  Symbol counts: {config['num_symbols_list']}\n")
        f.write(f"  Runs per test: {config['num_runs']}\n\n")
        
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 60 + "\n")
        
        avg_speedup = np.mean(results[2])
        max_speedup = np.max(results[2])
        min_speedup = np.min(results[2])
        
        f.write(f"Average Speedup (NumPy/JAX): {avg_speedup:.2f}x\n")
        f.write(f"Maximum Speedup: {max_speedup:.2f}x\n")
        f.write(f"Minimum Speedup: {min_speedup:.2f}x\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 60 + "\n")
        for i, (np_res, jax_res) in enumerate(zip(results[0], results[1])):
            f.write(f"\nTest {i+1} - {np_res['num_symbols']} symbols:\n")
            f.write(f"  NumPy: {np_res['avg_time']*1000:.2f} ± {np_res['std_time']*1000:.2f} ms, BER: {np_res['avg_ber']:.2e}\n")
            f.write(f"  JAX:   {jax_res['avg_time']*1000:.2f} ± {jax_res['std_time']*1000:.2f} ms, BER: {jax_res['avg_ber']:.2e}\n")
            f.write(f"  Speedup: {results[2][i]:.2f}x\n")
    
    print(f"📝 Summary saved to: {summary_path}")
    
    return json_path, csv_path, summary_path

def run_real_benchmark():
    """Run real benchmark test and save results"""
    print("=" * 60)
    print("Optical Simulation Platform - QPSK B2B System Performance Test")
    print("=" * 60)
    
    # Test parameters
    config = {
        "modulation": "QPSK",
        "scenario": "Back-to-Back (B2B)",
        "snr_db": 10,
        "num_symbols_list": [1000, 5000, 10000, 50000],
        "num_runs": 10
    }
    
    print(f"\nTest configuration:")
    print(f"  Modulation: {config['modulation']}")
    print(f"  Scenario: {config['scenario']}")
    print(f"  SNR: {config['snr_db']} dB")
    print(f"  Symbol counts: {config['num_symbols_list']}")
    print(f"  Runs per test: {config['num_runs']}")
    
    # Run traditional platform test
    print("\n" + "-" * 40)
    print("Running traditional platform (NumPy) test...")
    print("-" * 40)
    start_time = time.time()
    numpy_results = benchmark_numpy(config['num_symbols_list'], config['num_runs'])
    numpy_total_time = time.time() - start_time
    
    # Run JAX platform test
    print("\n" + "-" * 40)
    print("Running JAX platform test...")
    print("-" * 40)
    start_time = time.time()
    jax_results = benchmark_jax(config['num_symbols_list'], config['num_runs'])
    jax_total_time = time.time() - start_time
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Performance Comparison Analysis")
    print("=" * 60)
    
    print("\nPerformance Comparison Table:")
    print("-" * 110)
    print(f"{'Symbols':<8} {'NumPy Time(ms)':<15} {'JAX Time(ms)':<15} {'Speedup':<10} {'NumPy BER':<12} {'JAX BER':<12} {'Theory BER':<12}")
    print("-" * 110)
    
    speedups = []
    theory_ber = 0.5 * erfc(np.sqrt(10 ** (config['snr_db'] / 10)))
    
    for np_res, jax_res in zip(numpy_results, jax_results):
        speedup = np_res['avg_time'] / jax_res['avg_time']
        speedups.append(speedup)
        
        print(f"{np_res['num_symbols']:<8} "
              f"{np_res['avg_time']*1000:<15.2f} "
              f"{jax_res['avg_time']*1000:<15.2f} "
              f"{speedup:<10.2f}x "
              f"{np_res['avg_ber']:<12.2e} "
              f"{jax_res['avg_ber']:<12.2e} "
              f"{theory_ber:<12.2e}")
    
    print("-" * 110)
    print(f"Total runtime: NumPy={numpy_total_time:.2f}s, JAX={jax_total_time:.2f}s")
    
    # Save results to files
    print("\n" + "=" * 60)
    print("Saving Results to Files")
    print("=" * 60)
    
    output_dir = os.path.join("results", "testlevel1")
    json_path, csv_path, summary_path = save_results_to_files(
        (numpy_results, jax_results, speedups, theory_ber),
        config,
        output_dir
    )
    
    # Plot results
    plot_comparison_results(numpy_results, jax_results, speedups, theory_ber, output_dir)
    
    return numpy_results, jax_results, speedups, json_path, csv_path, summary_path

def plot_comparison_results(numpy_results, jax_results, speedups, theory_ber, output_dir):
    """Plot comparison results and save to testlevel1 folder"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Runtime comparison
    ax1 = axes[0, 0]
    symbols = [r['num_symbols'] for r in numpy_results]
    np_times = [r['avg_time']*1000 for r in numpy_results]
    jax_times = [r['avg_time']*1000 for r in jax_results]
    
    ax1.plot(symbols, np_times, 'o-', label='NumPy', linewidth=2, markersize=8)
    ax1.plot(symbols, jax_times, 's-', label='JAX', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Symbols', fontsize=12)
    ax1.set_ylabel('Runtime (ms)', fontsize=12)
    ax1.set_title('Runtime Comparison (QPSK B2B)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(speedups)), speedups, color='skyblue', edgecolor='black')
    ax2.set_xticks(range(len(symbols)))
    ax2.set_xticklabels([f'{s}' for s in symbols])
    ax2.set_xlabel('Number of Symbols', fontsize=12)
    ax2.set_ylabel('Speedup (NumPy Time / JAX Time)', fontsize=12)
    ax2.set_title('Speedup Analysis', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, speedup in enumerate(speedups):
        ax2.text(i, speedup + 0.1, f'{speedup:.1f}x', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: BER comparison
    ax3 = axes[1, 0]
    np_bers = [r['avg_ber'] for r in numpy_results]
    jax_bers = [r['avg_ber'] for r in jax_results]
    
    width = 0.35
    x = np.arange(len(symbols))
    ax3.bar(x - width/2, np_bers, width, label='NumPy Simulation', alpha=0.8)
    ax3.bar(x + width/2, jax_bers, width, label='JAX Simulation', alpha=0.8)
    
    # Add theoretical BER line
    ax3.axhline(y=theory_ber, color='red', linestyle='--', linewidth=2, 
                label=f'Theory BER={theory_ber:.2e}')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{s}' for s in symbols])
    ax3.set_xlabel('Number of Symbols', fontsize=12)
    ax3.set_ylabel('Average BER', fontsize=12)
    ax3.set_title('BER Consistency Verification', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Relative error
    ax4 = axes[1, 1]
    np_errors = [abs(ber - theory_ber) / theory_ber * 100 for ber in np_bers]
    jax_errors = [abs(ber - theory_ber) / theory_ber * 100 for ber in jax_bers]
    
    ax4.plot(symbols, np_errors, 'o-', label='NumPy Relative Error', linewidth=2)
    ax4.plot(symbols, jax_errors, 's-', label='JAX Relative Error', linewidth=2)
    ax4.set_xscale('log')
    ax4.set_xlabel('Number of Symbols', fontsize=12)
    ax4.set_ylabel('Relative Error (%)', fontsize=12)
    ax4.set_title('BER Relative Error Analysis', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"benchmark_plots_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Plots saved to: {plot_path}")
    
    # Also save individual plots
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        fig_single = plt.figure(figsize=(8, 6))
        ax_single = fig_single.add_subplot(111)
        
        # Recreate each plot individually
        if i == 0:  # Runtime comparison
            ax_single.plot(symbols, np_times, 'o-', label='NumPy', linewidth=2, markersize=8)
            ax_single.plot(symbols, jax_times, 's-', label='JAX', linewidth=2, markersize=8)
            ax_single.set_xscale('log')
            ax_single.set_yscale('log')
            ax_single.set_xlabel('Number of Symbols')
            ax_single.set_ylabel('Runtime (ms)')
            ax_single.set_title('Runtime Comparison (QPSK B2B)')
            ax_single.legend()
            ax_single.grid(True, alpha=0.3)
            filename = "plot1_runtime_comparison.png"
            
        elif i == 1:  # Speedup
            ax_single.bar(range(len(speedups)), speedups, color='skyblue', edgecolor='black')
            ax_single.set_xticks(range(len(symbols)))
            ax_single.set_xticklabels([f'{s}' for s in symbols])
            ax_single.set_xlabel('Number of Symbols')
            ax_single.set_ylabel('Speedup (NumPy Time / JAX Time)')
            ax_single.set_title('Speedup Analysis')
            ax_single.grid(True, alpha=0.3, axis='y')
            for j, speedup in enumerate(speedups):
                ax_single.text(j, speedup + 0.1, f'{speedup:.1f}x', 
                              ha='center', va='bottom', fontsize=10)
            filename = "plot2_speedup_analysis.png"
            
        elif i == 2:  # BER comparison
            ax_single.bar(x - width/2, np_bers, width, label='NumPy Simulation', alpha=0.8)
            ax_single.bar(x + width/2, jax_bers, width, label='JAX Simulation', alpha=0.8)
            ax_single.axhline(y=theory_ber, color='red', linestyle='--', linewidth=2, 
                            label=f'Theory BER={theory_ber:.2e}')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels([f'{s}' for s in symbols])
            ax_single.set_xlabel('Number of Symbols')
            ax_single.set_ylabel('Average BER')
            ax_single.set_title('BER Consistency Verification')
            ax_single.set_yscale('log')
            ax_single.legend()
            ax_single.grid(True, alpha=0.3, axis='y')
            filename = "plot3_ber_comparison.png"
            
        else:  # Relative error
            ax_single.plot(symbols, np_errors, 'o-', label='NumPy Relative Error', linewidth=2)
            ax_single.plot(symbols, jax_errors, 's-', label='JAX Relative Error', linewidth=2)
            ax_single.set_xscale('log')
            ax_single.set_xlabel('Number of Symbols')
            ax_single.set_ylabel('Relative Error (%)')
            ax_single.set_title('BER Relative Error Analysis')
            ax_single.legend()
            ax_single.grid(True, alpha=0.3)
            filename = "plot4_relative_error.png"
        
        fig_single.tight_layout()
        individual_path = os.path.join(output_dir, filename)
        fig_single.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig_single)
    
    plt.show()
    
    return plot_path

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs(os.path.join("results", "testlevel1"), exist_ok=True)
    
    print("Starting real QPSK B2B system benchmark test...")
    try:
        numpy_results, jax_results, speedups, json_path, csv_path, summary_path = run_real_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ Benchmark test completed successfully!")
        print("=" * 60)
        print(f"\nResults saved in: results\\testlevel1\\")
        print(f"  • {os.path.basename(json_path)} - Raw data in JSON format")
        print(f"  • {os.path.basename(csv_path)} - Data in CSV format for analysis")
        print(f"  • {os.path.basename(summary_path)} - Text summary of results")
        print(f"  • benchmark_plots_*.png - Visualizations")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark test: {e}")
        import traceback
        traceback.print_exc()