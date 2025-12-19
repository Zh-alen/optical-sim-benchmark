import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 确保导入路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from traditional_platform.level2_linear import benchmark_level2_numpy
    from jax_platform.level2_linear import benchmark_level2_jax
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def create_level2_test_cases():
    fiber_lengths = [0, 20, 40, 60, 80, 100, 120]
    return [{
        'name': f'{l}km', 
        'fiber_length_km': l, 
        'num_symbols': 16384,
        'samples_per_symbol': 16, 
        'snr_db': 7.5,        # 较低的SNR以确保产生误码
        'num_runs': 3
    } for l in fiber_lengths]

def plot_benchmark_results(np_res, jax_res, output_path):
    lengths = [r['fiber_length_km'] for r in np_res]
    np_times = [r['avg_time'] * 1000 for r in np_res]
    jax_times = [r['avg_time'] * 1000 for r in jax_res]
    
    # 获取原始BER
    np_bers = [r['avg_ber'] for r in np_res]
    jax_bers = [r['avg_ber'] for r in jax_res]
    
    # 绘图保护：处理BER为0的情况，方便对数显示
    np_bers_plot = [max(b, 1e-6) for b in np_bers]
    jax_bers_plot = [max(b, 1e-6) for b in jax_bers]
    
    speedups = [n / j for n, j in zip(np_times, jax_times)]

    # 调整画布大小，增加一行以容纳分开的BER图
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 第一行：性能对比
    # 1. 计算时间
    axs[0, 0].plot(lengths, np_times, 'o-', label='NumPy (CPU)', linewidth=2)
    axs[0, 0].plot(lengths, jax_times, 's-', label='JAX (Accelerated)', linewidth=2)
    axs[0, 0].set_xlabel('Fiber Length (km)')
    axs[0, 0].set_ylabel('Execution Time (ms)')
    axs[0, 0].set_title('Computational Time Comparison', fontsize=12, fontweight='bold')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. 加速倍数
    axs[0, 1].bar(np.array(lengths).astype(str), speedups, color='teal', alpha=0.7)
    axs[0, 1].axhline(y=1, color='r', linestyle='--')
    axs[0, 1].set_ylabel('Speedup Factor (x)')
    axs[0, 1].set_title('JAX Acceleration Ratio', fontsize=12, fontweight='bold')
    for i, v in enumerate(speedups):
        axs[0, 1].text(i, v + 0.1, f'{v:.1f}x', ha='center')

    # 第二行：分开显示 BER
    # 3. NumPy BER 详情
    axs[1, 0].semilogy(lengths, np_bers_plot, 'o-r', label='NumPy BER', linewidth=2)
    axs[1, 0].set_xlabel('Fiber Length (km)')
    axs[1, 0].set_ylabel('Bit Error Rate (BER)')
    axs[1, 0].set_title('NumPy BER Performance (Detail)', fontsize=12, color='red', fontweight='bold')
    axs[1, 0].grid(True, which="both", ls="-", alpha=0.2)
    axs[1, 0].legend()

    # 4. JAX BER 详情
    axs[1, 1].semilogy(lengths, jax_bers_plot, 's-g', label='JAX BER', linewidth=2)
    axs[1, 1].set_xlabel('Fiber Length (km)')
    axs[1, 1].set_ylabel('Bit Error Rate (BER)')
    axs[1, 1].set_title('JAX BER Performance (Detail)', fontsize=12, color='green', fontweight='bold')
    axs[1, 1].grid(True, which="both", ls="-", alpha=0.2)
    axs[1, 1].legend()

    # 第三行：数值差异分析
    # 5. BER 直接对比（放在一起看差距）
    axs[2, 0].semilogy(lengths, np_bers_plot, 'o--', label='NumPy', alpha=0.6)
    axs[2, 0].semilogy(lengths, jax_bers_plot, 'x-', label='JAX', linewidth=2)
    axs[2, 0].set_title('BER Comparison (Overlaid)', fontsize=12, fontweight='bold')
    axs[2, 0].legend()
    axs[2, 0].grid(True, which="both", ls="-", alpha=0.1)

    # 6. 绝对误差分析
    diff = [abs(n - j) for n, j in zip(np_bers, jax_bers)]
    axs[2, 1].plot(lengths, diff, 'D-', color='purple', label='|NP - JAX|')
    axs[2, 1].set_xlabel('Fiber Length (km)')
    axs[2, 1].set_ylabel('Absolute BER Difference')
    axs[2, 1].set_title('Numerical Deviation Analysis', fontsize=12, fontweight='bold')
    axs[2, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Optical Simulation Benchmark: Detached BER Analysis', fontsize=16, y=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 详细对比图像已生成: {output_path}")

def run_level2_benchmark():
    print("=" * 80)
    print("LEVEL 2 BENCHMARK: DETACHED PLOTTING MODE")
    print("=" * 80)
    
    test_cases = create_level2_test_cases()
    
    print("\n>>> Running NumPy Tests...")
    np_res = benchmark_level2_numpy(test_cases)
    
    print("\n>>> Running JAX Tests...")
    jax_res = benchmark_level2_jax(test_cases)
    
    os.makedirs(os.path.join("results", "testlevel2"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join("results", "testlevel2", f"detailed_analysis_{timestamp}.png")
    
    plot_benchmark_results(np_res, jax_res, plot_filename)

if __name__ == "__main__":
    run_level2_benchmark()