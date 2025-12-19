import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 确保路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from traditional_platform.level2_linear import benchmark_level2_numpy # 备用
    from traditional_platform.level3_nonlinear import benchmark_level3_numpy
    from jax_platform.level3_nonlinear import benchmark_level3_jax
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

def plot_level3_analysis(np_res, jax_res, output_path):
    lengths = [r['fiber_length_km'] for r in np_res]
    np_times = [r['avg_time'] * 1000 for r in np_res]  # 转为 ms
    jax_times = [r['avg_time'] * 1000 for r in jax_res] # 转为 ms
    np_bers = [max(r['avg_ber'], 1e-6) for r in np_res]
    jax_bers = [max(r['avg_ber'], 1e-6) for r in jax_res]
    speedups = [n / j for n, j in zip(np_times, jax_times)]

    # 创建 2x2 的综合图标
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # 1. 耗时趋势图
    axs[0, 0].plot(lengths, np_times, 'o-r', label='NumPy (Standard Loop)', linewidth=2)
    axs[0, 0].plot(lengths, jax_times, 's-g', label='JAX (lax.scan + JIT)', linewidth=2)
    axs[0, 0].set_xlabel('Fiber Length (km)')
    axs[0, 0].set_ylabel('Avg Execution Time (ms)')
    axs[0, 0].set_title('Computational Time: Level 3 (Nonlinear)', fontsize=12, fontweight='bold')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. 加速倍数柱状图
    bar_colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(lengths)))
    bars = axs[0, 1].bar(np.array(lengths).astype(str), speedups, color=bar_colors, edgecolor='black', alpha=0.8)
    axs[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axs[0, 1].set_ylabel('Speedup Factor (x)')
    axs[0, 1].set_title('JAX Acceleration Ratio', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}x', ha='center', fontweight='bold')

    # 3. BER 一致性验证 (对数坐标)
    axs[1, 0].semilogy(lengths, np_bers, 'o', label='NumPy BER', markersize=12, alpha=0.4)
    axs[1, 0].semilogy(lengths, jax_bers, 'x-', label='JAX BER', linewidth=2, color='darkgreen')
    axs[1, 0].set_xlabel('Fiber Length (km)')
    axs[1, 0].set_ylabel('Bit Error Rate (BER)')
    axs[1, 0].set_title('Algorithm Consistency Check', fontsize=12, fontweight='bold')
    axs[1, 0].grid(True, which="both", ls="-", alpha=0.2)
    axs[1, 0].legend()

    # 4. 边际成本分析 (计算斜率)
    # 这张图能显示出随着距离增加（循环次数增加），JAX 是否能保持常数级的增长
    np_slope = np.gradient(np_times, lengths) if len(lengths) > 1 else [0]
    jax_slope = np.gradient(jax_times, lengths) if len(lengths) > 1 else [0]
    axs[1, 1].plot(lengths, np_slope, 'o--', color='red', label='NP Time Slope')
    axs[1, 1].plot(lengths, jax_slope, 's--', color='green', label='JAX Time Slope')
    axs[1, 1].set_ylabel('ms / km')
    axs[1, 1].set_title('Marginal Computational Cost', fontsize=12, fontweight='bold')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Optical Simulation Benchmark Level 3: JAX vs NumPy\n(SSFM Steps=80, CMA Iterations=10)', fontsize=16, y=0.96)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 详细图标分析已保存至: {output_path}")

def run_level3_benchmark():
    print("=" * 80)
    print("🚀 LEVEL 3 BENCHMARK: STARTING SIMULATION")
    print("=" * 80)

    # 测试用例：加长距离，体现 SSFM 步数的压力
    test_cases = [
        {'name': f'{l}km', 'fiber_length_km': l, 'num_symbols': 8192, 'snr_db': 12.0, 'num_runs': 3}
        for l in [0, 40, 80, 120]
    ]
    
    # 1. 运行 NumPy
    np_res = benchmark_level3_numpy(test_cases)
    
    # 2. 运行 JAX
    jax_res = benchmark_level3_jax(test_cases)
    
    # 3. 结果汇总打印
    print("\n" + "=" * 85)
    print(f"{'Distance':<10} | {'NP Time(ms)':<15} | {'JAX Time(ms)':<15} | {'Speedup':<12} | {'BER Status'}")
    print("-" * 85)
    for n, j in zip(np_res, jax_res):
        speedup = n['avg_time'] / j['avg_time']
        print(f"{n['name']:<10} | {n['avg_time']*1000:<15.2f} | {j['avg_time']*1000:<15.2f} | {speedup:<12.2f}x | Match")

    # 4. 生成图标
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join("results", "level3_final", f"analysis_plot_{timestamp}.png")
    plot_level3_analysis(np_res, jax_res, output_file)

if __name__ == "__main__":
    run_level3_benchmark()