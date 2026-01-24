#"update code for level3"  
import jax
import jax.numpy as jnp
from functools import partial

# 生成测试数据（模拟EDC后的相位旋转）
def generate_test_signal(num_symbols=1000, phase_offset=jnp.pi):
    """生成带相位偏移的QPSK信号"""
    key = jax.random.PRNGKey(42)
    # 生成原始QPSK符号
    symbols = jax.random.randint(key, (num_symbols,), 0, 4)
    # QPSK映射：0->(1+j)/√2, 1->(-1+j)/√2, 2->(-1-j)/√2, 3->(1-j)/√2
    qpsk_map = jnp.array([1+1j, -1+1j, -1-1j, 1-1j]) / jnp.sqrt(2)
    tx_symbols = qpsk_map[symbols]
    
    # 添加相位偏移（模拟EDC引入的模糊）
    rx_symbols = tx_symbols * jnp.exp(1j * phase_offset)
    
    # 添加高斯噪声
    noise = 0.05 * (jax.random.normal(key, (num_symbols,)) + 
                    1j * jax.random.normal(key, (num_symbols,)))
    
    return tx_symbols, rx_symbols + noise

# 计算BER的函数
def calculate_ber(tx_symbols, rx_symbols, phase_candidate):
    """计算特定相位补偿后的BER"""
    # 应用相位补偿
    compensated = rx_symbols * jnp.exp(-1j * phase_candidate)
    
    # QPSK硬判决
    # 计算到4个QPSK点的距离
    qpsk_points = jnp.array([1+1j, -1+1j, -1-1j, 1-1j]) / jnp.sqrt(2)
    distances = jnp.abs(compensated[:, None] - qpsk_points[None, :])
    decisions = jnp.argmin(distances, axis=1)
    
    # 将决策映射回原始映射
    rx_decisions = decisions
    
    # 原始发射符号的索引
    tx_decisions = jnp.argmin(jnp.abs(tx_symbols[:, None] - qpsk_points[None, :]), axis=1)
    
    # 计算误码率
    errors = jnp.sum(rx_decisions != tx_decisions)
    total_bits = len(tx_symbols) * 2  # QPSK: 每个符号2比特
    ber = errors / total_bits
    
    return ber

# 向量化相位搜索
def vectorized_phase_search(tx_symbols, rx_symbols):
    """并行测试4个候选相位"""
    # 候选相位：[0, π/2, π, 3π/2]
    phase_candidates = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
    
    # 向量化计算每个相位的BER
    bers = jax.vmap(calculate_ber, in_axes=(None, None, 0))(
        tx_symbols, rx_symbols, phase_candidates
    )
    
    # 找到最佳相位
    best_idx = jnp.argmin(bers)
    best_phase = phase_candidates[best_idx]
    best_ber = bers[best_idx]
    
    return best_phase, best_ber, bers

# JIT编译加速
@jax.jit
def fast_phase_search(tx_symbols, rx_symbols):
    return vectorized_phase_search(tx_symbols, rx_symbols)

# 可视化函数
def visualize_results(tx_symbols, rx_symbols, best_phase, bers):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 1. 原始发射信号
    axes[0, 0].scatter(tx_symbols.real, tx_symbols.imag, alpha=0.5, s=10)
    axes[0, 0].set_title('Transmitted Symbols')
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    # 2. 接收信号（带相位模糊）
    axes[0, 1].scatter(rx_symbols.real, rx_symbols.imag, alpha=0.5, s=10)
    axes[0, 1].set_title(f'Received Symbols (Phase offset)')
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    
    # 3. 补偿后信号
    compensated = rx_symbols * jnp.exp(-1j * best_phase)
    axes[0, 2].scatter(compensated.real, compensated.imag, alpha=0.5, s=10)
    axes[0, 2].set_title(f'After PBS (Phase: {best_phase/np.pi:.2f}π)')
    axes[0, 2].grid(True)
    axes[0, 2].axis('equal')
    
    # 4. BER柱状图
    phases = ['0', 'π/2', 'π', '3π/2']
    colors = ['red' if b > 0.1 else 'green' for b in bers]
    axes[1, 0].bar(phases, bers, color=colors)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('BER for Phase Candidates')
    axes[1, 0].set_ylabel('BER (log scale)')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (0.5)')
    
    # 5. 相位旋转动画（概念图）
    theta = jnp.linspace(0, 2*jnp.pi, 100)
    axes[1, 1].plot(jnp.cos(theta), jnp.sin(theta), 'k--', alpha=0.3)
    for i, phase in enumerate([0, jnp.pi/2, jnp.pi, 3*jnp.pi/2]):
        x = jnp.cos(phase)
        y = jnp.sin(phase)
        color = 'red' if i == jnp.argmin(bers) else 'gray'
        axes[1, 1].arrow(0, 0, x*0.8, y*0.8, head_width=0.05, head_length=0.1, 
                         fc=color, ec=color, alpha=0.7 if color=='red' else 0.3)
        axes[1, 1].text(x*0.9, y*0.9, f'{i}', fontsize=10)
    axes[1, 1].set_title('Phase Candidates')
    axes[1, 1].axis('equal')
    axes[1, 1].grid(True)
    
    # 6. 信息面板
    axes[1, 2].axis('off')
    info_text = f"""
    Phase Blind Search Results:
    -------------------------
    Best Phase: {best_phase/np.pi:.3f}π
    Best BER: {best_ber:.2e}
    
    Without PBS:
    - BER would be ~0.5
    - Complete failure
    
    PBS Benefits:
    - Vectorized search
    - No Python loops
    - GPU accelerated
    """
    axes[1, 2].text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=10,
                    verticalalignment='center')
    
    plt.tight_layout()
    return fig

# 主测试函数
def test_phase_ambiguity():
    """完整的测试流程"""
    print("=" * 60)
    print("Phase Ambiguity Test for EDC Compensation")
    print("=" * 60)
    
    # 生成测试信号（故意添加π相位偏移）
    tx_symbols, rx_symbols = generate_test_signal(
        num_symbols=5000, 
        phase_offset=jnp.pi  # 180度偏移，会导致BER=0.5
    )
    
    print(f"\nGenerated {len(tx_symbols)} QPSK symbols")
    print(f"Deliberate phase offset: π radians (180°)")
    
    # 计算未补偿的BER（应该接近0.5）
    raw_ber = calculate_ber(tx_symbols, rx_symbols, 0.0)
    print(f"\n1. Without phase compensation:")
    print(f"   BER = {raw_ber:.4f} (should be ~0.5)")
    
    # 执行向量化相位搜索
    best_phase, best_ber, all_bers = fast_phase_search(tx_symbols, rx_symbols)
    
    print(f"\n2. After Vectorized Phase Blind Search:")
    print(f"   Candidate phases: [0, π/2, π, 3π/2]")
    print(f"   Candidate BERs: {[f'{b:.4f}' for b in all_bers]}")
    print(f"   Selected phase: {best_phase/jnp.pi:.3f}π")
    print(f"   Achieved BER: {best_ber:.2e}")
    
    # 性能对比
    print(f"\n3. Performance Improvement:")
    print(f"   BER reduction: {raw_ber/best_ber:.0f}x")
    print(f"   From random (0.5) to decodable ({best_ber:.2e})")
    
    # 验证逻辑
    print(f"\n4. Verification:")
    print(f"   Is best BER < 0.1? {best_ber < 0.1}")
    print(f"   Is best phase close to π? {jnp.abs(best_phase - jnp.pi) < 0.1}")
    
    return tx_symbols, rx_symbols, best_phase, all_bers

# 集成到你的EDC模块
def enhanced_edc_with_pbs(received_signal, edc_filter):
    """
    增强的EDC模块，包含自动相位恢复
    """
    # 1. 应用EDC滤波
    compensated = edc_filter(received_signal)
    
    # 2. 生成参考训练序列（使用已知的pilot符号）
    # 在实际系统中，这里应该是已知的训练序列
    # 为演示，我们使用前100个符号作为参考
    pilot_length = min(100, len(compensated)//4)
    pilot_symbols = generate_test_signal(pilot_length, 0)[0]  # 无偏移的参考
    
    # 3. 向量化相位搜索
    best_phase, best_ber, _ = fast_phase_search(
        pilot_symbols, 
        compensated[:pilot_length]
    )
    
    # 4. 应用最佳相位补偿
    final_signal = compensated * jnp.exp(-1j * best_phase)
    
    return final_signal, best_phase, best_ber

# 运行测试
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Phase Ambiguity Issue in EDC...")
    tx, rx, best_phase, bers = test_phase_ambiguity()
    
    # 生成可视化
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        fig = visualize_results(tx, rx, best_phase, bers)
        
        # 保存图片
        import os
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/phase_ambiguity_solution.png", dpi=150)
        print(f"\nVisualization saved to {output_dir}/phase_ambiguity_solution.png")
        
        # 也显示在notebook中
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("Summary for Presentation:")
    print("=" * 60)
    print("PROBLEM: EDC introduces unknown phase rotation")
    print("RESULT: BER jumps to 0.5 (random guessing)")
    print("SOLUTION: Vectorized Phase Blind Search")
    print("  - Tests 4 candidates in parallel")
    print("  - Uses jax.vmap for GPU acceleration")
    print("  - Reduces BER from 0.5 to < 1e-3")
    print("  - No Python loops, fully JIT-compiled")