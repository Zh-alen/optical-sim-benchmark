import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp_signal

# =========================================================================
# 1. 基础配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results', 'pam4_refined')
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# =========================================================================
# 2. 升余弦成形工具 (解决 ISI 的关键)
# =========================================================================
def rcosine_design(sps, alpha=0.5, span=10):
    """ 生成升余弦脉冲响应 """
    n = sps * span
    t = np.arange(-n/2, n/2 + 1) / sps
    # 处理分母为 0 的情况
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t)**2)
    # 修正特殊点
    h[np.isnan(h)] = 0
    idx = np.where(np.abs(np.abs(2 * alpha * t) - 1) < 1e-10)
    h[idx] = np.pi/4 * np.sinc(1/(2*alpha))
    return h / np.sqrt(np.sum(h**2))

# =========================================================================
# 3. 主仿真逻辑
# =========================================================================
def main():
    print("--- PAM4 Refined System Simulation ---")
    
    # 参数设置
    num_syms = 70000
    sps = 2  # 提高采样率让眼图更美观
    alpha = 1 # 滚降系数
    key = jax.random.PRNGKey(42)

    # Step 1: 符号生成与归一化
    # 标准 PAM4 电平: -3, -1, 1, 3
    mapping = jnp.array([-3., -1., 1., 3.])
    indices = jax.random.randint(key, (num_syms,), 0, 4)
    tx_syms = mapping[indices]
    tx_syms = tx_syms / jnp.std(tx_syms) # 归一化功率为 1

    # Step 2: 脉冲成形 (替代之前的 Butterworth)
    print("Step 2: Pulse Shaping & Transmission...")
    tx_upsampled = jnp.zeros(num_syms * sps)
    tx_upsampled = tx_upsampled.at[::sps].set(tx_syms)
    
    h = rcosine_design(sps, alpha)
    # 卷积生成连续波形
    rx_waveform = np.convolve(np.array(tx_upsampled), h, mode='same')
    
    # 添加极小噪声 (模拟高 SNR 场景)
    noise = np.random.normal(0, 0.02, rx_waveform.shape)
    rx_noisy = rx_waveform + noise

    # Step 3: 接收采样 (完美对齐)
    # 升余弦成形的优势在于：在每个符号的正中心采样，ISI 为 0
    rx_sampled = rx_noisy[::sps]

    # Step 4: 性能评估
    mse = np.mean((np.array(tx_syms) - rx_sampled)**2)
    snr_val = 10 * np.log10(1.0 / (mse + 1e-12))
    print(f"Result: PAM4 SNR = {snr_val:.2f} dB")

    # Step 5: 绘图
    print("Step 4: Generating Plots...")
    
    # 1. 接收电平图 (现在应该是 4 条非常细的线)
    plt.figure(figsize=(14, 6))
    plt.plot(rx_sampled[:500], 'o', markersize=2, alpha=0.6)
    plt.title(f"PAM4 Received Levels (SNR: {snr_val:.2f} dB)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURE_PATH, 'refined_levels.png'))

    # 2. 完美眼图
    plt.figure(figsize=(10, 6))
    trace_len = 3 * sps
    # 丢弃头尾不稳定的部分
    plot_data = rx_noisy[sps*10 : -sps*10]
    t = np.linspace(0, 2, trace_len)
    
    for i in range(0, len(plot_data) - trace_len, sps):
        plt.plot(t, plot_data[i:i+trace_len], color='blue', alpha=0.03, linewidth=0.5)
    
    plt.title("Refined PAM4 Eye Diagram (ISI-Free)")
    plt.xlabel("Symbol Period")
    plt.savefig(os.path.join(FIGURE_PATH, 'refined_eye.png'), dpi=300)
    plt.close()
    
    print(f"Success! Figures saved in {FIGURE_PATH}")

if __name__ == "__main__":
    main()