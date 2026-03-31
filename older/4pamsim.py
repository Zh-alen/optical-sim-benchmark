import os
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import scipy.signal as sp_signal

# =========================================================================
# 导入新版 Commplax 模块
# =========================================================================
from commplax import sym_map, adaptive_kernel

# =========================================================================
# 1. 路径设置
# =========================================================================
save_dir = r"C:\optical_sim_benchmark\result\PAM4NEW"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# =========================================================================
# 2. 系统参数设置
# =========================================================================
M = 4          # PAM4
sps = 16       # 每符号采样数 (主要用于画高精度眼图)
num_symbols = 70000
snr_db = 20    # 信噪比

# =========================================================================
# 3. 信号生成 (利用新版 sym_map)
# =========================================================================
print("正在生成 PAM4 信号...")
key_tx, key_noise = jr.split(jr.PRNGKey(42))

# 【新版特性 1】：直接使用 sym_map.const 获取归一化 PAM4 星座点
pam4_const_norm = sym_map.const('PAM4', norm=True)

# 生成随机符号索引，直接映射为发送符号
tx_indices = jr.randint(key_tx, (num_symbols,), 0, M)
tx_symbols = pam4_const_norm[tx_indices]

# =========================================================================
# 4. 模拟信道 (加 AWGN 噪声)
# =========================================================================
# PAM4 是实数基带信号，直接加实数噪声
noise_power = 10**(-snr_db / 10)
noise_std = jnp.sqrt(noise_power)
rx_symbols = tx_symbols + noise_std * jr.normal(key_noise, tx_symbols.shape)

# =========================================================================
# 5. 计算 BER (利用新版 decision 向量化)
# =========================================================================
print("正在计算误码率...")
def calculate_pam4_ber(rx, tx, const):
    """
    【新版特性 2】：
    新版 commplax 的 adaptive_kernel.decision 内部通过 const[:, None] - v[None, :] 
    实现了矩阵广播。你不需要 for 循环，直接把 70000 长度的 1D 数组传进去，
    JAX 会瞬间完成所有点的判决！
    """
    decided = adaptive_kernel.decision(const, rx)
    
    # 比较判决后的符号与发送的归一化符号 (使用 isclose 防止浮点精度误判)
    ser = jnp.mean(~jnp.isclose(decided, tx, atol=1e-5))
    
    # PAM4 格雷映射下的 BER 近似
    ber = ser / jnp.log2(M)
    return ber

ber_val = calculate_pam4_ber(rx_symbols, tx_symbols, pam4_const_norm)
print(f"Estimated BER: {ber_val:.2e}")

# =========================================================================
# 6. 绘图与保存
# =========================================================================
print("正在生成分析图表...")

# (1) 信号 Level 图 (直方图分布)
plt.figure(figsize=(8, 5))
# JAX 数组转为 NumPy 用于 Matplotlib 画图
plt.hist(np.array(rx_symbols), bins=150, density=True, alpha=0.7, color='skyblue', label='Rx Signal')
for level in pam4_const_norm:
    plt.axvline(level, color='red', linestyle='--', alpha=0.8, linewidth=2)
plt.title(f"PAM4 Signal Levels Distribution (SNR = {snr_db} dB)")
plt.xlabel("Amplitude")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "signal_levels.png"), dpi=300)
plt.close()

# (2) 生成极度平滑的准真实眼图
def generate_smooth_waveform(symbols, sps):
    """
    【眼图优化】：
    通过 插 0 并经过数字低通滤波器 (FIR)，
    生成真实物理世界中连续平滑的基带波形，彻底解决原来 repeat 造成的阶梯状眼图。
    """
    # 1. 插零上采样 (Upsampling)
    up_sig = np.zeros(len(symbols) * sps)
    up_sig[::sps] = np.array(symbols)
    
    # 2. 设计低通滤波器 (FIR)
    num_taps = sps * 6 + 1
    cutoff = 1.0 / sps
    taps = sp_signal.firwin(num_taps, cutoff, window='hamming')
    
    # 3. 滤波并补偿增益
    waveform = sp_signal.lfilter(taps, 1.0, up_sig) * sps
    
    # 4. 消除滤波器带来的群延迟，对齐眼图中心
    delay = (num_taps - 1) // 2
    waveform = np.roll(waveform, -delay)
    return waveform

def save_eye_diagram(waveform, sps, path):
    num_eyes = 3 # PAM4 在图上通常展示 3 个横向眼宽
    samples_per_window = int(sps * num_eyes)
    
    # 只取前 10000 个符号画图，避免线条过多糊成一团
    plot_samples = 10000 * sps 
    sig_segment = waveform[:plot_samples]
    
    num_frames = len(sig_segment) // samples_per_window
    frames = sig_segment[:num_frames * samples_per_window].reshape((num_frames, samples_per_window))
    
    plt.figure(figsize=(10, 5))
    t = np.linspace(0, num_eyes, samples_per_window)
    
    # 叠加画图模拟示波器余晖效应 (alpha 调低)
    plt.plot(t, frames.T, color='tab:blue', alpha=0.04, linewidth=1)
    
    plt.title(f"PAM4 Smoothed Eye Diagram ({sps} SPS)")
    plt.xlabel("Time (Symbol Periods)")
    plt.ylabel("Amplitude")
    # PAM4 归一化后幅度通常在 ±1.35 左右
    plt.ylim([-1.6, 1.6]) 
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# 生成并保存眼图
rx_waveform_smooth = generate_smooth_waveform(rx_symbols, sps)
save_eye_diagram(rx_waveform_smooth, sps, os.path.join(save_dir, "eye_diagram.png"))

# =========================================================================
# (3) 星座图 (Constellation Diagram)
# =========================================================================
plt.figure(figsize=(10, 3))
# 取前 5000 个点画图即可，防止重叠过于严重
plot_points = 5000
# PAM4 是一维信号，所以 Q 路（虚部）设定为 0
plt.scatter(np.array(rx_symbols[:plot_points]), np.zeros(plot_points), 
            s=5, alpha=0.4, color='tab:blue', label='Rx Symbols')
# 画出理想的 4 个参考星座点
plt.scatter(np.array(pam4_const_norm), np.zeros(M), 
            s=100, c='red', marker='x', linewidths=2, label='Ideal Constellation')

plt.title(f"PAM4 Constellation Diagram (SNR = {snr_db} dB)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
# 限制 Y 轴让点在图中央显示得更美观
plt.ylim([-0.5, 0.5])
# 隐藏 Y 轴的刻度，因为一维信号的 Y 轴其实没有实际意义
plt.yticks([]) 
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "constellation.png"), dpi=300)
plt.close()

print(f"成功！仿真完成，图片已保存至: {save_dir}")