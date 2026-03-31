#32qam
import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.signal as sp_signal
import numpy as np

# =========================================================================
# 路径配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results_32qam', 'figure')
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# =========================================================================
# 核心评价与绘图函数
# =========================================================================
def get_snr(ref, rec):
    rec = rec / (jnp.sqrt(jnp.mean(jnp.abs(rec)**2)) + 1e-12)
    ref = ref / (jnp.sqrt(jnp.mean(jnp.abs(ref)**2)) + 1e-12)
    best_snr = -100.0
    best_rec = rec
    for shift in range(-5, 6): 
        r_sh = jnp.roll(rec, shift)
        # 32QAM 依然具有 90 度旋转对称性
        for angle in [0, jnp.pi/2, jnp.pi, 3*jnp.pi/2]:
            temp_rec = r_sh * jnp.exp(-1j * angle)
            phase_offset = jnp.angle(jnp.mean(temp_rec * jnp.conj(ref)))
            temp_rec_final = temp_rec * jnp.exp(-1j * phase_offset)
            mse = jnp.mean(jnp.abs(ref - temp_rec_final)**2)
            snr = 10 * jnp.log10(1.0 / (mse + 1e-12))
            if snr > best_snr:
                best_snr = snr
                best_rec = temp_rec_final
    return best_snr, best_rec

def plot_psd(sig, fs, name='Signal PSD', filename='psd_analysis.png'):
    plt.figure(figsize=(10, 5))
    for i in range(sig.shape[1]):
        f, Pxx_den = sp_signal.welch(np.array(sig[:, i]), fs, nperseg=1024)
        plt.semilogy(f / 1e9, Pxx_den, label=f'Pol {i}')
    plt.title(name)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('PSD (V**2/Hz)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename), dpi=300)
    plt.close()

def plot_eye(sig, sps=2, name='Eye Diagram', amplitude_limit=2.0, filename='eye_diagram.png'):
    plt.figure(figsize=(12, 5))
    num_points = int(2 * sps) + 1 
    num_traces = 800
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        start = 20000 if sig.shape[0] > 20000 else 0
        data = np.array(sig[start:, i].real)
        traces = []
        for j in range(0, min(len(data) - num_points, num_traces * sps), sps):
            traces.append(data[j : j + num_points])
        if len(traces) > 0:
            t = np.linspace(0, 2, num_points)
            for trace in traces:
                ax.plot(t, trace, 'b-', alpha=0.05, linewidth=0.5)
            ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlim([0, 2])
        ax.set_ylim([-amplitude_limit, amplitude_limit])
        ax.set_title(f'{name} - Pol {i}')
        ax.set_xlabel('Time (Symbol Period)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename), dpi=300)
    plt.close()

# =========================================================================
# 物理损伤模块
# =========================================================================
def apply_chromatic_dispersion(sig, baud_rate, sps, D=17.0, L=80.0, lambda_=1550e-9):
    fs = baud_rate * sps
    c = 299792458.0
    beta2 = -(lambda_**2) / (2 * jnp.pi * c) * (D * 1e-6)
    f = jnp.fft.fftfreq(sig.shape[0], 1/fs)
    omega = 2 * jnp.pi * f
    H = jnp.exp(-1j * (beta2 / 2) * (L * 1e3) * omega**2)
    sig_f = jnp.fft.fft(sig, axis=0)
    return jnp.fft.ifft(sig_f * H[:, None], axis=0)

def get_safe_rrc_taps(sps, num_taps, beta=0.1):
    t = np.arange(num_taps) - (num_taps - 1) / 2.0
    t = t / sps
    h = np.zeros_like(t)
    for i, val in enumerate(t):
        if val == 0.0:
            h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif np.abs(np.abs(val) - 1.0 / (4 * beta)) < 1e-6:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + 
                                          (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * val * (1 - beta)) + 4 * beta * val * np.cos(np.pi * val * (1 + beta))
            den = np.pi * val * (1 - (4 * beta * val)**2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))

# =========================================================================
# JAX 极速自适应 DSP 算法核心 (专为 32QAM 打造)
# =========================================================================
@jax.jit(static_argnames=['taps', 'train_len'])
def qam32_lms_equalizer(rx, tx_train, const, taps=5, lr_da=2e-3, lr_dd=5e-4, train_len=10000):
    n_syms = rx.shape[0]
    pad = taps // 2
    rx_pad = jnp.pad(rx, ((pad, pad), (0, 0)))
    
    def step(W, i):
        x = jax.lax.dynamic_slice(rx_pad, (i, 0), (taps, 2))
        x = jnp.flip(x, axis=0)
        
        y0 = jnp.sum(W[0, 0] * x[:, 0] + W[0, 1] * x[:, 1])
        y1 = jnp.sum(W[1, 0] * x[:, 0] + W[1, 1] * x[:, 1])
        y = jnp.array([y0, y1])
        
        d_da = tx_train[i]
        
        # 32QAM 密集二维复平面距离判决
        dist0 = jnp.abs(y0 - const)
        dist1 = jnp.abs(y1 - const)
        d_dd = jnp.array([const[jnp.argmin(dist0)], const[jnp.argmin(dist1)]])
        
        is_train = i < train_len
        d = jnp.where(is_train, d_da, d_dd)
        lr = jnp.where(is_train, lr_da, lr_dd)
        
        e = d - y
        
        W = W.at[0, 0].add(lr * e[0] * jnp.conj(x[:, 0]))
        W = W.at[0, 1].add(lr * e[0] * jnp.conj(x[:, 1]))
        W = W.at[1, 0].add(lr * e[1] * jnp.conj(x[:, 0]))
        W = W.at[1, 1].add(lr * e[1] * jnp.conj(x[:, 1]))
        
        return W, y
        
    W0 = jnp.zeros((2, 2, taps), dtype=jnp.complex64)
    W0 = W0.at[0, 0, taps//2].set(1.0)
    W0 = W0.at[1, 1, taps//2].set(1.0)
    
    _, rx_eq = jax.lax.scan(step, W0, jnp.arange(n_syms))
    return rx_eq

@jax.jit
def dd_pll(rx_syms, const, mu=0.015):
    """判决引导锁相环 (Decision-Directed PLL)，专门对付十字形 32QAM"""
    def step(phi, y_in):
        # 补偿相位
        y_rot = y_in * jnp.exp(-1j * phi)
        # 硬判决找到最近的理想星座点
        idx = jnp.argmin(jnp.abs(y_rot - const))
        d = const[idx]
        # 计算当前旋转点与理想点之间的相位差
        err = jnp.angle(y_rot * jnp.conj(d))
        # 环路滤波更新
        phi_next = phi + mu * err
        return phi_next, y_rot
    
    _, rx_out = jax.lax.scan(step, 0.0, rx_syms)
    return rx_out

# =========================================================================
# 主仿真逻辑
# =========================================================================
def main():
    start_time = time.time()

    print("Step 1: 32QAM Signal Generation & Pulse Shaping...")
    # 【暴增数据量】：131072 个符号！
    num_syms = 131072 
    baud_rate = 32e9 
    
    # 手动精细生成 32QAM 十字形星座图 (Cross-QAM)
    grid = np.array([-5, -3, -1, 1, 3, 5])
    X, Y = np.meshgrid(grid, grid)
    c_raw = (X + 1j*Y).flatten()
    # 剔除四个角落的点 (±5 ±5j)，剩下 36-4=32 个点
    corners = [5+5j, 5-5j, -5+5j, -5-5j]
    c_32qam = np.array([pt for pt in c_raw if pt not in corners])
    const = jnp.array(c_32qam)
    const = const / jnp.sqrt(jnp.mean(jnp.abs(const)**2)) # 严格归一化

    key = jax.random.PRNGKey(42)
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 32)
    tx_syms = const[tx_indices]

    rrc_taps_len = 65
    h_rrc = get_safe_rrc_taps(sps=2, num_taps=rrc_taps_len, beta=0.1)
    tx_sig = sp_signal.upfirdn(np.array(h_rrc), np.array(tx_syms), up=2, axis=0)
    delay = (rrc_taps_len - 1) // 2
    rx_2sps = jnp.array(tx_sig[delay : delay + num_syms * 2])

    print("Step 2: Realistic Channel Impairments...")
    theta = jnp.pi / 4 * 0.6
    J_rot = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
                       [-jnp.sin(theta), jnp.cos(theta)]])
    rx_2sps = rx_2sps @ J_rot.T
    
    lw = 10e3  
    Ts = 1.0 / (baud_rate * 2)
    sigma = jnp.sqrt(2 * jnp.pi * lw * Ts)
    phase_steps = sigma * jax.random.normal(jax.random.PRNGKey(99), (rx_2sps.shape[0], 1))
    rx_2sps = rx_2sps * jnp.exp(1j * jnp.cumsum(phase_steps, axis=0))
    
    rx_2sps = apply_chromatic_dispersion(rx_2sps, baud_rate, sps=2, D=17.0, L=80.0)

    # 注意：32QAM 星座点极度密集，抗噪极差。我们将 AWGN 噪声设定为 0.02
    noise = (jax.random.normal(jax.random.PRNGKey(123), rx_2sps.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(124), rx_2sps.shape)) * 0.02
    rx_2sps += noise

    print("Generating Pre-DSP Plots...")
    plot_psd(rx_2sps, fs=baud_rate*2, name='PSD Before DSP', filename='01_psd_before_dsp.png')
    plot_eye(rx_2sps, sps=2, name='32QAM Eye Before DSP', filename='02_eye_before_dsp.png')

    print("Step 2.5: Static CDC & Rx Matched Filtering...")
    rx_2sps = apply_chromatic_dispersion(rx_2sps, baud_rate, sps=2, D=17.0, L=-80.0)
    rx_mf = sp_signal.upfirdn(np.array(h_rrc), np.array(rx_2sps), up=1, axis=0)
    rx_2sps = jnp.array(rx_mf[delay : delay + num_syms * 2])
    
    plot_eye(rx_2sps, sps=2, name='32QAM Eye After CDC & MF', filename='02.5_eye_after_cdc_mf.png')

    print(f"Step 3: JAX LMS Equalization ({num_syms} symbols!)...")
    rx_1sps = rx_2sps[::2] 
    rx_1sps = rx_1sps / jnp.sqrt(jnp.mean(jnp.abs(rx_1sps)**2, axis=0))

    # Taps 增加到 5，加长训练序列到 10000 保证完美解开密集的偏振串扰
    rx_eq = qam32_lms_equalizer(rx_1sps, tx_syms, const, taps=5, train_len=10000)

    print("Step 4: Discarding convergence transient & Normalization...")
    start_idx = 15000 
    rx_eq_cut = rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + rx_eq_cut.shape[0]]

    rx_eq_cut = rx_eq_cut / jnp.sqrt(jnp.mean(jnp.abs(rx_eq_cut)**2, axis=0))

    print("Step 5: Decision-Directed Carrier Phase Recovery (DD-PLL)...")
    rx_cpr_list = []
    for i in range(2):
        pol_cpr_out = dd_pll(rx_eq_cut[:, i], const, mu=0.015)
        rx_cpr_list.append(pol_cpr_out)
        
    rx_cpr_all = jnp.stack(rx_cpr_list, axis=1)

    print("Step 6: Performance Evaluation & Alignment...")
    aligned_list = []
    snr_vals = []
    for i in range(2):
        snr_v, aligned_sig = get_snr(tx_final[:, i], rx_cpr_all[:, i])
        print(f"   Pol {i} Final SNR: {snr_v:.2f} dB")
        aligned_list.append(aligned_sig)
        snr_vals.append(snr_v)
    
    rx_aligned_all = jnp.stack(aligned_list, axis=1)

    print("Generating Final DSP Plots & Diagnostics...")
    
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(rx_eq_cut[-8000:, 0].real, rx_eq_cut[-8000:, 0].imag, s=1, alpha=0.3, color='tab:blue')
    ax1.set_title("1. After JAX LMS (Ring Structure)")
    ax1.axis('equal'); ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(rx_aligned_all[-8000:, 0].real, rx_aligned_all[-8000:, 0].imag, s=1, alpha=0.3, color='tab:green')
    ax2.set_title(f"2. Final 32QAM\nSNR: {float(snr_vals[0]):.2f} dB")
    ax2.axis('equal')
    ax2.scatter(const.real, const.imag, c='red', marker='x', s=30, alpha=0.9) 
    ax2.grid(True, linestyle=':', alpha=0.6)

    phase_diff = jnp.unwrap(jnp.angle(rx_cpr_all[-2000:, 0] * jnp.conj(rx_eq_cut[-2000:, 0])))
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(phase_diff, color='tab:orange', linewidth=1)
    ax3.set_title("3. DD-PLL Phase Track")
    ax3.set_xlabel("Symbols"); ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, '00_diagnostic_plot.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        ax.scatter(rx_aligned_all[-8000:, i].real, rx_aligned_all[-8000:, i].imag, s=1, alpha=0.3)
        ax.set_title(f'32QAM Constellation Pol {i}\nSNR: {float(snr_vals[i]):.2f} dB')
        ax.axis('equal')
        ax.scatter(const.real, const.imag, c='red', marker='x', s=30, alpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.savefig(os.path.join(FIGURE_PATH, '03_constellations.png'), dpi=300)
    plt.close()

    plot_psd(rx_aligned_all, fs=baud_rate, name='PSD After DSP', filename='04_psd_after_dsp.png')
    plot_eye(rx_aligned_all, sps=1, name='Eye After DSP (Recovered)', amplitude_limit=1.5, filename='05_eye_after_dsp.png')

    end_time = time.time()
    
    print("\n" + "="*55)
    print(f"32QAM 极速仿真结束")
    print(f"处理数据量: {num_syms} 个符号")
    print(f"总共耗时: {end_time - start_time:.2f} 秒")
    print(f"图像已保存至目录:\n   {FIGURE_PATH}\n")
    print("本次生成的图像文件清单:")
    if os.path.exists(FIGURE_PATH):
        saved_files = [f for f in os.listdir(FIGURE_PATH) if f.endswith('.png')]
        for idx, file_name in enumerate(sorted(saved_files), 1):
            print(f"  [{idx}] {file_name}")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()