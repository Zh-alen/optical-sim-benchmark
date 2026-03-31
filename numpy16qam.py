import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp_signal

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results_numpy', 'figure')
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# =========================================================================
# 基础工具函数 (NumPy 版)
# =========================================================================
def get_snr_numpy(ref, rec):
    rec = rec / (np.sqrt(np.mean(np.abs(rec)**2)) + 1e-12)
    ref = ref / (np.sqrt(np.mean(np.abs(ref)**2)) + 1e-12)
    best_snr = -100.0
    best_rec = rec
    for shift in range(-5, 6):
        r_sh = np.roll(rec, shift)
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            temp_rec = r_sh * np.exp(-1j * angle)
            phase_offset = np.angle(np.mean(temp_rec * np.conj(ref)))
            temp_rec_final = temp_rec * np.exp(-1j * phase_offset)
            mse = np.mean(np.abs(ref - temp_rec_final)**2)
            snr = 10 * np.log10(1.0 / (mse + 1e-12))
            if snr > best_snr:
                best_snr = snr
                best_rec = temp_rec_final
    return best_snr, best_rec

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

def apply_cd_numpy(sig, baud_rate, sps, D=17.0, L=80.0, lambda_=1550e-9):
    fs = baud_rate * sps
    c = 299792458.0
    beta2 = -(lambda_**2) / (2 * np.pi * c) * (D * 1e-6)
    f = np.fft.fftfreq(sig.shape[0], 1/fs)
    omega = 2 * np.pi * f
    H = np.exp(-1j * (beta2 / 2) * (L * 1e3) * omega**2)
    sig_f = np.fft.fft(sig, axis=0)
    return np.fft.ifft(sig_f * H[:, np.newaxis], axis=0)

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

def plot_eye(sig, sps=2, name='Eye Diagram', amplitude_limit=1.5, filename='eye_diagram.png'):
    plt.figure(figsize=(12, 5))
    num_points = int(2 * sps) + 1 
    num_traces = 800
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        start = 10000 if sig.shape[0] > 10000 else 0
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
# 核心算法复刻: MIMO-CMA & CPR
# =========================================================================

def cma_equalizer_numpy(signal, taps=3, lr=2e-3, R2=1.32):
    n_samples = signal.shape[0]
    w = np.zeros((2, 2, taps), dtype=complex)
    w[0, 0, taps//2] = 1.0
    w[1, 1, taps//2] = 1.0
    
    out = np.zeros_like(signal, dtype=complex)
    for i in range(taps, n_samples):
        x = signal[i-taps+1 : i+1, :].T 
        x = x[:, ::-1] 
        
        y = np.zeros(2, dtype=complex)
        y[0] = np.sum(w[0, 0] * x[0] + w[0, 1] * x[1])
        y[1] = np.sum(w[1, 0] * x[0] + w[1, 1] * x[1])
        out[i] = y
        
        for p in range(2):
            err = R2 - np.abs(y[p])**2
            w[p, 0] += lr * err * y[p] * np.conj(x[0])
            w[p, 1] += lr * err * y[p] * np.conj(x[1])
            
    return out

def cpr_4th_power_numpy(sig, mu=0.02):
    n = len(sig)
    out = np.zeros(n, dtype=complex)
    phi = 0.0
    for i in range(n):
        out[i] = sig[i] * np.exp(-1j * phi)
        err = np.angle(out[i]**4) / 4.0
        phi += mu * err
    return out

# =========================================================================
# 主仿真逻辑
# =========================================================================
def main():
    start_time = time.time()
    print("Step 1: Signal Generation (NumPy)...")
    
    num_syms = 32768
    baud_rate = 32e9
    points = np.array([-3, -1, 1, 3])
    const = np.array([complex(i, j) for i in points for j in points])
    const /= np.sqrt(np.mean(np.abs(const)**2))
    
    np.random.seed(42)
    tx_indices = np.random.randint(0, 16, (num_syms, 2))
    tx_syms = const[tx_indices]

    h_rrc = get_safe_rrc_taps(sps=2, num_taps=65, beta=0.1)
    tx_sig = sp_signal.upfirdn(h_rrc, tx_syms, up=2, axis=0)
    delay = 32
    rx_2sps = tx_sig[delay : delay + num_syms * 2]

    print("Step 2: Channel Impairments...")
    theta = np.pi / 4 * 0.6
    J_rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rx_2sps = rx_2sps @ J_rot.T
    
    lw = 10e3
    sigma = np.sqrt(2 * np.pi * lw * (1.0 / (baud_rate * 2)))
    phase_noise = np.exp(1j * np.cumsum(sigma * np.random.standard_normal((rx_2sps.shape[0], 1)), axis=0))
    rx_2sps *= phase_noise
    
    rx_2sps = apply_cd_numpy(rx_2sps, baud_rate, sps=2, D=17.0, L=80.0)
    
    noise = (np.random.standard_normal(rx_2sps.shape) + 1j * np.random.standard_normal(rx_2sps.shape)) * 0.04
    rx_2sps += noise

    # 【新增图像】：DSP 前的信道状态
    print("Generating Pre-DSP Plots...")
    plot_psd(rx_2sps, fs=baud_rate*2, name='PSD Before DSP', filename='01_psd_before_dsp.png')
    plot_eye(rx_2sps, sps=2, name='Eye Before DSP', filename='02_eye_before_dsp.png')

    print("Step 3: Rx DSP (CDC + CMA + CPR)...")
    rx_2sps = apply_cd_numpy(rx_2sps, baud_rate, sps=2, D=17.0, L=-80.0)
    rx_mf = sp_signal.upfirdn(h_rrc, rx_2sps, up=1, axis=0)
    rx_2sps = rx_mf[delay : delay + num_syms * 2]
    
    # 【新增图像】：静态色散补偿与匹配滤波后的眼图
    plot_eye(rx_2sps, sps=2, name='Eye After CDC & MF', filename='02.5_eye_after_cdc_mf.png')
    
    rx_1sps = rx_2sps[::2]
    rx_1sps /= np.sqrt(np.mean(np.abs(rx_1sps)**2, axis=0))

    print("   Running CMA Equalizer (Pass 1 & 2)...")
    rx_eq = cma_equalizer_numpy(rx_1sps, taps=3, lr=2e-3)
    rx_eq = cma_equalizer_numpy(rx_eq, taps=3, lr=1e-3)

    start_idx = 3000
    rx_eq_cut = rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + rx_eq_cut.shape[0]]
    rx_eq_cut /= np.sqrt(np.mean(np.abs(rx_eq_cut)**2, axis=0))

    print("   Running CPR (Carrier Phase Recovery)...")
    rx_cpr = np.zeros_like(rx_eq_cut)
    for i in range(2):
        rx_cpr[:, i] = cpr_4th_power_numpy(rx_eq_cut[:, i], mu=0.02)

    print("Step 4: Evaluation & Alignment...")
    aligned_list = []
    snr_vals = []
    for i in range(2):
        snr_v, aligned_sig = get_snr_numpy(tx_final[:, i], rx_cpr[:, i])
        print(f"  Pol {i} SNR: {snr_v:.2f} dB")
        snr_vals.append(snr_v)
        aligned_list.append(aligned_sig)
    
    rx_final = np.stack(aligned_list, axis=1)

    # =========================================================================
    # 生成所有图像
    # =========================================================================
    print("Generating Final DSP Plots & Diagnostics...")
    
    # 【新增图像】：三联诊断图
    plt.figure(figsize=(15, 5))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(rx_eq_cut[-4000:, 0].real, rx_eq_cut[-4000:, 0].imag, s=1, alpha=0.5, color='tab:blue')
    ax1.set_title("1. After CMA (Before CPR)")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(rx_final[-4000:, 0].real, rx_final[-4000:, 0].imag, s=1, alpha=0.5, color='tab:green')
    ax2.set_title(f"2. After CPR & Align\nFinal SNR: {float(snr_vals[0]):.2f} dB")
    ax2.axis('equal')
    ax2.scatter(const.real, const.imag, c='red', marker='x', s=20, alpha=0.7) 
    ax2.grid(True, linestyle=':', alpha=0.6)

    phase_diff = np.unwrap(np.angle(rx_cpr[-1000:, 0] * np.conj(rx_eq_cut[-1000:, 0])))
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(phase_diff, color='tab:orange', linewidth=1)
    ax3.set_title("3. Phase Track (Last 1000 syms)")
    ax3.set_xlabel("Symbols")
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, '00_diagnostic_plot.png'), dpi=300)
    plt.close()

    # 常规双偏振星座图
    plt.figure(figsize=(10, 5))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        ax.scatter(rx_final[-4000:, i].real, rx_final[-4000:, i].imag, s=1, alpha=0.5)
        ax.scatter(const.real, const.imag, c='red', marker='x', s=20, alpha=0.7)
        ax.set_title(f'Constellation Pol {i}\nSNR: {snr_vals[i]:.2f} dB')
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, '03_constellations.png'), dpi=300)
    plt.close()

    # 【新增图像】：DSP 后频谱与眼图
    plot_psd(rx_final, fs=baud_rate, name='PSD After DSP', filename='04_psd_after_dsp.png')
    plot_eye(rx_final, sps=1, name='Eye After DSP (Recovered)', amplitude_limit=1.5, filename='05_eye_after_dsp.png')

    end_time = time.time()
    
    print("\n" + "="*55)
    print(f"NumPy 仿真完美结束！")
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