#32qam numpy
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp_signal

# =========================================================================
# 路径配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results_32qam_numpy', 'figure')
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
# 纯 NumPy 自适应 DSP 算法核心 (慢速 32QAM 版)
# =========================================================================
def qam32_lms_equalizer_numpy(rx, tx_train, const, taps=5, lr_da=2e-3, lr_dd=5e-4, train_len=10000):
    n_syms = rx.shape[0]
    pad = taps // 2
    rx_pad = np.pad(rx, ((pad, pad), (0, 0)), mode='constant')
    
    W = np.zeros((2, 2, taps), dtype=complex)
    W[0, 0, taps//2] = 1.0
    W[1, 1, taps//2] = 1.0
    
    out = np.zeros_like(rx, dtype=complex)
    
    # 死亡 for 循环：要在 Python 里跑 13 万次！
    for i in range(n_syms):
        x = rx_pad[i : i+taps, :]
        x = x[::-1, :] 
        
        y0 = np.sum(W[0, 0] * x[:, 0] + W[0, 1] * x[:, 1])
        y1 = np.sum(W[1, 0] * x[:, 0] + W[1, 1] * x[:, 1])
        y = np.array([y0, y1])
        out[i] = y
        
        if i < train_len:
            d = tx_train[i]
            lr = lr_da
        else:
            # 极其耗时的 32QAM 全遍历复平面距离硬判决
            dist0 = np.abs(y0 - const)
            dist1 = np.abs(y1 - const)
            d = np.array([const[np.argmin(dist0)], const[np.argmin(dist1)]])
            lr = lr_dd
            
        e = d - y
        
        W[0, 0] += lr * e[0] * np.conj(x[:, 0])
        W[0, 1] += lr * e[0] * np.conj(x[:, 1])
        W[1, 0] += lr * e[1] * np.conj(x[:, 0])
        W[1, 1] += lr * e[1] * np.conj(x[:, 1])
        
    return out

def dd_pll_numpy(rx_syms, const, mu=0.015):
    """纯 NumPy 版判决引导锁相环 (DD-PLL)"""
    n = len(rx_syms)
    out = np.zeros(n, dtype=complex)
    phi = 0.0
    for i in range(n):
        y_rot = rx_syms[i] * np.exp(-1j * phi)
        # 硬判决找到最近的理想星座点
        idx = np.argmin(np.abs(y_rot - const))
        d = const[idx]
        err = np.angle(y_rot * np.conj(d))
        phi += mu * err
        out[i] = y_rot
    return out

# =========================================================================
# 主仿真逻辑
# =========================================================================
def main():
    start_time = time.time()

    print("Step 1: 32QAM Signal Generation (NumPy)...")
    num_syms = 131072 
    baud_rate = 32e9 
    
    # 手动精细生成 32QAM 十字形星座图 (Cross-QAM)
    grid = np.array([-5, -3, -1, 1, 3, 5])
    X, Y = np.meshgrid(grid, grid)
    c_raw = (X + 1j*Y).flatten()
    corners = [5+5j, 5-5j, -5+5j, -5-5j]
    c_32qam = np.array([pt for pt in c_raw if pt not in corners])
    const = np.array(c_32qam, dtype=complex)
    const = const / np.sqrt(np.mean(np.abs(const)**2))

    np.random.seed(42)
    tx_indices = np.random.randint(0, 32, (num_syms, 2))
    tx_syms = const[tx_indices]

    h_rrc = get_safe_rrc_taps(sps=2, num_taps=65, beta=0.1)
    tx_sig = sp_signal.upfirdn(h_rrc, tx_syms, up=2, axis=0)
    delay = 32
    rx_2sps = tx_sig[delay : delay + num_syms * 2]

    print("Step 2: Realistic Channel Impairments...")
    theta = np.pi / 4 * 0.6
    J_rot = np.array([[np.cos(theta), np.sin(theta)],
                       [-np.sin(theta), np.cos(theta)]])
    rx_2sps = rx_2sps @ J_rot.T
    
    lw = 10e3  
    sigma = np.sqrt(2 * np.pi * lw * (1.0 / (baud_rate * 2)))
    # 注意避免 *= 原地赋值报错
    phase_noise = np.exp(1j * np.cumsum(sigma * np.random.standard_normal((rx_2sps.shape[0], 1)), axis=0))
    rx_2sps = rx_2sps * phase_noise
    
    rx_2sps = apply_cd_numpy(rx_2sps, baud_rate, sps=2, D=17.0, L=80.0)

    noise = (np.random.standard_normal(rx_2sps.shape) + 
             1j * np.random.standard_normal(rx_2sps.shape)) * 0.02
    rx_2sps += noise

    print("Generating Pre-DSP Plots...")
    plot_psd(rx_2sps, fs=baud_rate*2, name='PSD Before DSP', filename='01_psd_before_dsp.png')
    plot_eye(rx_2sps, sps=2, name='32QAM Eye Before DSP', filename='02_eye_before_dsp.png')

    print("Step 2.5: Static CDC & Rx Matched Filtering...")
    rx_2sps = apply_cd_numpy(rx_2sps, baud_rate, sps=2, D=17.0, L=-80.0)
    rx_mf = sp_signal.upfirdn(h_rrc, rx_2sps, up=1, axis=0)
    rx_2sps = rx_mf[delay : delay + num_syms * 2]
    
    plot_eye(rx_2sps, sps=2, name='32QAM Eye After CDC & MF', filename='02.5_eye_after_cdc_mf.png')

    print(f"Step 3: NumPy DA/DD-LMS Equalization ({num_syms} symbols!)...")
    print("   [WARNING] This pure Python loop will take several minutes to run. Please wait...")
    rx_1sps = rx_2sps[::2] 
    rx_1sps /= np.sqrt(np.mean(np.abs(rx_1sps)**2, axis=0))

    # 漫长的等待阶段开始
    t0_lms = time.time()
    rx_eq = qam32_lms_equalizer_numpy(rx_1sps, tx_syms, const, taps=5, train_len=10000)
    print(f"   --> LMS Equalization finished in {time.time() - t0_lms:.2f} seconds!")

    print("Step 4: Discarding convergence transient & Normalization...")
    start_idx = 15000 
    rx_eq_cut = rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + rx_eq_cut.shape[0]]

    rx_eq_cut /= np.sqrt(np.mean(np.abs(rx_eq_cut)**2, axis=0))

    print("Step 5: Decision-Directed Carrier Phase Recovery (DD-PLL)...")
    t0_pll = time.time()
    rx_cpr = np.zeros_like(rx_eq_cut)
    for i in range(2):
        rx_cpr[:, i] = dd_pll_numpy(rx_eq_cut[:, i], const, mu=0.015)
    print(f"   --> DD-PLL finished in {time.time() - t0_pll:.2f} seconds!")

    print("Step 6: Performance Evaluation & Alignment...")
    aligned_list = []
    snr_vals = []
    for i in range(2):
        snr_v, aligned_sig = get_snr_numpy(tx_final[:, i], rx_cpr[:, i])
        print(f"   Pol {i} Final SNR: {snr_v:.2f} dB")
        aligned_list.append(aligned_sig)
        snr_vals.append(snr_v)
    
    rx_aligned_all = np.stack(aligned_list, axis=1)

    print("Generating Final DSP Plots & Diagnostics...")
    
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(rx_eq_cut[-8000:, 0].real, rx_eq_cut[-8000:, 0].imag, s=1, alpha=0.3, color='tab:blue')
    ax1.set_title("1. After NumPy LMS (Ring Structure)")
    ax1.axis('equal'); ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(rx_aligned_all[-8000:, 0].real, rx_aligned_all[-8000:, 0].imag, s=1, alpha=0.3, color='tab:green')
    ax2.set_title(f"2. Final 32QAM\nSNR: {float(snr_vals[0]):.2f} dB")
    ax2.axis('equal')
    ax2.scatter(const.real, const.imag, c='red', marker='x', s=30, alpha=0.9) 
    ax2.grid(True, linestyle=':', alpha=0.6)

    phase_diff = np.unwrap(np.angle(rx_cpr[-2000:, 0] * np.conj(rx_eq_cut[-2000:, 0])))
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
    print(f"NumPy 32QAM ")
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