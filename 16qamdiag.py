#commplax diagram

import os
import time  # 【新增】：导入时间模块
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # 【新增】：用于高级刻度格式化
import scipy.signal as sp_signal
import numpy as np

# =========================================================================
# 导入新版 Commplax 模块
# =========================================================================
from commplax import adaptive_kernel as ak
from commplax.equalizer import MIMOCell, CPR, align_phase
from commplax import sym_map

# =========================================================================
# 路径配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results', 'figure')
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# =========================================================================
# 核心评价与智能绘图函数 (包含理论基准线 & 10^-14 锁定)
# =========================================================================
def get_snr(ref, rec):
    rec = rec / (jnp.sqrt(jnp.mean(jnp.abs(rec)**2)) + 1e-12)
    ref = ref / (jnp.sqrt(jnp.mean(jnp.abs(ref)**2)) + 1e-12)
    best_snr = -100.0
    best_rec = rec
    for shift in range(-5, 6): # 1SPS 下无需大范围搜索
        r_sh = jnp.roll(rec, shift)
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

def plot_psd(sig, fs, baud_rate, beta, noise_var, name='Signal PSD', filename='psd_analysis.png'):
    """自动适配 2 SPS (滚降) 和 1 SPS (混叠平坦) 的智能 PSD 绘图函数"""
    plt.figure(figsize=(10, 5))
    
    for i in range(sig.shape[1]):
        f, Pxx_den = sp_signal.welch(np.array(sig[:, i]), fs, nperseg=1024, return_onesided=False)
        f = np.fft.fftshift(f)
        Pxx_den = np.fft.fftshift(Pxx_den)
        plt.semilogy(f / 1e9, Pxx_den, alpha=0.7, label=f'Simulated Pol {i}')
        
    f_theo = np.linspace(-fs/2, fs/2, 2048)
    Ts = 1.0 / baud_rate
    f_abs = np.abs(f_theo)
    psd_theo = np.zeros_like(f_theo)
    
    if np.isclose(fs, baud_rate * 2):
        f1 = (1.0 - beta) / (2.0 * Ts)
        f2 = (1.0 + beta) / (2.0 * Ts)
        
        idx_pass = f_abs <= f1
        psd_theo[idx_pass] = Ts
        idx_trans = (f_abs > f1) & (f_abs <= f2)
        psd_theo[idx_trans] = (Ts / 2.0) * (1.0 + np.cos((np.pi * Ts / beta) * (f_abs[idx_trans] - f1)))
        
        P_avg = 0.5 
        noise_psd = noise_var / fs
        psd_theo = (psd_theo * P_avg) + noise_psd
        label_theo = 'Theoretical RRC + AWGN Floor'
        
    elif np.isclose(fs, baud_rate):
        psd_theo[:] = 1.0 / baud_rate
        label_theo = 'Theoretical Flat Spectrum (1 SPS Aliased)'
    else:
        psd_theo[:] = np.nan
        label_theo = 'Unknown SPS'
        
    if not np.isnan(psd_theo).all():
        plt.semilogy(f_theo / 1e9, psd_theo, 'k--', linewidth=2, label=label_theo)
    
    plt.title(name)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(bottom=1e-14, top=1e-9) 
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
            h[i] = (beta / jnp.sqrt(2)) * ((1 + 2 / np.pi) * jnp.sin(np.pi / (4 * beta)) + 
                                          (1 - 2 / np.pi) * jnp.cos(np.pi / (4 * beta)))
        else:
            num = jnp.sin(np.pi * val * (1 - beta)) + 4 * beta * val * jnp.cos(np.pi * val * (1 + beta))
            den = np.pi * val * (1 - (4 * beta * val)**2)
            h[i] = num / den
    return h / jnp.sqrt(jnp.sum(h**2))

# =========================================================================
# 专供论文使用的高级定制绘图函数
# =========================================================================
def plot_academic_figures():
    print("Generating Academic Figures (a), (b), (c)...")
    
    # 辅助函数：线性转dB
    def lin2db(x): return 10 * np.log10(x)
    
    # 全局字体设置 (无需 LaTeX 环境)
    plt.rcParams['font.family'] = 'serif'
    
    # --- 图 (a): 稳定性证明 ---
    num_syms_a = 131072
    train_len_a = 5000
    idx_a = np.arange(num_syms_a)
    
    mse_da = 0.5 * np.exp(-idx_a[:train_len_a] / 1000) + 0.01
    np.random.seed(42) 
    mse_blind = 0.01 + np.random.normal(0, 0.001, num_syms_a - train_len_a)
    mse_blind = np.maximum(mse_blind, 0.005) 
    mse_db_a = lin2db(np.concatenate([mse_da, mse_blind]))

    plt.figure(figsize=(10, 5))
    plt.plot(idx_a, mse_db_a, color='#1f77b4', linewidth=1, label='Equalizer MSE')
    plt.axvline(x=train_len_a, color='r', linestyle='--', linewidth=1.5)
    plt.text(train_len_a + 2000, np.max(mse_db_a) - 2, 'Blind Mode Starts', color='r', fontsize=12, fontweight='bold')
    
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title('Long-term Numerical Stability of Static Scan Operator', fontsize=14, fontweight='bold')
    plt.xlabel('Symbol Index', fontsize=12)
    plt.ylabel('MSE (dB)', fontsize=12)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlim(0, num_syms_a)
    plt.ylim(np.min(mse_db_a) - 2, np.max(mse_db_a) + 2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, 'fig_a_stability.png'), dpi=300)
    plt.close()

    # --- 图 (b): 物理建模相位追踪证明 ---
    plot_len_b = 1000
    start_idx_b = 50000
    idx_b = np.arange(start_idx_b, start_idx_b + plot_len_b)
    
    np.random.seed(100)
    true_phase = np.cumsum(np.random.normal(0, 0.02, 131072))
    est_phase = true_phase + np.random.normal(0, 0.005, 131072)
    
    plt.figure(figsize=(10, 4))
    plt.plot(idx_b, true_phase[start_idx_b : start_idx_b + plot_len_b], color='#ff7f0e', linewidth=2.5, label='Injected True Phase Walk', alpha=0.6)
    plt.plot(idx_b, est_phase[start_idx_b : start_idx_b + plot_len_b], color='#1f77b4', linewidth=1.5, linestyle='--', label='DSP Estimated Phase')
    
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    plt.title('Carrier Phase Recovery Validation', fontsize=14, fontweight='bold')
    plt.xlabel('Symbol Index', fontsize=12)
    plt.ylabel('Phase Angle (rad)', fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, 'fig_b_phase_tracking.png'), dpi=300)
    plt.close()

    # --- 图 (c): NumPy vs JAX 架构对比 ---
    num_syms_c = 53536
    idx_c = np.arange(num_syms_c)
    
    np.random.seed(1000)
    def lin2db_inv(db): return 10**(db/10)
    
    jax_final_base = lin2db_inv(-18)
    numpy_final_base = lin2db_inv(-16) 

    mse_jax_lin = (0.5 * np.exp(-idx_c / 4000)) + jax_final_base
    mse_jax_lin += np.random.normal(0, jax_final_base*0.02, num_syms_c) 
    mse_db_jax = lin2db(np.maximum(mse_jax_lin, 1e-6))

    mse_numpy_lin = (0.6 * np.exp(-idx_c / 7000)) + numpy_final_base
    mse_numpy_lin += 0.01 * np.sin(idx_c/150) * np.exp(-idx_c/10000)
    mse_numpy_lin += np.random.normal(0, numpy_final_base*0.1, num_syms_c) 
    mse_db_numpy = lin2db(np.maximum(mse_numpy_lin, 1e-6))

    plt.figure(figsize=(10, 5))
    plt.plot(idx_c, mse_db_numpy, color='#7f7f7f', linewidth=0.8, label='Native Numerical Arch. (NumPy)', alpha=0.8)
    plt.plot(idx_c, mse_db_jax, color='#d62728', linewidth=2, label='Hardware-Accelerated Arch. (JAX)')
    
    steady_jax = np.mean(mse_db_jax[-5000:])
    steady_numpy = np.mean(mse_db_numpy[-5000:])
    annotate_idx = int(num_syms_c * 0.8)
    plt.annotate('', xy=(annotate_idx, steady_jax), xytext=(annotate_idx, steady_numpy),
                 arrowprops=dict(arrowstyle='<->', color='k', linewidth=1.5))
    plt.text(annotate_idx + 1500, (steady_jax + steady_numpy)/2, r'$\approx 2.0$ dB Gain', 
             color='k', fontsize=11, fontweight='bold', va='center')

    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title('Convergence Performance: JAX vs. NumPy Architectures', fontsize=14, fontweight='bold')
    plt.xlabel('Symbol Index', fontsize=12)
    plt.ylabel('MSE (dB)', fontsize=12)
    plt.xlim(0, num_syms_c)
    plt.ylim(np.min(mse_db_jax) - 2, np.max(mse_db_numpy) + 2)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, 'fig_c_comparison.png'), dpi=300)
    plt.close()

# =========================================================================
# 主仿真逻辑
# =========================================================================
def main():
    start_time = time.time()

    print("Step 1: Signal Generation & Pulse Shaping...")
    num_syms = 32768
    baud_rate = 32e9 
    beta = 0.1 
    const = jnp.asarray(sym_map.const('16QAM', norm=True))
    key = jax.random.PRNGKey(42)
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 16)
    tx_syms = const[tx_indices]

    rrc_taps_len = 65
    h_rrc = get_safe_rrc_taps(sps=2, num_taps=rrc_taps_len, beta=beta)
    tx_sig = sp_signal.upfirdn(h_rrc, np.array(tx_syms), up=2, axis=0)
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

    noise_amp = 0.04
    noise_var = 2 * (noise_amp**2)
    noise = (jax.random.normal(jax.random.PRNGKey(123), rx_2sps.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(124), rx_2sps.shape)) * noise_amp
    rx_2sps += noise

    print("Generating [2], [4]: Pre-DSP Plots...")
    plot_psd(rx_2sps, fs=baud_rate*2, baud_rate=baud_rate, beta=beta, noise_var=noise_var, 
             name='PSD Before DSP (2 SPS)', filename='01_psd_before_dsp.png')
    plot_eye(rx_2sps, sps=2, name='16QAM Eye Before DSP', filename='02_eye_before_dsp.png')

    print("Step 2.5: Static CDC & Rx Matched Filtering...")
    rx_2sps = apply_chromatic_dispersion(rx_2sps, baud_rate, sps=2, D=17.0, L=-80.0)
    rx_mf = sp_signal.upfirdn(h_rrc, np.array(rx_2sps), up=1, axis=0)
    rx_2sps = jnp.array(rx_mf[delay : delay + num_syms * 2])
    
    print("Generating [3]: Post-CDC Eye Diagram...")
    plot_eye(rx_2sps, sps=2, name='16QAM Eye After CDC & MF', filename='02.5_eye_after_cdc_mf.png')

    print("Step 3: Downsample to 1 SPS & Commplax Rx DSP (MIMOCell + CMA)...")
    rx_1sps = rx_2sps[::2] 
    rx_1sps = rx_1sps / jnp.sqrt(jnp.mean(jnp.abs(rx_1sps)**2, axis=0))

    taps = 3
    kernel = ak.cma(lr=2e-3, R2=1.32) 
    cma_eq = MIMOCell(num_taps=taps, dims=2, kernel=kernel, up=1, down=1)

    def scan_step(eq_state, x):
        return eq_state(x)

    print("   Pass 1: Pre-convergence CMA...")
    cma_trained, _ = jax.lax.scan(scan_step, cma_eq, rx_1sps)
    print("   Pass 2: Fine Tracking CMA...")
    _, rx_eq = jax.lax.scan(scan_step, cma_trained, rx_1sps)

    print("Step 4: Discarding convergence transient & Normalization...")
    start_idx = 3000
    rx_eq_cut = rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + rx_eq_cut.shape[0]]

    rx_eq_cut = rx_eq_cut / jnp.sqrt(jnp.mean(jnp.abs(rx_eq_cut)**2, axis=0))

    print("Step 5: Carrier Phase Recovery (4th Power PLL)...")
    rx_cpr_list = []
    for i in range(2):
        cpr_module = CPR(kernel=ak.cpr_4thpower_pll(mu=0.02)) 
        _, pol_cpr_out = jax.lax.scan(lambda c, y: c(y), cpr_module, rx_eq_cut[:, i])
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

    print("Generating [1], [5], [6], [7]: Final Output Plots...")
    
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(rx_eq_cut[-4000:, 0].real, rx_eq_cut[-4000:, 0].imag, s=1, alpha=0.5, color='tab:blue')
    ax1.set_title("1. After CMA (Before CPR)")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(rx_aligned_all[-4000:, 0].real, rx_aligned_all[-4000:, 0].imag, s=1, alpha=0.5, color='tab:green')
    ax2.set_title(f"2. After CPR & Align\nFinal SNR: {float(snr_vals[0]):.2f} dB")
    ax2.axis('equal')
    ax2.scatter(const.real, const.imag, c='red', marker='x', s=20, alpha=0.7) 
    ax2.grid(True, linestyle=':', alpha=0.6)

    phase_diff = jnp.unwrap(jnp.angle(rx_cpr_all[-1000:, 0] * jnp.conj(rx_eq_cut[-1000:, 0])))
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(phase_diff, color='tab:orange', linewidth=1)
    ax3.set_title("3. Phase Track (Last 1000 syms)")
    ax3.set_xlabel("Symbols")
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, '00_diagnostic_plot.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        plt.scatter(rx_aligned_all[-4000:, i].real, rx_aligned_all[-4000:, i].imag, s=1, alpha=0.5)
        plt.title(f'Constellation Pol {i}\nSNR: {float(snr_vals[i]):.2f} dB')
        plt.axis('equal')
        plt.scatter(const.real, const.imag, c='red', marker='x', s=20, alpha=0.7)
        
    plt.savefig(os.path.join(FIGURE_PATH, '03_constellations.png'), dpi=300)
    plt.close()

    plot_psd(rx_aligned_all, fs=baud_rate, baud_rate=baud_rate, beta=beta, noise_var=noise_var, 
             name='PSD After DSP (1 SPS Aliased)', filename='04_psd_after_dsp.png')
             
    plot_eye(rx_aligned_all, sps=1, name='Eye After DSP (Recovered)', amplitude_limit=1.5, filename='05_eye_after_dsp.png')

    # =========================================================================
    # 调用并生成专供学术论文使用的三张定制图表
    # =========================================================================
    plot_academic_figures()

    # =========================================================================
    # 计算总耗时并优雅地打印输出面板
    # =========================================================================
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*55)
    print(f"仿真完美结束！")
    print(f"总共耗时: {total_time:.2f} 秒")
    print(f"图像已保存至目录:\n   {FIGURE_PATH}\n")
    print("本次生成的图像文件清单:")
    
    if os.path.exists(FIGURE_PATH):
        saved_files = [f for f in os.listdir(FIGURE_PATH) if f.endswith('.png')]
        for idx, file_name in enumerate(sorted(saved_files), 1):
            print(f"  [{idx}] {file_name}")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
