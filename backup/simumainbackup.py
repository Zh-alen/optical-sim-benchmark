import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
# 核心评价与绘图函数
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
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + 
                                          (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * val * (1 - beta)) + 4 * beta * val * np.cos(np.pi * val * (1 + beta))
            den = np.pi * val * (1 - (4 * beta * val)**2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))

# =========================================================================
# 主仿真逻辑
# =========================================================================
def main():
    print("Step 1: Signal Generation & Pulse Shaping...")
    num_syms = 32768
    baud_rate = 32e9 
    const = jnp.asarray(sym_map.const('16QAM', norm=True))
    key = jax.random.PRNGKey(42)
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 16)
    tx_syms = const[tx_indices]

    rrc_taps_len = 65
    h_rrc = get_safe_rrc_taps(sps=2, num_taps=rrc_taps_len, beta=0.1)
    tx_sig = sp_signal.upfirdn(h_rrc, np.array(tx_syms), up=2, axis=0)
    delay = (rrc_taps_len - 1) // 2
    rx_2sps = jnp.array(tx_sig[delay : delay + num_syms * 2])

    print("Step 2: Realistic Channel Impairments...")
    # 偏振旋转
    theta = jnp.pi / 4 * 0.6
    J_rot = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
                       [-jnp.sin(theta), jnp.cos(theta)]])
    rx_2sps = rx_2sps @ J_rot.T
    
    # 激光器相位噪声 (降低到 10kHz 标准水平测试锁相环)
    lw = 10e3  
    Ts = 1.0 / (baud_rate * 2)
    sigma = jnp.sqrt(2 * jnp.pi * lw * Ts)
    phase_steps = sigma * jax.random.normal(jax.random.PRNGKey(99), (rx_2sps.shape[0], 1))
    rx_2sps = rx_2sps * jnp.exp(1j * jnp.cumsum(phase_steps, axis=0))
    
    rx_2sps = apply_chromatic_dispersion(rx_2sps, baud_rate, sps=2, D=17.0, L=80.0)

    # AWGN 噪声
    noise = (jax.random.normal(jax.random.PRNGKey(123), rx_2sps.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(124), rx_2sps.shape)) * 0.04
    rx_2sps += noise

    print("Step 2.5: Static CDC & Rx Matched Filtering...")
    rx_2sps = apply_chromatic_dispersion(rx_2sps, baud_rate, sps=2, D=17.0, L=-80.0)
    rx_mf = sp_signal.upfirdn(h_rrc, np.array(rx_2sps), up=1, axis=0)
    rx_2sps = jnp.array(rx_mf[delay : delay + num_syms * 2])
    

    print("Step 3: Downsample to 1 SPS & Commplax Rx DSP (MIMOCell + CMA)...")
    rx_1sps = rx_2sps[::2] 
    rx_1sps = rx_1sps / jnp.sqrt(jnp.mean(jnp.abs(rx_1sps)**2, axis=0))

    # 抽头数降至 3，仅用于 1 SPS 信号的偏振追踪，拒绝过拟合！
    taps = 3
    kernel = ak.cma(lr=2e-3, R2=1.32) # 抽头少，学习率可以大胆提高
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

    # =========================================================================
    # 生成诊断图
    # =========================================================================
    print("Generating Diagnostic Plots...")
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
    diagnostic_path = os.path.join(FIGURE_PATH, '00_diagnostic_plot.png')
    plt.savefig(diagnostic_path, dpi=300)
    plt.close()
    
    
    print("\n" + "="*50)
    print(f"仿真结束，所有图像已保存至目录:\n{FIGURE_PATH}\n")
    print("本次生成的图像文件清单:")
    
    # 动态读取并打印文件夹下的所有 .png 文件
    saved_files = [f for f in os.listdir(FIGURE_PATH) if f.endswith('.png')]
    for idx, file_name in enumerate(sorted(saved_files), 1):
        print(f"  [{idx}] {file_name}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()