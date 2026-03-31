import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.signal as sp_signal
import numpy as np

# =========================================================================
# 导入 Commplax 模块
# =========================================================================
from commplax import adaptive_kernel as ak
from commplax.equalizer import CPR
from commplax import sym_map

# =========================================================================
# 路径配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results_pam4', 'figure')
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
        for angle in [0, jnp.pi]: # PAM4 只有 0 和 180 度翻转
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
# 【终极核心】：原生 JAX 手写 DA/DD-LMS 均衡器 (破除 CMA 奇点)
# =========================================================================
@jax.jit(static_argnames=['taps', 'train_len'])
def pam4_lms_equalizer(rx, tx_train, const, taps=3, lr_da=5e-3, lr_dd=1e-3, train_len=5000):
    n_syms = rx.shape[0]
    pad = taps // 2
    rx_pad = jnp.pad(rx, ((pad, pad), (0, 0)))
    
    def step(W, i):
        # 提取滑动窗口并翻转卷积
        x = jax.lax.dynamic_slice(rx_pad, (i, 0), (taps, 2))
        x = jnp.flip(x, axis=0)
        
        # MIMO 线性组合
        y0 = jnp.sum(W[0, 0] * x[:, 0] + W[0, 1] * x[:, 1])
        y1 = jnp.sum(W[1, 0] * x[:, 0] + W[1, 1] * x[:, 1])
        y = jnp.array([y0, y1])
        
        # --- 误差计算引擎 ---
        # 1. 训练模式 (DA): 用已知的真实发送符号
        d_da = tx_train[i]
        
        # 2. 判决模式 (DD): 找离当前 y 最近的理想 PAM4 点
        dist0 = jnp.abs(y0 - const)
        dist1 = jnp.abs(y1 - const)
        d_dd = jnp.array([const[jnp.argmin(dist0)], const[jnp.argmin(dist1)]])
        
        # 3. 动态切换：前 5000 个点用 DA 训练解开偏振，之后用 DD 盲跟
        is_train = i < train_len
        d = jnp.where(is_train, d_da, d_dd)
        lr = jnp.where(is_train, lr_da, lr_dd)
        
        e = d - y
        
        # 抽头权重更新
        W = W.at[0, 0].add(lr * e[0] * jnp.conj(x[:, 0]))
        W = W.at[0, 1].add(lr * e[0] * jnp.conj(x[:, 1]))
        W = W.at[1, 0].add(lr * e[1] * jnp.conj(x[:, 0]))
        W = W.at[1, 1].add(lr * e[1] * jnp.conj(x[:, 1]))
        
        return W, y
        
    # 初始化 MIMO 权重矩阵
    W0 = jnp.zeros((2, 2, taps), dtype=jnp.complex64)
    W0 = W0.at[0, 0, taps//2].set(1.0)
    W0 = W0.at[1, 1, taps//2].set(1.0)
    
    _, rx_eq = jax.lax.scan(step, W0, jnp.arange(n_syms))
    return rx_eq

# =========================================================================
# 主仿真逻辑
# =========================================================================
def main():
    start_time = time.time()

    print("Step 1: DP-PAM4 Signal Generation & Pulse Shaping...")
    num_syms = 32768
    baud_rate = 32e9 
    
    # 获取归一化 PAM4 星座点
    const = jnp.asarray(sym_map.const('PAM4', norm=True))
    key = jax.random.PRNGKey(42)
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 4)
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

    noise = (jax.random.normal(jax.random.PRNGKey(123), rx_2sps.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(124), rx_2sps.shape)) * 0.04
    rx_2sps += noise

    print("Generating Pre-DSP Plots...")
    plot_psd(rx_2sps, fs=baud_rate*2, name='PSD Before DSP', filename='01_psd_before_dsp.png')
    plot_eye(rx_2sps, sps=2, name='Eye Before DSP', filename='02_eye_before_dsp.png')

    print("Step 2.5: Static CDC & Rx Matched Filtering...")
    rx_2sps = apply_chromatic_dispersion(rx_2sps, baud_rate, sps=2, D=17.0, L=-80.0)
    rx_mf = sp_signal.upfirdn(np.array(h_rrc), np.array(rx_2sps), up=1, axis=0)
    rx_2sps = jnp.array(rx_mf[delay : delay + num_syms * 2])
    
    plot_eye(rx_2sps, sps=2, name='Eye After CDC & MF', filename='02.5_eye_after_cdc_mf.png')

    print("Step 3: Custom JAX DA/DD-LMS Equalization (Bypassing CMA Singularity)...")
    rx_1sps = rx_2sps[::2] 
    rx_1sps = rx_1sps / jnp.sqrt(jnp.mean(jnp.abs(rx_1sps)**2, axis=0))

    # 使用极致高效的混合 LMS 完美分离偏振并锁定实轴
    rx_eq = pam4_lms_equalizer(rx_1sps, tx_syms, const, taps=3)

    print("Step 4: Discarding convergence transient & Normalization...")
    start_idx = 6000 
    rx_eq_cut = rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + rx_eq_cut.shape[0]]

    rx_eq_cut = rx_eq_cut / jnp.sqrt(jnp.mean(jnp.abs(rx_eq_cut)**2, axis=0))

    print("Step 5: Carrier Phase Recovery...")
    rx_cpr_list = []
    for i in range(2):
        cpr_module = CPR(kernel=ak.cpr_4thpower_pll(mu=0.01)) 
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

    print("Generating Final DSP Plots & Diagnostics...")
    
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(rx_eq_cut[-4000:, 0].real, rx_eq_cut[-4000:, 0].imag, s=1, alpha=0.5, color='tab:blue')
    ax1.set_title("1. After JAX LMS (Perfect I-channel)")
    ax1.axis('equal'); ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(rx_aligned_all[-4000:, 0].real, rx_aligned_all[-4000:, 0].imag, s=1, alpha=0.5, color='tab:green')
    ax2.set_title(f"2. Final DP-PAM4\nSNR: {float(snr_vals[0]):.2f} dB")
    ax2.axis('equal')
    ax2.scatter(const.real, jnp.zeros_like(const.real), c='red', marker='x', s=40, alpha=0.9) 
    ax2.grid(True, linestyle=':', alpha=0.6)

    phase_diff = jnp.unwrap(jnp.angle(rx_cpr_all[-1000:, 0] * jnp.conj(rx_eq_cut[-1000:, 0])))
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(phase_diff, color='tab:orange', linewidth=1)
    ax3.set_title("3. Phase Track Residual")
    ax3.set_xlabel("Symbols"); ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, '00_diagnostic_plot.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        ax.scatter(rx_aligned_all[-4000:, i].real, rx_aligned_all[-4000:, i].imag, s=1, alpha=0.5)
        ax.set_title(f'PAM4 Constellation Pol {i}\nSNR: {float(snr_vals[i]):.2f} dB')
        ax.set_ylim([-0.5, 0.5]) 
        ax.scatter(const.real, jnp.zeros_like(const.real), c='red', marker='x', s=40, alpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.savefig(os.path.join(FIGURE_PATH, '03_constellations.png'), dpi=300)
    plt.close()

    plot_psd(rx_aligned_all, fs=baud_rate, name='PSD After DSP', filename='04_psd_after_dsp.png')
    plot_eye(rx_aligned_all, sps=1, name='Eye After DSP (Recovered)', amplitude_limit=2.0, filename='05_eye_after_dsp.png')

    end_time = time.time()
    
    print("\n" + "="*55)
    print(f"DP-PAM4 仿真完美结束！")
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