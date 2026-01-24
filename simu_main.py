import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from commplax import adaptive_filter as af
from commplax import sym_map
import scipy.signal as sp_signal
import numpy as np

# =========================================================================
# 路径配置
# =========================================================================
FIGURE_PATH = r'C:\optical_sim_benchmark\results\figure'
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# =========================================================================
# 绘图工具函数
# =========================================================================

def plot_psd(sig, fs, name='Signal PSD', filename='psd_analysis.png'):
    """生成功率谱密度图并保存"""
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
    print(f"  Saved: {filename}")

def plot_eye(sig, sps=2, name='Eye Diagram', amplitude_limit=1.5, filename='eye_diagram.png'):
    """绘制眼图并保存"""
    plt.figure(figsize=(12, 5))
    samples_per_eye = sps * 2
    num_traces = 800
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        start = 10000 if sig.shape[0] > 10000 else 0
        # 确保数据长度足够
        end = min(start + num_traces * samples_per_eye, sig.shape[0])
        data = np.array(sig[start:end, i].real)
        
        # 截取整数倍轨迹
        num_actual_traces = data.size // samples_per_eye
        reshaped = data[:num_actual_traces * samples_per_eye].reshape(-1, samples_per_eye)
        
        t = np.linspace(0, 2, samples_per_eye)
        for trace in reshaped:
            ax.plot(t, trace, 'b-', alpha=0.05, linewidth=0.5)
        
        ax.set_xlim([0, 2])
        ax.set_ylim([-amplitude_limit, amplitude_limit])
        ax.set_title(f'{name} - Pol {i}')
        ax.set_xlabel('Time (Symbol Period)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, filename), dpi=300)
    print(f"  Saved: {filename}")

# =========================================================================
# 主仿真逻辑
# =========================================================================

def main():
    import numpy as np # 用于 welch 等兼容
    
    # 1. Tx: 16QAM 生成
    print("Step 1: Signal Generation...")
    num_syms = 32768
    const = jnp.asarray(sym_map.const('16QAM', norm=True))
    key = jax.random.PRNGKey(42)
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 16)
    tx_syms = const[tx_indices]

    # 2. Channel: 2 SPS 并加噪
    print("Step 2: Channel Impairments...")
    rx_2sps = jnp.zeros((num_syms * 2, 2), dtype=tx_syms.dtype)
    rx_2sps = rx_2sps.at[::2, :].set(tx_syms) 
    noise = (jax.random.normal(jax.random.PRNGKey(123), rx_2sps.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(124), rx_2sps.shape)) * 0.03
    rx_2sps += noise

    # --- 绘图：DSP 前 ---
    print("Plotting Before DSP status...")
    plot_psd(rx_2sps, fs=64e9, name='PSD Before DSP', filename='01_psd_before_dsp.png')
    plot_eye(rx_2sps, sps=2, name='Eye Before DSP', filename='02_eye_before_dsp.png')

    # 3. Rx DSP: MIMO CMA
    print("Step 3: Commplax Rx DSP...")
    taps = 41
    cma_af = af.cma(lr=1e-4, R2=1.32)
    mimo_state = cma_af.init(taps=taps, dims=2)
    rx_framed = af.frame(rx_2sps, taps=taps, sps=2)

    def scan_train(state, inp):
        new_state, _ = cma_af.update(0, state, inp)
        return new_state, None

    def scan_apply(state, inp):
        _, out = cma_af.apply(state, inp)
        sig = out[0] if isinstance(out, (tuple, list)) else out
        return state, jnp.atleast_1d(sig)

    print("  Training & Applying CMA...")
    mimo_state, _ = jax.lax.scan(scan_train, mimo_state, rx_framed)
    _, rx_eq = jax.lax.scan(scan_apply, mimo_state, rx_framed)

    # 4. Evaluation
    print("Step 4: Performance Evaluation...")
    if rx_eq.ndim == 3: rx_eq = jnp.squeeze(rx_eq, axis=1)

    start_idx = num_syms // 3
    rx_final = rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + rx_final.shape[0]]

    def get_snr(ref, rec):
        rec = rec / (jnp.sqrt(jnp.mean(jnp.abs(rec)**2)) + 1e-12)
        ref = ref / (jnp.sqrt(jnp.mean(jnp.abs(ref)**2)) + 1e-12)
        best_snr, best_rec = -100.0, rec
        for angle in [0, jnp.pi/2, jnp.pi, 3*jnp.pi/2]:
            temp_rec = rec * jnp.exp(-1j * angle)
            fine_angle = jnp.angle(jnp.mean(temp_rec * jnp.conj(ref)))
            temp_rec = temp_rec * jnp.exp(-1j * fine_angle)
            mse = jnp.mean(jnp.abs(ref - temp_rec)**2)
            snr = 10 * jnp.log10(1.0 / (mse + 1e-12))
            if snr > best_snr: best_snr, best_rec = snr, temp_rec
        return best_snr, best_rec

    # 处理对齐信号
    aligned_list = []
    for i in range(2):
        snr_v, aligned_sig = get_snr(tx_final[:, i], rx_final[:, i])
        print(f"  Pol {i} SNR: {snr_v:.2f} dB")
        aligned_list.append(aligned_sig)
    
    rx_aligned_all = jnp.stack(aligned_list, axis=1)

    # --- 绘图：DSP 后 ---
    # 1. 星座图
    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.scatter(rx_aligned_all[-3000:, i].real, rx_aligned_all[-3000:, i].imag, s=1, alpha=0.5)
        plt.title(f'Constellation Pol {i}')
        plt.axis('equal')
    plt.savefig(os.path.join(FIGURE_PATH, '03_constellations.png'), dpi=300)
    print("  Saved: 03_constellations.png")

    # 2. 后处理 PSD
    plot_psd(rx_aligned_all, fs=32e9, name='PSD After DSP', filename='04_psd_after_dsp.png')
    
    # 3. 后处理眼图 (由于是 1 SPS，我们调小 limit 观察点阵开启度)
    plot_eye(rx_aligned_all, sps=1, name='Eye After DSP', amplitude_limit=1.5, filename='05_eye_after_dsp.png')

    print(f"\nAll plots are saved in: {FIGURE_PATH}")
    plt.show()

if __name__ == "__main__":
    main()