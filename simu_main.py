import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from commplax import adaptive_filter as af
from commplax import sym_map
import scipy.signal as sp_signal
import numpy as np

# =========================================================================
# 1. 路径配置
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(BASE_DIR, 'results', 'figure')
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)

# =========================================================================
# 2. 终极对齐函数 (落实师兄建议：搜索位移 + 最小 MSE 角度)
# =========================================================================

def get_snr_ultimate(ref, rec):
    # 归一化
    rec = rec / (jnp.sqrt(jnp.mean(jnp.abs(rec)**2)) + 1e-12)
    ref = ref / (jnp.sqrt(jnp.mean(jnp.abs(ref)**2)) + 1e-12)
    
    best_snr = -100.0
    best_rec = rec
    
    # 扩大搜索范围：符号位移 (-20 到 20)
    for shift in range(-20, 21):
        r_shifted = jnp.roll(rec, shift)
        
        # 针对 16QAM 的相位模糊搜索
        for angle_q in [0, jnp.pi/2, jnp.pi, 3*jnp.pi/2]:
            tmp = r_shifted * jnp.exp(-1j * angle_q)
            
            # 解析解：直接找到使 MSE 最小的残余角度 phi
            # 这里的 phi 补偿了师兄提到的 Constant Phase Error
            phase_err = jnp.angle(jnp.mean(tmp * jnp.conj(ref)))
            tmp_final = tmp * jnp.exp(-1j * phase_err)
            
            mse = jnp.mean(jnp.abs(ref - tmp_final)**2)
            snr = 10 * jnp.log10(1.0 / (mse + 1e-12))
            
            if snr > best_snr:
                best_snr = snr
                best_rec = tmp_final
                
    return best_snr, best_rec

# =========================================================================
# 3. 绘图工具
# =========================================================================

def plot_psd(sig, fs, name='Signal PSD', filename='psd_analysis.png'):
    plt.figure(figsize=(10, 5))
    for i in range(sig.shape[1]):
        f, Pxx_den = sp_signal.welch(np.array(sig[:, i]), fs, nperseg=1024)
        plt.semilogy(f / 1e9, Pxx_den, label=f'Pol {i}')
    plt.title(name); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(FIGURE_PATH, filename)); plt.close()

def plot_eye(sig, sps=1, name='Eye Diagram', filename='eye_diagram.png'):
    plt.figure(figsize=(12, 5))
    num_points = int(2 * sps) + 1 
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        data = np.array(sig[-2000:, i].real)
        for j in range(0, len(data)-num_points, sps):
            ax.plot(np.linspace(0, 2, num_points), data[j:j+num_points], 'b-', alpha=0.03)
        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.3)
        ax.set_title(f'{name} - Pol {i}'); ax.set_ylim([-1.5, 1.5])
    plt.tight_layout(); plt.savefig(os.path.join(FIGURE_PATH, filename)); plt.close()

# =========================================================================
# 4. 主仿真逻辑
# =========================================================================

def main():
    # Step 1: Tx
    print("Step 1: Signal Generation...")
    num_syms = 32768
    const = jnp.asarray(sym_map.const('16QAM', norm=True))
    key = jax.random.PRNGKey(42)
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 16)
    tx_syms = const[tx_indices]

    # Step 2: Channel
    print("Step 2: Adding Noise...")
    rx_2sps = jnp.zeros((num_syms * 2, 2), dtype=tx_syms.dtype)
    rx_2sps = rx_2sps.at[::2, :].set(tx_syms) 
    noise = (jax.random.normal(jax.random.PRNGKey(7), rx_2sps.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(8), rx_2sps.shape)) * 0.02
    rx_2sps += noise

    # Step 3: Rx DSP (标准 CMA)
    print("Step 3: Commplax Rx DSP...")
    taps = 41
    cma_af = af.cma(lr=1e-4, R2=1.32)
    
    # 使用标准初始化，不手动修改 w 以免报错
    mimo_state = cma_af.init(taps=taps, dims=2)
    rx_framed = af.frame(rx_2sps, taps=taps, sps=2)

    def scan_train(state, inp):
        new_state, _ = cma_af.update(0, state, inp)
        return new_state, None

    def scan_apply(state, inp):
        _, out = cma_af.apply(state, inp)
        # 兼容不同版本的 commplax 输出
        sig = out[0] if isinstance(out, (tuple, list)) else out
        return state, jnp.atleast_1d(sig)

    print("   Training...")
    mimo_state, _ = jax.lax.scan(scan_train, mimo_state, rx_framed)
    # 增加一次训练，让系数更稳
    mimo_state, _ = jax.lax.scan(scan_train, mimo_state, rx_framed)
    
    print("   Applying...")
    _, rx_eq = jax.lax.scan(scan_apply, mimo_state, rx_framed)

    # Step 4: 对齐与 SNR
    print("Step 4: Ultimate Alignment (Shift + Phase)...")
    if rx_eq.ndim == 3: rx_eq = jnp.squeeze(rx_eq, axis=1)

    # 丢弃收敛期数据
    rx_final = rx_eq[num_syms // 2:]
    tx_final = tx_syms[num_syms // 2 : num_syms // 2 + rx_final.shape[0]]

    aligned_list = []
    for i in range(2):
        snr_v, aligned_sig = get_snr_ultimate(tx_final[:, i], rx_final[:, i])
        print(f"   Pol {i} SNR: {snr_v:.2f} dB")
        aligned_list.append(aligned_sig)
    
    rx_aligned_all = jnp.stack(aligned_list, axis=1)

    # Step 5: 绘图
    print("Step 5: Final Plotting...")
    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.scatter(rx_aligned_all[-2000:, i].real, rx_aligned_all[-2000:, i].imag, s=1, alpha=0.5)
        plt.scatter(const.real, const.imag, c='red', marker='x', s=20)
        plt.title(f'Pol {i} Constellation'); plt.axis('equal')
    plt.savefig(os.path.join(FIGURE_PATH, '03_constellations.png'), dpi=300)
    
    plot_eye(rx_aligned_all, sps=1, name='Eye After DSP', filename='05_eye_after_dsp.png')
    print(f"Success! Figures saved in {FIGURE_PATH}")

if __name__ == "__main__":
    main()