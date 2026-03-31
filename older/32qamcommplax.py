import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# 导入路径兼容性处理
try:
    from commplax import adaptive_filter as af
except ImportError:
    from commplax._deprecated import adaptive_filter as af

# =========================================================================
# 1. 核心工具：手动定义 32QAM 与 全角度对齐器
# =========================================================================

def get_32qam_const():
    # 生成 6x6 网格并去除四个角，得到标准的 32QAM 十字星座图
    x = jnp.arange(-5, 6, 2)
    y = jnp.arange(-5, 6, 2)
    xv, yv = jnp.meshgrid(x, y)
    full_grid = (xv + 1j * yv).flatten()
    corners = [5+5j, 5-5j, -5+5j, -5-5j]
    mask = jnp.array([jnp.all(jnp.abs(p - jnp.array(corners)) > 0.1) for p in full_grid])
    c32 = full_grid[mask]
    return c32 / jnp.sqrt(jnp.mean(jnp.abs(c32)**2))

def get_snr_ultimate_scan(ref, rec):
    """
    通过 0-360 度全方位扫描寻找最小 MSE 角度，修复 Pol 0 扭转问题
    """
    rec = rec / (jnp.sqrt(jnp.mean(jnp.abs(rec)**2)) + 1e-12)
    ref = ref / (jnp.sqrt(jnp.mean(jnp.abs(ref)**2)) + 1e-12)
    
    best_snr = -100.0
    best_rec = rec
    
    # 搜索范围：符号位移 (-15 到 15)
    for shift in range(-15, 16):
        r_sh = jnp.roll(rec, shift)
        
        # 核心修复：以 1 度为步长扫描 0-360 度，寻找使 MSE 最小的角度
        angles = jnp.linspace(0, 2 * jnp.pi, 360)
        for angle in angles:
            tmp = r_sh * jnp.exp(-1j * angle)
            # 补偿细微的残余相位误差 (CPE)
            phase_err = jnp.angle(jnp.mean(tmp * jnp.conj(ref)))
            tmp_final = tmp * jnp.exp(-1j * phase_err)
            
            mse = jnp.mean(jnp.abs(ref - tmp_final)**2)
            snr = 10 * jnp.log10(1.0 / (mse + 1e-12))
            
            if snr > best_snr:
                best_snr = snr
                best_rec = tmp_final
                
    return best_snr, best_rec

# =========================================================================
# 2. 主仿真程序
# =========================================================================

def main():
    print("--- 32QAM 双偏振全流程仿真 (含眼图生成) ---")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FIGURE_PATH = os.path.join(BASE_DIR, 'results', '32qam_final_check')
    if not os.path.exists(FIGURE_PATH): os.makedirs(FIGURE_PATH)

    num_syms = 32768
    sps = 2
    key = jax.random.PRNGKey(42)

    # Step 1: Tx 信号生成
    const = get_32qam_const()
    tx_indices = jax.random.randint(key, (num_syms, 2), 0, 32)
    tx_syms = const[tx_indices]

    # Step 2: 信道加载 (AWGN)
    rx_2sps = jnp.zeros((num_syms * sps, 2), dtype=tx_syms.dtype)
    rx_2sps = rx_2sps.at[::sps, :].set(tx_syms)
    # 注入适量噪声
    noise_std = 0.012 
    noise = (jax.random.normal(key, rx_2sps.shape) + 1j * jax.random.normal(key, rx_2sps.shape)) * noise_std
    rx_2sps += noise

    # Step 3: CMA 均衡
    cma_af = af.cma(lr=1e-4, R2=1.45) 
    mimo_state = cma_af.init(taps=41, dims=2)
    rx_framed = af.frame(rx_2sps, taps=41, sps=sps)

    print("正在运行 CMA 均衡器...")
    def scan_train(state, inp):
        new_state, _ = cma_af.update(0, state, inp)
        return new_state, None
    mimo_state, _ = jax.lax.scan(scan_train, mimo_state, rx_framed)
    
    def scan_apply(state, inp):
        _, out = cma_af.apply(state, inp)
        sig = out[0] if isinstance(out, (tuple, list)) else out
        return state, jnp.atleast_1d(sig)
    _, rx_eq = jax.lax.scan(scan_apply, mimo_state, rx_framed)

    # Step 4: 对齐与全角度旋转修复
    if rx_eq.ndim == 3: rx_eq = jnp.squeeze(rx_eq, axis=1)
    rx_final = rx_eq[5000:] 
    tx_final = tx_syms[5000 : 5000 + rx_final.shape[0]]

    print("正在进行全角度扫描修复 Pol 0 扭转...")
    aligned_list = []
    for i in range(2):
        snr_v, aligned_sig = get_snr_ultimate_scan(tx_final[:, i], rx_final[:, i])
        print(f"Pol {i} 最终对齐 SNR: {snr_v:.2f} dB")
        aligned_list.append(aligned_sig)
    
    rx_aligned_all = jnp.stack(aligned_list, axis=1)

    # Step 5: 绘图 (星座图 + 眼图)
    print("生成分析图表...")
    plt.figure(figsize=(16, 6))

    # 1. 星座图 (Pol 0)
    plt.subplot(1, 3, 1)
    plt.scatter(rx_aligned_all[-2000:, 0].real, rx_aligned_all[-2000:, 0].imag, s=1, alpha=0.5)
    plt.scatter(const.real, const.imag, c='red', marker='x', s=15)
    plt.title(f'32QAM Pol 0 Constellation\nSNR: {float(snr_v):.2f} dB')
    plt.axis('equal')

    # 2. I路眼图 (实部)
    plt.subplot(1, 3, 2)
    eye_data = rx_aligned_all[-1000:, 0].real
    t = np.linspace(0, 1, 10)
    for j in range(len(eye_data)-1):
        # 线性插值模拟轨迹
        y = np.interp(t, [0, 1], [eye_data[j], eye_data[j+1]])
        plt.plot(t, y, color='tab:blue', alpha=0.05, linewidth=0.5)
    plt.title("32QAM Eye Diagram (I-ch)")
    plt.ylim([-1.5, 1.5]); plt.grid(True, alpha=0.2)

    # 3. Q路眼图 (虚部)
    plt.subplot(1, 3, 3)
    eye_data_q = rx_aligned_all[-1000:, 0].imag
    for j in range(len(eye_data_q)-1):
        y = np.interp(t, [0, 1], [eye_data_q[j], eye_data_q[j+1]])
        plt.plot(t, y, color='tab:green', alpha=0.05, linewidth=0.5)
    plt.title("32QAM Eye Diagram (Q-ch)")
    plt.ylim([-1.5, 1.5]); plt.grid(True, alpha=0.2)

    plt.tight_layout()
    save_name = os.path.join(FIGURE_PATH, '32qam_final_analysis.png')
    plt.savefig(save_name, dpi=300)
    print(f"成功！结果已保存至: {save_name}")
    plt.show()

if __name__ == "__main__":
    main()