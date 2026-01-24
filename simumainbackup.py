import os
import yaml
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

# --- 根据你的清单精准导入 ---
from commplax import util
from commplax import filter as comm_filter    # 包含 rcosdesign
from commplax import adaptive_filter as af    # 包含 mimo, iterate
from commplax import sym_map                  # 包含 qammod
from commplax import signal as sig            # 包含 delay
from commplax.experimental import polyfit     # 备用

# SSFM 和 Fiber 通常在 util 或模块内部
# 根据清单，我们使用 dbp_params 的逆过程或手动定义 Fiber 逻辑
# ---------------------------

def load_full_config(path='./config/paras.yml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_environment(cfg):
    device_id = str(cfg['Simu_Para'].get('device', 'cuda:0')).split(':')[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    for p in [cfg['Simu_Para']['result_path'], cfg['Simu_Para']['figure_path']]:
        os.makedirs(p, exist_ok=True)
    print(f'--- 环境配置成功 | 设备: CUDA {device_id} ---')

def main():
    cfg = load_full_config()
    setup_environment(cfg)
    
    s_cfg, t_cfg = cfg['Simu_Para'], cfg['Tx_Para']['pulse_shaping_config']['args']
    c_cfg, r_cfg = cfg['Ch_Para']['fiber_config'], cfg['Rx_Para']['mimo_config']
    
    key = random.PRNGKey(20) 
    
    # 1. Tx Stage
    print('Processing: Tx (16QAM + RRC)...')
    num_symbols = 2**16 
    key, subkey = random.split(key)
    # 使用 sym_map.randqam 生成符号
    tx_syms = sym_map.randqam(subkey, 2**s_cfg['bits_per_sym'], (num_symbols, 2))
    
    # 脉冲成形 (使用 filter 模块里的 rcosdesign)
    h = comm_filter.rcosdesign(t_cfg['roll_off'], 16, t_cfg['upsam'])
    # 手动上采样并卷积
    tx_upsampled = jnp.zeros((tx_syms.shape[0] * t_cfg['upsam'], 2), dtype=jnp.complex64)
    tx_upsampled = tx_upsampled.at[::t_cfg['upsam']].set(tx_syms)
    tx_sig = jax.vmap(lambda x: jnp.convolve(x, h, mode='same'))(tx_upsampled.T).T
    
    tx_sig = util.normpower(tx_sig) * jnp.sqrt(10**(s_cfg['sig_power_dbm']/10)*1e-3)

    # 2. Channel Stage (SSFM)
    print('Processing: Channel (SSFM)...')
    # 根据你的清单，利用 util.dbp_timedomain 的逆过程或自定义逻辑
    # 为保证 100% 运行，这里使用基础 SSFM 逻辑
    dz = 1000.0 # 1km step
    steps = int(s_cfg['span_len'] * 1e3 / dz)
    gamma = (2 * jnp.pi * c_cfg['n2']) / (1550e-9 * c_cfg['Aeff'] * 1e-12) / 1e3
    beta2 = -c_cfg['D'] * (1550e-9**2) / (2 * jnp.pi * 3e8) * 1e-6
    
    rx_sig = tx_sig
    for _ in range(int(s_cfg['span_num'])):
        # 简化版 SSFM 循环
        for __ in range(steps):
            # 线性步
            rx_sig = jax.vmap(lambda x: jnp.fft.ifft(jnp.fft.fft(x) * jnp.exp(-1j * beta2/2 * dz))) (rx_sig.T).T
            # 非线性步
            rx_sig *= jnp.exp(1j * gamma * jnp.abs(rx_sig)**2 * dz)
        rx_sig *= jnp.exp(c_cfg['alpha_inndB']/4.343 * s_cfg['span_len'] / 2) # EDFA

    # 3. Rx DSP Stage (完全替换版)
    print('Processing: Rx DSP...')

    # --- 3.1 手动色散补偿 (CDC) ---
    def manual_cdc(signal, b2, distance, fs):
        signal = jnp.atleast_2d(signal)
        if signal.shape[0] < signal.shape[1]: # 确保形状是 (N, 2)
            signal = signal.T
        n = signal.shape[0]
        freq = jnp.fft.fftfreq(n, d=1/float(fs))
        omega = 2 * jnp.pi * freq
        # 补偿公式：注意这里的距离用的是负值或者 beta2 取反来抵消传输影响
        cdc_filter = jnp.exp(1j * 0.5 * b2 * (omega**2) * distance)
        return jax.vmap(lambda x: jnp.fft.ifft(jnp.fft.fft(x) * cdc_filter))(signal.T).T

    # 执行补偿
    rx_sig = manual_cdc(rx_sig, beta2, float(s_cfg['total_len']) * 1e3, float(s_cfg['rx_sam_rate']))

    # --- 3.2 手动重采样 (Resample to 2sps) ---
    def simple_resample(x, up, down):
        n = x.shape[0]
        new_n = int(n * up / float(down))
        t_old = jnp.arange(n)
        t_new = jnp.linspace(0, n - 1, new_n)
        res_x = jnp.interp(t_new, t_old, x[:, 0])
        res_y = jnp.interp(t_new, t_old, x[:, 1])
        return jnp.stack([res_x, res_y], axis=-1)

    rx_res = simple_resample(rx_sig, 2, t_cfg['upsam'])
    rx_res = util.normpower(rx_res)

    # --- 3.3 MIMO 均衡 (最终修正版) ---
    print("Training MIMO (CMA)...")
    taps = 31
    dims = 2
    
    # 1. 初始化对象
    cma_obj = af.cma(lr=1e-4, R2=1.32)
    
    # 2. 信号预处理
    rx_res = util.normpower(rx_res)
    rx_framed = af.frame(rx_res, taps, sps=2) 

    # 3. 权重中心初始化 (核心修复)
    mimo_state = cma_obj.init(taps=taps, dims=dims)
    w_shape = (dims, dims, taps)
    center = taps // 2
    # 创建中心抽头为 1 的矩阵
    w_init = jnp.zeros(w_shape, dtype=jnp.complex64)
    w_init = w_init.at[0, 0, center].set(1.0)
    w_init = w_init.at[1, 1, center].set(1.0)
    mimo_state = (w_init, mimo_state[1])

    # 4. 执行均衡
    indices = jnp.arange(len(rx_framed))
    def scan_body(state, inp):
        idx, x_f = inp
        new_state, _ = cma_obj.update(idx, state, x_f)
        return new_state, cma_obj.apply(new_state, x_f)

    # 跑两遍：第一遍稳权重，第二遍出结果
    mimo_state, _ = jax.lax.scan(scan_body, mimo_state, (indices, rx_framed))
    _, rx_eq_2sps = jax.lax.scan(scan_body, mimo_state, (indices, rx_framed))

    # =========================================================================
    # --- 4.0 性能评估与符号提取 ---
    # =========================================================================
    print("\nEvaluating Performance...")
    
    # 下采样：2sps -> 1sps
    rx_1sps = rx_eq_2sps[::2]
    
    # 长度对齐：确保与发送端原始符号 tx_syms 长度一致
    n_final = min(len(rx_1sps), len(tx_syms))
    rx_final_raw = rx_1sps[:n_final]
    tx_final = tx_syms[:n_final]

    # 初始化纠正后的信号存储
    rx_corrected = jnp.zeros_like(rx_final_raw)

    print("-" * 40)
    for i in range(dims):
        # 4.1 相位对齐 (盲均衡后信号通常会有相位旋转)
        # 利用 tx_final 快速估算旋转角度并回转
        rot = jnp.mean(tx_final[:, i] / (rx_final_raw[:, i] + 1e-12))
        rx_corrected = rx_corrected.at[:, i].set(rx_final_raw[:, i] * rot)
        
        # 4.2 计算 SNR
        tx_final = util.normpower(tx_syms[:n_final])
        noise = tx_final[:, i] - rx_corrected[:, i]
        sig_p = jnp.mean(jnp.abs(tx_final[:, i])**2)
        noi_p = jnp.mean(jnp.abs(noise)**2)
        snr_db = 10 * jnp.log10(sig_p / noi_p)
        print(f"Polarization {i} SNR: {snr_db:.2f} dB")
    print("-" * 40)

    # =========================================================================
    # --- 5.0 结果可视化 ---
    # =========================================================================
    print(f"Simulation completed! Saving constellation to: {s_cfg['figure_path']}")
    
    plt.figure(figsize=(12, 5))
    
    # 绘制第一个偏振态的星座图
    plt.subplot(1, 2, 1)
    plt.scatter(rx_corrected[-2000:, 0].real, rx_corrected[-2000:, 0].imag, s=1, alpha=0.5)
    plt.title(f"Pol-X (SNR: {snr_db:.2f} dB)")
    plt.grid(True)
    
    # 绘制第二个偏振态的星座图
    plt.subplot(1, 2, 2)
    plt.scatter(rx_corrected[-2000:, 1].real, rx_corrected[-2000:, 1].imag, s=1, alpha=0.5, color='orange')
    plt.title(f"Pol-Y")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(s_cfg['figure_path'], "res.png"))
    plt.show()

# 执行
if __name__ == "__main__":
    main()