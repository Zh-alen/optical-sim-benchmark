import jax
import jax.numpy as jnp
from jax import random, lax
import time
import numpy as np

# 物理常数
C = 299792458
LAMBDA_0 = 1550e-9

# ================== 核心 DSP 函数 ==================

@jax.jit
def apply_edc(signal, distance_km, fs):
    """全频域色散补偿"""
    beta2 = -17.0 * (LAMBDA_0**2) / (2 * jnp.pi * C) * 1e-6
    N = signal.shape[0]
    omega = 2 * jnp.pi * jnp.fft.fftfreq(N, 1/fs)
    h_omega = jnp.exp(1j * 0.5 * beta2 * (omega**2) * (distance_km * 1e3))
    return jnp.fft.ifft(jnp.fft.fft(signal) * h_omega)

@jax.jit(static_argnums=(0, 2))
def rrc_taps(sps, alpha, span):
    """根升余弦滤波器"""
    t = jnp.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    denom = 1.0 - (2.0 * alpha * t)**2
    h = jnp.where(
        jnp.abs(denom) < 1e-10,
        (alpha / jnp.sqrt(2.0)) * ((1.0 + 2.0 / jnp.pi) * jnp.sin(jnp.pi / (4.0 * alpha)) + (1.0 - 2.0 / jnp.pi) * jnp.cos(jnp.pi / (4.0 * alpha))),
        jnp.sinc(t) * jnp.cos(jnp.pi * alpha * t) / denom
    )
    return h / jnp.sqrt(jnp.sum(h**2))

# ================== 物理信道 (SSFM) ==================

def ssfm_propagation(signal, distance_km, sps, baud_rate, D=17.0, gamma=1.3, n_steps=100):
    """对称分步傅里叶法"""
    dz = (distance_km * 1e3) / n_steps
    fs = baud_rate * sps
    beta2 = -D * (LAMBDA_0**2) / (2 * jnp.pi * C) * 1e-6
    N = signal.shape[0]
    omega = 2 * jnp.pi * jnp.fft.fftfreq(N, 1/fs)
    exp_linear = jnp.exp(-1j * 0.5 * beta2 * (omega**2) * (dz / 2))
    
    def ssfm_body(carry, _):
        # Linear -> Nonlinear -> Linear
        A = jnp.fft.ifft(jnp.fft.fft(carry) * exp_linear)
        A = A * jnp.exp(1j * gamma * jnp.abs(A)**2 * dz)
        A = jnp.fft.ifft(jnp.fft.fft(A) * exp_linear)
        return A, None
        
    final_signal, _ = lax.scan(ssfm_body, signal, None, length=n_steps)
    return final_signal

# ================== 仿真主函数 ==================

@jax.jit
def simulate_level3_core(bits, distance_km, snr_db, key):
    sps, baud_rate = 16, 32e9
    num_symbols = bits.shape[0] // 2
    span = 64
    
    # 1. 发射机：调制与 RRC 成形
    symbols = ((1 - 2. * bits[0::2]) + 1j * (1 - 2. * bits[1::2])) / jnp.sqrt(2.0)
    h_rrc = rrc_taps(sps, 0.25, span)
    sig_tx = jnp.convolve(jnp.zeros(num_symbols * sps, dtype=jnp.complex64).at[::sps].set(symbols), h_rrc, mode='same')
    
    # 2. 信道：SSFM 传播
    sig_ch = ssfm_propagation(sig_tx, distance_km, sps, baud_rate)
    
    # 3. 噪声：AWGN
    snr_lin = 10**(snr_db / 10.0)
    sig_pwr = jnp.mean(jnp.abs(sig_ch)**2)
    noise_std = jnp.sqrt(sig_pwr / (snr_lin * sps))
    k1, k2 = random.split(key)
    noise = (noise_std / jnp.sqrt(2)) * (random.normal(k1, sig_ch.shape) + 1j*random.normal(k2, sig_ch.shape))
    sig_noisy = sig_ch + noise
    
    # 4. 接收机 DSP
    # A. EDC 补偿
    sig_edc = apply_edc(sig_noisy, distance_km, baud_rate * sps)
    
    # B. 匹配滤波 (可选，为对齐方便此处直接抽样)
    # RRC mode='same' 的中心点延迟为 (span * sps) // 2
    delay = (span * sps) // 2
    sig_1sps = sig_edc[delay::sps]
    
    # C. 数据辅助相位补偿 (Data-Aided CPR)
    # 截取对应长度的原始符号
    min_len = min(len(sig_1sps), len(symbols))
    rx_slice = sig_1sps[:min_len]
    tx_slice = symbols[:min_len]
    
    # 计算平均相位误差并补偿
    phase_error = jnp.angle(jnp.mean(rx_slice * jnp.conj(tx_slice)))
    sig_corrected = rx_slice * jnp.exp(-1j * phase_error)
    
    # 5. 判决与 BER 计算 (避开边界效应)
    margin = 100
    res = sig_corrected[margin:-margin]
    target_bits_slice = bits[margin*2 : (margin + len(res))*2]
    
    recovered_bits = jnp.stack([jnp.real(res) < 0, 
                               jnp.imag(res) < 0], 
                              axis=1).flatten().astype(jnp.int32)
    
    return jnp.mean(recovered_bits != target_bits_slice)

# ================== 外部调用接口 ==================

def benchmark_level3_jax(test_cases):
    print("🚀 Level 3 JAX (Data-Aided Precision) 启动...")
    # 预热编译
    _ = simulate_level3_core(jnp.zeros(10000, dtype=jnp.int32), 80.0, 15.0, random.PRNGKey(0))
    
    results = []
    for case in test_cases:
        print(f"  正在执行方案: {case['name']}")
        num_sym = max(case['num_symbols'], 5000)
        
        times, bers = [], []
        for run in range(case['num_runs']):
            # 这里的随机比特必须与 simulate 内部逻辑一致
            bits = random.randint(random.PRNGKey(run), (2 * num_sym,), 0, 2)
            
            start = time.perf_counter()
            ber = simulate_level3_core(bits, float(case['fiber_length_km']), float(case['snr_db']), random.PRNGKey(run+100))
            ber.block_until_ready()
            
            times.append(time.perf_counter() - start)
            bers.append(ber)
            
        results.append({
            'name': case['name'], 
            'fiber_length_km': case['fiber_length_km'], 
            'avg_time': np.mean(times), 
            'avg_ber': float(np.mean(bers))
        })
    return results