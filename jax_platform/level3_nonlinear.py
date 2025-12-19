import jax
import jax.numpy as jnp
from jax import random, lax
import time
import numpy as np

# 物理常数
C = 299792458
LAMBDA_0 = 1550e-9

# 修复后的装饰器：sps 是索引0, span 是索引2
@jax.jit(static_argnums=(0, 2))
def rrc_taps(sps, alpha, span):
    """手动实现 RRC 滤波器，并将参数设为静态以确定数组形状"""
    t = jnp.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    denom = 1.0 - (2.0 * alpha * t)**2
    h = jnp.where(
        jnp.abs(denom) < 1e-10,
        (alpha / jnp.sqrt(2.0)) * ((1.0 + 2.0 / jnp.pi) * jnp.sin(jnp.pi / (4.0 * alpha)) + (1.0 - 2.0 / jnp.pi) * jnp.cos(jnp.pi / (4.0 * alpha))),
        jnp.sinc(t) * jnp.cos(jnp.pi * alpha * t) / denom
    )
    return h / jnp.sqrt(jnp.sum(h**2))

# --- 1. 物理信道: 基于 lax.scan 的高性能 SSFM ---
def ssfm_propagation(signal, distance_km, sps, baud_rate, D=17.0, gamma=1.3, n_steps=80):
    dz = (distance_km * 1e3) / n_steps
    fs = baud_rate * sps
    beta2 = -D * (LAMBDA_0**2) / (2 * jnp.pi * C) * 1e-6
    
    N = signal.shape[0]
    omega = 2 * jnp.pi * jnp.fft.fftfreq(N, 1/fs)
    exp_linear = jnp.exp(-1j * 0.5 * beta2 * (omega**2) * (dz / 2))
    
    def ssfm_body(carry, _):
        # 1. 线性前半步
        A = jnp.fft.ifft(jnp.fft.fft(carry) * exp_linear)
        # 2. 非线性步 (SPM)
        A = A * jnp.exp(1j * gamma * jnp.abs(A)**2 * dz)
        # 3. 线性后半步
        A = jnp.fft.ifft(jnp.fft.fft(A) * exp_linear)
        return A, None

    final_signal, _ = lax.scan(ssfm_body, signal, None, length=n_steps)
    return final_signal

# --- 2. 接收端 DSP: 自适应 CMA 均衡器 ---
def cma_equalizer(signal, n_taps=11, mu=1e-3, n_iter=10):
    w = jnp.zeros(n_taps, dtype=jnp.complex64).at[n_taps // 2].set(1.0)
    
    def cma_loop(carry_w, _):
        y = jnp.convolve(signal, carry_w, mode='same')
        err = 1.0 - jnp.abs(y)**2
        # 简化版梯度更新，用于 Benchmark
        grad = -mu * jnp.mean(err) * carry_w 
        return carry_w - grad, None

    final_w, _ = lax.scan(cma_loop, w, None, length=n_iter)
    return jnp.convolve(signal, final_w, mode='same')

@jax.jit
def simulate_level3_core(bits, distance_km, snr_db, key):
    sps, baud_rate = 16, 32e9
    num_symbols = bits.shape[0] // 2
    
    # 调制
    symbols = ((1 - 2. * bits[0::2]) + 1j * (1 - 2. * bits[1::2])) / jnp.sqrt(2.0)
    
    # 脉冲成形
    h_rrc = rrc_taps(sps, 0.25, 16)
    sig_up = jnp.zeros(num_symbols * sps, dtype=jnp.complex64).at[::sps].set(symbols)
    sig_tx = jnp.convolve(sig_up, h_rrc, mode='same')
    
    # 物理信道
    sig_ch = ssfm_propagation(sig_tx, distance_km, sps, baud_rate, n_steps=80)
    
    # 加噪
    snr_lin = 10**(snr_db / 10.0)
    noise_pwr = jnp.mean(jnp.abs(sig_ch)**2) / (snr_lin * sps)
    k1, k2 = random.split(key)
    noise = jnp.sqrt(noise_pwr/2) * (random.normal(k1, sig_ch.shape) + 1j*random.normal(k2, sig_ch.shape))
    sig_noisy = sig_ch + noise
    
    # 接收 DSP: EDC + CMA
    beta2 = -17.0 * (LAMBDA_0**2) / (2 * jnp.pi * C) * 1e-6
    N = sig_noisy.shape[0]
    omega = 2 * jnp.pi * jnp.fft.fftfreq(N, 1/(baud_rate*sps))
    sig_edc = jnp.fft.ifft(jnp.fft.fft(sig_noisy) * jnp.exp(1j * 0.5 * beta2 * (omega**2) * (distance_km * 1e3)))
    sig_cma = cma_equalizer(sig_edc, n_taps=11, mu=1e-3, n_iter=10)
    
    # 匹配滤波与解调
    sig_rx = jnp.convolve(sig_cma, h_rrc, mode='same')
    sig_samples = sig_rx[::sps][:num_symbols]
    recovered_bits = jnp.stack([jnp.real(sig_samples) < 0, jnp.imag(sig_samples) < 0], axis=1).flatten().astype(jnp.int32)
    return jnp.mean(recovered_bits != bits)

def benchmark_level3_jax(test_cases):
    results = []
    print("🚀 Level 3 JAX 启动 (SSFM + CMA)...")
    # 预热
    _ = simulate_level3_core(jnp.zeros(2048, dtype=jnp.int32), 80.0, 10.0, random.PRNGKey(0))
    
    for case in test_cases:
        print(f"  正在运行 JAX: {case['name']}")
        times, bers = [], []
        for run in range(case['num_runs']):
            bits = random.randint(random.PRNGKey(run), (2 * case['num_symbols'],), 0, 2)
            start = time.perf_counter()
            ber = simulate_level3_core(bits, float(case['fiber_length_km']), float(case['snr_db']), random.PRNGKey(run+42))
            times.append(time.perf_counter() - start)
            bers.append(ber)
        results.append({'name': case['name'], 'fiber_length_km': case['fiber_length_km'], 'avg_time': np.mean(times), 'avg_ber': float(np.mean(bers))})
    return results