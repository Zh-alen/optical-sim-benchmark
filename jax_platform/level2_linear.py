import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np

# 参数定义
SPS = 16
BAUD_RATE = 32e9

@jax.jit(static_argnums=(0, 1, 2))
def rrc_taps(sps, alpha, span):
    """手动实现 RRC 滤波器"""
    t = jnp.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    denom = 1.0 - (2.0 * alpha * t)**2
    h = jnp.where(
        jnp.abs(denom) < 1e-10,
        (alpha / jnp.sqrt(2.0)) * ((1.0 + 2.0 / jnp.pi) * jnp.sin(jnp.pi / (4.0 * alpha)) + (1.0 - 2.0 / jnp.pi) * jnp.cos(jnp.pi / (4.0 * alpha))),
        jnp.sinc(t) * jnp.cos(jnp.pi * alpha * t) / denom
    )
    return h / jnp.sqrt(jnp.sum(h**2))

def simulate_level2_jax_core(bits, fiber_length_km, snr_db, key):
    num_symbols = bits.shape[0] // 2
    
    # 1. QPSK 调制
    symbols = (1 - 2. * bits[0::2]) + 1j * (1 - 2. * bits[1::2])
    symbols = symbols / jnp.sqrt(2.0)
    
    # 2. 脉冲成形
    h_rrc = rrc_taps(SPS, 0.25, 16)
    signal_up = jnp.zeros(num_symbols * SPS, dtype=jnp.complex64)
    signal_up = signal_up.at[::SPS].set(symbols)
    signal_tx = jnp.convolve(signal_up, h_rrc, mode='same')
    
    # 3. 物理传输
    z = fiber_length_km * 1e3
    D = 17.0
    c = 299792458
    lmbd = 1550e-9
    beta2 = -D * (lmbd**2) / (2 * jnp.pi * c) * 1e-6
    
    N = signal_tx.shape[0]
    freqs = jnp.fft.fftfreq(N, 1/(BAUD_RATE * SPS))
    omega = 2 * jnp.pi * freqs
    H_fiber = jnp.exp(-1j * 0.5 * beta2 * (omega**2) * z)
    signal_ch = jnp.fft.ifft(jnp.fft.fft(signal_tx) * H_fiber)
    
    # 4. 加噪
    snr_linear = 10**(snr_db / 10.0)
    sig_pwr = jnp.mean(jnp.abs(signal_ch)**2)
    noise_pwr = sig_pwr / (snr_linear * SPS)
    k1, k2 = random.split(key)
    noise = jnp.sqrt(noise_pwr/2) * (random.normal(k1, signal_ch.shape) + 1j*random.normal(k2, signal_ch.shape))
    signal_noisy = signal_ch + noise
    
    # 5. 接收端 EDC 补偿
    H_edc = jnp.exp(1j * 0.5 * beta2 * (omega**2) * z)
    signal_edc = jnp.fft.ifft(jnp.fft.fft(signal_noisy) * H_edc)
    
    # 6. 匹配滤波与采样
    signal_rx = jnp.convolve(signal_edc, h_rrc, mode='same')
    signal_samples = signal_rx[::SPS][:num_symbols]
    
    # 7. 解调
    recovered_bits_re = (jnp.real(signal_samples) < 0).astype(jnp.int32)
    recovered_bits_im = (jnp.imag(signal_samples) < 0).astype(jnp.int32)
    recovered_bits = jnp.empty_like(bits)
    recovered_bits = recovered_bits.at[0::2].set(recovered_bits_re)
    recovered_bits = recovered_bits.at[1::2].set(recovered_bits_im)
    
    errors = jnp.sum(recovered_bits != bits)
    return errors / bits.shape[0]

def simulate_level2_jax(num_symbols=4096, fiber_length_km=80, snr_db=10, seed=42):
    key = random.PRNGKey(seed)
    bk, sk = random.split(key)
    bits = random.randint(bk, (2 * num_symbols,), 0, 2)
    run_sim = jax.jit(simulate_level2_jax_core)
    ber = run_sim(bits, float(fiber_length_km), float(snr_db), sk)
    return float(ber)

def benchmark_level2_jax(test_cases):
    results = []
    print("🚀 JAX (Optimized DSP Mode) 启动...")
    _ = simulate_level2_jax(1024, 0, 10)
    for case in test_cases:
        print(f"  正在运行 JAX 测试: {case['name']}")
        times, bers = [], []
        for run in range(case['num_runs']):
            start = time.perf_counter()
            ber = simulate_level2_jax(case['num_symbols'], case['fiber_length_km'], case['snr_db'], run)
            times.append(time.perf_counter() - start)
            bers.append(ber)
        results.append({'name': case['name'], 'fiber_length_km': case['fiber_length_km'],
                        'num_symbols': case['num_symbols'], 'avg_time': np.mean(times),
                        'avg_ber': np.mean(bers), 'snr_db': case['snr_db']})
    return results