import jax
import jax.numpy as jnp
import time

def benchmark_level2_jax_raw(sps, baud_rate, distance, snr_db, key):
    """
    Level 2 线性仿真核心逻辑
    sps: Samples per symbol (static)
    baud_rate: 波特率 (static)
    distance: 传输距离 (static)
    """
    # 仿真参数
    num_symbols = 8192
    
    # 1. 信号生成
    bits = jax.random.randint(key, (2 * num_symbols,), 0, 2)
    symbols = ((1 - 2. * bits[0::2]) + 1j * (1 - 2. * bits[1::2])) / jnp.sqrt(2.0)
    
    # 脉冲成形 (简化版)
    sig_tx = jnp.zeros(num_symbols * sps, dtype=jnp.complex64).at[::sps].set(symbols)
    
    # 2. 线性信道 (色散 CD)
    # 这里的物理常数可以写在函数内或作为参数
    c = 299792458
    lambda_0 = 1550e-9
    beta2 = -17.0 * (lambda_0**2) / (2 * jnp.pi * c) * 1e-6
    
    fs = baud_rate * sps
    omega = 2 * jnp.pi * jnp.fft.fftfreq(sig_tx.shape[0], 1/fs)
    
    # 色散相移
    cd_phi = 0.5 * beta2 * (omega**2) * (distance * 1e3)
    sig_ch = jnp.fft.ifft(jnp.fft.fft(sig_tx) * jnp.exp(-1j * cd_phi))
    
    # 3. 加噪
    snr_linear = 10**(snr_db / 10.0)
    p_sig = jnp.mean(jnp.abs(sig_ch)**2)
    p_noise = p_sig / snr_linear
    noise = jnp.sqrt(p_noise/2) * (jax.random.normal(key, sig_ch.shape) + 1j * jax.random.normal(key, sig_ch.shape))
    sig_rx = sig_ch + noise
    
    # 4. EDC 补偿 (逆向色散)
    sig_edc = jnp.fft.ifft(jnp.fft.fft(sig_rx) * jnp.exp(1j * cd_phi))
    
    # 5. 判决与 BER 计算
    sig_samples = sig_edc[::sps]
    recovered_bits = jnp.stack([jnp.real(sig_samples) < 0, jnp.imag(sig_samples) < 0], axis=1).flatten().astype(jnp.int32)
    ber = jnp.mean(recovered_bits != bits)
    
    return ber

# --- 方法二的核心：显式调用 jax.jit ---
# static_argnums 指定哪些参数在编译时视为常量
# 这里我们将 sps(0), baud_rate(1), distance(2) 设为静态参数
benchmark_level2_jax = jax.jit(
    benchmark_level2_jax_raw, 
    static_argnums=(0, 1, 2)
)

print("JAX Level 2 模块已成功加载 (采用显式 JIT 包装)")