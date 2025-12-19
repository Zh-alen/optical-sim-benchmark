import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
from scipy.special import erfc

@jax.jit
def qpsk_modulation_jax(bits):
    """JAX版本的QPSK调制"""
    real_part = 1 - 2 * bits[::2].astype(jnp.float32)
    imag_part = 1 - 2 * bits[1::2].astype(jnp.float32)
    return (real_part + 1j * imag_part) / jnp.sqrt(2.0)

@jax.jit
def add_awgn_jax(signal, snr_db, key):
    """JAX版本的AWGN噪声"""
    signal_power = jnp.mean(jnp.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    noise_real_key, noise_imag_key = random.split(key)
    noise_real = random.normal(noise_real_key, signal.shape) * jnp.sqrt(noise_power/2)
    noise_imag = random.normal(noise_imag_key, signal.shape) * jnp.sqrt(noise_power/2)
    
    return signal + (noise_real + 1j * noise_imag)

@jax.jit
def qpsk_demodulation_jax(received):
    """JAX版本的QPSK解调"""
    decisions = jnp.sign(jnp.real(received)) + 1j * jnp.sign(jnp.imag(received))
    return decisions / jnp.sqrt(2.0)

@jax.jit
def calculate_ber_jax(original_bits, received_symbols):
    """JAX版本的BER计算"""
    decisions = qpsk_demodulation_jax(received_symbols)
    
    # 向量化的比特判决
    real_bits = (jnp.real(decisions) < 0).astype(jnp.int32)
    imag_bits = (jnp.imag(decisions) < 0).astype(jnp.int32)
    
    # 交错合并实部和虚部比特
    detected_bits = jnp.zeros(2 * len(decisions), dtype=jnp.int32)
    detected_bits = detected_bits.at[::2].set(real_bits)
    detected_bits = detected_bits.at[1::2].set(imag_bits)
    
    errors = jnp.sum(detected_bits != original_bits)
    return errors / len(original_bits)

@jax.jit
def run_b2b_jax_core(bits, snr_db, key):
    """JAX编译的核心仿真函数"""
    # 调制
    tx_symbols = qpsk_modulation_jax(bits)
    
    # 加噪声
    rx_symbols = add_awgn_jax(tx_symbols, snr_db, key)
    
    # 计算BER
    ber = calculate_ber_jax(bits, rx_symbols)
    
    return ber

def run_b2b_jax(num_symbols=1000, snr_db=10, seed=42):
    """完整的JAX B2B仿真"""
    key = random.PRNGKey(seed)
    
    # 生成随机比特
    bits_key, sim_key = random.split(key)
    bits = random.randint(bits_key, (2 * num_symbols,), 0, 2)
    
    # 运行仿真
    ber = run_b2b_jax_core(bits, snr_db, sim_key)
    
    # 理论值
    ber_theory = 0.5 * erfc(np.sqrt(10 ** (snr_db / 10)))
    
    return float(ber), ber_theory

def benchmark_jax(num_symbols_list=[1000, 5000, 10000], num_runs=5):
    """真实的JAX基准测试"""
    results = []
    
    # 预热编译
    print("预热JAX编译...")
    test_key = random.PRNGKey(0)
    test_bits = random.randint(test_key, (200,), 0, 2)
    _ = run_b2b_jax_core(test_bits, 10, test_key).block_until_ready()
    print("JAX编译完成")
    
    for num_symbols in num_symbols_list:
        print(f"测试符号数: {num_symbols}")
        
        times = []
        bers = []
        
        for run in range(num_runs):
            start = time.perf_counter()
            ber, theory = run_b2b_jax(num_symbols, snr_db=10, seed=run)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            bers.append(ber)
            
            if run == 0:
                print(f"  仿真BER: {ber:.2e}, 理论BER: {theory:.2e}")
        
        avg_time = np.mean(times)
        avg_ber = np.mean(bers)
        std_time = np.std(times)
        
        print(f"  平均时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print()
        
        results.append({
            'num_symbols': num_symbols,
            'avg_time': avg_time,
            'std_time': std_time,
            'avg_ber': avg_ber
        })
    
    return results