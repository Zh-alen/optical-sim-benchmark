import numpy as np
import time
from scipy.special import erfc

def qpsk_modulation(bits):
    """QPSK调制"""
    # 将比特映射到QPSK符号
    symbols = (1 - 2 * bits[::2]) + 1j * (1 - 2 * bits[1::2])
    return symbols / np.sqrt(2)  # 归一化功率

def add_awgn(signal, snr_db):
    """添加AWGN噪声"""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise_real = np.random.randn(len(signal)) * np.sqrt(noise_power/2)
    noise_imag = np.random.randn(len(signal)) * np.sqrt(noise_power/2)
    return signal + (noise_real + 1j * noise_imag)

def qpsk_demodulation(received):
    """QPSK解调"""
    # 判决到最近的星座点
    decisions = np.sign(np.real(received)) + 1j * np.sign(np.imag(received))
    return decisions / np.sqrt(2)

def calculate_ber(original_bits, received_symbols):
    """计算BER"""
    # 解调得到判决符号
    decisions = qpsk_demodulation(received_symbols)
    
    # 将符号映射回比特
    detected_bits = np.zeros(len(original_bits))
    for i in range(len(decisions)):
        # QPSK到比特的映射
        real_bit = 0 if np.real(decisions[i]) > 0 else 1
        imag_bit = 0 if np.imag(decisions[i]) > 0 else 1
        detected_bits[2*i] = real_bit
        detected_bits[2*i+1] = imag_bit
    
    # 计算误码数
    errors = np.sum(detected_bits != original_bits)
    return errors / len(original_bits)

def run_b2b_numpy(num_symbols=1000, snr_db=10):
    """完整的B2B仿真"""
    # 1. 生成随机比特
    bits = np.random.randint(0, 2, 2 * num_symbols)
    
    # 2. QPSK调制
    tx_symbols = qpsk_modulation(bits)
    
    # 3. 通过AWGN信道
    rx_symbols = add_awgn(tx_symbols, snr_db)
    
    # 4. 计算BER
    ber = calculate_ber(bits, rx_symbols)
    
    # 5. 理论BER（用于验证）
    ber_theory = 0.5 * erfc(np.sqrt(10 ** (snr_db / 10)))
    
    return ber, ber_theory

def benchmark_numpy(num_symbols_list=[1000, 5000, 10000], num_runs=5):
    """真实的NumPy基准测试"""
    results = []
    
    for num_symbols in num_symbols_list:
        print(f"测试符号数: {num_symbols}")
        
        times = []
        bers = []
        
        for run in range(num_runs):
            start = time.perf_counter()
            ber, theory = run_b2b_numpy(num_symbols, snr_db=10)
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
