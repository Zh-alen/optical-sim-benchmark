import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
import time

def apply_chromatic_dispersion(signal_t, sample_rate, fiber_length, inverse=False):
    D, lambda0, c = 17, 1550e-9, 3e8
    beta2 = -(D * 1e-6) * (lambda0**2) / (2 * np.pi * c)
    N = len(signal_t)
    dt = 1 / sample_rate
    omega = 2 * np.pi * fftfreq(N, dt)
    z = fiber_length * 1000 * (-1 if inverse else 1)
    H = np.exp(-1j * beta2 * omega**2 * z / 2)
    return ifft(fft(signal_t) * H)

def root_raised_cosine_filter(samples_per_symbol=16, roll_off=0.25):
    filter_span = 8
    t = np.arange(-filter_span*samples_per_symbol, filter_span*samples_per_symbol + 1) / samples_per_symbol
    h = np.sinc(t) * np.cos(np.pi * roll_off * t) / (1 - (4 * roll_off * t)**2 + 1e-12)
    return h / np.sqrt(np.sum(h**2))

def simulate_level2_numpy(num_symbols=4096, samples_per_symbol=16, fiber_length_km=80, snr_db=10, seed=42):
    np.random.seed(seed)
    bits = np.random.randint(0, 2, 2 * num_symbols)
    symbols = ((1 - 2*bits[::2]) + 1j*(1 - 2*bits[1::2])) / np.sqrt(2)
    
    signal_up = np.repeat(symbols, samples_per_symbol)
    rrc_filter = root_raised_cosine_filter(samples_per_symbol)
    signal_tx = signal.convolve(signal_up, rrc_filter, mode='same')
    
    sample_rate = 32e9 * samples_per_symbol
    signal_ch = apply_chromatic_dispersion(signal_tx, sample_rate, fiber_length_km) if fiber_length_km > 0 else signal_tx
    
    # 噪声
    snr_linear = 10**(snr_db/10)
    noise_std = np.sqrt(np.mean(np.abs(signal_ch)**2) / (2 * snr_linear))
    signal_noisy = signal_ch + noise_std * (np.random.randn(len(signal_ch)) + 1j*np.random.randn(len(signal_ch)))
    
    # 接收端色散补偿 (EDC)
    signal_edc = apply_chromatic_dispersion(signal_noisy, sample_rate, fiber_length_km, inverse=True) if fiber_length_km > 0 else signal_noisy
    signal_rx = signal.convolve(signal_edc, rrc_filter, mode='same')
    
    signal_down = signal_rx[len(rrc_filter)//2::samples_per_symbol][:num_symbols]
    decisions = (np.sign(np.real(signal_down)) + 1j*np.sign(np.imag(signal_down))) / np.sqrt(2)
    
    return np.sum(decisions != symbols[:len(decisions)]) / len(decisions)

def benchmark_level2_numpy(test_cases):
    results = []
    for case in test_cases:
        print(f"  测试用例: {case['name']} (NumPy)")
        times, bers = [], []
        for run in range(case['num_runs']):
            start = time.perf_counter()
            ber = simulate_level2_numpy(case['num_symbols'], case['samples_per_symbol'], case['fiber_length_km'], case['snr_db'], run)
            times.append(time.perf_counter() - start)
            bers.append(ber)
        results.append({
            'name': case['name'], 'fiber_length_km': case['fiber_length_km'],
            'num_symbols': case['num_symbols'], 'avg_time': np.mean(times),
            'std_time': np.std(times), 'avg_ber': np.mean(bers), 'snr_db': case['snr_db']
        })
    return results