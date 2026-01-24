import numpy as np
import time

C, LAMBDA_0 = 299792458, 1550e-9

def rrc_taps_numpy(sps, alpha, span):
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1.0 - (2.0 * alpha * t)**2
        h = np.sinc(t) * np.cos(np.pi * alpha * t) / denom
        h[np.abs(denom) < 1e-10] = (alpha / np.sqrt(2.0)) * ((1.0+2.0/np.pi)*np.sin(np.pi/(4*alpha)) + (1.0-2.0/np.pi)*np.cos(np.pi/(4*alpha)))
    return h / np.sqrt(np.sum(h**2))

def simulate_level3_numpy(num_symbols, distance_km, snr_db, seed):
    np.random.seed(seed)
    sps, baud_rate = 16, 32e9
    bits = np.random.randint(0, 2, 2 * num_symbols)
    symbols = ((1 - 2. * bits[0::2]) + 1j * (1 - 2. * bits[1::2])) / np.sqrt(2.0)
    
    h_rrc = rrc_taps_numpy(sps, 0.25, 16)
    sig_up = np.zeros(num_symbols * sps, dtype=np.complex64)
    sig_up[::sps] = symbols
    sig_tx = np.convolve(sig_up, h_rrc, mode='same')
    
    # SSFM 循环
    n_steps, dz = 80, (distance_km * 1e3) / 80
    beta2 = -17.0 * (LAMBDA_0**2) / (2 * np.pi * C) * 1e-6
    omega = 2 * np.pi * np.fft.fftfreq(len(sig_tx), 1/(baud_rate*sps))
    exp_l = np.exp(-1j * 0.5 * beta2 * (omega**2) * (dz / 2))
    
    A = sig_tx.copy()
    for _ in range(n_steps):
        A = np.fft.ifft(np.fft.fft(A) * exp_l)
        A = A * np.exp(1j * 1.3 * np.abs(A)**2 * dz)
        A = np.fft.ifft(np.fft.fft(A) * exp_l)
    
    # 噪声
    snr_lin = 10**(snr_db / 10.0)
    noise = np.sqrt(np.mean(np.abs(A)**2)/(snr_lin*sps)/2) * (np.random.randn(len(A)) + 1j*np.random.randn(len(A)))
    sig_noisy = A + noise
    
    # EDC + CMA
    sig_edc = np.fft.ifft(np.fft.fft(sig_noisy) * np.exp(1j * 0.5 * beta2 * (omega**2) * (distance_km * 1e3)))
    w = np.zeros(11, dtype=np.complex64); w[5] = 1.0
    for _ in range(10):
        y = np.convolve(sig_edc, w, mode='same')
        w = w - (-1e-3 * np.mean(1.0 - np.abs(y)**2) * w)
    
    sig_rx = np.convolve(np.convolve(sig_edc, w, mode='same'), h_rrc, mode='same')
    sig_samples = sig_rx[::sps][:num_symbols]
    recovered_bits = np.stack([np.real(sig_samples) < 0, np.imag(sig_samples) < 0], axis=1).flatten().astype(np.int32)
    return np.mean(recovered_bits != bits)

def benchmark_level3_numpy(test_cases):
    results = []
    print(">>> NumPy Level 3 启动...")
    for case in test_cases:
        print(f"  正在运行 NumPy: {case['name']}")
        times, bers = [], []
        for run in range(case['num_runs']):
            start = time.perf_counter()
            ber = simulate_level3_numpy(case['num_symbols'], case['fiber_length_km'], case['snr_db'], run)
            times.append(time.perf_counter() - start)
            bers.append(ber)
        results.append({'name':case['name'], 'fiber_length_km':case['fiber_length_km'], 'avg_time':np.mean(times), 'avg_ber':float(np.mean(bers))})
    return results