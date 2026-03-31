import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Coherent DP-PAM4 Optical Communication System: Pure NumPy Interactive Comparison Version

    This is a DP-PAM4 simulator built purely using `NumPy` and `SciPy`, without relying on any deep learning frameworks.

    In this version, the core DA/DD-LMS equalizer is calculated symbol-by-symbol using a **native Python `for` loop**. Please feel free to scroll through the console below and experience the **real "lag" without JAX compilation acceleration**!
        """
    )
    return (mo,)


@app.cell
def _():
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.signal as sp_signal

    def get_snr_pam4_numpy(ref, rec):
        rec = rec / (np.sqrt(np.mean(np.abs(rec)**2)) + 1e-12)
        ref = ref / (np.sqrt(np.mean(np.abs(ref)**2)) + 1e-12)
        best_snr = -100.0
        best_rec = rec
        for shift in range(-5, 6):
            r_sh = np.roll(rec, shift)
            for angle in [0, np.pi]:
                temp_rec = r_sh * np.exp(-1j * angle)
                phase_offset = np.angle(np.mean(temp_rec * np.conj(ref)))
                temp_rec_final = temp_rec * np.exp(-1j * phase_offset)
                mse = np.mean(np.abs(ref - temp_rec_final)**2)
                snr = 10 * np.log10(1.0 / (mse + 1e-12))
                if snr > best_snr:
                    best_snr = snr
                    best_rec = temp_rec_final
        return best_snr, best_rec

    def get_safe_rrc_taps(sps, num_taps, beta=0.1):
        t = np.arange(num_taps) - (num_taps - 1) / 2.0
        t = t / sps
        h = np.zeros_like(t)
        for i, val in enumerate(t):
            if val == 0.0:
                h[i] = 1.0 - beta + (4 * beta / np.pi)
            elif np.abs(np.abs(val) - 1.0 / (4 * beta)) < 1e-6:
                h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + 
                                              (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            else:
                num = np.sin(np.pi * val * (1 - beta)) + 4 * beta * val * np.cos(np.pi * val * (1 + beta))
                den = np.pi * val * (1 - (4 * beta * val)**2)
                h[i] = num / den
        return h / np.sqrt(np.sum(h**2))

    def apply_cd_numpy(sig, baud_rate, sps, D=17.0, L=80.0, lambda_=1550e-9):
        fs = baud_rate * sps
        c = 299792458.0
        beta2 = -(lambda_**2) / (2 * np.pi * c) * (D * 1e-6)
        f = np.fft.fftfreq(sig.shape[0], 1/fs)
        omega = 2 * np.pi * f
        H = np.exp(-1j * (beta2 / 2) * (L * 1e3) * omega**2)
        sig_f = np.fft.fft(sig, axis=0)
        return np.fft.ifft(sig_f * H[:, np.newaxis], axis=0)

    def pam4_lms_equalizer_numpy(rx, tx_train, const, taps=3, lr_da=5e-3, lr_dd=1e-3, train_len=5000):
        n_syms = rx.shape[0]
        pad = taps // 2
        rx_pad = np.pad(rx, ((pad, pad), (0, 0)), mode='constant')

        W = np.zeros((2, 2, taps), dtype=complex)
        W[0, 0, taps//2] = 1.0
        W[1, 1, taps//2] = 1.0

        out = np.zeros_like(rx, dtype=complex)

        for i in range(n_syms):
            x = rx_pad[i : i+taps, :]
            x = x[::-1, :] 

            y0 = np.sum(W[0, 0] * x[:, 0] + W[0, 1] * x[:, 1])
            y1 = np.sum(W[1, 0] * x[:, 0] + W[1, 1] * x[:, 1])
            y = np.array([y0, y1])
            out[i] = y

            if i < train_len:
                d = tx_train[i]
                lr = lr_da
            else:
                dist0 = np.abs(y0 - const)
                dist1 = np.abs(y1 - const)
                d = np.array([const[np.argmin(dist0)], const[np.argmin(dist1)]])
                lr = lr_dd

            e = d - y

            W[0, 0] += lr * e[0] * np.conj(x[:, 0])
            W[0, 1] += lr * e[0] * np.conj(x[:, 1])
            W[1, 0] += lr * e[1] * np.conj(x[:, 0])
            W[1, 1] += lr * e[1] * np.conj(x[:, 1])

        return out

    def cpr_4th_power_numpy(sig, mu=0.01):
        n = len(sig)
        out = np.zeros(n, dtype=complex)
        phi = 0.0
        for i in range(n):
            out[i] = sig[i] * np.exp(-1j * phi)
            err = np.angle(out[i]**4) / 4.0
            phi += mu * err
        return out

    def get_ber(tx_aligned, rx_aligned, const):
        """完全使用 NumPy 实现的 BER 计算"""
        dist_tx = np.abs(tx_aligned[:, None] - const[None, :])
        tx_idx = np.argmin(dist_tx, axis=-1)

        dist_rx = np.abs(rx_aligned[:, None] - const[None, :])
        rx_idx = np.argmin(dist_rx, axis=-1)

        ser_errors = np.sum(tx_idx != rx_idx)
        total_syms = tx_idx.shape[0]
        ser = ser_errors / total_syms

        M = len(const)
        num_bits = int(np.log2(M))

        rx_bits = np.array([(rx_idx >> i) & 1 for i in range(num_bits)])
        tx_bits = np.array([(tx_idx >> i) & 1 for i in range(num_bits)])

        ber_errors = np.sum(rx_bits != tx_bits)
        total_bits = total_syms * num_bits
        ber = ber_errors / total_bits

        return float(ber), int(ber_errors), total_bits, float(ser)

    return (
        apply_cd_numpy,
        cpr_4th_power_numpy,
        get_ber,
        get_safe_rrc_taps,
        get_snr_pam4_numpy,
        np,
        pam4_lms_equalizer_numpy,
        plt,
        sp_signal,
        time,
    )


@app.cell
def _(mo):
    D_slider = mo.ui.slider(start=0, stop=20, step=0.5, value=17.0, label="色散系数 (ps/nm/km)")
    L_slider = mo.ui.slider(start=0, stop=200, step=10, value=80.0, label="光纤长度 (km)")
    lw_slider = mo.ui.slider(start=0, stop=100e3, step=5e3, value=10e3, label="激光器线宽 (Hz)") 
    snr_slider = mo.ui.slider(start=0.005, stop=0.05, step=0.01, value=0.04, label="噪声幅度")

    mo.md(f"""
    ### 交互式控制台 (NumPy)

    | 物理参数 | 实时控制滑块 |
    | :--- | :--- |
    | **光纤色散系数** | {D_slider} |
    | **光纤传输距离** | {L_slider} |
    | **激光器线宽 (相位噪声)** | {lw_slider} |
    | **信道 AWGN 噪声幅度** | {snr_slider} |
    """)
    return D_slider, L_slider, lw_slider, snr_slider


@app.cell
def _(get_safe_rrc_taps, np, sp_signal):
    num_syms = 32768
    baud_rate = 32e9
    _pam_levels = np.array([-3, -1, 1, 3], dtype=complex)
    const = _pam_levels / np.sqrt(np.mean(np.abs(_pam_levels)**2))

    np.random.seed(42)
    _tx_indices = np.random.randint(0, 4, (num_syms, 2))
    tx_syms = const[_tx_indices]

    h_rrc = get_safe_rrc_taps(sps=2, num_taps=65, beta=0.1)
    _tx_sig = sp_signal.upfirdn(h_rrc, tx_syms, up=2, axis=0)
    delay = 32
    tx_2sps = _tx_sig[delay : delay + num_syms * 2]
    return baud_rate, const, delay, h_rrc, num_syms, tx_2sps, tx_syms


@app.cell
def _(
    D_slider,
    L_slider,
    apply_cd_numpy,
    baud_rate,
    delay,
    h_rrc,
    lw_slider,
    np,
    num_syms,
    plt,
    snr_slider,
    sp_signal,
    tx_2sps,
):
    _theta = np.pi / 4 * 0.6
    _J_rot = np.array([[np.cos(_theta), np.sin(_theta)], [-np.sin(_theta), np.cos(_theta)]])
    _rx_rot = tx_2sps @ _J_rot.T

    _sigma = np.sqrt(2 * np.pi * lw_slider.value * (1.0 / (baud_rate * 2)))
    # 固定噪声种子以防止每次重绘时眼图乱跳
    np.random.seed(99)
    _phase_noise = np.exp(1j * np.cumsum(_sigma * np.random.standard_normal((_rx_rot.shape[0], 1)), axis=0))
    _rx_pn = _rx_rot * _phase_noise

    _rx_cd = apply_cd_numpy(_rx_pn, baud_rate, sps=2, D=D_slider.value, L=L_slider.value)

    np.random.seed(123)
    _noise = (np.random.standard_normal(_rx_cd.shape) + 1j * np.random.standard_normal(_rx_cd.shape)) * snr_slider.value
    rx_channel_out = _rx_cd + _noise

    # 接收端 CDC 与 MF
    _rx_cdc = apply_cd_numpy(rx_channel_out, baud_rate, sps=2, D=D_slider.value, L=-L_slider.value)
    _rx_mf = sp_signal.upfirdn(h_rrc, _rx_cdc, up=1, axis=0)
    rx_mf_aligned = _rx_mf[delay : delay + num_syms * 2]

    fig_eye_pam4_np, _ax_eye = plt.subplots(1, 2, figsize=(10, 4))
    for _i in range(2):
        _traces = rx_mf_aligned[10000:10000+400*2, _i].real.reshape(-1, 5)
        _ax_eye[_i].plot(np.linspace(0, 2, 5), _traces.T, 'b-', alpha=0.1)
        _ax_eye[_i].set_title(f'PAM4 Eye After CDC & MF - Pol {_i}')
        _ax_eye[_i].grid(True, linestyle=':', alpha=0.5)
    fig_eye_pam4_np
    return (rx_mf_aligned,)


@app.cell
def _(const, mo, np, pam4_lms_equalizer_numpy, rx_mf_aligned, time, tx_syms):
    _start_lms = time.time()

    _rx_1sps = rx_mf_aligned[::2]
    _rx_1sps /= np.sqrt(np.mean(np.abs(_rx_1sps)**2, axis=0))

    # 这将是一段漫长的 Python for 循环等待...
    _rx_eq = pam4_lms_equalizer_numpy(_rx_1sps, tx_syms, const, taps=3, train_len=5000)

    start_idx = 6000
    _rx_eq_cut_temp = _rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + _rx_eq_cut_temp.shape[0]]

    rx_eq_cut = _rx_eq_cut_temp / np.sqrt(np.mean(np.abs(_rx_eq_cut_temp)**2, axis=0))

    _end_lms = time.time()
    lms_time = _end_lms - _start_lms

    mo.md(f"### LMS opration time: **{lms_time:.2f} s**")
    return rx_eq_cut, tx_final


@app.cell
def _(cpr_4th_power_numpy, np, rx_eq_cut):
    _rx_cpr_list = []
    for _i in range(2):
        _rx_cpr_list.append(cpr_4th_power_numpy(rx_eq_cut[:, _i], mu=0.01))

    rx_cpr_all = np.stack(_rx_cpr_list, axis=1)
    return (rx_cpr_all,)


@app.cell
def _(
    const,
    get_ber,
    get_snr_pam4_numpy,
    mo,
    np,
    plt,
    rx_cpr_all,
    rx_eq_cut,
    tx_final,
):
    _aligned_list = []
    snr_vals = []
    ber_vals = []
    error_counts = []

    for _eval_idx in range(2):
        # Phase and time alignment via SNR calculation
        _snr_v, _aligned_sig = get_snr_pam4_numpy(tx_final[:, _eval_idx], rx_cpr_all[:, _eval_idx])

        # Calculate Bit Error Rate (BER) based on the aligned signals
        _ber, _err_count, _total_bits, _ser = get_ber(tx_final[:, _eval_idx], _aligned_sig, const)

        _aligned_list.append(_aligned_sig)
        snr_vals.append(_snr_v)
        ber_vals.append(_ber)
        error_counts.append(_err_count)

    rx_aligned_all = np.stack(_aligned_list, axis=1)



    # 1. 把 Markdown 赋值给变量
    performance_report = mo.md(f"""
    ### Final System Performance Report

    | Polarization | SNR | BER | Error Count | Total Bits Evaluated |
    | :---: | :---: | :---: | :---: | :---: |
    | **Pol 0 (X-Pol)** | **{snr_vals[0]:.2f} dB** | **{ber_vals[0]:.2e}** | {error_counts[0]} | {_total_bits} |
    | **Pol 1 (Y-Pol)** | **{snr_vals[1]:.2f} dB** | **{ber_vals[1]:.2e}** | {error_counts[1]} | {_total_bits} |

    *Note: Modern coherent optical communication systems typically require a Pre-FEC (Forward Error Correction) BER threshold of $2 \\times 10^{{-2}}$ for error-free transmission.*
    """)

    fig_diag_np = plt.figure(figsize=(15, 5))

    _ax1 = plt.subplot(1, 3, 1)
    _ax1.scatter(rx_eq_cut[-4000:, 0].real, rx_eq_cut[-4000:, 0].imag, s=1, alpha=0.5, color='tab:blue')
    _ax1.set_title("1. After DA/DD-LMS (NumPy)")
    _ax1.axis('equal'); _ax1.grid(True, linestyle=':', alpha=0.6)

    _ax2 = plt.subplot(1, 3, 2)
    _ax2.scatter(rx_aligned_all[-4000:, 0].real, rx_aligned_all[-4000:, 0].imag, s=1, alpha=0.5, color='tab:green')
    _ax2.set_title(f"2. Final DP-PAM4\nSNR: {snr_vals[0]:.2f} dB")
    _ax2.axis('equal'); _ax2.set_ylim([-0.5, 0.5])
    _ax2.scatter(const.real, np.zeros_like(const.real), c='red', marker='x', s=40, alpha=0.9) 
    _ax2.grid(True, linestyle=':', alpha=0.6)

    _phase_diff = np.unwrap(np.angle(rx_cpr_all[-1000:, 0] * np.conj(rx_eq_cut[-1000:, 0])))
    _ax3 = plt.subplot(1, 3, 3)
    _ax3.plot(_phase_diff, color='tab:orange', linewidth=1)
    _ax3.set_title("3. Phase Track Residual")
    _ax3.set_xlabel("Symbols"); _ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    mo.vstack([performance_report, fig_diag_np])
    return


if __name__ == "__main__":
    app.run()
