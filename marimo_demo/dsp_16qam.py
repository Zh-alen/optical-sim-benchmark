import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
    # Coherent Optical Communication DSP Algorithm Practical Tutorial

    Welcome to the JAX/Commplax optical communication simulation tutorial! In this interactive notebook, we will simulate a **32 GBaud 16QAM** signal, and after it has undergone physical damage, use a DSP algorithm to recover it.

    """
    )
    return (mo,)


@app.cell
def _():
    import time
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import scipy.signal as sp_signal
    import numpy as np

    from commplax import adaptive_kernel as ak
    from commplax.equalizer import MIMOCell, CPR
    from commplax import sym_map

    def get_snr(ref, rec):
        rec = rec / (jnp.sqrt(jnp.mean(jnp.abs(rec)**2)) + 1e-12)
        ref = ref / (jnp.sqrt(jnp.mean(jnp.abs(ref)**2)) + 1e-12)
        best_snr = -100.0
        best_rec = rec
        for shift in range(-5, 6):
            r_sh = jnp.roll(rec, shift)
            for angle in [0, jnp.pi/2, jnp.pi, 3*jnp.pi/2]:
                temp_rec = r_sh * jnp.exp(-1j * angle)
                phase_offset = jnp.angle(jnp.mean(temp_rec * jnp.conj(ref)))
                temp_rec_final = temp_rec * jnp.exp(-1j * phase_offset)
                mse = jnp.mean(jnp.abs(ref - temp_rec_final)**2)
                snr = 10 * jnp.log10(1.0 / (mse + 1e-12))
                if snr > best_snr:
                    best_snr = snr
                    best_rec = temp_rec_final
        return best_snr, best_rec

    def apply_chromatic_dispersion(sig, baud_rate, sps, D=17.0, L=80.0, lambda_=1550e-9):
        fs = baud_rate * sps
        c = 299792458.0
        beta2 = -(lambda_**2) / (2 * jnp.pi * c) * (D * 1e-6)
        f = jnp.fft.fftfreq(sig.shape[0], 1/fs)
        omega = 2 * jnp.pi * f
        H = jnp.exp(-1j * (beta2 / 2) * (L * 1e3) * omega**2)
        sig_f = jnp.fft.fft(sig, axis=0)
        return jnp.fft.ifft(sig_f * H[:, None], axis=0)

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

    def get_ber(tx_aligned, rx_aligned, const):
        """
        计算接收信号的误码率 (BER) 和符号误码率 (SER)
        tx_aligned: 对齐后的理想发送符号 (1D array)
        rx_aligned: 均衡并相位恢复后的接收符号 (1D array)
        const: 标准星座图复数点坐标
        """
        # 1. 找到发送信号和接收信号各自对应的最近星座点索引 (硬判决)
        # 利用 JAX 的广播机制，瞬间计算所有点到所有星座点的欧氏距离
        dist_tx = jnp.abs(tx_aligned[:, None] - const[None, :])
        tx_idx = jnp.argmin(dist_tx, axis=-1)

        dist_rx = jnp.abs(rx_aligned[:, None] - const[None, :])
        rx_idx = jnp.argmin(dist_rx, axis=-1)

        # 计算符号错误率 (SER)
        ser_errors = jnp.sum(tx_idx != rx_idx)
        total_syms = tx_idx.shape[0]
        ser = ser_errors / total_syms

        # 2. 将星座点索引转换为二进制比特 (Bit Demapping)
        M = len(const)
        num_bits = int(np.log2(M)) # 例如 32QAM 就是 5 bits, 16QAM 就是 4 bits

        # 利用位运算提取每一位比特
        rx_bits = jnp.array([(rx_idx >> i) & 1 for i in range(num_bits)])
        tx_bits = jnp.array([(tx_idx >> i) & 1 for i in range(num_bits)])

        # 3. 计算误码率 (BER)
        ber_errors = jnp.sum(rx_bits != tx_bits)
        total_bits = total_syms * num_bits
        ber = ber_errors / total_bits

        return float(ber), int(ber_errors), total_bits, float(ser)

    return (
        CPR,
        MIMOCell,
        ak,
        apply_chromatic_dispersion,
        get_ber,
        get_safe_rrc_taps,
        get_snr,
        jax,
        jnp,
        np,
        plt,
        sp_signal,
        sym_map,
    )


@app.cell
def _(get_safe_rrc_taps, jax, jnp, np, sp_signal, sym_map):
    num_syms = 32768
    baud_rate = 32e9 
    const = jnp.asarray(sym_map.const('16QAM', norm=True))
    _key = jax.random.PRNGKey(42)
    _tx_indices = jax.random.randint(_key, (num_syms, 2), 0, 16)
    tx_syms = const[_tx_indices]

    rrc_taps_len = 65
    h_rrc = get_safe_rrc_taps(sps=2, num_taps=rrc_taps_len, beta=0.1)

    _tx_sig = sp_signal.upfirdn(h_rrc, np.array(tx_syms), up=2, axis=0)
    delay = (rrc_taps_len - 1) // 2
    tx_2sps = jnp.array(_tx_sig[delay : delay + num_syms * 2])
    return baud_rate, const, delay, h_rrc, num_syms, tx_2sps, tx_syms


@app.cell
def _(mo):
    D_slider = mo.ui.slider(start=0, stop=20, step=0.5, value=17.0, label="Dispersion (ps/nm/km)")
    L_slider = mo.ui.slider(start=0, stop=200, step=10, value=80.0, label="Fiber Length (km)")
    lw_slider = mo.ui.slider(start=0, stop=100e3, step=5e3, value=10e3, label="Laser Linewidth (Hz)") 
    snr_slider = mo.ui.slider(start=0.005, stop=0.05, step=0.005, value=0.04, label="Noise Level")

    # Display them in the marimo page using Markdown
    mo.md(f"""
    ### Interactive Physical Impairments Console
    Drag the sliders below, and the underlying fiber impairments, DSP algorithms, and the final constellations will be **recalculated and rendered in real-time**!

    | Physical Parameters | Real-time Control Sliders |
    | :--- | :--- |
    | **Fiber Dispersion Coefficient** | {D_slider} |
    | **Fiber Transmission Distance** | {L_slider} |
    | **Laser Linewidth (Phase Noise)** | {lw_slider} |
    | **Channel AWGN Noise Level** | {snr_slider} |
    """)
    return D_slider, L_slider, lw_slider, snr_slider


@app.cell
def _(
    D_slider,
    L_slider,
    apply_chromatic_dispersion,
    baud_rate,
    jax,
    jnp,
    lw_slider,
    snr_slider,
    tx_2sps,
):
    _theta = jnp.pi / 4 * 0.6
    _J_rot = jnp.array([[jnp.cos(_theta), jnp.sin(_theta)],
                       [-jnp.sin(_theta), jnp.cos(_theta)]])
    _rx_rot = tx_2sps @ _J_rot.T


    _lw = lw_slider.value  
    _Ts = 1.0 / (baud_rate * 2)
    _sigma = jnp.sqrt(2 * jnp.pi * _lw * _Ts)
    _phase_steps = _sigma * jax.random.normal(jax.random.PRNGKey(99), (_rx_rot.shape[0], 1))
    _rx_pn = _rx_rot * jnp.exp(1j * jnp.cumsum(_phase_steps, axis=0))


    _rx_cd = apply_chromatic_dispersion(_rx_pn, baud_rate, sps=2, D=D_slider.value, L=L_slider.value)


    _noise = (jax.random.normal(jax.random.PRNGKey(123), _rx_cd.shape) + 
             1j * jax.random.normal(jax.random.PRNGKey(124), _rx_cd.shape)) * snr_slider.value
    rx_channel_out = _rx_cd + _noise
    return (rx_channel_out,)


@app.cell
def _(
    D_slider,
    L_slider,
    apply_chromatic_dispersion,
    baud_rate,
    delay,
    h_rrc,
    jnp,
    np,
    num_syms,
    plt,
    rx_channel_out,
    sp_signal,
):

    _rx_cdc = apply_chromatic_dispersion(rx_channel_out, baud_rate, sps=2, D=D_slider.value, L=-L_slider.value)
    _rx_mf = sp_signal.upfirdn(h_rrc, np.array(_rx_cdc), up=1, axis=0)
    rx_mf_aligned = jnp.array(_rx_mf[delay : delay + num_syms * 2])

    fig_eye, _ax_eye = plt.subplots(1, 2, figsize=(10, 4))
    for _i in range(2):
        _traces = rx_mf_aligned[10000:10000+400*2, _i].real.reshape(-1, 5)
        _ax_eye[_i].plot(np.linspace(0, 2, 5), _traces.T, 'b-', alpha=0.1)
        _ax_eye[_i].set_title(f'Eye After CDC & MF - Pol {_i}')
        _ax_eye[_i].grid(True, linestyle=':', alpha=0.5)
    fig_eye
    return (rx_mf_aligned,)


@app.cell
def _(MIMOCell, ak, jax, jnp, rx_mf_aligned, tx_syms):
    _rx_1sps = rx_mf_aligned[::2] 
    _rx_1sps = _rx_1sps / jnp.sqrt(jnp.mean(jnp.abs(_rx_1sps)**2, axis=0))

    _kernel = ak.cma(lr=2e-3, R2=1.32) 
    _cma_eq = MIMOCell(num_taps=3, dims=2, kernel=_kernel, up=1, down=1)

    def scan_step(eq_state, x):
        return eq_state(x)

    _cma_trained, _ = jax.lax.scan(scan_step, _cma_eq, _rx_1sps)
    _, _rx_eq = jax.lax.scan(scan_step, _cma_trained, _rx_1sps)

    start_idx = 3000
    _rx_eq_cut_temp = _rx_eq[start_idx:]
    tx_final = tx_syms[start_idx:start_idx + _rx_eq_cut_temp.shape[0]]

    rx_eq_cut = _rx_eq_cut_temp / jnp.sqrt(jnp.mean(jnp.abs(_rx_eq_cut_temp)**2, axis=0))
    return rx_eq_cut, tx_final


@app.cell
def _(CPR, ak, jax, jnp, rx_eq_cut):
    _rx_cpr_list = []
    for _pol_idx in range(2):
        _cpr_module = CPR(kernel=ak.cpr_4thpower_pll(mu=0.02)) 
        _, _pol_cpr_out = jax.lax.scan(lambda c, y: c(y), _cpr_module, rx_eq_cut[:, _pol_idx])
        _rx_cpr_list.append(_pol_cpr_out)

    rx_cpr_all = jnp.stack(_rx_cpr_list, axis=1)
    return (rx_cpr_all,)


@app.cell
def _(const, get_ber, get_snr, jnp, mo, plt, rx_cpr_all, rx_eq_cut, tx_final):
    _aligned_list = []
    snr_vals = []
    ber_vals = []
    error_counts = []

    for _eval_idx in range(2):
        # Phase and time alignment via SNR calculation
        _snr_v, _aligned_sig = get_snr(tx_final[:, _eval_idx], rx_cpr_all[:, _eval_idx])

        # Calculate Bit Error Rate (BER) based on the aligned signals
        _ber, _err_count, _total_bits, _ser = get_ber(tx_final[:, _eval_idx], _aligned_sig, const)

        _aligned_list.append(_aligned_sig)
        snr_vals.append(_snr_v)
        ber_vals.append(_ber)
        error_counts.append(_err_count)

    rx_aligned_all = jnp.stack(_aligned_list, axis=1)

    # 1. Assign the Markdown output to a variable
    performance_report = mo.md(f"""
    ### Final System Performance Report

    | Polarization | SNR | BER | Error Count | Total Bits Evaluated |
    | :---: | :---: | :---: | :---: | :---: |
    | **Pol 0 (X-Pol)** | **{snr_vals[0]:.2f} dB** | **{ber_vals[0]:.2e}** | {error_counts[0]} | {_total_bits} |
    | **Pol 1 (Y-Pol)** | **{snr_vals[1]:.2f} dB** | **{ber_vals[1]:.2e}** | {error_counts[1]} | {_total_bits} |

    *Note: Modern coherent optical communication systems typically require a Pre-FEC (Forward Error Correction) BER threshold of $2 \\times 10^{{-2}}$ for error-free transmission.*
    """)

    fig_diag = plt.figure(figsize=(15, 5))

    _ax1 = plt.subplot(1, 3, 1)
    _ax1.scatter(rx_eq_cut[-4000:, 0].real, rx_eq_cut[-4000:, 0].imag, s=2, alpha=0.3, color='tab:blue')
    _ax1.set_title("1. After CMA (Before CPR)")
    _ax1.axis('equal'); _ax1.grid(True, linestyle=':')

    _ax2 = plt.subplot(1, 3, 2)
    _ax2.scatter(rx_aligned_all[-4000:, 0].real, rx_aligned_all[-4000:, 0].imag, s=2, alpha=0.3, color='tab:green')
    _ax2.scatter(const.real, const.imag, c='red', marker='x', s=30, alpha=0.8)
    _ax2.set_title(f"2. Final Constellation\nSNR: {float(snr_vals[0]):.2f} dB")
    _ax2.axis('equal'); _ax2.grid(True, linestyle=':')

    _phase_diff = jnp.unwrap(jnp.angle(rx_cpr_all[-1000:, 0] * jnp.conj(rx_eq_cut[-1000:, 0])))
    _ax3 = plt.subplot(1, 3, 3)
    _ax3.plot(_phase_diff, color='tab:orange', linewidth=1.5)
    _ax3.set_title("3. PLL Phase Track")
    _ax3.set_xlabel("Symbols"); _ax3.grid(True, linestyle=':')

    plt.tight_layout()

    # 2. Use mo.vstack as the final expression to render both objects vertically
    mo.vstack([performance_report, fig_diag])
    return


if __name__ == "__main__":
    app.run()
