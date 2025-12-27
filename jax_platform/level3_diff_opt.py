import jax
import jax.numpy as jnp
from jax import random, lax, value_and_grad
import time

# 物理常数
C = 299792458
LAMBDA_0 = 1550e-9

@jax.jit(static_argnums=(0, 2))
def rrc_taps(sps, alpha, span):
    t = jnp.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    denom = 1.0 - (2.0 * alpha * t)**2
    h = jnp.where(
        jnp.abs(denom) < 1e-10,
        (alpha / jnp.sqrt(2.0)) * ((1.0 + 2.0 / jnp.pi) * jnp.sin(jnp.pi / (4.0 * alpha)) + (1.0 - 2.0 / jnp.pi) * jnp.cos(jnp.pi / (4.0 * alpha))),
        jnp.sinc(t) * jnp.cos(jnp.pi * alpha * t) / denom
    )
    return h / jnp.sqrt(jnp.sum(h**2))

def ssfm_propagation(signal, distance_km, sps, baud_rate, D=17.0, gamma=1.3, n_steps=80):
    dz = (distance_km * 1e3) / n_steps
    fs = baud_rate * sps
    beta2 = -D * (LAMBDA_0**2) / (2 * jnp.pi * C) * 1e-6
    omega = 2 * jnp.pi * jnp.fft.fftfreq(signal.shape[0], 1/fs)
    exp_linear = jnp.exp(-1j * 0.5 * beta2 * (omega**2) * (dz / 2))
    
    def ssfm_body(carry, _):
        A = jnp.fft.ifft(jnp.fft.fft(carry) * exp_linear)
        A = A * jnp.exp(1j * gamma * jnp.abs(A)**2 * dz)
        A = jnp.fft.ifft(jnp.fft.fft(A) * exp_linear)
        return A, None
    final_signal, _ = lax.scan(ssfm_body, signal, None, length=n_steps)
    return final_signal

def cma_core(mu, signal, n_taps=11, n_iter=10):
    w = jnp.zeros(n_taps, dtype=jnp.complex64).at[n_taps // 2].set(1.0)
    def cma_step(carry_w, _):
        y = jnp.convolve(signal, carry_w, mode='same')
        err = 1.0 - jnp.abs(y)**2
        new_w = carry_w + mu * jnp.mean(err) * carry_w
        return new_w, None
    final_w, _ = lax.scan(cma_step, w, None, length=n_iter)
    return jnp.convolve(signal, final_w, mode='same')

def cma_loss_fn(mu, signal):
    equalized_sig = cma_core(mu, signal)
    return jnp.mean((jnp.abs(equalized_sig)**2 - 1.0)**2)

@jax.jit
def train_step(mu, signal, lr=1e-5):
    loss, grad = value_and_grad(cma_loss_fn)(mu, signal)
    return mu - lr * grad, loss

def simulate_with_optimization(num_symbols=8192, distance=80.0, snr_db=12.0):
    key = random.PRNGKey(42)
    sps, baud_rate = 16, 32e9
    
    # 信号生成
    bits = random.randint(key, (2 * num_symbols,), 0, 2)
    symbols = ((1 - 2. * bits[0::2]) + 1j * (1 - 2. * bits[1::2])) / jnp.sqrt(2.0)
    h_rrc = rrc_taps(sps, 0.25, 16)
    sig_tx = jnp.convolve(jnp.zeros(num_symbols*sps, dtype=jnp.complex64).at[::sps].set(symbols), h_rrc, mode='same')
    
    # 信道与初步补偿
    sig_ch = ssfm_propagation(sig_tx, distance, sps, baud_rate)
    beta2 = -17.0 * (LAMBDA_0**2) / (2 * jnp.pi * C) * 1e-6
    omega = 2 * jnp.pi * jnp.fft.fftfreq(sig_ch.shape[0], 1/(baud_rate*sps))
    sig_edc = jnp.fft.ifft(jnp.fft.fft(sig_ch) * jnp.exp(1j * 0.5 * beta2 * (omega**2) * (distance * 1e3)))

    # 自动优化 mu
    mu_opt = 1e-4
    for _ in range(30):
        mu_opt, _ = train_step(mu_opt, sig_edc)
    
    # 最终输出与 BER 计算
    sig_final = cma_core(mu_opt, sig_edc)
    sig_rx = jnp.convolve(sig_final, h_rrc, mode='same')
    sig_samples = sig_rx[::sps][:num_symbols]
    recovered_bits = jnp.stack([jnp.real(sig_samples) < 0, jnp.imag(sig_samples) < 0], axis=1).flatten().astype(jnp.int32)
    ber = jnp.mean(recovered_bits != bits)
    
    return ber, mu_opt