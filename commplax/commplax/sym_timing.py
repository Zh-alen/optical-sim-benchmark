# Copyright 2026 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
from jax import lax, numpy as jnp
import numpy as np
import equinox as eqx
from equinox import field
from jaxtyping import Array, Float, Int, PyTree
from typing import Any, TypeVar, Callable, Optional, Tuple, Union
import dataclasses as dc
from commplax.jax_util import default_complexing_dtype, default_floating_dtype
from commplax.sym_map import square_qam_decision


InitFn = Callable
ApplyFn = Callable


@dc.dataclass
class SymbolTimingSync():
    """Container for symbol timing synchronization init/apply functions."""
    init: InitFn = None
    apply: ApplyFn = None

    def __iter__(self):
        return iter((self.init, self.apply))


@dc.dataclass
class TEDKernel():
    """Container for baud-rate TED init/update functions."""
    init: Callable = None    # (dims, dtype) -> state (PyTree)
    update: Callable = None  # (state, rx) -> (state, error)


class SymSync(eqx.Module):
    """Symbol timing synchronizer using Gardner TED with interpolation control.

    Recovers symbol timing from a 2 samples-per-symbol (sps) input signal.
    Uses the Gardner Timing Error Detector (TED) and cubic Farrow interpolation
    to track and correct timing offset.

    The synchronizer operates as follows:
    1. Input samples are buffered in a 4-sample FIFO
    2. Cubic interpolation computes samples at fractional delay μ
    3. NCO (η) tracks timing phase, fires strobe at symbol times
    4. Gardner TED computes timing error at symbol strobes
    5. PI loop filter adjusts NCO rate to track timing

    Args:
        kernel: Timing sync kernel (default: symbol_timing_sync())
        state: Initial state tuple (μ, η, strobe, vi, B)
        fifo: Initial sample buffer (default: zeros)
        dtype: Complex dtype for signal processing
        kernel_kwds: Additional arguments for kernel creation

    Returns:
        When called with input sample:
        - Updated SymSync module
        - Tuple (y, e, μ):
            - y: Symbol output (NaN if not at symbol time)
            - e: Timing error from Gardner TED
            - μ: Fractional interpolation delay [0, 1)

    Example:
        Process a 2 sps signal::

            ss = SymSync()
            outputs = []
            for x in signal_2sps:
                ss, (y, e, mu) = ss(x)
                if not jnp.isnan(y):
                    outputs.append(y)

        Or with lax.scan::

            _, (ys, es, mus) = lax.scan(
                lambda s, x: s(x), ss, signal_2sps)
            symbols = ys[~jnp.isnan(ys)]

    Note:
        - Input must be at 2 samples per symbol
        - Output rate is 1 symbol per 2 input samples (on average)
        - Loop converges within ~50-100 symbols typically
    """
    fifo: Array
    state: PyTree
    kernel: PyTree = field(static=True)

    def __init__(self, kernel=None, state=None, fifo=None, dtype=None, kernel_kwds={}):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.kernel = symbol_timing_sync(**kernel_kwds) if kernel is None else kernel
        self.state = self.kernel.init() if state is None else state
        self.fifo = jnp.zeros(4, dtype=dtype) if fifo is None else fifo

    def __call__(self, input):
        """Process one input sample at 2 sps.

        Args:
            input: Single complex sample at 2 samples per symbol

        Returns:
            Tuple of (updated_module, outputs):
            - updated_module: SymSync with updated state
            - outputs: (y, e, μ) where y is symbol (or NaN), e is timing error
        """
        # Shift FIFO and insert new sample
        fifo = jnp.roll(self.fifo, -1, axis=0).at[-1:].set(input)
        # Apply timing sync kernel
        state, out = self.kernel.apply(self.state, fifo)
        ss = dc.replace(self, fifo=fifo, state=state)
        # out = (y, e, μ): symbol output, timing error, fractional delay
        return ss, out


class TEDCell(eqx.Module):
    """Generic baud-rate TED with PI loop filter.

    Wraps any TED kernel (e.g. mm_ted) with a proportional-integral loop
    filter.  Each call processes one symbol-rate sample and returns
    ``tau = Kp * e + integrator``.

    **Loop order** depends on how the caller uses the output:

    - **2nd-order (recommended)**: add an external NCO
      (``tau_pos += tau_delta``) so the PI output is a *rate* correction.
      Use gains from :func:`ted_pi_gains` (negated).  The double
      integration (PI integrator + NCO) gives 40 dB/dec TED-noise
      rejection above the loop bandwidth.

    - **1st-order**: use the PI output directly as a *position*.
      Set ``Ki = -4 × Bn / baud``, ``Kp = 2 × Ki``.  Simpler but only
      20 dB/dec TED-noise rejection.

    **Sign convention**: the gains must be **negative** when
    :class:`ResampleCell` is used (positive tau = sample later) and the
    TED produces positive error for late sampling (e.g. MM TED with
    negative Kd).  This matches :class:`SymSync` which uses negative K1/K2.

    Args:
        kernel: TEDKernel providing init() and update() methods
        Kp: Proportional loop gain
        Ki: Integral loop gain
        dims: Number of polarizations (default 2)
        dtype: Complex dtype for TED state
        ted_state: Initial TED state (default: from kernel.init)
        integrator: Initial PI integrator value (default 0)

    Returns:
        When called with rx (dims,):
        (updated_module, tau) where tau is PI output (rate or position)

    Example::

        # 2nd-order PLL (recommended): PI + NCO
        kernel = mm_ted(const=const_16qam, Kd=Kd_mm)
        Kp, Ki = ted_pi_gains(3e6, baud)
        Kp, Ki = -Kp, -Ki           # negate for correct feedback
        cell = TEDCell(kernel, Kp=Kp, Ki=Ki)

        # In the TR scan loop:
        cell, tau_delta = cell(rx)   # PI output = rate
        tau_pos = tau_pos + tau_delta  # NCO integration
    """
    ted_state: PyTree
    integrator: Array
    Kp: float = field(static=True)
    Ki: float = field(static=True)
    kernel: TEDKernel = field(static=True)

    def __init__(self, kernel, Kp, Ki, dims=2, dtype=None,
                 ted_state=None, integrator=None):
        self.kernel = kernel
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.ted_state = kernel.init(dims, dtype) if ted_state is None else ted_state
        self.integrator = jnp.array(0.0) if integrator is None else jnp.asarray(integrator)

    def __call__(self, rx):
        ted_state_new, e = self.kernel.update(self.ted_state, rx)
        integrator_new = self.integrator + self.Ki * e
        tau = self.Kp * e + integrator_new
        return dc.replace(self, ted_state=ted_state_new,
                          integrator=integrator_new), tau


# Cubic Farrow interpolation coefficients (module-level constant).
# After flip: μ=0 → fifo[1], μ=1 → fifo[2].
_FARROW_B = 1/2 * jnp.flip(jnp.array(
    [[ 1, -1, -1,  1],
     [-1,  3, -1, -1],
     [ 0,  0,  2,  0]], dtype=float),
    axis=1)


class FarrowResampleCell(eqx.Module):
    """Decimating cubic Farrow interpolator with timing correction.

    Downsamples an N sps signal to 1 sps using cubic Farrow interpolation.
    A counter fires every ``sps`` input samples, producing one output.
    The interpolation position is shifted by ``tau`` (timing correction
    in UI) relative to the nominal sample point.

    Uses a 4-sample FIFO for the cubic (4-point) Farrow filter.

    Args:
        sps: Samples per symbol (default 2).
        dims: Signal dimensions (default 2 for dual-pol).
        dtype: Complex dtype.

    Returns:
        (updated_module, (y, valid))
    """
    fifo: Array       # (4, dims) last 4 input samples
    counter: Array    # modulo sps
    sps: int = field(static=True)

    def __init__(self, sps=2, dims=2, dtype=None, fifo=None, counter=None):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.sps = sps
        self.fifo = jnp.zeros((4, dims), dtype=dtype) if fifo is None else fifo
        self.counter = jnp.array(0) if counter is None else counter

    def __call__(self, x, tau):
        fifo_new = jnp.concatenate([self.fifo[1:], x[None]], axis=0)
        valid = self.counter == self.sps - 1

        # fifo_new = [s_{t-3}, s_{t-2}, s_{t-1}, s_t]
        # Piecewise Farrow: keep mu ∈ [0,1] for each segment so the
        # cubic stays within its designed interpolation range.
        delta = jnp.clip(tau * self.sps, -1.0, 1.0)

        # Early (delta ≤ 0): interpolate fifo[1]–fifo[2] (s_{t-2} to s_{t-1})
        # u = 1+delta: delta=-1→u=0→fifo[1], delta=0→u=1→fifo[2]
        u_early = 1.0 + delta
        m_early = jnp.array([u_early**2, u_early, 1.0])
        y_early = jnp.dot(m_early, jnp.dot(_FARROW_B, fifo_new))

        # Late (delta > 0): interpolate fifo[2]–fifo[3] (s_{t-1} to s_t)
        # Shift FIFO by 1; extrapolate missing future sample linearly.
        s_ext = 2 * fifo_new[3] - fifo_new[2]
        fifo_late = jnp.stack([fifo_new[1], fifo_new[2], fifo_new[3], s_ext])
        u_late = delta  # delta=0→u=0→fifo[2]=s_{t-1}, delta=1→u=1→fifo[3]=s_t
        m_late = jnp.array([u_late**2, u_late, 1.0])
        y_late = jnp.dot(m_late, jnp.dot(_FARROW_B, fifo_late))

        y = jnp.where(delta <= 0, y_early, y_late)

        counter_new = (self.counter + 1) % self.sps
        cell = dc.replace(self, fifo=fifo_new, counter=counter_new)
        return cell, (y, valid)


ResampleCell = FarrowResampleCell


def symbol_timing_sync():
    """Create Gardner-based symbol timing synchronizer for 2 sps signals.

    Implements the Gardner Timing Error Detector (TED) with cubic Farrow
    interpolation for symbol timing recovery. Operates on 2 samples per
    symbol input and outputs 1 symbol per 2 inputs on average.

    Algorithm overview:
        1. **Interpolation**: Cubic Farrow structure computes sample at
           fractional delay μ: xI = (b @ fifo) @ [μ², μ, 1]

        2. **NCO (Numerically Controlled Oscillator)**:
           - η tracks timing phase, decrements by W each sample
           - W = 0.5 + v (nominal rate + loop filter output)
           - When η < 0: symbol strobe fires, η wraps by +1

        3. **Gardner TED**: At symbol time (strobe=2), computes error:
           e = Re{x_mid * (x_prev - x_curr)}
           where x_mid is transition sample, x_prev/x_curr are symbols

        4. **Loop filter**: 2nd-order PI controller
           - vp = K1 * e (proportional)
           - vi += K2 * e (integral)
           - v = vp + vi adjusts NCO rate W

    State variables:
        - μ (mu): Fractional interpolation delay [0, 1)
        - η (eta): NCO timing phase accumulator
        - strobe: Output indicator
            - 0: Idle (between samples)
            - 1: Transition time (midpoint)
            - 2: Symbol time, compute TED (first symbol after transition)
            - 3: Symbol time (subsequent)
        - vi: Loop filter integrator
        - B: TED buffer [transition_sample, prev_symbol_sample]

    Loop gains:
        K1 = -2.46e-3 (proportional)
        K2 = -8.2e-6 (integral)
        These provide ~1% loop bandwidth with critical damping.

    Returns:
        SymbolTimingSync with init() and apply() methods:
        - init(dtype): Initialize state
        - apply(state, fifo): Process 4-sample FIFO, return (new_state, outputs)

    References:
        [1] Rice, Michael. "Digital Communications: A Discrete-Time Approach"
            pp. 493, Gardner timing recovery implementation

    Note:
        - Input must be 2 sps after matched filtering
        - Output y is NaN when strobe < 2 (not at symbol time)
        - Convergence typically within 50-100 symbols
    """
    # PI loop filter gains (negative for correct feedback sign)
    # K1: proportional gain, K2: integral gain
    # These values give ~1% normalized loop bandwidth
    K1 = -2.46e-3
    K2 = -8.2e-6

    # Cubic Farrow interpolation coefficients
    # Computes xI = (b @ x) @ [μ², μ, 1] for fractional delay μ
    # Rows correspond to μ² coefficient, μ coefficient, constant term
    b = 1/2 * jnp.flip(jnp.array(
        [[ 1, -1, -1,  1],   # μ² coefficients
         [-1,  3, -1, -1],   # μ coefficients
         [ 0,  0,  2,  0]],  # constant (μ⁰) coefficients
        dtype=default_complexing_dtype()),
        axis=1)

    def init(dtype=None):
        """Initialize timing synchronizer state.

        Args:
            dtype: Complex dtype for buffers

        Returns:
            State tuple: (μ_next, η_next, strobe, vi, B)
        """
        dtype = default_complexing_dtype() if dtype is None else dtype
        η_next = 0.     # NCO timing phase
        μ_next = 0.     # Fractional interpolation delay
        strobe = 0      # Output strobe indicator
        B = jnp.zeros(2, dtype=dtype)  # TED buffer: [transition, prev_symbol]
        vi = 0.         # Loop filter integrator
        state = μ_next, η_next, strobe, vi, B
        return state

    def apply(state, x):
        """Process one sample through timing synchronizer.

        Args:
            state: (μ_next, η_next, strobe, vi, B) state tuple
            x: 4-sample FIFO buffer (newest sample last)

        Returns:
            (new_state, outputs) where outputs = (y, e, μ):
            - y: Interpolated symbol (NaN if not at symbol time)
            - e: Timing error from Gardner TED
            - μ: Current fractional delay
        """
        μ_next, η_next, strobe, vi, B = state

        μ = μ_next
        η = η_next

        # Cubic interpolation at fractional delay μ
        # m = [μ², μ, 1] for polynomial evaluation
        m = jnp.power(μ, jnp.arange(2,-1,-1))
        xI = jnp.dot(jnp.dot(b, x), m)

        # Output symbol only at strobe (strobe >= 2)
        y = jnp.where(strobe//2 == 1, xI, jnp.nan)

        # Gardner TED: e = Re{mid * (prev - curr)} at symbol strobe
        # B[0] = transition sample, B[1] = previous symbol, xI = current symbol
        e  = jnp.where(
            strobe == 2,
            B[0].real * (B[1].real - xI.real) + B[0].imag * (B[1].imag - xI.imag),
            0.,
            )

        # PI loop filter
        vp = K1 * e           # Proportional term
        vi = vi + K2 * e      # Integral term (accumulates)
        v = vp + vi           # Total loop filter output
        W = 1 / 2 + v         # NCO step: nominal (0.5) + correction

        # Update TED buffer based on strobe state
        # strobe 1,2: shift in new sample (transition or symbol)
        # strobe 0: hold
        # strobe 3: reset prev_symbol
        B = lax.cond(
            (strobe == 1) | (strobe == 2),
            lambda *_: jnp.array([xI, B[0]], dtype=B.dtype),
            lambda *_: lax.cond(
                strobe == 0,
                lambda *_: B,
                lambda *_: jnp.array([xI, 0.], dtype=B.dtype),
            )
        )

        # NCO update: decrement η by step W
        η_next = η - W

        # Strobe logic: when η crosses zero, fire strobe and wrap
        # μ_next = η/W gives fractional delay for interpolation
        η_next, strobe, μ_next = lax.cond(
            η_next < 0,
            lambda *_: (η_next+1, 2+strobe//2, η/W),  # Strobe fired, wrap η
            lambda *_: (η_next,   0+strobe//2, μ),    # No strobe, hold μ
        )

        state = μ_next, η_next, strobe, vi, B
        return state, (y, e, μ)

    return SymbolTimingSync(init, apply)


def mm_ted(const, Kd):
    """Create a Mueller-Muller Timing Error Detector kernel.

    The MM TED computes timing error from the difference between
    current and previous hard decisions crossed with received samples.
    Error is averaged across polarizations and normalized by Kd.

    Args:
        const: 1-D constellation points (e.g. 16QAM)
        Kd: TED gain normalization (from pulse shape derivative at T)

    Returns:
        TEDKernel with init() and update() methods.

    Example::

        kernel = mm_ted(const=const_16qam, Kd=Kd_mm)
        cell = TEDCell(kernel, Kp=0.01, Ki=1e-5)
    """
    const = jnp.asarray(const)
    Kd = float(Kd)

    def init(dims=2, dtype=None):
        dtype = default_complexing_dtype() if dtype is None else dtype
        return (jnp.zeros(dims, dtype=dtype),   # prev_rx
                jnp.zeros(dims, dtype=dtype))    # prev_dec

    def update(state, rx):
        prev_rx, prev_dec = state
        dec = const[jnp.argmin(jnp.abs(rx[:, None] - const[None, :]), axis=1)]
        e_raw = jnp.mean(jnp.real(
            jnp.conj(prev_dec) * rx - jnp.conj(dec) * prev_rx))
        return (rx, dec), e_raw / Kd

    return TEDKernel(init, update)


def mm_ted_fast(const, Kd):
    """Create a fast Mueller-Muller TED using PAM-rounded decisions.

    Drop-in replacement for :func:`mm_ted` that uses
    :func:`~commplax.sym_map.square_qam_decision` (O(1) per symbol)
    instead of argmin over the constellation (O(M)).

    Args:
        const: 1-D constellation points (e.g. 16QAM, may be normalized).
        Kd: TED gain normalization (from pulse shape derivative at T).

    Returns:
        TEDKernel with init() and update() methods.
    """
    const = jnp.asarray(const)
    L = len(const)
    Kd = float(Kd)
    # Scale factor: pamdecision expects {-3,-1,1,3,...} grid
    std_pow = 2.0 * (L - 1) / 3
    const_pow = float(jnp.mean(jnp.abs(const) ** 2))
    scale = jnp.sqrt(std_pow / const_pow)

    def init(dims=2, dtype=None):
        dtype = default_complexing_dtype() if dtype is None else dtype
        return (jnp.zeros(dims, dtype=dtype),
                jnp.zeros(dims, dtype=dtype))

    def update(state, rx):
        prev_rx, prev_dec = state
        dec = square_qam_decision(rx * scale, L) / scale
        e_raw = jnp.mean(jnp.real(
            jnp.conj(prev_dec) * rx - jnp.conj(dec) * prev_rx))
        return (rx, dec), e_raw / Kd

    return TEDKernel(init, update)


def mm_ted_gain(pulse, sps, const):
    """Compute the Mueller-Muller TED gain Kd from pulse shape parameters.

    Args:
        pulse: Tuple describing the pulse shape, e.g. ``('rcos', rolloff)``.
        sps: Samples per symbol.
        const: 1-D constellation points (e.g. 16QAM).

    Returns:
        Kd (float): TED gain normalization constant.
    """
    shape, *params = pulse
    ref_pow = np.mean(np.abs(np.asarray(const))**2)
    if shape == 'rcos':
        rolloff = params[0]
        from commplax.filter import rcosdesign
        h_rrc = rcosdesign(rolloff, 32, sps, shape='sqrt')
        h_rrc /= np.sqrt(np.sum(h_rrc**2))
        h_rc = np.convolve(h_rrc, h_rrc)
        center = len(h_rc) // 2
        # Finite difference gives derivative per sample; multiply by sps
        # to convert to derivative per UI so that e/Kd = τ_UI.
        dp_T = (h_rc[center + sps + 1] - h_rc[center + sps - 1]) / 2
        Kd = 2 * ref_pow * dp_T * sps
    else:
        raise ValueError(f"Unknown pulse shape: {shape!r}")
    return float(Kd)


def ted_pi_gains(Bn, baud, zeta=None):
    """Compute PI loop filter gains for a 2nd-order timing loop.

    Args:
        Bn: Loop noise bandwidth (Hz).
        baud: Symbol rate (Hz).
        zeta: Damping factor. Default ``1/sqrt(2)`` (critical damping).

    Returns:
        (Kp, Ki): Proportional and integral gains (floats).
    """
    if zeta is None:
        zeta = 1 / np.sqrt(2)
    theta_n = Bn / baud
    denom = 1 + 2 * zeta * theta_n
    Kp = 4 * zeta * theta_n / denom
    Ki = 4 * theta_n**2 / denom**2
    return float(Kp), float(Ki)


def centroid_ted(num_taps):
    """Create a centroid-based Timing Error Detector (TED).

    Returns a pure function that computes timing error from an equalizer's
    tap energy centroid. The centroid shifts proportionally to sampling
    phase error, providing a feedback signal for timing recovery.

    The TED accounts for MIMOCell's reversed tap indexing: physical delay
    k corresponds to state index (num_taps - 1 - k).

    Args:
        num_taps: Number of equalizer taps (must match MIMOCell's num_taps)

    Returns:
        A function (eq: MIMOCell) -> timing_error (scalar float).
        Positive error means sampling early (centroid > center).

    Example::

        ted = centroid_ted(11)
        error = ted(mimo_cell)  # scalar timing error
    """
    center = (num_taps - 1) / 2.0
    k = num_taps - 1 - jnp.arange(num_taps)

    def ted(eq):
        # state[0] shape: (dims, dims, num_taps) for up=1,
        # or (dims, dims, num_taps, up) for up>1.
        # Sum energy over all axes except the tap axis (always second-to-last
        # for up>1, or last for up=1). Use dynamic approach: sum all axes
        # except the last, which is always the tap dimension after h_phase
        # reindexing in MIMOCell. For up=1 the shape is (dims, dims, num_taps).
        taps = eq.state[0]
        tap_energy = jnp.sum(jnp.abs(taps) ** 2, axis=tuple(range(taps.ndim - 1)))
        centroid = jnp.sum(k * tap_energy) / (jnp.sum(tap_energy) + 1e-10)
        return centroid - center

    return ted


class TimingLoop(eqx.Module):
    """Closed-loop timing recovery using resampler + MIMO equalizer feedback.

    Composes a VarRateResampler with a MIMOCell equalizer in a feedback loop.
    The equalizer's tap centroid (via a pluggable TED) drives a PI loop
    filter that adjusts the resampler's sampling phase.

    Architecture::

        N sps --> Resampler(eps) --> 1 sps --> MIMOCell --> output
                       ^                          |
                       +--- kp*e + integrator ---+

    The resampler ratio should be set to ``1 / sps`` so that N sps input is
    decimated to 1 sps before the equalizer.  The loop operates
    sample-by-sample: each input sample produces either a valid 1 sps output
    (after resampling + equalization) or NaN (when the resampler skips).
    Feedback uses the *previous* timing error since the resampler must run
    before the equalizer.

    The loop filter is PI (proportional-integral):
    ``eps = -(kp * timing_error + integrator)``, where the integrator
    accumulates ``ki * timing_error`` on each valid output.  With the
    default ``ki=0`` the loop is P-only, which suffices when the nominal
    resampler ratio exactly matches the true sps.  Set ``ki`` to a small
    positive value (e.g. 1e-5) to track residual rate mismatch.

    Args:
        resampler: VarRateResampler with ratio = 1/sps (e.g. 0.5 for 2 sps,
            0.8 for 1.25 sps). The resampler operates on scalar samples;
            for multi-dimensional inputs (dims > 1), resample each dimension
            independently before feeding into this loop.
        equalizer: MIMOCell configured for 1/1 operation (up=1, down=1)
        kp: Proportional loop gain. Default 0.5.
        ki: Integral loop gain. Default 0 (P-only). Use a very small value
            (e.g. 1e-5) when the nominal ratio has residual rate error.
        ted: Timing error detector function (eq -> error). Default centroid_ted.
        timing_error: Initial timing error state. Default 0.
        integrator: Initial integrator state. Default 0.

    Returns:
        When called with a single input sample:
        (updated_loop, (y, timing_error, valid)) where:
        - y: equalized symbol output, shape (dims,) (NaN if resampler skipped)
        - timing_error: current timing error from TED
        - valid: boolean, True if output is valid

    Example::

        from commplax.resampler import VarRateResampler
        from commplax.equalizer import MIMOCell
        from commplax import adaptive_kernel as ak, module as mod

        # 2 sps input
        loop = TimingLoop(
            resampler=VarRateResampler(ratio=0.5),
            equalizer=MIMOCell(11, dims=1, up=1, down=1,
                               kernel=ak.rls_cma(const=const), update_mode=1),
            kp=0.5,
        )
        loop_final, (y, te, valid) = mod.scan_with()(loop, signal_2sps)
        y_valid = y[valid]

        # 1.25 sps with integral term for rate tracking
        loop = TimingLoop(
            resampler=VarRateResampler(ratio=0.8),
            equalizer=MIMOCell(11, dims=1, up=1, down=1,
                               kernel=ak.rls_cma(const=const), update_mode=1),
            kp=0.5,
            ki=1e-5,
        )
    """
    resampler: eqx.Module
    equalizer: eqx.Module
    timing_error: Array
    integrator: Array
    kp: float = field(static=True)
    ki: float = field(static=True)
    ted: Callable = field(static=True)

    def __init__(self, resampler, equalizer, kp=0.5, ki=0.0, ted=None,
                 timing_error=None, integrator=None):
        self.resampler = resampler
        self.equalizer = equalizer
        self.kp = kp
        self.ki = ki
        self.ted = centroid_ted(equalizer.state[0].shape[2]) if ted is None else ted
        self.timing_error = jnp.array(0.0) if timing_error is None else jnp.asarray(timing_error)
        self.integrator = jnp.array(0.0) if integrator is None else jnp.asarray(integrator)

    def __call__(self, x):
        """Process one input sample through the timing loop.

        Args:
            x: Single complex input sample (at N sps)

        Returns:
            Tuple of (updated_loop, (y, timing_error, valid)):
            - y: Equalized output, shape (dims,) (NaN if resampler skipped)
            - timing_error: TED output after equalization
            - valid: True if a symbol was produced
        """
        dims = self.equalizer.fifo.shape[1]

        # 1. PI loop filter: eps from previous timing error + integrator
        #    positive error -> sampling early -> negative eps -> delay output
        eps = -(self.kp * self.timing_error + self.integrator)

        # 2. Resample with phase adjustment
        rsplr = dc.replace(self.resampler, acc_phase=eps)
        rsplr_new, y_arr = rsplr(x)
        y_1sps = y_arr[0]
        valid = ~jnp.isnan(y_1sps)

        # 3. Conditionally run equalizer + TED (skip on NaN to protect state)
        def _update(eq, te):
            eq_new, y_eq = eq((jnp.atleast_1d(y_1sps), jnp.zeros(dims)))
            te_new = self.ted(eq_new)
            return eq_new, y_eq, te_new

        def _hold(eq, te):
            return eq, jnp.full(dims, jnp.nan + 0j, dtype=eq.fifo.dtype), te

        eq_new, y_eq, te_new = lax.cond(valid, _update, _hold, self.equalizer, self.timing_error)

        # 4. Update integrator (only on valid outputs)
        integrator_new = jnp.where(valid,
                                   self.integrator + self.ki * te_new,
                                   self.integrator)

        # 5. Return updated loop and outputs
        loop = dc.replace(self, resampler=rsplr_new, equalizer=eq_new,
                          timing_error=te_new, integrator=integrator_new)
        return loop, (y_eq, te_new, valid)
