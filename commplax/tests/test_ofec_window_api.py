"""Tests for OFEC sliding-window decoder API.

Tests the high-level ofec_decode_window / ofec_rx_window wrappers and
the OFEC_WINDOW_PRESET.  Each test is atomic: tests preset values,
function signatures, resolver behaviour, and basic decode sanity
independently.
"""

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import pytest

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Preset tests (no JAX compilation, instant)
# ---------------------------------------------------------------------------

class TestWindowPreset:
    """Verify OFEC_WINDOW_PRESET has correct structure and values."""

    def test_preset_is_dict(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert isinstance(OFEC_WINDOW_PRESET, dict)

    def test_preset_has_window_true(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['window'] is True

    def test_preset_osd1_enabled(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['use_osd1'] is True

    def test_preset_siso_enabled(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['siso'] is True

    def test_preset_r_anchor_is_graduated(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        ra = OFEC_WINDOW_PRESET['r_anchor']
        assert isinstance(ra, tuple)
        assert len(ra) == 6
        # Must be non-decreasing
        for i in range(1, len(ra)):
            assert ra[i] >= ra[i - 1]

    def test_preset_ss_r_anchor_above_schedule(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['window_ss_r_anchor'] >= max(
            OFEC_WINDOW_PRESET['r_anchor'])

    def test_preset_window_size(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['window_size'] == 80

    def test_preset_advance(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['window_advance'] == 2

    def test_preset_alpha_beta_values(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['alpha'] == 0.3
        assert OFEC_WINDOW_PRESET['beta'] == 4.0

    def test_preset_hd_iterations(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert OFEC_WINDOW_PRESET['hd_iterations'] == 2
        assert OFEC_WINDOW_PRESET['hd_max_t'] == 1


# ---------------------------------------------------------------------------
# Resolver tests (no JAX compilation, instant)
# ---------------------------------------------------------------------------

class TestWindowResolver:
    """Verify _resolve_window_preset produces correct kwargs."""

    def test_resolver_sets_num_coder_blocks_1600zr(self):
        from commplax.fec.ofec import _resolve_window_preset
        kw = _resolve_window_preset('1600ZR+', sd_iterations=3)
        assert kw['num_coder_blocks'] == 42

    def test_resolver_sets_num_coder_blocks_800zr(self):
        from commplax.fec.ofec import _resolve_window_preset
        kw = _resolve_window_preset('800ZR', sd_iterations=3)
        assert kw['num_coder_blocks'] == 84

    def test_resolver_sets_sd_iterations(self):
        from commplax.fec.ofec import _resolve_window_preset
        for sd in [1, 3, 5]:
            kw = _resolve_window_preset('1600ZR+', sd_iterations=sd)
            assert kw['sd_iterations'] == sd

    def test_resolver_generates_tp_matrices(self):
        from commplax.fec.ofec import _resolve_window_preset
        kw = _resolve_window_preset('1600ZR+', sd_iterations=3)
        assert 'tp_matrices' in kw
        assert len(kw['tp_matrices']) == 3

    def test_resolver_override_takes_precedence(self):
        from commplax.fec.ofec import _resolve_window_preset
        kw = _resolve_window_preset('1600ZR+', sd_iterations=3, alpha=0.5)
        assert kw['alpha'] == 0.5

    def test_resolver_preserves_window_true(self):
        from commplax.fec.ofec import _resolve_window_preset
        kw = _resolve_window_preset('1600ZR+', sd_iterations=3)
        assert kw['window'] is True

    def test_resolver_invalid_mode_raises(self):
        from commplax.fec.ofec import _resolve_window_preset
        with pytest.raises(ValueError, match="Unknown mode"):
            _resolve_window_preset('invalid', sd_iterations=3)


# ---------------------------------------------------------------------------
# Import / export tests (no JAX compilation, instant)
# ---------------------------------------------------------------------------

class TestWindowExports:
    """Verify functions and preset are importable from commplax.fec."""

    def test_import_ofec_decode_window(self):
        from commplax.fec import ofec_decode_window
        assert callable(ofec_decode_window)

    def test_import_ofec_rx_window(self):
        from commplax.fec import ofec_rx_window
        assert callable(ofec_rx_window)

    def test_import_window_preset(self):
        from commplax.fec import OFEC_WINDOW_PRESET
        assert isinstance(OFEC_WINDOW_PRESET, dict)

    def test_sd_preset_differs_from_window_preset(self):
        from commplax.fec import OFEC_SD_PRESET, OFEC_WINDOW_PRESET
        assert 'window' not in OFEC_SD_PRESET or not OFEC_SD_PRESET.get('window')
        assert OFEC_WINDOW_PRESET['window'] is True


# ---------------------------------------------------------------------------
# Functional decode tests (require JAX compilation, marked slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoded_data():
    """Generate a single-frame 1600ZR+ encoded bitstream with BPSK LLRs."""
    from commplax.fec.ofec import (
        ofec_frame_adapt_jax, ofec_encode_jax,
        _OFEC_PAYLOAD_BITS, OFEC_CODER_BLOCKS_1600ZR,
    )
    rng = np.random.default_rng(123)
    payload = rng.integers(0, 2, _OFEC_PAYLOAD_BITS, dtype=np.int32)
    adapted = np.array(ofec_frame_adapt_jax(jnp.array(payload)))
    num_blocks = OFEC_CODER_BLOCKS_1600ZR
    encoded = np.array(ofec_encode_jax(jnp.array(adapted),
                                       num_coder_blocks=num_blocks))
    # BPSK + mild noise (well below threshold for quick test)
    sigma = 0.40
    bpsk = 1.0 - 2.0 * encoded.astype(np.float64)
    noise = rng.normal(0, sigma, len(encoded))
    llrs = (2.0 * (bpsk + noise) / (sigma ** 2)).astype(np.float32)
    return {
        'payload': payload,
        'llrs': jnp.array(llrs),
        'num_blocks': num_blocks,
    }


@pytest.mark.slow
class TestWindowDecode:
    """Functional tests for ofec_decode_window."""

    def test_decode_returns_tuple_of_two(self, encoded_data):
        from commplax.fec import ofec_decode_window
        result = ofec_decode_window(encoded_data['llrs'], mode='1600ZR+')
        assert len(result) == 2

    def test_decode_output_shape(self, encoded_data):
        from commplax.fec import ofec_decode_window
        from commplax.fec.ofec import OFEC_CODER_BLOCKS_1600ZR, OFEC_NUM_ENCODERS
        decoded, _ = ofec_decode_window(encoded_data['llrs'], mode='1600ZR+')
        # full pipeline: 8 encoders × num_blocks × 3552 info bits each
        expected_len = OFEC_NUM_ENCODERS * OFEC_CODER_BLOCKS_1600ZR * 3552
        assert decoded.shape == (expected_len,)

    def test_decode_output_is_binary(self, encoded_data):
        from commplax.fec import ofec_decode_window
        decoded, _ = ofec_decode_window(encoded_data['llrs'], mode='1600ZR+')
        decoded_np = np.array(decoded)
        assert set(np.unique(decoded_np)).issubset({0, 1})

    def test_corrections_is_scalar(self, encoded_data):
        from commplax.fec import ofec_decode_window
        _, corrections = ofec_decode_window(encoded_data['llrs'], mode='1600ZR+')
        assert corrections.shape == ()
        assert corrections.dtype == jnp.int32

    def test_sd_iters_override(self, encoded_data):
        """sd_iters=1 should also produce valid output (different quality)."""
        from commplax.fec import ofec_decode_window
        from commplax.fec.ofec import OFEC_NUM_ENCODERS
        decoded, corr = ofec_decode_window(
            encoded_data['llrs'], mode='1600ZR+', sd_iters=1)
        expected_len = OFEC_NUM_ENCODERS * encoded_data['num_blocks'] * 3552
        assert decoded.shape == (expected_len,)


@pytest.mark.slow
class TestWindowRx:
    """Functional tests for ofec_rx_window."""

    def test_rx_returns_tuple_of_three(self, encoded_data):
        from commplax.fec import ofec_rx_window
        result = ofec_rx_window(encoded_data['llrs'], mode='1600ZR+')
        assert len(result) == 3

    def test_rx_payload_shape(self, encoded_data):
        from commplax.fec import ofec_rx_window
        from commplax.fec.ofec import _OFEC_PAYLOAD_BITS
        payload, crc_ok, corrections = ofec_rx_window(
            encoded_data['llrs'], mode='1600ZR+')
        assert payload.shape == (_OFEC_PAYLOAD_BITS,)

    def test_rx_crc_shape(self, encoded_data):
        from commplax.fec import ofec_rx_window
        payload, crc_ok, corrections = ofec_rx_window(
            encoded_data['llrs'], mode='1600ZR+')
        assert crc_ok.shape == (29,)

    def test_rx_payload_is_binary(self, encoded_data):
        from commplax.fec import ofec_rx_window
        payload, _, _ = ofec_rx_window(encoded_data['llrs'], mode='1600ZR+')
        payload_np = np.array(payload)
        assert set(np.unique(payload_np)).issubset({0, 1})
