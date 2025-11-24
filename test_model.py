# test_aio_model.py
# PyTest tests for aio_squared_loss_no_fb.py

import numpy as np
import tensorflow as tf
import importlib

# Import your script as a module
# Make sure your file is named exactly aio_squared_loss_no_fb.py
module_name = "main"
m = importlib.import_module(module_name)

def test_import_success():
    """Check the model file imports without errors."""
    assert hasattr(m, "single_residual_parts")
    assert hasattr(m, "policy_net")
    assert hasattr(m, "value_net")

def test_policy_output_ranges():
    """Ensure policy transforms outputs to valid ranges: a ∈ (0,1), k' ∈ [K_MIN,K_MAX]."""
    k = tf.constant([[1.0]], dtype=tf.float32)
    z = tf.constant([[1.0]], dtype=tf.float32)
    state = tf.concat([k, z], axis=1)

    a_raw, kraw = m.policy_net(state)
    a, kprime = m.policy_transform(a_raw, kraw)

    a_val = float(a.numpy()[0][0])
    kprime_val = float(kprime.numpy()[0][0])

    assert 0.0 < a_val < 1.0
    assert m.K_MIN <= kprime_val <= m.K_MAX

def test_single_residual_shapes():
    """Check shapes of residual outputs."""
    k = tf.random.uniform((16,1), minval=m.K_MIN, maxval=m.K_MAX)
    z = tf.random.uniform((16,1), minval=m.Z_MIN, maxval=m.Z_MAX)
    eps = tf.random.normal((16,1), stddev=m.sigma_eps)

    R, V, c, kp, i, pi, knext, ns = m.single_residual_parts(k, z, eps)

    assert R.shape == (16,)
    assert V.shape == (16,)
    assert c.shape == (16,)
    assert kp.shape == (16,)
    assert i.shape == (16,)
    assert pi.shape == (16,)
    assert knext.shape == (16,)
    assert ns.shape == (16, 2)

    # ensure no NaNs
    for arr in [R, V, c, kp, i, pi, knext]:
        assert not tf.math.reduce_any(tf.math.is_nan(arr))

def test_loss_non_negative():
    """Loss must always be non-negative because it is integrand^2 mean."""
    loss, mR1, mR2, mV = m.compute_loss(128)

    loss_val = float(loss.numpy())
    assert loss_val >= 0.0
    assert not np.isnan(loss_val)

def test_value_network_bounded():
    """Check that bounded value_from_raw never produces NaN or extremely large values."""
    raw = tf.random.normal((32,1))
    bounded = m.value_from_raw(raw)

    assert not tf.reduce_any(tf.math.is_nan(bounded))
    assert tf.reduce_max(tf.abs(bounded)) < m.V_SCALE * 1.1

