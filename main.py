# aio_squared_loss_no_fb.py
# All-in-one Bellman residual (squared integrand, no FB), stabilized

import time
import numpy as np
import tensorflow as tf

# -------------------------
# Reproducibility
# -------------------------
tf.random.set_seed(0)
np.random.seed(0)

# -------------------------
# Model & training hyperparams
# -------------------------
beta = 0.96
alpha = 0.33
delta = 0.08

rho = 0.90
sigma_eps = 0.02

# state bounds
K_MIN, K_MAX = 0.1, 20.0
Z_MIN, Z_MAX = 0.5, 1.5

# training
BATCH = 1024
LR = 1e-5
STEPS = 8000
TAU = 0.01
CLIP_NORM = 1.0

# all-in-one scalar (fixed)
V_CONST = 0.10

# value scaling
V_SCALE = 5.0

# reward scaling to keep magnitudes small (optional)
REWARD_SCALE = 0.1

# -------------------------


# Value network: same canonical MLP
value_net = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])

# Target network copy (for V_next)
target_value_net = tf.keras.models.clone_model(value_net)
target_value_net.set_weights(value_net.get_weights())

def structure_policy_net():
    policy_in = tf.keras.Input(shape=(2,))
    h = tf.keras.layers.Dense(64, activation='relu')(policy_in)
    h = tf.keras.layers.Dense(64, activation='relu')(h)
    a_raw = tf.keras.layers.Dense(1, activation=None, name='a_raw')(h)   # investment share_raw
    kraw  = tf.keras.layers.Dense(1, activation=None, name='k_raw')(h)   # optional direct k' raw
    return tf.keras.Model(policy_in, outputs=[a_raw, kraw])
policy_net = structure_policy_net()

optimizer = tf.keras.optimizers.Adam(LR)

# -------------------------
# Economic primitives / helpers
# -------------------------
@tf.function
def production(k, z):
    return z * tf.pow(k, alpha)

@tf.function
def transition_z(z, eps):
    z_ss = 1.0
    return rho * z + (1.0 - rho) * z_ss + eps

@tf.function
def value_from_raw(raw):
    return V_SCALE * tf.tanh(raw / (V_SCALE + 1e-9))

@tf.function
def utility_log(c):
    return tf.math.log(tf.maximum(c, 1e-8))

@tf.function
def policy_transform(a_raw, kraw):
    # investment share a in [0,1]
    a = tf.sigmoid(a_raw)
    # chosen next capital k' bounded in [K_MIN,K_MAX]
    kprime = K_MIN + (K_MAX - K_MIN) * tf.sigmoid(kraw)
    return a, kprime

# -------------------------
# Single residual parts (returns vectors)
# -------------------------
@tf.function
def single_residual_parts(k, z, eps):
    # k,z shape (B,1)
    state = tf.concat([k, z], axis=1)  # (B,2)

    a_raw, kraw = policy_net(state)    # (B,1) each
    a, kprime = policy_transform(a_raw, kraw)  # (B,1)

    pi_val = production(k, z)          # (B,1)
    i = a * pi_val                     # (B,1)
    k_next = (1.0 - delta) * k + i     # (B,1)

    c = pi_val - i
    c = tf.maximum(c, 1e-8)

    u = utility_log(c) * REWARD_SCALE  # scaled utility

    # next z
    z_next = transition_z(z, eps)
    next_state = tf.concat([k_next, z_next], axis=1)

    # bounded value evaluations
    raw_V_current = value_net(state)           # (B,1)
    raw_V_next    = target_value_net(next_state)  # (B,1)

    V_current = tf.squeeze(value_from_raw(raw_V_current), axis=1)  # (B,)
    V_next = tf.squeeze(value_from_raw(raw_V_next), axis=1)        # (B,)

    # residual (vector)
    R = V_current - (tf.squeeze(u, axis=1) + beta * V_next)

    return R, V_current, tf.squeeze(c, axis=1), tf.squeeze(kprime, axis=1), tf.squeeze(i, axis=1), tf.squeeze(pi_val, axis=1), tf.squeeze(k_next, axis=1), next_state

# -------------------------
# All-in-one integrand (use two independent eps draws),
# then square integrand to make loss non-negative.
# -------------------------
@tf.function
def compute_loss(batch_size):
    # sample states
    k = tf.random.uniform((batch_size,1), minval=K_MIN, maxval=K_MAX)
    z = tf.random.uniform((batch_size,1), minval=Z_MIN, maxval=Z_MAX)

    # independent shocks
    eps1 = tf.random.normal((batch_size,1), stddev=sigma_eps)
    eps2 = tf.random.normal((batch_size,1), stddev=sigma_eps)

    R1, V1, c1, kprime1, i1, pi1, knext1, ns1 = single_residual_parts(k, z, eps1)
    R2, V2, c2, kprime2, i2, pi2, knext2, ns2 = single_residual_parts(k, z, eps2)

    v = tf.cast(V_CONST, tf.float32)

    integrand = (R1 - v/2.0) * (R2 - v/2.0) - v * (V1 + v/4.0)   # (B,)

    # Non-negative loss: square the integrand and take mean
    loss_bellman = tf.reduce_mean(tf.square(integrand))

    mean_abs_R1 = tf.reduce_mean(tf.abs(R1))
    mean_abs_R2 = tf.reduce_mean(tf.abs(R2))
    mean_V = tf.reduce_mean(V1)

    return loss_bellman, mean_abs_R1, mean_abs_R2, mean_V

# -------------------------
# Training loop
# -------------------------
trainable_vars = policy_net.trainable_weights + value_net.trainable_weights

start = time.time()
for step in range(1, STEPS + 1):
    with tf.GradientTape() as tape:
        loss, mR1, mR2, mV = compute_loss(BATCH)

    grads = tape.gradient(loss, trainable_vars)
    grads = [None if g is None else tf.clip_by_norm(g, CLIP_NORM) for g in grads]
    optimizer.apply_gradients(zip(grads, trainable_vars))

    # target soft update
    w = value_net.get_weights()
    tw = target_value_net.get_weights()
    new_tw = []
    for w_i, tw_i in zip(w, tw):
        new_tw.append(TAU * w_i + (1.0 - TAU) * tw_i)
    target_value_net.set_weights(new_tw)

    if step % 200 == 0 or step == 1:
        t = time.time() - start
        print(f"step={step:5d} | loss={loss.numpy():.6f} | mean|R1|={mR1.numpy():.4f} | mean|R2|={mR2.numpy():.4f} | meanV={mV.numpy():.4f} | time={t:.1f}s")

# ============================================================
# 9. Policy & Value Evaluation + Plotting (inside main())
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def evaluate_and_plot():
    """Evaluate trained networks and generate plots for the report."""

    # --------- Build grid ----------
    k_grid = np.linspace(K_MIN, K_MAX, 200).reshape(-1, 1).astype(np.float32)
    z_grid = np.full_like(k_grid, 1.0, dtype=np.float32)
    state_grid = np.concatenate([k_grid, z_grid], axis=1)

    # --------- Evaluate policy ---------
    a_raw_grid, kraw_grid = policy_net(state_grid)
    a_grid, kprime_grid = policy_transform(a_raw_grid, kraw_grid)

    a_grid = a_grid.numpy().reshape(-1)
    kprime_grid = kprime_grid.numpy().reshape(-1)

    # --------- Evaluate value function ---------
    V_grid = value_net(state_grid).numpy().reshape(-1)

    # -----------------------------------------------------
    # Print sample values
    # -----------------------------------------------------
    print("\nSample policy evaluation (k, a, k'):")
    idxs = np.linspace(0, len(k_grid)-1, 10, dtype=int)
    for i in idxs:
        print(f"k={k_grid[i,0]:.3f}  a={a_grid[i]:.3f}  k'={kprime_grid[i]:.3f}")

    # -----------------------------------------------------
    # Plot 1: Policy a(k)
    # -----------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(k_grid, a_grid, linewidth=2)
    plt.xlabel("Capital k")
    plt.ylabel("Policy a(k)")
    plt.title("Optimal Policy Function a(k) at z = 1")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Plot 2: Next-period Capital k′(k)
    # -----------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(k_grid, kprime_grid, linewidth=2)
    plt.xlabel("Capital k")
    plt.ylabel("Next-period Capital k′")
    plt.title("Capital Transition Function k′(k) at z = 1")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # Plot 3: Value Function V(k, 1)
    # -----------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(k_grid, V_grid, linewidth=2)
    plt.xlabel("Capital k")
    plt.ylabel("Value V(k, z=1)")
    plt.title("Value Function V(k, z=1)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Standard Python main guard
# ============================================================

def main():
    print("Running policy and value evaluation...")
    evaluate_and_plot()
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()



