"""Mean flow matching implementation

See https://arxiv.org/pdf/2505.13447."""

import functools
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import flax.linen as nn
import optax
import chex
from jaxtyping import Array, Float

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def position_embedding(embedding_dim: int, min_period: float, max_period: float, t: Float[Array, ""]) -> Float[Array, "out"]:
    chex.assert_rank(t, 0)
    fraction = jnp.linspace(0.0, 1.0, num=embedding_dim // 2)
    period = min_period * ((max_period / min_period) ** fraction)
    x = t / period * 2 * jnp.pi
    return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)

    # This works better than sin/cos rotary position embeddings:
    # return t * jnp.arange(64)

POSEMB = functools.partial(position_embedding, 128, 4e-3, 4.0)

class Flow(nn.Module):
    dim: int = 2
    h: int = 512

    @nn.compact
    def __call__(self, x_t: Float[Array, "batch dim"], r: Float[Array, "batch"], t: Float[Array, "batch"]):
        # Concatenate time and state
        inputs = jnp.concatenate([x_t, vmap(POSEMB)(r), vmap(POSEMB)(t)], axis=-1)

        x = nn.Dense(self.h)(inputs)
        x = nn.relu(x)
        x = nn.Dense(self.h)(x)
        x = nn.relu(x)
        x = nn.Dense(self.h)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        return x

@functools.partial(jit, static_argnames=("flow",))
def loss_fn(params, key, flow: Flow):
    """See Algorithm 1 in https://arxiv.org/pdf/2505.13447"""
    # Generate data
    x_1_np, _ = make_moons(1024, noise=0.05)
    x_1 = jnp.array(x_1_np, dtype=jnp.float32)

    # Generate noise
    key, subkey = random.split(key)
    x_0 = random.normal(subkey, x_1.shape)

    # Sample random times
    key, subkey1, subkey2 = random.split(key, 3)
    ta = random.uniform(subkey1, (len(x_1), ))
    tb = random.uniform(subkey2, (len(x_1), ))
    r = jnp.minimum(ta, tb)
    t = jnp.maximum(ta, tb)

    # Linear interpolation
    x_t = (1 - t[:, jnp.newaxis]) * x_1 + t[:, jnp.newaxis] * x_0
    v = x_0 - x_1

    u, dudt = jax.jvp(lambda z, r, t: flow.apply(params, z, r, t), (x_t, r, t), (v, jnp.zeros_like(r), jnp.ones_like(t)))
    u_tgt = v - (t - r)[:, jnp.newaxis] * dudt

    # MSE loss
    loss = jnp.mean((u - jax.lax.stop_gradient(u_tgt)) ** 2)
    return loss, key

# Initialize model and optimizer
key = random.PRNGKey(42)
flow = Flow()

# Initialize parameters
key, init_key = random.split(key)
dummy_t = jnp.ones((1, ))
dummy_x = jnp.ones((1, 2))
params = flow.init(init_key, dummy_x, dummy_t, dummy_t)

# Initialize optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Initialize EMA parameters
ema_decay = 0.999
ema_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

# JIT compile the loss and update functions
@jit
def compute_loss_and_grads(params, key):
    return jax.value_and_grad(lambda p: loss_fn(p, key, flow)[0], has_aux=False)(params)

@jit
def update_step(params, opt_state, grads):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

@jit
def update_ema(ema_params, new_params, decay):
    """Update EMA parameters"""
    return jax.tree_util.tree_map(lambda ema, new: decay * ema + (1 - decay) * new, ema_params, new_params)

# Training loop
losses = []
ema_losses = []
for i in range(10_000):
    key, subkey = random.split(key)
    loss_val, grads = compute_loss_and_grads(params, subkey)
    params, opt_state = update_step(params, opt_state, grads)

    # Update EMA parameters
    ema_params = update_ema(ema_params, params, ema_decay)

    # Compute EMA loss
    key, ema_subkey = random.split(key)
    ema_loss_val, _ = loss_fn(ema_params, ema_subkey, flow)

    losses.append(float(loss_val))
    ema_losses.append(float(ema_loss_val))

    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss_val:.6f}, EMA Loss: {ema_loss_val:.6f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss', alpha=0.7)
plt.plot(ema_losses, label='EMA Loss', alpha=0.7)
plt.title('Training Loss Comparison')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('mean_flow_training_loss_jax.png', dpi=300, bbox_inches='tight')
plt.close()

#########################################################
# Sampling

n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30 / 8 * n_steps, 4), sharex=True, sharey=True)
time_steps = jnp.linspace(0, 1.0, n_steps + 1)

# Solve ODE using diffrax with EMA parameters
key, sample_key = random.split(key)
e = random.normal(sample_key, (1024, 2))

# Plot evolution
for i in range(n_steps + 1):
    x_t = e - flow.apply(ema_params, e, r=jnp.zeros((1024,)), t=time_steps[i] * jnp.ones((1024,)))
    axes[i].scatter(x_t[:, 0], x_t[:, 1], s=10, alpha=0.5)
    axes[i].set_title(f't = {time_steps[i]:.2f}')
    axes[i].set_xlim(-3.0, 3.0)
    axes[i].set_ylim(-3.0, 3.0)

plt.tight_layout()
plt.savefig('mean_flow_sampling_evolution_jax.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training completed and visualizations saved!")
print("Final sampling performed using EMA parameters for improved stability.")
