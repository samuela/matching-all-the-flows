import jax
import jax.numpy as jnp
from jax import random, jit
import flax.linen as nn
import optax
import diffrax

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class Flow(nn.Module):
    dim: int = 2
    h: int = 128

    @nn.compact
    def __call__(self, t, x_t):
        # Concatenate time and state
        inputs = jnp.concatenate([t, x_t], axis=-1)

        x = nn.Dense(self.h)(inputs)
        x = nn.relu(x)
        x = nn.Dense(self.h)(x)
        x = nn.relu(x)
        x = nn.Dense(self.h)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        return x

def loss_fn(params, key, flow):
    """Compute flow matching loss"""
    # Generate data
    x_1_np, _ = make_moons(1024, noise=0.05)
    x_1 = jnp.array(x_1_np, dtype=jnp.float32)

    # Generate noise
    key, subkey = random.split(key)
    x_0 = random.normal(subkey, x_1.shape)

    # Sample random times
    key, subkey = random.split(key)
    t = random.uniform(subkey, (len(x_1), 1))

    # Linear interpolation
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0

    # Predict velocity
    pred_dx_t = flow.apply(params, t, x_t)

    # MSE loss
    loss = jnp.mean((pred_dx_t - dx_t) ** 2)
    return loss, key

# Initialize model and optimizer
key = random.PRNGKey(42)
flow = Flow()

# Initialize parameters
key, init_key = random.split(key)
dummy_t = jnp.ones((1, 1))
dummy_x = jnp.ones((1, 2))
params = flow.init(init_key, dummy_t, dummy_x)

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
plt.savefig('training_loss_jax.png', dpi=300, bbox_inches='tight')
plt.close()

#########################################################
# Sampling

def vector_field(t, y, args):
    """Vector field function for diffrax ODE solver"""
    params = args
    flow = Flow()

    # Expand t to match batch size
    t_expanded = jnp.broadcast_to(jnp.array([[t]]), (y.shape[0], 1))

    return flow.apply(params, t_expanded, y)

def solve_ode(params, x_init, t_span):
    """Solve ODE using diffrax with Dopri5 method"""
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=0.01,
        y0=x_init,
        args=params,
        saveat=diffrax.SaveAt(ts=t_span),
    )

    return solution.ys

key, sample_key = random.split(key)
x_init = random.normal(sample_key, (1024, 2))
n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30 / 8 * n_steps, 4), sharex=True, sharey=True)
time_steps = jnp.linspace(0, 1.0, n_steps + 1)

# Solve ODE using diffrax with EMA parameters
solution = solve_ode(ema_params, x_init, time_steps)

# Plot evolution
for i in range(n_steps + 1):
    x_t = solution[i]
    axes[i].scatter(x_t[:, 0], x_t[:, 1], s=10, alpha=0.5)
    axes[i].set_title(f't = {time_steps[i]:.2f}')
    axes[i].set_xlim(-3.0, 3.0)
    axes[i].set_ylim(-3.0, 3.0)

plt.tight_layout()
plt.savefig('flow_sampling_evolution_jax.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training completed and visualizations saved!")
print("Final sampling performed using EMA parameters for improved stability.")
