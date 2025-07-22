from logging import logMultiprocessing
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from functools import partial
from flax.training import train_state

class DiscreteFlow(nn.Module):
    vocab_size: int
    num_hidden_units: int

    @nn.compact
    def __call__(self, x_t, t):
        # x_t shape: (batch_size, 2)
        # t shape: (batch_size,)
        batch_size = x_t.shape[0]

        # Embed discrete tokens
        embedded = nn.Embed(self.vocab_size, self.num_hidden_units)(x_t)  # (batch_size, 2, num_hidden_units)
        embedded_flat = embedded.reshape(batch_size, -1)  # (batch_size, 2 * num_hidden_units)

        # Concatenate time and embedded tokens
        t_expanded = t[:, None]  # (batch_size, 1)
        net_input = jnp.concatenate([t_expanded, embedded_flat], axis=-1)

        # Forward through network
        x = nn.Dense(self.num_hidden_units)(net_input)
        x = nn.relu(x)
        x = nn.Dense(self.num_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_hidden_units)(x)
        x = nn.relu(x)
        logits = nn.Dense(2 * self.vocab_size)(x)  # (batch_size, 2 * vocab_size)

        # Reshape to (batch_size, 2, vocab_size)
        return logits.reshape(batch_size, 2, self.vocab_size)

def create_train_state(rng, model, learning_rate):
    """Creates initial training state."""
    # Initialize with dummy inputs
    dummy_x = jnp.ones((1, 2), dtype=jnp.int32)
    dummy_t = jnp.ones((1,))

    params = model.init(rng, dummy_x, dummy_t)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnums=(0,))
def train_step(model, state, x_t, x_1, t):
    """Performs a single training step."""

    def loss_fn(params):
        logits = model.apply(params, x_t, t)
        # Flatten for cross entropy: (batch_size * 2, vocab_size) and (batch_size * 2,)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = x_1.reshape(-1)

        # Compute cross entropy loss
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.mean(jnp.take_along_axis(log_probs, targets_flat[:, jnp.newaxis], axis=1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def generate_batch(rng, batch_size, vocab_size):
    """Generate a batch of training data."""
    # Generate moon dataset
    x_1_continuous, _ = make_moons(batch_size, noise=0.05)

    # Discretize to vocabulary
    x_1 = jnp.round(jnp.clip(x_1_continuous * 35 + 50, min=0.0, max=vocab_size - 1)).astype(jnp.int32)

    # Generate random starting points
    rng, rng_x0, rng_t, rng_mask = jax.random.split(rng, 4)
    x_0 = jax.random.randint(rng_x0, (batch_size, 2), 0, vocab_size)

    # Sample time and create interpolated states
    t = jax.random.uniform(rng_t, (batch_size,))
    mask = jax.random.uniform(rng_mask, (batch_size, 2)) < t[:, None]
    x_t = jnp.where(mask, x_1, x_0)

    return rng, x_t, x_1, t

################################################################################
# Training the model on a synthetic dataset

# Hyperparameters
batch_size = 512
vocab_size = 128
num_hidden_units = 128
learning_rate = 0.001
num_iterations = 10_000

# Initialize model and training state
rng = jax.random.PRNGKey(42)
model = DiscreteFlow(vocab_size=vocab_size, num_hidden_units=num_hidden_units)

rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng, model, learning_rate)

# Track training losses
training_losses = []

print("Starting training...")
for i in range(num_iterations):
    # Generate batch
    rng, x_t, x_1, t = generate_batch(rng, batch_size, vocab_size)

    # Training step
    state, loss = train_step(model, state, x_t, x_1, t)

    # Store loss value
    training_losses.append(float(loss))

    if (i + 1) % 1000 == 0:
        print(f"Iteration {i + 1}, Loss: {loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(training_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy Loss')
plt.grid(True, alpha=0.3)
plt.savefig('discrete_flow_training_loss_jax.png', dpi=150, bbox_inches='tight')
plt.close()
print("Training loss plot saved as 'discrete_flow_training_loss_jax.png'")

################################################################################
# Sampling from the trained model
print("Starting sampling...")

def sample_step(x_t, t, h, params, rng_key):
    """Single sampling step."""
    logits = model.apply(params, x_t, jnp.full((x_t.shape[0],), t))
    p1 = jax.nn.softmax(logits, axis=-1)

    # One-hot encoding of current state
    one_hot_x_t = jax.nn.one_hot(x_t, vocab_size)

    # Compute flow direction
    u = (p1 - one_hot_x_t) / (1.0 - t)

    # Update probabilities
    new_probs = one_hot_x_t + h * u

    # Sample from categorical distribution
    x_new = jax.random.categorical(rng_key, logits=jnp.log(new_probs + 1e-8), axis=-1)
    return x_new

def safe_sample_step(x_t, t, h, params, rng_key):
    """Eq 6.7"""
    logits = model.apply(params, x_t, jnp.full((x_t.shape[0],), t))
    p1 = jax.nn.softmax(logits, axis=-1)

    # One-hot encoding of current state
    one_hot_x_t = jax.nn.one_hot(x_t, vocab_size)

    # Compute flow direction
    u = (p1 - one_hot_x_t) / (1.0 - t)

    exphu = jnp.exp(h * u)                                 # (..., 2, vocab_size)
    new_probs = one_hot_x_t * exphu + (1 - one_hot_x_t) * u * (1 - jnp.take_along_axis(exphu, x_t[..., jnp.newaxis], axis=-1)) / jnp.abs(1e-8 + jnp.take_along_axis(u, x_t[..., jnp.newaxis], axis=-1))  # (..., 2, vocab_size)

    # Sample from categorical distribution
    x_new = jax.random.categorical(rng_key, logits=jnp.log(new_probs + 1e-8), axis=-1)
    return x_new


# Initialize sampling
rng, sample_rng = jax.random.split(rng)
x_t = jax.random.randint(sample_rng, (512, 2), 0, vocab_size)
t = 0.0
results = [(x_t, t)]

while t < 1.0 - 1e-3:
    rng, step_rng = jax.random.split(rng)
    h = min(0.05, 1.0 - t)
    x_t = safe_sample_step(x_t, t,h, state.params, step_rng)
    t += h
    results.append((x_t, t))

# Plot sampling evolution
fig, axes = plt.subplots(1, min(len(results), 10), figsize=(15, 2), sharex=True, sharey=True)

for (x_t, t), ax in zip(results[-len(axes):], axes):
    ax.scatter(x_t[:, 0], x_t[:, 1], s=10, alpha=0.5)
    ax.set_title(f't={t:.2f}')

plt.tight_layout()
plt.savefig('discrete_flow_sampling_evolution_jax.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sampling evolution plot saved as 'discrete_flow_sampling_evolution_jax.png'")
