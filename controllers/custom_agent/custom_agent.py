from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Importar el paquete comun (agent_runner, config) igual que los demas agentes.
_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import agent_runner  # noqa: E402
import config        # noqa: E402

ROBOT_NAME = "agente propio"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "custom_model.keras")

TRAIN_EPISODES = int(os.environ.get("EPISODES", "300"))
GAMMA = float(os.environ.get("GAMMA", "0.99"))

# ---------------------------------------------------------------------------
# NUCLEO DEL TALLER 2 (intacto): modelo actor-critic + descubrimiento.
# ---------------------------------------------------------------------------
def create_model(n_inputs, n_actions):
    inputs = layers.Input(shape=(n_inputs,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    actor = layers.Dense(n_actions, activation="softmax")(x)   # politica (Actor)
    critic = layers.Dense(1)(x)                                # valor de estado (Critic)
    return tf.keras.Model(inputs=inputs, outputs=[actor, critic])

class Curiosity:
    """Descubrimiento por novedad (intacto del Taller 2).

    Opera ahora sobre la observacion de 7 dims (ya normalizada) en vez de la RAM
    de Atari; el mecanismo es identico: recompensa intrinseca ~ distancia al
    estado mas cercano visto.
    """
    def __init__(self, max_size=2000, scale=0.5):
        self.memory = []
        self.max_size = max_size
        self.scale = scale

    def bonus(self, s):
        if len(self.memory) == 0:
            self.memory.append(s.copy())
            return self.scale
        distances = np.linalg.norm(np.array(self.memory) - s, axis=1)
        min_dist = float(np.min(distances))
        self.memory.append(s.copy())
        if len(self.memory) > self.max_size:
            self.memory.pop(0)
        return self.scale * (min_dist / (min_dist + 1.0))

def preprocess(obs):
    # La observacion del entorno ya viene normalizada (Box de 7 floats); NO se
    # divide entre 255 como en Atari. Solo se asegura el dtype.
    return np.asarray(obs, dtype=np.float32)

# ---------------------------------------------------------------------------
# Entrenamiento Monte Carlo por episodio (formulacion del Taller 2, intacta).
# ---------------------------------------------------------------------------
def train(model, env, episodes=TRAIN_EPISODES, gamma=GAMMA, save_path=MODEL_PATH):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    huber_loss = tf.keras.losses.Huber()
    curiosity = Curiosity()
    avg_history = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0.0
        states, actions, rewards = [], [], []

        with tf.GradientTape() as tape:
            terminated = truncated = False
            while not (terminated or truncated):
                state_tf = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action_probs, _ = model(state_tf)                      # Actor
                action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs))

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = preprocess(next_obs)

                bonus = curiosity.bonus(next_state)                    # descubrimiento
                states.append(state_tf)
                actions.append(action)
                rewards.append(reward + bonus)                         # recompensa combinada

                state = next_state
                total_reward += reward                                 # solo recompensa real

            returns = []
            reward_sum = 0.0
            for r in rewards[::-1]:
                reward_sum = r + gamma * reward_sum
                returns.insert(0, reward_sum)
            returns = tf.convert_to_tensor(returns, dtype=tf.float32)

            states_tensor = tf.concat(states, axis=0)
            probs, values = model(states_tensor)
            values = tf.squeeze(values)

            advantage = returns - values
            advantage_norm = (advantage - tf.math.reduce_mean(advantage)) / (
                tf.math.reduce_std(advantage) + 1e-8
            )

            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
            indices = tf.range(len(actions))
            action_indices = tf.stack([indices, actions_tensor], axis=1)
            chosen_probs = tf.gather_nd(probs, action_indices)

            actor_loss = tf.math.reduce_mean(
                -tf.math.log(chosen_probs + 1e-8) * tf.stop_gradient(advantage_norm)
            )
            entropy = tf.math.reduce_mean(
                -tf.math.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
            )
            actor_loss = actor_loss - 0.01 * entropy
            critic_loss = huber_loss(returns, values)
            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        avg_history.append(total_reward)
        average = np.mean(avg_history[-20:])
        print(f"[custom] Episodio {ep}/{episodes} - Recompensa: {total_reward:.1f} "
              f"- Promedio (ult. 20): {average:.1f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[custom] Modelo guardado en {save_path}")

def run_race(model, env):
    """Corre un episodio determinista para la carrera (modo race)."""
    obs, _ = env.reset()
    state = preprocess(obs)
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        state_tf = tf.expand_dims(tf.convert_to_tensor(state), 0)
        probs, _ = model(state_tf)
        action = int(np.argmax(np.squeeze(probs)))   # determinista para competir
        obs, reward, terminated, truncated, info = env.step(action)
        state = preprocess(obs)
        total_reward += float(reward)
    print(f"[custom] carrera terminada - recompensa: {total_reward:.1f}")

def _build_backend():
    from controller import Supervisor
    return Supervisor()

def main():
    mode = agent_runner.get_mode()           # misma convencion que PPO/DQN/A2C
    backend = _build_backend()
    env = agent_runner.make_env(ROBOT_NAME, backend)

    n_inputs = int(env.observation_space.shape[0])   # 7
    n_actions = int(env.action_space.n)              # 3

    if mode == "train":
        model = create_model(n_inputs, n_actions)
        train(model, env, episodes=TRAIN_EPISODES)
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        run_race(model, env)

if __name__ == "__main__":
    main()