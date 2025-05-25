import tensorflow as tf
import numpy as np

class PPOAgent:
    def __init__(self, n_assets, window_size, features_per_asset=5, gamma=0.99, clip_ratio=0.2, actor_lr=3e-4, critic_lr=1e-3, entropy_coeff=0.01):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.n_assets = n_assets
        self.features_per_asset = features_per_asset
        self.entropy_coeff = entropy_coeff

        # Build actor and critic networks
        self.actor = self._build_actor(window_size, n_assets, features_per_asset)
        self.critic = self._build_critic(window_size, n_assets, features_per_asset)

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def _build_actor(self, window_size, n_assets, features_per_asset):
        inputs = tf.keras.Input(shape=(window_size, n_assets * features_per_asset))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(n_assets, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self, window_size, n_assets, features_per_asset):
        inputs = tf.keras.Input(shape=(window_size, n_assets * features_per_asset))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def get_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        probs = self.actor(state)[0].numpy()

        # Adjust concentration for Dirichlet sampling to encourage exploration
        concentration = probs * 10 + 1e-3
        action = np.random.dirichlet(concentration)
        return action, probs

    def train(self, states, actions, rewards, old_probs):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        old_probs = np.array(old_probs, dtype=np.float32)

        # Compute returns and advantages
        values = self.critic(states).numpy().flatten()
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            advantages[t] = returns[t] - values[t]

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        with tf.GradientTape(persistent=True) as tape:
            new_probs = self.actor(states)
            new_probs_selected = tf.reduce_sum(new_probs * actions, axis=1)
            old_probs_selected = tf.reduce_sum(old_probs * actions, axis=1)
            ratio = new_probs_selected / (old_probs_selected + 1e-8)

            # Clipped surrogate objective
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # Entropy regularization
            entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1)
            entropy_bonus = self.entropy_coeff * tf.reduce_mean(entropy)
            actor_loss -= entropy_bonus

            # Critic loss
            critic_values = self.critic(states)[:, 0]
            critic_loss = tf.reduce_mean(tf.square(returns - critic_values))

        # Compute gradients and update networks
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        del tape
