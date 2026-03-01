"""
DQN Agent with Variational Autoencoder State Representation for Chef's Hat Gym (v3 API)
=========================================================================================
Variant 6: Generative AI Augmentation (Student ID mod 7 = 6)

This agent uses a Variational Autoencoder (VAE) to learn compressed
latent representations of game states, which are then used as input
to the DQN for policy learning. The VAE also enables data augmentation
through latent space sampling.

Student: Avinash Megavatn (ID: 16829749)
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer


# ============================================================================
# Variational Autoencoder for State Representation
# ============================================================================

class StateVAE(nn.Module):
    """
    Variational Autoencoder that learns a compressed representation
    of Chef's Hat game states. The latent space captures the essential
    game dynamics in a lower-dimensional manifold.
    """

    def __init__(self, input_dim=28, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z

    def get_latent(self, x):
        """Get the latent representation (mean) without sampling."""
        mu, _ = self.encode(x)
        return mu

    @staticmethod
    def vae_loss(reconstruction, original, mu, logvar, beta=1.0):
        """Compute VAE loss: reconstruction + KL divergence."""
        recon_loss = F.mse_loss(reconstruction, original, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss


# ============================================================================
# DQN Networks
# ============================================================================

class DQNWithLatent(nn.Module):
    """DQN that operates on VAE latent representations."""

    def __init__(self, latent_dim=16, action_dim=200, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z):
        return self.network(z)


class DQNRaw(nn.Module):
    """Standard DQN operating on raw state (for comparison)."""

    def __init__(self, state_dim=28, action_dim=200, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# DQN Agent with VAE
# ============================================================================

class DQNAgentWithVAE(ChefsHatPlayer):
    """
    DQN Agent with VAE-based state representation for Chef's Hat Gym.

    The agent first trains a VAE to learn compressed game state representations,
    then uses these latent representations as input to the DQN policy network.
    This tests whether generative AI components can improve RL learning.
    """

    def __init__(
        self,
        name,
        use_vae=True,
        log_directory="",
        verbose_log=False,
        verbose_console=False,
        # DQN hyperparameters
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.999,
        batch_size=64,
        memory_size=50000,
        target_update_freq=100,
        # VAE hyperparameters
        vae_latent_dim=16,
        vae_lr=1e-3,
        vae_beta=1.0,
        vae_pretrain_steps=500,
    ):
        suffix = "DQN_VAE" if use_vae else "DQN_Raw"
        super().__init__(
            suffix,
            name,
            this_agent_folder="",
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vae = use_vae

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.vae_beta = vae_beta
        self.vae_pretrain_steps = vae_pretrain_steps

        # Dimensions
        self.state_dim = 28
        self.action_dim = 200
        self.latent_dim = vae_latent_dim

        # VAE
        self.vae = StateVAE(self.state_dim, self.latent_dim).to(self.device)
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=vae_lr)
        self.vae_trained = False
        self.state_buffer = deque(maxlen=10000)  # Buffer for VAE pre-training

        # DQN networks
        if use_vae:
            self.policy_net = DQNWithLatent(self.latent_dim, self.action_dim).to(self.device)
            self.target_net = DQNWithLatent(self.latent_dim, self.action_dim).to(self.device)
        else:
            self.policy_net = DQNRaw(self.state_dim, self.action_dim).to(self.device)
            self.target_net = DQNRaw(self.state_dim, self.action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.dqn_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # Tracking
        self.current_state = None
        self.current_action = None
        self.current_action_mask = None
        self.my_player_index = -1
        self.prev_game_score = [0, 0, 0, 0]
        self.steps = 0
        self.training_losses = []
        self.vae_losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.match_wins = []
        self.match_positions = []
        self.latent_representations = []  # For t-SNE visualization

    def _extract_state(self, observations):
        board = observations[:11]
        hand = observations[11:28]
        return np.concatenate([board, hand])

    def _extract_action_mask(self, observations):
        return observations[28:]

    def _get_dqn_input(self, state_tensor):
        """Convert raw state to DQN input (latent if VAE, raw otherwise)."""
        if self.use_vae and self.vae_trained:
            with torch.no_grad():
                latent = self.vae.get_latent(state_tensor)
            return latent
        elif self.use_vae:
            # VAE not trained yet, use zeros as placeholder
            return torch.zeros(
                state_tensor.shape[0] if state_tensor.dim() > 1 else 1,
                self.latent_dim,
            ).to(self.device)
        else:
            return state_tensor

    def _pretrain_vae(self):
        """Pre-train VAE on collected state data."""
        if len(self.state_buffer) < self.batch_size * 2:
            return

        self.vae.train()

        for step in range(self.vae_pretrain_steps):
            batch_states = random.sample(list(self.state_buffer), self.batch_size)
            states = torch.FloatTensor(np.array(batch_states)).to(self.device)

            recon, mu, logvar, z = self.vae(states)
            loss = StateVAE.vae_loss(recon, states, mu, logvar, self.vae_beta)

            self.vae_optimizer.zero_grad()
            loss.backward()
            self.vae_optimizer.step()

            if (step + 1) % 100 == 0:
                self.vae_losses.append(loss.item() / self.batch_size)

        self.vae_trained = True

    def _train_vae_online(self, states_batch):
        """Online VAE training during DQN training."""
        if not self.use_vae:
            return

        self.vae.train()
        states = torch.FloatTensor(np.array(states_batch)).to(self.device)

        recon, mu, logvar, z = self.vae(states)
        loss = StateVAE.vae_loss(recon, states, mu, logvar, self.vae_beta)

        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

        self.vae_losses.append(loss.item() / len(states_batch))

    def get_action(self, observations):
        state = self._extract_state(observations)
        action_mask = self._extract_action_mask(observations)
        valid_indices = np.where(action_mask == 1)[0]

        self.current_state = state.copy()
        self.current_action_mask = action_mask.copy()

        # Collect states for VAE training
        self.state_buffer.append(state)

        # Check if we should pre-train VAE
        if (self.use_vae and not self.vae_trained
                and len(self.state_buffer) >= self.vae_pretrain_steps):
            self._pretrain_vae()

        # Epsilon-greedy
        if random.random() < self.epsilon:
            chosen_idx = random.choice(valid_indices.tolist())
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dqn_input = self._get_dqn_input(state_tensor)

            with torch.no_grad():
                q_values = self.policy_net(dqn_input).squeeze(0).cpu().numpy()

            q_values[action_mask == 0] = -1e9
            chosen_idx = int(valid_indices[np.argmax(q_values[valid_indices])])

        self.current_action = chosen_idx

        # Store latent representation for visualization
        if self.use_vae and self.vae_trained and len(self.latent_representations) < 5000:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                latent = self.vae.get_latent(state_tensor).cpu().numpy().flatten()
            self.latent_representations.append(latent)

        action = np.zeros(self.action_dim)
        action[chosen_idx] = 1
        return action

    def get_exhanged_cards(self, cards, amount):
        cards_sorted = sorted(range(len(cards)), key=lambda i: cards[i])
        return [cards[i] for i in cards_sorted[:amount]]

    def update_exchange_cards(self, cards_sent, cards_received):
        pass

    def do_special_action(self, info, specialAction):
        return True

    def get_reward(self, info):
        """Intermediate reward only. Terminal reward is in update_end_match."""
        return -0.001

    def update_start_match(self, cards, players, starting_player):
        self.current_state = None
        self.current_action = None
        self.current_episode_reward = 0.0
        self.prev_game_score = [0, 0, 0, 0]
        my_name = self.get_name()
        if isinstance(players, list):
            for i, p in enumerate(players):
                if p == my_name:
                    self.my_player_index = i
                    break

    def update_my_action(self, envInfo):
        reward = self.get_reward(envInfo)
        self.current_episode_reward += reward

        obs_after = envInfo.get("Observation_After", None)
        if obs_after is not None:
            next_state = self._extract_state(obs_after)
            next_mask = self._extract_action_mask(obs_after)
        else:
            next_state = self.current_state
            next_mask = self.current_action_mask

        if self.current_state is not None and self.current_action is not None:
            self.memory.append((
                self.current_state,
                self.current_action_mask,
                self.current_action,
                reward,
                next_state,
                next_mask,
                False,
            ))

        self.steps += 1
        if len(self.memory) >= self.batch_size:
            loss = self._train_step()
            self.training_losses.append(loss)

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_action_others(self, envInfo):
        pass

    def update_end_match(self, envInfo):
        game_score = envInfo.get("Game_Score", [0]*4)
        if 0 <= self.my_player_index < len(game_score):
            my_match_score = game_score[self.my_player_index] - self.prev_game_score[self.my_player_index]
            position = 3 - my_match_score
        else:
            position = 3
        self.prev_game_score = [int(s) for s in game_score]

        position_rewards = {0: 1.0, 1: 0.3, 2: -0.3, 3: -1.0}
        terminal_reward = position_rewards.get(position, -1.0)

        if self.current_state is not None and self.current_action is not None:
            self.memory.append((
                self.current_state, self.current_action_mask,
                self.current_action, terminal_reward,
                self.current_state, self.current_action_mask, True,
            ))

        self.current_episode_reward += terminal_reward
        self.match_positions.append(position)
        self.match_wins.append(1 if position == 0 else 0)
        self.episode_rewards.append(self.current_episode_reward)

        self.current_state = None
        self.current_action = None
        self.current_episode_reward = 0.0

    def update_game_over(self):
        pass

    def observe_special_action(self, action_type, player):
        pass

    def _train_step(self):
        """Train DQN (and optionally VAE) on a mini-batch."""
        batch = random.sample(self.memory, self.batch_size)
        states, masks, actions, rewards, next_states, next_masks, dones = zip(*batch)

        states_np = np.array(states)
        next_states_np = np.array(next_states)

        # Train VAE online (every 10 steps)
        if self.use_vae and self.steps % 10 == 0:
            self._train_vae_online(states_np)

        states_t = torch.FloatTensor(states_np).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states_np).to(self.device)
        next_masks_t = torch.FloatTensor(np.array(next_masks)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Convert to DQN input space
        dqn_states = self._get_dqn_input(states_t)
        dqn_next_states = self._get_dqn_input(next_states_t)

        # DQN training
        current_q = self.policy_net(dqn_states).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(dqn_next_states)
            next_q[next_masks_t == 0] = -1e9
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.dqn_optimizer.step()

        return loss.item()

    def generate_augmented_states(self, num_samples=100):
        """
        Use the VAE to generate synthetic game states by sampling
        from the learned latent space.
        """
        if not self.vae_trained:
            return None

        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated_states = self.vae.decode(z).cpu().numpy()
        return generated_states

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "vae": self.vae.state_dict(),
            "dqn_optimizer": self.dqn_optimizer.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "use_vae": self.use_vae,
            "vae_trained": self.vae_trained,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.vae.load_state_dict(checkpoint["vae"])
        self.dqn_optimizer.load_state_dict(checkpoint["dqn_optimizer"])
        self.vae_optimizer.load_state_dict(checkpoint["vae_optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.vae_trained = checkpoint.get("vae_trained", True)

    def get_metrics(self):
        return {
            "training_losses": self.training_losses,
            "vae_losses": self.vae_losses,
            "episode_rewards": self.episode_rewards,
            "match_wins": self.match_wins,
            "match_positions": self.match_positions,
            "latent_representations": self.latent_representations,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "use_vae": self.use_vae,
        }
