"""
DQN Agent with Opponent Modelling for Chef's Hat Gym (v3 API)
==============================================================
Variant 1: Opponent Modelling (Student ID mod 7 = 1)

Student: Praveen K Gandikota (ID: 16829772)
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


class DuelingDQN(nn.Module):
    """Dueling DQN network with separate value and advantage streams."""

    def __init__(self, state_dim=28, action_dim=200, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class OpponentModel(nn.Module):
    """Predicts opponent action distributions from game state."""

    def __init__(self, input_dim=28, action_dim=200, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgentWithOpponentModelling(ChefsHatPlayer):
    """
    DQN Agent with Opponent Modelling for Chef's Hat Gym.
    Uses Dueling Double DQN with an opponent behaviour prediction model.
    """

    def __init__(
        self,
        name,
        log_directory="",
        verbose_log=False,
        verbose_console=False,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        batch_size=64,
        memory_size=50000,
        target_update_freq=100,
        tau=0.005,
        opponent_lr=1e-3,
        use_opponent_model=True,
    ):
        super().__init__(
            "DQN_OppModel",
            name,
            this_agent_folder="",
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.use_opponent_model = use_opponent_model

        self.state_dim = 28
        self.action_dim = 200

        self.policy_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.opponent_model = OpponentModel(self.state_dim, self.action_dim).to(self.device)
        self.opponent_optimizer = optim.Adam(self.opponent_model.parameters(), lr=opponent_lr)
        self.opponent_memory = deque(maxlen=10000)

        self.memory = deque(maxlen=memory_size)

        # Episode state
        self.current_state = None
        self.current_action = None
        self.current_action_mask = None
        self.my_player_index = -1
        self.steps = 0

        # Metrics
        self.training_losses = []
        self.opponent_losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.match_wins = []
        self.match_positions = []

    def _extract_state(self, observations):
        board = observations[:11]
        hand = observations[11:28]
        return np.concatenate([board, hand])

    def _extract_action_mask(self, observations):
        return observations[28:]

    def get_action(self, observations):
        state = self._extract_state(observations)
        action_mask = self._extract_action_mask(observations)
        valid_indices = np.where(action_mask == 1)[0]

        self.current_state = state.copy()
        self.current_action_mask = action_mask.copy()

        if random.random() < self.epsilon:
            chosen_idx = random.choice(valid_indices.tolist())
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
                if self.use_opponent_model and len(self.opponent_memory) > 100:
                    opponent_pred = self.opponent_model(state_tensor).squeeze(0)
                    opponent_probs = F.softmax(opponent_pred, dim=-1)
                    q_values = q_values + 0.1 * (1.0 - opponent_probs)
                q_values_np = q_values.cpu().numpy()
            q_values_np[action_mask == 0] = -1e9
            chosen_idx = int(valid_indices[np.argmax(q_values_np[valid_indices])])

        self.current_action = chosen_idx
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
        player_idx = self.my_player_index
        if player_idx < 0:
            player_idx = info.get("Author_Index", 0)

        finished_players = info.get("Finished_Players", [False]*4)
        is_finished = finished_players[player_idx] if player_idx < len(finished_players) else False

        if is_finished:
            match_score = info.get("Match_Score", [-1]*4)
            score = match_score[player_idx] if player_idx < len(match_score) else 0
            position = 3 - score  # 3->0(1st), 2->1(2nd), 1->2(3rd), 0->3(4th)
            if position == 0:
                return 1.0
            elif position == 1:
                return 0.3
            elif position == 2:
                return -0.3
            else:
                return -1.0
        return -0.001

    def update_start_match(self, cards, players, starting_player):
        self.current_state = None
        self.current_action = None
        self.current_episode_reward = 0.0
        # Determine our player index from name
        my_name = self.get_name()
        if isinstance(players, list):
            for i, pname in enumerate(players):
                if pname == my_name:
                    self.my_player_index = i
                    break

    def update_my_action(self, envInfo):
        reward = self.get_reward(envInfo)
        self.current_episode_reward += reward

        next_state = self.current_state
        finished_players = envInfo.get("Finished_Players", [False]*4)
        done = finished_players[self.my_player_index] if self.my_player_index >= 0 and self.my_player_index < len(finished_players) else False

        if self.current_state is not None and self.current_action is not None:
            self.memory.append((
                self.current_state,
                self.current_action_mask,
                self.current_action,
                reward,
                next_state,
                self.current_action_mask,
                done,
            ))

        self.steps += 1
        if len(self.memory) >= self.batch_size:
            loss = self._train_step()
            self.training_losses.append(loss)

        if self.steps % self.target_update_freq == 0:
            self._soft_update_target()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_action_others(self, envInfo):
        other_player = envInfo.get("Author_Index", -1)
        action_idx = envInfo.get("Action_Index", -1)

        if other_player >= 0 and action_idx >= 0 and self.current_state is not None:
            opponent_state = self.current_state.copy()
            self.opponent_memory.append((opponent_state, action_idx))

            if len(self.opponent_memory) >= self.batch_size and self.steps % 10 == 0:
                self._train_opponent_model()

    def update_end_match(self, envInfo):
        match_score = envInfo.get("Match_Score", [-1]*4)
        if self.my_player_index >= 0 and self.my_player_index < len(match_score):
            score = match_score[self.my_player_index]
            position = 3 - score
        else:
            position = 3

        self.match_positions.append(position)
        self.match_wins.append(1 if position == 0 else 0)
        self.episode_rewards.append(self.current_episode_reward)

    def update_game_over(self):
        pass

    def observe_special_action(self, action_type, player):
        pass

    def _train_step(self):
        batch = random.sample(self.memory, self.batch_size)
        states, action_masks, actions, rewards, next_states, next_masks, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        next_masks = torch.FloatTensor(np.array(next_masks)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_policy[next_masks == 0] = -1e9
            next_actions = next_q_policy.argmax(dim=1)
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def _train_opponent_model(self):
        if len(self.opponent_memory) < self.batch_size:
            return
        batch = random.sample(list(self.opponent_memory), min(self.batch_size, len(self.opponent_memory)))
        states, actions = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        predictions = self.opponent_model(states)
        loss = F.cross_entropy(predictions, actions)
        self.opponent_optimizer.zero_grad()
        loss.backward()
        self.opponent_optimizer.step()
        self.opponent_losses.append(loss.item())

    def _soft_update_target(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "opponent_model": self.opponent_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.opponent_model.load_state_dict(checkpoint["opponent_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

    def get_metrics(self):
        return {
            "training_losses": self.training_losses,
            "opponent_losses": self.opponent_losses,
            "episode_rewards": self.episode_rewards,
            "match_wins": self.match_wins,
            "match_positions": self.match_positions,
            "epsilon": self.epsilon,
            "steps": self.steps,
        }
