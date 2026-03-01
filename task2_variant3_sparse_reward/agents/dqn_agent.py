"""
DQN Agent with Multiple Reward Strategies for Chef's Hat Gym (v3 API)
======================================================================
Variant 3: Sparse / Delayed Reward (Student ID mod 7 = 3)

Student: Furkhanuddin Mohammad (ID: 16667990)
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
# Reward Functions
# ============================================================================

class SparseReward:
    """Original sparse reward: +1 for winning, -0.001 otherwise."""
    name = "Sparse"
    def get_reward(self, info, player_idx):
        finished = info.get("Finished_Players", [False]*4)
        if player_idx < len(finished) and finished[player_idx]:
            score = info.get("Match_Score", [-1]*4)[player_idx]
            if score == 3:
                return 1.0
            else:
                return -0.5
        return -0.001


class ShapedReward:
    """Shaped reward with intermediate signals for card plays."""
    name = "Shaped"
    def __init__(self):
        self.prev_cards = 17

    def get_reward(self, info, player_idx):
        reward = 0.0
        finished = info.get("Finished_Players", [False]*4)
        if player_idx < len(finished) and finished[player_idx]:
            score = info.get("Match_Score", [-1]*4)[player_idx]
            position = 3 - score
            position_rewards = {0: 1.0, 1: 0.3, 2: -0.3, 3: -1.0}
            reward += position_rewards.get(position, -1.0)
            self.prev_cards = 17
            return reward

        cards = info.get("Cards_Per_Player", [17]*4)
        if player_idx < len(cards):
            current = int(cards[player_idx])
            played = self.prev_cards - current
            if played > 0:
                reward += 0.02 * played
            elif played == 0:
                reward -= 0.01
            self.prev_cards = current
        return reward

    def reset(self):
        self.prev_cards = 17


class AuxiliaryReward:
    """Dense auxiliary reward with efficiency bonuses."""
    name = "Auxiliary"
    def __init__(self):
        self.prev_cards = 17
        self.actions_taken = 0
        self.passes = 0

    def get_reward(self, info, player_idx):
        reward = 0.0
        self.actions_taken += 1
        finished = info.get("Finished_Players", [False]*4)
        if player_idx < len(finished) and finished[player_idx]:
            score = info.get("Match_Score", [-1]*4)[player_idx]
            position = 3 - score
            reward += (3 - position) / 3.0
            reward += max(0, 1.0 - self.actions_taken / 100.0) * 0.2
            self.prev_cards = 17
            self.actions_taken = 0
            self.passes = 0
            return reward

        cards = info.get("Cards_Per_Player", [17]*4)
        if player_idx < len(cards):
            current = int(cards[player_idx])
            played = self.prev_cards - current
            if played > 0:
                reward += 0.03 * played
                self.passes = 0
            else:
                self.passes += 1
                reward -= 0.005 * self.passes
            self.prev_cards = current

        if info.get("Is_Pizza", False):
            reward += 0.05
        return reward

    def reset(self):
        self.prev_cards = 17
        self.actions_taken = 0
        self.passes = 0


class PerformanceScoreReward:
    """Reward using environment performance score."""
    name = "PerformanceScore"
    def get_reward(self, info, player_idx):
        finished = info.get("Finished_Players", [False]*4)
        if player_idx < len(finished) and finished[player_idx]:
            score = info.get("Match_Score", [-1]*4)[player_idx]
            position = 3 - score
            perf = info.get("Game_Performance_Score", [0]*4)
            perf_val = perf[player_idx] if player_idx < len(perf) else 0
            return (3 - position) / 3.0 + float(perf_val) * 0.5
        return -0.001

    def reset(self):
        pass


# ============================================================================
# DQN Network
# ============================================================================

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=28, action_dim=200, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgentSparseReward(ChefsHatPlayer):
    """DQN Agent with configurable reward strategy."""

    def __init__(self, name, reward_type="sparse", log_directory="",
                 verbose_log=False, verbose_console=False,
                 learning_rate=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 batch_size=64, memory_size=50000, target_update_freq=200):
        super().__init__(
            f"DQN_{reward_type}", name,
            this_agent_folder="", verbose_console=verbose_console,
            verbose_log=verbose_log, log_directory=log_directory,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        reward_map = {"sparse": SparseReward, "shaped": ShapedReward,
                      "auxiliary": AuxiliaryReward, "performance": PerformanceScoreReward}
        self.reward_type = reward_type
        self.reward_function = reward_map.get(reward_type, SparseReward)()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.state_dim = 28
        self.action_dim = 200
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=memory_size)
        self.current_state = None
        self.current_action = None
        self.current_action_mask = None
        self.my_player_index = -1
        self.steps = 0

        self.training_losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.match_wins = []
        self.match_positions = []
        self.reward_history = []

    def _extract_state(self, obs):
        return np.concatenate([obs[:11], obs[11:28]])

    def _extract_action_mask(self, obs):
        return obs[28:]

    def get_action(self, observations):
        state = self._extract_state(observations)
        mask = self._extract_action_mask(observations)
        valid = np.where(mask == 1)[0]
        self.current_state = state.copy()
        self.current_action_mask = mask.copy()

        if random.random() < self.epsilon:
            chosen = random.choice(valid.tolist())
        else:
            st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                qv = self.policy_net(st).squeeze(0).cpu().numpy()
            qv[mask == 0] = -1e9
            chosen = int(valid[np.argmax(qv[valid])])

        self.current_action = chosen
        a = np.zeros(self.action_dim)
        a[chosen] = 1
        return a

    def get_exhanged_cards(self, cards, amount):
        idx = sorted(range(len(cards)), key=lambda i: cards[i])
        return [cards[i] for i in idx[:amount]]

    def update_exchange_cards(self, cards_sent, cards_received):
        pass

    def do_special_action(self, info, specialAction):
        return True

    def get_reward(self, info):
        return self.reward_function.get_reward(info, self.my_player_index)

    def update_start_match(self, cards, players, starting_player):
        self.current_state = None
        self.current_action = None
        self.current_episode_reward = 0.0
        if hasattr(self.reward_function, 'reset'):
            self.reward_function.reset()
        my_name = self.get_name()
        if isinstance(players, list):
            for i, p in enumerate(players):
                if p == my_name:
                    self.my_player_index = i
                    break

    def update_my_action(self, envInfo):
        reward = self.get_reward(envInfo)
        self.current_episode_reward += reward
        self.reward_history.append(reward)

        finished = envInfo.get("Finished_Players", [False]*4)
        done = finished[self.my_player_index] if 0 <= self.my_player_index < len(finished) else False

        if self.current_state is not None and self.current_action is not None:
            self.memory.append((self.current_state, self.current_action_mask,
                                self.current_action, reward, self.current_state,
                                self.current_action_mask, done))

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
        score = envInfo.get("Match_Score", [-1]*4)
        if 0 <= self.my_player_index < len(score):
            position = 3 - score[self.my_player_index]
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
        s, m, a, r, ns, nm, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        nm = torch.FloatTensor(np.array(nm)).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        cq = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_net(ns)
            nq[nm == 0] = -1e9
            tq = r + (1 - d) * self.gamma * nq.max(1)[0]

        loss = F.smooth_l1_loss(cq, tq)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"policy_net": self.policy_net.state_dict(),
                     "target_net": self.target_net.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "epsilon": self.epsilon, "steps": self.steps,
                     "reward_type": self.reward_type}, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps = ckpt["steps"]

    def get_metrics(self):
        return {"training_losses": self.training_losses, "episode_rewards": self.episode_rewards,
                "match_wins": self.match_wins, "match_positions": self.match_positions,
                "reward_history": self.reward_history, "epsilon": self.epsilon,
                "steps": self.steps, "reward_type": self.reward_type}
