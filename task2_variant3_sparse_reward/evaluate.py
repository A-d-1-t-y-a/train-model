"""
Evaluation Script for DQN Agent with Reward Shaping
=====================================================
Loads trained models for each reward strategy and evaluates them.

Student: Furkhanuddin Mohammad (ID: 16667990)
"""

import os
import sys
import io
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.bool8 = np.bool_  # Fix gym/numpy 2.x compatibility

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon

from agents.dqn_agent import DQNAgentSparseReward


class SuppressStdout:
    """Context manager to suppress stdout (AgentRandon prints every action)."""
    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._original


def evaluate_agent(model_path, reward_type, num_games=50, matches_per_game=10, save_dir="eval_results"):
    """Evaluate a trained agent."""
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    agent = DQNAgentSparseReward(
        name=f"Eval_{reward_type}",
        reward_type=reward_type,
        log_directory=log_dir,
        epsilon_start=0.05,
    )
    agent.load_model(model_path)
    agent.epsilon = 0.05

    print(f"\nEvaluating {reward_type} agent over {num_games} games...")

    for game_idx in range(num_games):
        opponents = [
            AgentRandon(
                name=f"R{game_idx}_{i}",
                log_directory=log_dir,
                verbose_console=False,
                verbose_log=False,
            )
            for i in range(3)
        ]

        room = ChefsHatRoomLocal(
            f"eval_{reward_type}_{game_idx}",
            game_type=ChefsHatEnv.GAMETYPE["MATCHES"],
            stop_criteria=matches_per_game,
            max_rounds=-1,
            verbose_console=False,
            verbose_log=False,
            game_verbose_console=False,
            game_verbose_log=False,
            save_dataset=False,
            log_directory=log_dir,
            timeout_player_response=60,
        )

        room.add_player(agent)
        for opp in opponents:
            room.add_player(opp)

        try:
            with SuppressStdout():
                room.start_new_game()
        except Exception as e:
            print(f"  Game {game_idx} failed: {e}")

        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx+1}/{num_games} games")

    positions = agent.match_positions
    wins = agent.match_wins
    total = len(positions)
    win_count = sum(wins)

    results = {
        "reward_type": reward_type,
        "total_matches": total,
        "wins": win_count,
        "win_rate": win_count / total if total > 0 else 0,
        "avg_position": float(np.mean(positions)) if positions else -1,
    }

    print(f"  {reward_type}: Win rate = {results['win_rate']:.2%}, Avg position = {results['avg_position']:.2f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing trained models")
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--matches-per-game", type=int, default=10)
    args = parser.parse_args()

    reward_types = ["sparse", "shaped", "auxiliary", "performance"]
    all_results = {}

    for rt in reward_types:
        model_path = os.path.join(args.results_dir, f"reward_{rt}_model.pth")
        if os.path.exists(model_path):
            results = evaluate_agent(
                model_path, rt, args.num_games, args.matches_per_game
            )
            all_results[rt] = results

    # Save combined results
    with open(os.path.join(args.results_dir, "eval_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.results_dir}/eval_comparison.json")
