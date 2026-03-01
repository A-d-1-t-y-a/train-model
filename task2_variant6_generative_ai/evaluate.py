"""
Evaluation Script for DQN Agent with VAE Augmentation
=======================================================
Student: Avinash Megavatn (ID: 16829749)
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

from agents.dqn_agent import DQNAgentWithVAE


class SuppressStdout:
    """Context manager to suppress stdout (AgentRandon prints every action)."""
    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._original


def evaluate_agent(model_path, use_vae, num_games=50, matches_per_game=10, save_dir="eval_results"):
    """Evaluate a trained agent."""
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    label = "VAE" if use_vae else "Raw"

    agent = DQNAgentWithVAE(
        name=f"Eval_{label}",
        use_vae=use_vae,
        log_directory=log_dir,
        epsilon_start=0.05,
    )
    agent.load_model(model_path)
    agent.epsilon = 0.05

    print(f"\nEvaluating {label} agent over {num_games} games...")

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
            f"eval_{label}_{game_idx}",
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
        "method": label,
        "use_vae": use_vae,
        "total_matches": total,
        "wins": win_count,
        "win_rate": win_count / total if total > 0 else 0,
        "avg_position": float(np.mean(positions)) if positions else -1,
    }

    print(f"  {label}: Win rate = {results['win_rate']:.2%}, Avg position = {results['avg_position']:.2f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--matches-per-game", type=int, default=10)
    args = parser.parse_args()

    all_results = {}

    for name, use_vae in [("dqn_raw", False), ("dqn_vae", True)]:
        model_path = os.path.join(args.results_dir, f"{name}_model.pth")
        if os.path.exists(model_path):
            results = evaluate_agent(
                model_path, use_vae, args.num_games, args.matches_per_game
            )
            all_results[name] = results

    with open(os.path.join(args.results_dir, "eval_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.results_dir}/eval_comparison.json")
