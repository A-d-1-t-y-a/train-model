"""
Evaluation Script for Trained DQN Agent with Opponent Modelling
================================================================
Loads a trained model and evaluates against various opponent types.

Student: Praveen K Gandikota (ID: 16829772)
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

from agents.dqn_agent import DQNAgentWithOpponentModelling


class SuppressStdout:
    """Context manager to suppress stdout (AgentRandon prints every action)."""
    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._original


def evaluate_agent(model_path, num_games=50, matches_per_game=10, save_dir="eval_results"):
    """Evaluate a trained agent against random opponents."""
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Load trained agent
    agent = DQNAgentWithOpponentModelling(
        name="Trained_DQN",
        log_directory=log_dir,
        epsilon_start=0.05,  # Minimal exploration during evaluation
        use_opponent_model=True,
    )
    agent.load_model(model_path)
    agent.epsilon = 0.05

    print(f"Loaded model from: {model_path}")
    print(f"Evaluating over {num_games} games ({matches_per_game} matches each)...")

    all_scores = []

    for game_idx in range(num_games):
        opponents = [
            AgentRandon(
                name=f"Eval_R{game_idx}_{i}",
                log_directory=log_dir,
                verbose_console=False,
                verbose_log=False,
            )
            for i in range(3)
        ]

        room = ChefsHatRoomLocal(
            f"eval_game_{game_idx}",
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
                info = room.start_new_game()
            scores = info.get("Game_Score", [0, 0, 0, 0])
            all_scores.append(scores)
        except Exception as e:
            print(f"  Game {game_idx} failed: {e}")

        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx+1}/{num_games} games")

    # Analysis
    positions = agent.match_positions
    wins = agent.match_wins
    total = len(positions)
    win_count = sum(wins)

    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    print(f"  Total matches: {total}")
    print(f"  Wins (1st place): {win_count} ({win_count/total*100:.1f}%)" if total > 0 else "  No matches completed")
    print(f"  Average position: {np.mean(positions):.2f}" if positions else "  No positions recorded")
    if positions:
        print(f"  Position distribution:")
        for pos in range(4):
            count = positions.count(pos)
            print(f"    {pos+1}st: {count} ({count/total*100:.1f}%)")

    # Save results
    results = {
        "total_matches": total,
        "wins": win_count,
        "win_rate": win_count / total if total > 0 else 0,
        "avg_position": float(np.mean(positions)) if positions else -1,
        "position_distribution": {
            str(i+1): positions.count(i) for i in range(4)
        } if positions else {},
    }

    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    if positions:
        fig, ax = plt.subplots(figsize=(8, 5))
        pos_counts = [positions.count(i) for i in range(4)]
        labels = ["1st Place", "2nd Place", "3rd Place", "4th Place"]
        colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
        ax.bar(labels, pos_counts, color=colors, edgecolor="black")
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Agent Performance Distribution\n(Win Rate: {win_count/total*100:.1f}%)", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(pos_counts):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "eval_positions.png"), dpi=300)
        plt.close()

    print(f"\n  Results saved to: {save_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--matches-per-game", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="eval_results")
    args = parser.parse_args()

    evaluate_agent(args.model, args.num_games, args.matches_per_game, args.save_dir)
