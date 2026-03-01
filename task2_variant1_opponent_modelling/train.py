"""
Training Script for DQN Agent with Opponent Modelling
=====================================================
Variant 1: Opponent Modelling (Student ID mod 7 = 0 or 1)

Student: Praveen K Gandikota (ID: 16829772)
"""

import os
import sys
import io
import json
import time
import pickle
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


def run_training_game(agent, opponent_agents, room_name, num_matches=10, log_dir="temp/"):
    """Run a single training game (multiple matches)."""
    room = ChefsHatRoomLocal(
        room_name,
        game_type=ChefsHatEnv.GAMETYPE["MATCHES"],
        stop_criteria=num_matches,
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
    for opp in opponent_agents:
        room.add_player(opp)

    with SuppressStdout():
        info = room.start_new_game()
    return info


def create_random_opponents(log_dir, suffix=""):
    """Create 3 random baseline opponents."""
    return [
        AgentRandon(name=f"R{suffix}_{i}", log_directory=log_dir, verbose_console=False, verbose_log=False)
        for i in range(3)
    ]


def create_mixed_opponents(log_dir, trained_agent_path=None, suffix=""):
    """Create mixed opponents: 1 trained DQN + 2 random."""
    opponents = [
        AgentRandon(name=f"MR{suffix}_{i}", log_directory=log_dir, verbose_console=False, verbose_log=False)
        for i in range(2)
    ]
    dqn_opponent = DQNAgentWithOpponentModelling(
        name=f"DQN_Opp{suffix}",
        log_directory=log_dir,
        epsilon_start=0.3,
        use_opponent_model=False,
    )
    if trained_agent_path and os.path.exists(trained_agent_path):
        dqn_opponent.load_model(trained_agent_path)
        dqn_opponent.epsilon = 0.1
    opponents.append(dqn_opponent)
    return opponents


def run_experiment(experiment_name, num_games, matches_per_game,
                   opponent_type="random", use_opponent_model=True,
                   save_dir="results", trained_opponent_path=None):
    """Run a full training experiment."""
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"  Games: {num_games}, Matches/game: {matches_per_game}")
    print(f"  Opponents: {opponent_type}, Opponent model: {use_opponent_model}")
    print(f"{'='*60}")

    agent = DQNAgentWithOpponentModelling(
        name="DQN_Agent",
        log_directory=log_dir,
        use_opponent_model=use_opponent_model,
    )

    all_game_scores = []
    start_time = time.time()

    for game_idx in range(num_games):
        if opponent_type == "random":
            opponents = create_random_opponents(log_dir, suffix=f"g{game_idx}")
        elif opponent_type == "mixed":
            opponents = create_mixed_opponents(log_dir, trained_opponent_path, suffix=f"g{game_idx}")
        else:
            opponents = create_random_opponents(log_dir, suffix=f"g{game_idx}")

        room_name = f"{experiment_name}_g{game_idx}"

        try:
            info = run_training_game(agent, opponents, room_name, matches_per_game, log_dir)
            scores = info.get("Game_Score", [0, 0, 0, 0])
            all_game_scores.append(scores)

            if (game_idx + 1) % max(1, num_games // 10) == 0 or game_idx == 0:
                win_rate = np.mean(agent.match_wins[-matches_per_game:]) if agent.match_wins else 0
                print(f"  Game {game_idx+1}/{num_games} | Win rate: {win_rate:.2%} | Eps: {agent.epsilon:.4f}")
        except Exception as e:
            print(f"  Game {game_idx+1} failed: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    model_path = os.path.join(save_dir, f"{experiment_name}_model.pth")
    agent.save_model(model_path)

    metrics = agent.get_metrics()
    metrics["game_scores"] = all_game_scores
    metrics["experiment_name"] = experiment_name
    metrics["training_time"] = elapsed

    metrics_path = os.path.join(save_dir, f"{experiment_name}_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    return agent, metrics


def plot_results(all_metrics, save_dir="results"):
    """Generate comparison plots."""
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    # Plot 1: Win Rate
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        wins = metrics["match_wins"]
        if len(wins) > 0:
            window = min(50, len(wins))
            smoothed = np.convolve(wins, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2)
    ax.set_xlabel("Match Number", fontsize=12)
    ax.set_ylabel("Win Rate (Moving Avg)", fontsize=12)
    ax.set_title("Win Rate: Opponent Modelling Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "win_rate_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Training Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        losses = metrics["training_losses"]
        if len(losses) > 0:
            window = min(100, len(losses))
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2, alpha=0.8)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("DQN Training Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "training_loss.png"), dpi=300)
    plt.close()

    # Plot 3: Position Distribution
    n = len(all_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, metrics) in zip(axes, all_metrics.items()):
        positions = metrics["match_positions"]
        if positions:
            pos_counts = [positions.count(i) for i in range(4)]
            colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
            ax.bar(["1st", "2nd", "3rd", "4th"], pos_counts, color=colors, edgecolor="black")
            ax.set_title(f"{name}", fontsize=11)
            ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "position_distribution.png"), dpi=300)
    plt.close()

    # Plot 4: Opponent Model Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for name, metrics in all_metrics.items():
        opp_losses = metrics.get("opponent_losses", [])
        if len(opp_losses) > 0:
            has_data = True
            window = min(50, len(opp_losses))
            smoothed = np.convolve(opp_losses, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2)
    if has_data:
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Opponent Model Prediction Loss", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "opponent_model_loss.png"), dpi=300)
    plt.close()

    # Plot 5: Episode Rewards
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        rewards = metrics["episode_rewards"]
        if len(rewards) > 0:
            window = min(20, len(rewards))
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Episode Rewards", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "episode_rewards.png"), dpi=300)
    plt.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Experiment':<30} {'Matches':>8} {'Wins':>6} {'Win%':>8} {'AvgPos':>8}")
    print(f"{'-'*70}")
    summary = {}
    for name, metrics in all_metrics.items():
        wins = metrics["match_wins"]
        positions = metrics["match_positions"]
        total = len(wins)
        wc = sum(wins)
        avg = np.mean(positions) if positions else -1
        summary[name] = {"total": total, "wins": wc, "win_rate": wc/total if total > 0 else 0, "avg_position": float(avg)}
        print(f"  {name:<28} {total:>8} {wc:>6} {wc/total*100 if total>0 else 0:>7.1f}% {avg:>8.2f}")

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPlots saved to {save_dir}/plots/")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train DQN with Opponent Modelling")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--matches-per-game", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.num_games = 5
        args.matches_per_game = 2

    all_metrics = {}

    _, m1 = run_experiment("exp1_no_opp_model", args.num_games, args.matches_per_game,
                           "random", False, args.save_dir)
    all_metrics["Random (No OppModel)"] = m1

    _, m2 = run_experiment("exp2_with_opp_model", args.num_games, args.matches_per_game,
                           "random", True, args.save_dir)
    all_metrics["Random (With OppModel)"] = m2

    trained_path = os.path.join(args.save_dir, "exp2_with_opp_model_model.pth")
    _, m3 = run_experiment("exp3_mixed_opp_model", args.num_games, args.matches_per_game,
                           "mixed", True, args.save_dir, trained_path)
    all_metrics["Mixed (With OppModel)"] = m3

    print("\nGenerating plots...")
    plot_results(all_metrics, args.save_dir)
    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
