"""
Training Script for DQN Agent with Reward Shaping Experiments
===============================================================
Variant 3: Sparse / Delayed Reward (Student ID mod 7 = 2 or 3)

This script trains DQN agents with different reward strategies to
investigate the impact of reward shaping on learning in Chef's Hat Gym.

Experiments:
1. Sparse Reward (baseline): Only +1 for winning
2. Shaped Reward: Intermediate signals for card plays and passes
3. Auxiliary Reward: Dense signals with efficiency bonuses
4. Performance Score Reward: Environment-provided scoring

Student: Furkhanuddin Mohammad (ID: 16667990)
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

from agents.dqn_agent import DQNAgentSparseReward


class SuppressStdout:
    """Context manager to suppress stdout (AgentRandon prints every action)."""
    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._original


def run_training_game(agent, room_name, num_matches=10, log_dir="temp/"):
    """Run a single training game against 3 random opponents."""
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
    for i in range(3):
        opp = AgentRandon(
            name=f"Random_{i}",
            log_directory=log_dir,
            verbose_console=False,
            verbose_log=False,
        )
        room.add_player(opp)

    with SuppressStdout():
        info = room.start_new_game()
    return info


def run_experiment(reward_type, num_games, matches_per_game, save_dir="results"):
    """Run a full training experiment with a specific reward type."""
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    experiment_name = f"reward_{reward_type}"

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"  Reward type: {reward_type}")
    print(f"  Games: {num_games}")
    print(f"  Matches/game: {matches_per_game}")

    agent = DQNAgentSparseReward(
        name=f"DQN_{reward_type}",
        reward_type=reward_type,
        log_directory=log_dir,
    )

    all_scores = []
    start_time = time.time()

    for game_idx in range(num_games):
        room_name = f"{experiment_name}_game_{game_idx}"

        try:
            info = run_training_game(
                agent, room_name, matches_per_game, log_dir
            )
            scores = info.get("Game_Score", [0, 0, 0, 0])
            all_scores.append(scores)

            if (game_idx + 1) % max(1, num_games // 10) == 0 or game_idx == 0:
                wins = agent.match_wins[-matches_per_game:]
                win_rate = np.mean(wins) if wins else 0
                avg_reward = np.mean(agent.episode_rewards[-matches_per_game:]) if agent.episode_rewards else 0
                print(
                    f"  Game {game_idx+1}/{num_games} | "
                    f"Win rate: {win_rate:.2%} | "
                    f"Avg reward: {avg_reward:.4f} | "
                    f"Epsilon: {agent.epsilon:.4f}"
                )
        except Exception as e:
            print(f"  Game {game_idx+1} failed: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\n  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save model
    model_path = os.path.join(save_dir, f"{experiment_name}_model.pth")
    agent.save_model(model_path)

    # Get metrics
    metrics = agent.get_metrics()
    metrics["game_scores"] = all_scores
    metrics["experiment_name"] = experiment_name
    metrics["training_time"] = elapsed

    metrics_path = os.path.join(save_dir, f"{experiment_name}_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    return agent, metrics


def plot_reward_comparison(all_metrics, save_dir="results"):
    """Generate comparison plots for reward shaping experiments."""
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    colors = {
        "Sparse": "#e74c3c",
        "Shaped": "#3498db",
        "Auxiliary": "#2ecc71",
        "PerformanceScore": "#9b59b6",
    }

    # Plot 1: Win Rate Learning Curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        wins = metrics["match_wins"]
        if len(wins) > 0:
            window = min(50, len(wins))
            smoothed = np.convolve(wins, np.ones(window)/window, mode="valid")
            color = colors.get(name, "#333333")
            ax.plot(smoothed, label=name, linewidth=2, color=color)

    ax.set_xlabel("Match Number", fontsize=12)
    ax.set_ylabel("Win Rate (Moving Avg)", fontsize=12)
    ax.set_title("Impact of Reward Shaping on Learning Speed", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "win_rate_comparison.png"), dpi=300)
    plt.close()
    print("  Saved: win_rate_comparison.png")

    # Plot 2: Cumulative Rewards
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        rewards = metrics["episode_rewards"]
        if len(rewards) > 0:
            window = min(20, len(rewards))
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            color = colors.get(name, "#333333")
            ax.plot(smoothed, label=name, linewidth=2, color=color)

    ax.set_xlabel("Episode (Match)", fontsize=12)
    ax.set_ylabel("Average Episode Reward", fontsize=12)
    ax.set_title("Episode Rewards by Reward Strategy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "episode_rewards.png"), dpi=300)
    plt.close()
    print("  Saved: episode_rewards.png")

    # Plot 3: Training Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        losses = metrics["training_losses"]
        if len(losses) > 0:
            window = min(100, len(losses))
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            color = colors.get(name, "#333333")
            ax.plot(smoothed, label=name, linewidth=2, color=color, alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("DQN Training Loss by Reward Strategy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=300)
    plt.close()
    print("  Saved: training_loss.png")

    # Plot 4: Position Distribution Comparison
    n_exp = len(all_metrics)
    fig, axes = plt.subplots(1, n_exp, figsize=(5*n_exp, 5))
    if n_exp == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, all_metrics.items()):
        positions = metrics["match_positions"]
        if positions:
            pos_counts = [positions.count(i) for i in range(4)]
            total = len(positions)
            pos_pcts = [c/total*100 for c in pos_counts]
            labels = ["1st", "2nd", "3rd", "4th"]
            bar_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
            bars = ax.bar(labels, pos_pcts, color=bar_colors, edgecolor="black")
            ax.set_title(f"{name}\nReward Strategy", fontsize=11)
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim(0, 60)
            for bar, pct in zip(bars, pos_pcts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f"{pct:.1f}%", ha="center", fontsize=9)

    plt.suptitle("Finishing Position Distribution by Reward Strategy", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "position_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: position_distribution.png")

    # Plot 5: Reward Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, metrics) in enumerate(all_metrics.items()):
        if idx >= 4:
            break
        reward_hist = metrics.get("reward_history", [])
        if reward_hist:
            ax = axes[idx]
            # Filter out the tiny -0.001 penalties for better visualization
            significant = [r for r in reward_hist if abs(r) > 0.005]
            if significant:
                ax.hist(significant, bins=50, color=colors.get(name, "#333"), alpha=0.7, edgecolor="black")
            ax.set_title(f"{name} Reward Distribution", fontsize=11)
            ax.set_xlabel("Reward Value")
            ax.set_ylabel("Frequency")
            ax.grid(alpha=0.3)

    plt.suptitle("Reward Signal Distribution by Strategy", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_distribution.png"), dpi=300)
    plt.close()
    print("  Saved: reward_distribution.png")

    # Summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'Matches':>10} {'Wins':>8} {'Win Rate':>10} {'Avg Pos':>10} {'Time(s)':>10}")
    print(f"{'-'*70}")

    summary = {}
    for name, metrics in all_metrics.items():
        wins = metrics["match_wins"]
        positions = metrics["match_positions"]
        total = len(wins)
        win_count = sum(wins)
        avg_pos = np.mean(positions) if positions else -1
        train_time = metrics.get("training_time", 0)

        summary[name] = {
            "total_matches": total,
            "wins": win_count,
            "win_rate": win_count / total if total > 0 else 0,
            "avg_position": float(avg_pos),
            "training_time": train_time,
        }

        print(f"{name:<20} {total:>10} {win_count:>8} {win_count/total*100 if total > 0 else 0:>9.1f}% {avg_pos:>10.2f} {train_time:>10.1f}")

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN with different reward strategies for Chef's Hat"
    )
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games per experiment")
    parser.add_argument("--matches-per-game", type=int, default=5,
                        help="Matches per game")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run")
    parser.add_argument("--reward-types", nargs="+",
                        default=["sparse", "shaped", "auxiliary", "performance"],
                        help="Reward types to test")
    args = parser.parse_args()

    if args.quick:
        args.num_games = 20
        args.matches_per_game = 5
        print("Quick mode: 20 games, 5 matches each")

    all_metrics = {}

    for reward_type in args.reward_types:
        _, metrics = run_experiment(
            reward_type=reward_type,
            num_games=args.num_games,
            matches_per_game=args.matches_per_game,
            save_dir=args.save_dir,
        )

        # Use the reward function's name for display
        display_name = metrics["reward_type"].capitalize()
        if display_name == "Performance":
            display_name = "PerformanceScore"
        all_metrics[display_name] = metrics

    # Generate comparison plots
    print("\n\nGenerating plots...")
    summary = plot_reward_comparison(all_metrics, args.save_dir)

    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {args.save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
