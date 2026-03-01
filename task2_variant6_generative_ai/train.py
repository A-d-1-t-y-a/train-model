"""
Training Script for DQN Agent with Generative AI Augmentation
================================================================
Variant 6: Generative AI Augmentation (Student ID mod 7 = 6)

This script trains and compares DQN agents with and without
VAE-based state representation learning.

Experiments:
1. DQN with raw state input (baseline)
2. DQN with VAE-encoded state input
3. Analysis of VAE latent space quality

Student: Avinash Megavatn (ID: 16829749)
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

from agents.dqn_agent import DQNAgentWithVAE


class SuppressStdout:
    """Context manager to suppress stdout (AgentRandon prints every action)."""
    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._original


def run_training_game(agent, room_name, num_matches=10, log_dir="temp/"):
    """Run a training game against 3 random opponents."""
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


def run_experiment(use_vae, num_games, matches_per_game, save_dir="results"):
    """Run a full training experiment."""
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    experiment_name = "dqn_vae" if use_vae else "dqn_raw"

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"  VAE enabled: {use_vae}")
    print(f"  Games: {num_games}")
    print(f"  Matches/game: {matches_per_game}")

    agent = DQNAgentWithVAE(
        name=f"Agent_{experiment_name}",
        use_vae=use_vae,
        log_directory=log_dir,
    )

    all_scores = []
    start_time = time.time()

    for game_idx in range(num_games):
        room_name = f"{experiment_name}_game_{game_idx}"

        try:
            info = run_training_game(agent, room_name, matches_per_game, log_dir)
            scores = info.get("Game_Score", [0, 0, 0, 0])
            all_scores.append(scores)

            if (game_idx + 1) % max(1, num_games // 10) == 0 or game_idx == 0:
                wins = agent.match_wins[-matches_per_game:]
                win_rate = np.mean(wins) if wins else 0
                print(
                    f"  Game {game_idx+1}/{num_games} | "
                    f"Win rate: {win_rate:.2%} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"VAE trained: {agent.vae_trained}"
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


def plot_results(all_metrics, save_dir="results"):
    """Generate comparison plots."""
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    colors = {"DQN + VAE": "#e74c3c", "DQN Raw": "#3498db"}

    # Plot 1: Win Rate Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        wins = metrics["match_wins"]
        if len(wins) > 0:
            window = min(50, len(wins))
            smoothed = np.convolve(wins, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2, color=colors.get(name, "#333"))

    ax.set_xlabel("Match Number", fontsize=12)
    ax.set_ylabel("Win Rate (Moving Avg)", fontsize=12)
    ax.set_title("DQN with VAE vs Raw State: Win Rate Over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "win_rate_comparison.png"), dpi=300)
    plt.close()
    print("  Saved: win_rate_comparison.png")

    # Plot 2: Training Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        losses = metrics["training_losses"]
        if len(losses) > 0:
            window = min(100, len(losses))
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2, color=colors.get(name, "#333"))

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("DQN Loss", fontsize=12)
    ax.set_title("DQN Training Loss: VAE vs Raw", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=300)
    plt.close()
    print("  Saved: training_loss.png")

    # Plot 3: VAE Loss (if applicable)
    for name, metrics in all_metrics.items():
        vae_losses = metrics.get("vae_losses", [])
        if len(vae_losses) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(vae_losses, linewidth=2, color="#e74c3c")
            ax.set_xlabel("Training Step", fontsize=12)
            ax.set_ylabel("VAE Loss", fontsize=12)
            ax.set_title("VAE Reconstruction + KL Loss Over Training", fontsize=14)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "vae_loss.png"), dpi=300)
            plt.close()
            print("  Saved: vae_loss.png")

    # Plot 4: Position Distribution
    n_exp = len(all_metrics)
    fig, axes = plt.subplots(1, n_exp, figsize=(6*n_exp, 5))
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
            ax.set_title(f"{name}", fontsize=12)
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim(0, 60)
            for bar, pct in zip(bars, pos_pcts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f"{pct:.1f}%", ha="center", fontsize=9)

    plt.suptitle("Position Distribution: VAE vs Raw", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "position_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: position_distribution.png")

    # Plot 5: Latent Space Visualization (PCA)
    for name, metrics in all_metrics.items():
        latent_reps = metrics.get("latent_representations", [])
        if len(latent_reps) > 100:
            from sklearn.decomposition import PCA

            latent_array = np.array(latent_reps[:2000])
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(latent_array)

            fig, ax = plt.subplots(figsize=(8, 8))
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=np.arange(len(reduced)), cmap="viridis",
                alpha=0.5, s=10,
            )
            plt.colorbar(scatter, label="Time Step", ax=ax)
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            ax.set_title("VAE Latent Space (PCA Projection)\nColor = Training Progress", fontsize=14)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "latent_space_pca.png"), dpi=300)
            plt.close()
            print("  Saved: latent_space_pca.png")

            # Explained variance
            fig, ax = plt.subplots(figsize=(8, 5))
            pca_full = PCA(n_components=min(16, latent_array.shape[1]))
            pca_full.fit(latent_array)
            ax.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                   pca_full.explained_variance_ratio_ * 100,
                   color="#3498db", edgecolor="black")
            ax.set_xlabel("Principal Component", fontsize=12)
            ax.set_ylabel("Explained Variance (%)", fontsize=12)
            ax.set_title("VAE Latent Space: PCA Explained Variance", fontsize=14)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "latent_variance.png"), dpi=300)
            plt.close()
            print("  Saved: latent_variance.png")

    # Plot 6: Episode Rewards
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, metrics in all_metrics.items():
        rewards = metrics["episode_rewards"]
        if len(rewards) > 0:
            window = min(20, len(rewards))
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=name, linewidth=2, color=colors.get(name, "#333"))

    ax.set_xlabel("Episode (Match)", fontsize=12)
    ax.set_ylabel("Average Episode Reward", fontsize=12)
    ax.set_title("Episode Rewards: VAE vs Raw", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "episode_rewards.png"), dpi=300)
    plt.close()
    print("  Saved: episode_rewards.png")

    # Summary
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Matches':>10} {'Wins':>8} {'Win Rate':>10} {'Avg Pos':>10} {'Time(s)':>10}")
    print(f"{'-'*70}")

    summary = {}
    for name, metrics in all_metrics.items():
        wins = metrics["match_wins"]
        positions = metrics["match_positions"]
        total = len(wins)
        win_count = sum(wins)
        avg_pos = np.mean(positions) if positions else -1

        summary[name] = {
            "total_matches": total,
            "wins": win_count,
            "win_rate": win_count / total if total > 0 else 0,
            "avg_position": float(avg_pos),
            "training_time": metrics.get("training_time", 0),
        }

        print(f"{name:<15} {total:>10} {win_count:>8} "
              f"{win_count/total*100 if total > 0 else 0:>9.1f}% "
              f"{avg_pos:>10.2f} {metrics.get('training_time', 0):>10.1f}")

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN with Generative AI Augmentation for Chef's Hat"
    )
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--matches-per-game", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.num_games = 20
        args.matches_per_game = 5
        print("Quick mode: 20 games, 5 matches each")

    all_metrics = {}

    # Experiment 1: DQN with raw state (baseline)
    _, metrics_raw = run_experiment(
        use_vae=False,
        num_games=args.num_games,
        matches_per_game=args.matches_per_game,
        save_dir=args.save_dir,
    )
    all_metrics["DQN Raw"] = metrics_raw

    # Experiment 2: DQN with VAE state representation
    _, metrics_vae = run_experiment(
        use_vae=True,
        num_games=args.num_games,
        matches_per_game=args.matches_per_game,
        save_dir=args.save_dir,
    )
    all_metrics["DQN + VAE"] = metrics_vae

    # Generate plots
    print("\n\nGenerating plots...")
    summary = plot_results(all_metrics, args.save_dir)

    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: {args.save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
