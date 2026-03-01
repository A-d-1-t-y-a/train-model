# Chef's Hat Gym - Generative AI Augmentation Variant

**Module**: 7043SCN - Generative AI and Reinforcement Learning
**Student**: Avinash Megavatn (ID: 16829749)
**Variant**: Generative AI Augmentation (ID mod 7 = 6)

## Assigned Variant

**ID mod 7 = 6 - Generative AI Augmentation Variant**: Incorporate a generative AI component (e.g., policy initialisation, opponent simulation, or representation learning) and critically assess its impact on performance and behaviour.

## Environment

This project uses [Chef's Hat Gym](https://github.com/pablovin/ChefsHatGYM), a competitive, turn-based, multi-agent card game environment. Key challenges include:
- Large discrete action space (200 actions)
- Sparse, delayed rewards
- Non-stationary multi-agent dynamics

## Approach

### Generative AI Component: Variational Autoencoder (VAE) for State Representation

A **Variational Autoencoder** learns compressed latent representations of game states. The hypothesis is that a learned representation can capture meaningful game dynamics more effectively than the raw 28-dimensional state vector, potentially improving DQN learning efficiency.

### Architecture

```
Raw State (28-dim) → VAE Encoder → Latent Space (16-dim) → DQN → Q-values (200-dim)
                          ↓
                     VAE Decoder → Reconstructed State (28-dim)
```

- **VAE Encoder**: 28 → 64 → 64 → 16 (mean) + 16 (log-variance)
- **VAE Decoder**: 16 → 64 → 64 → 28
- **DQN Policy**: 16 → 128 → 128 → 200 (with action masking)
- **VAE Loss**: Reconstruction (MSE) + KL Divergence (weighted by beta=1.0)

### State Representation
- **Raw state**: Board (11 values) + Hand (17 values) = 28 dimensions, normalised to [0,1]
- **Latent state**: 16-dimensional compressed representation learned by VAE

### Training Procedure
1. **Phase 1 - State Collection**: Play initial games, collecting states in a buffer
2. **Phase 2 - VAE Pre-training**: Train VAE on collected states (500 steps)
3. **Phase 3 - Joint Training**: Train DQN using VAE-encoded states, with online VAE updates

### Data Augmentation
The trained VAE can generate synthetic game states by sampling from the latent space, providing additional training data for the DQN.

## Experiments

| Experiment | DQN Input | VAE | Purpose |
|------------|-----------|-----|---------|
| Exp 1 (Baseline) | Raw 28-dim state | No | Baseline performance |
| Exp 2 (VAE) | VAE latent 16-dim | Yes | Test generative augmentation |

### Research Questions
1. Does VAE-based state compression improve DQN learning speed?
2. Does the learned latent space capture meaningful game structure?
3. What is the trade-off between representation quality and training overhead?
4. Can the VAE generate plausible synthetic game states?

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Full training
python train.py --num-games 100 --matches-per-game 5

# Quick test
python train.py --quick

# Custom settings
python train.py --num-games 200 --matches-per-game 10 --save-dir my_results
```

### Evaluation

```bash
python evaluate.py --results-dir results --num-games 50 --matches-per-game 10
```

## Results Interpretation

After training, results are saved in the `results/` directory:

- `plots/win_rate_comparison.png` - Win rate: VAE vs Raw DQN
- `plots/training_loss.png` - DQN training loss comparison
- `plots/vae_loss.png` - VAE reconstruction + KL loss
- `plots/position_distribution.png` - Finishing position distributions
- `plots/latent_space_pca.png` - PCA visualisation of VAE latent space
- `plots/latent_variance.png` - PCA explained variance analysis
- `plots/episode_rewards.png` - Episode reward curves
- `summary.json` - Quantitative comparison

### Key Metrics
- **Win Rate**: Match win percentage
- **VAE Loss**: Quality of state reconstruction
- **Latent Space Structure**: Visualised via PCA projection
- **Training Overhead**: Time comparison between VAE and raw DQN

## Project Structure

```
.
├── README.md
├── requirements.txt
├── agents/
│   └── dqn_agent.py         # DQN agent with VAE integration
├── train.py                  # Training: raw vs VAE comparison
├── evaluate.py               # Evaluation script
├── results/                  # Generated (after training)
│   ├── plots/
│   ├── summary.json
│   └── *.pth
└── plots/
```

## Limitations and Challenges

1. **VAE Quality**: The latent space may not capture all game-relevant features
2. **Training Overhead**: VAE adds computational cost; benefits may not justify cost for simple state spaces
3. **State Space Size**: With only 28 dimensions, compression to 16 may lose information
4. **Joint Training Instability**: Simultaneously training VAE and DQN can cause instability
5. **Generated States**: Synthetic states may not be physically valid game configurations

## Video Viva

[Link to video presentation]

## AI Use Declaration

This project was developed with assistance from Claude Code (Anthropic) for:
- VAE architecture implementation
- Code structure and boilerplate generation
- Documentation drafting

All code was reviewed, understood, and adapted by the student. The experimental design, analysis of generative AI impact, and conclusions are the student's own work.
