# Chef's Hat Gym - Opponent Modelling Variant

**Module**: 7043SCN - Generative AI and Reinforcement Learning
**Student**: Praveen K Gandikota (ID: 16829772)
**Variant**: Opponent Modelling (ID mod 7 = 1)

## Assigned Variant

**ID mod 7 = 1 - Opponent Modelling Variant**: Investigate the impact of different opponent behaviours, including training against varied baselines, explicit opponent modelling, or analysis of non-stationarity.

## Environment

This project uses [Chef's Hat Gym](https://github.com/pablovin/ChefsHatGYM), a competitive, turn-based, multi-agent card game environment. The game features:
- 4 players competing over multiple matches
- Large discrete action space (200 possible actions)
- Delayed rewards (assigned at match end)
- Role-based mechanics (Chef, Sous-Chef, Waiter, Dishwasher)

## Approach

### Algorithm: Dueling DQN with Opponent Modelling

The agent uses a **Dueling Double DQN** architecture enhanced with an **opponent behaviour prediction model**:

- **State representation**: Board state (11 values) + Player hand (17 values) = 28-dimensional vector, normalized to [0,1]
- **Action handling**: 200-action discrete space with invalid action masking using the environment's legal action mask
- **Reward**: `RewardOnlyWinning` - sparse reward (+1 for winning, -0.001 per step)
- **Opponent Model**: A neural network that learns to predict opponent action distributions from observed game states

### Key Design Choices

1. **Dueling Architecture**: Separates value and advantage estimation for better state evaluation
2. **Double DQN**: Reduces overestimation bias using separate selection and evaluation networks
3. **Opponent Modelling**: Tracks opponent state-action pairs and trains a prediction model to anticipate opponent behaviour
4. **Epsilon-greedy Exploration**: Starts at 1.0, decays to 0.05 with factor 0.9995

## Experiments

Three experiments are conducted:

| Experiment | Opponents | Opponent Model |
|------------|-----------|----------------|
| Exp 1 | Random agents | Disabled |
| Exp 2 | Random agents | Enabled |
| Exp 3 | Mixed (Random + DQN) | Enabled |

### Research Questions
1. Does explicit opponent modelling improve win rate against random opponents?
2. How does the agent adapt when facing diverse opponent types (non-stationarity)?
3. What is the impact of opponent diversity on learning stability?

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Full training (100 games per experiment)
python train.py --num-games 100 --matches-per-game 5

# Quick test run
python train.py --quick

# Custom configuration
python train.py --num-games 200 --matches-per-game 10 --save-dir my_results
```

### Evaluation

```bash
python evaluate.py --model results/exp2_random_with_opp_model_model.pth --num-games 50
```

## Results Interpretation

After training, results are saved in the `results/` directory:

- `plots/win_rate_comparison.png` - Win rate learning curves for all experiments
- `plots/training_loss.png` - DQN training loss over time
- `plots/position_distribution.png` - Distribution of finishing positions
- `plots/opponent_model_loss.png` - Opponent prediction model loss
- `plots/episode_rewards.png` - Cumulative rewards per episode
- `summary.json` - Quantitative summary of all experiments

### Key Metrics
- **Win Rate**: Percentage of matches where agent finishes 1st
- **Average Position**: Mean finishing position (0=1st, 3=4th)
- **Opponent Model Loss**: Cross-entropy loss of opponent behaviour prediction

## Project Structure

```
.
├── README.md
├── requirements.txt
├── agents/
│   └── dqn_agent.py         # DQN agent with opponent modelling
├── train.py                  # Training script with 3 experiments
├── evaluate.py               # Evaluation script
├── results/                  # Generated results (after training)
│   ├── plots/
│   ├── summary.json
│   └── *.pth (saved models)
└── plots/
```

## Limitations and Challenges

1. **Partial Observability**: The agent cannot see opponents' hands, limiting opponent modelling accuracy
2. **Sparse Rewards**: Learning signal only at match end makes credit assignment difficult
3. **Non-stationarity**: Opponent behaviour changes as agents learn, complicating convergence
4. **Computational Cost**: Training multiple experiments sequentially is time-consuming

## Video Viva

[Link to video presentation]

## AI Use Declaration

This project was developed with assistance from Claude Code (Anthropic) for:
- Code structure and boilerplate generation
- Debugging assistance
- Documentation drafting

All code was reviewed, understood, and adapted by the student. The experimental design, analysis, and conclusions are the student's own work.
