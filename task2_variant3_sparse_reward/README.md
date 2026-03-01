# Chef's Hat Gym - Sparse / Delayed Reward Variant

**Module**: 7043SCN - Generative AI and Reinforcement Learning
**Student**: Furkhanuddin Mohammad (ID: 16667990)
**Variant**: Sparse / Delayed Reward (ID mod 7 = 3)

## Assigned Variant

**ID mod 7 = 3 - Sparse / Delayed Reward Variant**: Focus on delayed match rewards, exploring techniques such as reward shaping, auxiliary rewards, or alternative credit-assignment strategies.

## Environment

This project uses [Chef's Hat Gym](https://github.com/pablovin/ChefsHatGYM), a competitive, turn-based, multi-agent card game environment. The game's reward structure presents a key challenge:
- Rewards are only assigned at match end (sparse)
- Many actions occur between the start and end of a match (delayed credit assignment)
- The agent must learn which intermediate actions contributed to the final outcome

## Approach

### Algorithm: DQN with Configurable Reward Functions

The agent uses a standard DQN architecture with four different reward strategies:

- **State representation**: Board state (11) + Player hand (17) = 28-dimensional normalised vector
- **Action handling**: 200 discrete actions with invalid action masking
- **Exploration**: Epsilon-greedy with exponential decay (1.0 → 0.05)

### Reward Strategies

| Strategy | Description | Signal Density |
|----------|-------------|----------------|
| **Sparse** | +1 for winning, -0.001 per step | Very sparse |
| **Shaped** | Graduated position reward + card play bonus + pass penalty | Medium |
| **Auxiliary** | Dense intermediate signals + efficiency bonus + pizza bonus | Dense |
| **PerformanceScore** | Environment performance metric + position reward | Sparse (end of match) |

### Shaped Reward Details
- Playing cards: +0.02 per card played
- Passing: -0.01 penalty
- Match end: Graduated (+1.0, +0.3, -0.3, -1.0) by position

### Auxiliary Reward Details
- Card reduction: +0.03 per card played
- Consecutive pass penalty: -0.005 * pass_count (increasing)
- Pizza completion: +0.05 bonus
- Efficiency bonus at match end: up to +0.2 for quick finishes
- Position reward: (3 - position) / 3.0

## Experiments

Four experiments are conducted, one per reward strategy, all against random opponents:

| Experiment | Reward Type | Hypothesis |
|------------|-------------|------------|
| Exp 1 | Sparse | Slow learning, baseline |
| Exp 2 | Shaped | Faster initial learning |
| Exp 3 | Auxiliary | Fastest learning, best exploration |
| Exp 4 | PerformanceScore | Balanced learning |

### Research Questions
1. Does reward shaping accelerate learning in Chef's Hat?
2. Which reward strategy leads to the highest win rate?
3. Do shaped rewards introduce any learning biases?
4. What is the trade-off between reward density and final performance?

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Full training (all 4 reward strategies)
python train.py --num-games 100 --matches-per-game 5

# Quick test run
python train.py --quick

# Specific reward types only
python train.py --reward-types sparse shaped --num-games 50
```

### Evaluation

```bash
python evaluate.py --results-dir results --num-games 50 --matches-per-game 10
```

## Results Interpretation

After training, results are saved in the `results/` directory:

- `plots/win_rate_comparison.png` - Win rate learning curves for all strategies
- `plots/episode_rewards.png` - Episode rewards over training
- `plots/training_loss.png` - DQN training loss comparison
- `plots/position_distribution.png` - Finishing position distributions
- `plots/reward_distribution.png` - Reward signal distributions
- `summary.json` - Quantitative summary

### Key Metrics
- **Win Rate**: Percentage of matches finishing 1st
- **Average Position**: Mean finishing position (lower is better)
- **Learning Speed**: How quickly win rate improves
- **Reward Distribution**: Density and variance of reward signals

## Project Structure

```
.
├── README.md
├── requirements.txt
├── agents/
│   └── dqn_agent.py         # DQN agent with 4 reward functions
├── train.py                  # Training with reward comparison
├── evaluate.py               # Evaluation script
├── results/                  # Generated (after training)
│   ├── plots/
│   ├── summary.json
│   └── *.pth
└── plots/
```

## Limitations and Challenges

1. **Credit Assignment**: Even with shaping, determining which action in a long game led to winning remains difficult
2. **Reward Hacking**: Shaped rewards may encourage sub-optimal strategies (e.g., playing any card just to reduce hand size)
3. **Hyperparameter Sensitivity**: Reward shaping coefficients significantly affect learning
4. **Non-stationarity**: Against learning opponents, reward statistics would shift

## Video Viva

[Link to video presentation]

## AI Use Declaration

This project was developed with assistance from Claude Code (Anthropic) for:
- Code structure and boilerplate generation
- Reward function implementation suggestions
- Documentation drafting

All code was reviewed, understood, and adapted by the student. The experimental design, reward function analysis, and conclusions are the student's own work.
