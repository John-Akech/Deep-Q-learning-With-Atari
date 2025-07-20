# DQN Atari Breakout Agent

This project implements a Deep Q-Network (DQN) agent using Stable Baselines3 to play Atari Breakout.

## Environment
- **Game**: ALE/Breakout-v5 (Atari Breakout)
- **Framework**: Stable Baselines3 with Gymnasium
- **Algorithm**: Deep Q-Network (DQN)

## Files
- `train.py` - Trains DQN agents with different hyperparameters and policies
- `play.py` - Loads the trained model and displays gameplay using GreedyQPolicy
- `dqn_model.zip` - Saved trained model (generated after training)
- `training_results.csv` - Detailed hyperparameter tuning results

## Setup and Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Playing
```bash
python play.py
```

## Hyperparameter Tuning Results

The following hyperparameter configurations were tested, comparing MLPPolicy vs CNNPolicy:

| Hyperparameter Set                                                                 | Noted Behavior                                                   | Avg Reward | Training Time |
|------------------------------------------------------------------------------------|------------------------------------------------------------------|------------|----------------|
| lr=0.0001, gamma=0.99, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (MLPPolicy)  | MLP struggles with visual input, poor spatial understanding       | 1.05       | ~3263.6s       |
| lr=0.0001, gamma=0.99, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (CNNPolicy)  | CNN baseline with optimal exploration schedule                    | 1.70       | ~3263.6s       |
| lr=0.00025, gamma=0.99, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (CNNPolicy) | Higher LR for faster convergence with CNN architecture            | 2.10       | ~3263.6s       |
| lr=0.0001, gamma=0.995, batch_size=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.2 (CNNPolicy) | Higher gamma + larger batch for long-term reward optimization     | 1.35       | ~3263.6s       |


## Results Analysis

### Policy Comparison
- **CNNPolicy significantly outperforms MLPPolicy** for Atari games
- CNNs can extract spatial features from game frames effectively
- MLPPolicy treats pixels as independent features, losing spatial relationships
- Visual pattern recognition is crucial for Atari game performance

### Hyperparameter Impact Analysis

**Learning Rate Impact:**
- **0.0001**: Stable learning for both MLP and CNN (1.05-1.70 reward)
- **0.00025**: Higher LR improved performance with CNN architecture (2.10 reward)
- **Conclusion**: Higher learning rates can benefit CNN architectures for Atari environments

**Policy Comparison:**
- **MLPPolicy**: Achieved 1.05 reward but struggles with spatial visual patterns
- **CNNPolicy**: Better performance (1.70) and better suited for visual processing
- **Conclusion**: CNN architecture is essential for Atari games due to superior visual pattern recognition

**Gamma and Batch Size:**
- **Higher gamma (0.995)**: Combined with larger batch size achieved 1.35 reward
- **Larger batch size (64)**: Showed moderate performance with high gamma
- **Conclusion**: Standard values (gamma=0.99, batch=32) provide better stability and performance

### Best Configuration
Based on actual results:
- **Policy**: CNN significantly outperformed MLP (2.10 vs 1.05 reward)
- **Learning Rate**: 0.00025 (best performance with CNN)
- **Gamma**: 0.99 (better than 0.995)
- **Batch Size**: 32 (more effective than 64)
- **Key Finding**: CNN with slightly higher learning rate provided the best performance

## Technical Implementation

### Training Details
- **Environment**: ALE/Breakout-v5 with AtariWrapper preprocessing
- **Training Steps**: 100,000 timesteps per configuration (~30 minute runtime)
- **Buffer Size**: 20,000 experiences
- **Learning Starts**: 1,000 timesteps (initial exploration)
- **Evaluation**: 20 episodes with deterministic policy (up to 1,000 steps each)

### Play Script Features
- Loads best trained model (`dqn_model.zip`)
- Uses **GreedyQPolicy** (deterministic=True) for evaluation
- Displays game using `env.render()` for real-time visualization
- Runs multiple episodes with performance statistics

## Actual Performance Results
- **MLPPolicy**: 1.05 average reward (~17 minutes training)
- **CNNPolicy**: 1.70-2.10 average reward (~52-57 minutes training)
- **Total Training Time**: ~3-4 hours for all configurations
- **Best Model**: Saved as `dqn_model.zip` (Config 3 - CNN with lr=0.00025)

## Video Demonstration
[[Video showing the trained agent playing Breakout](https://drive.google.com/file/d/1grnGlZfJy7i4geigkSCd1QRdwedspxIg/view?usp=sharing)]

## Play.py Test Results
```
Episode 1 completed: Reward = 2.0, Steps = 24
Episode 2 completed: Reward = 1.0, Steps = 12
Episode 3 completed: Reward = 2.0, Steps = 23
Episode 4 completed: Reward = 3.0, Steps = 41
Episode 5 completed: Reward = 1.0, Steps = 17
Episode 6 completed: Reward = 4.0, Steps = 41
Episode 7 completed: Reward = 3.0, Steps = 33
Episode 8 completed: Reward = 3.0, Steps = 36
Episode 9 completed: Reward = 3.0, Steps = 33
Episode 10 completed: Reward = 2.0, Steps = 24
Episode 11 completed: Reward = 7.0, Steps = 64
Episode 12 completed: Reward = 3.0, Steps = 36
Episode 13 completed: Reward = 1.0, Steps = 17
Episode 14 completed: Reward = 3.0, Steps = 43
Episode 15 completed: Reward = 2.0, Steps = 25

Average reward: 2.80
```
The agent successfully learned basic gameplay - Episode 2 demonstrates ball contact and scoring.

## Key Findings
1. **CNN architecture** significantly outperformed MLP for visual game tasks
2. **Moderately higher learning rate (0.00025)** with CNN achieved best performance (2.10 reward)
3. **Standard gamma (0.99)** provided better stability than higher values
4. **Agent successfully learned** basic Breakout gameplay (ball contact, scoring)
5. **Training time varies** significantly based on hyperparameter complexity

## Submission Files
- `train.py` - Training script with hyperparameter tuning
- `play.py` - Playing script with GreedyQPolicy
- `dqn_model.zip` - Trained model file
- `training_results.csv` - Detailed hyperparameter results
- `requirements.txt` - Dependencies
- `README.md` - This documentation

## Group Members

1. John Akech 
2. Eliane Munezero
3. Kuir Juach Thuch

## Team Contributions

### John Akech
- Set up the Atari environment and wrappers
- Implemented the DQN training script (train.py)
- Developed the play.py script for model evaluation
- Assisted with hyperparameter analysis


### Eliane Munezero
- Conducted hyperparameter tuning experiments
- Analyzed results and documented findings
- Created the project documentation and README


### Kuir Juach Thuch
- Performed model evaluation and testing
- Created visualizations and result analysis
