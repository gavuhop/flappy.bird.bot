# Flappy Bird AI with PyTorch

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird using PyTorch.

## Overview

The project consists of three main components:

1. **Flappy Bird Game**: A simple implementation of the Flappy Bird game using Pygame.
2. **DQN Agent**: A Deep Q-Network agent implemented in PyTorch that learns to play the game.
3. **Training and Testing**: Scripts to train and test the agent.

## Requirements

- Python 3.10+
- Pygame 2.6.1
- NumPy 1.23.5
- PyTorch 2.1.0
- Protobuf 3.20.3+

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/flappy-bird-bot.git
   cd flappy-bird-bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

To train a new agent:

```
python main.py
```

Then select the "train" mode when prompted. The agent will be trained for 1000 episodes, and the weights will be saved every 100 episodes in the "weights" folder.

### Testing

To test a trained agent:

```
python main.py
```

Then select the "test" mode when prompted and choose a weights file from the list.

## Project Structure

- `main.py`: Main script for training and testing the agent.
- `agent/dqn_agent.py`: Implementation of the DQN agent using PyTorch.
- `game/flappy_bird.py`: Implementation of the Flappy Bird game using Pygame.
- `weights/`: Directory where trained weights are saved.
- `test_pytorch.py`: Script to test if PyTorch is working correctly.
- `test_dqn.py`: Script to test the DQN agent implementation.

## How It Works

The DQN agent learns to play Flappy Bird by:

1. Observing the current state of the game (bird position, velocity, pipe positions).
2. Choosing an action (flap or do nothing) based on the current state.
3. Receiving a reward based on the outcome of the action.
4. Learning from the experience using a neural network.

The agent uses experience replay to learn from past experiences and a target network to stabilize training.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
