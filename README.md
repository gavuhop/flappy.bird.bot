# Flappy Bird AI

This project implements a Flappy Bird game with an AI that learns to play using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Requirements

- Python 3.6+
- Pygame
- NEAT-Python

You can install the required packages using pip:

```bash
pip install pygame neat-python
```

## Project Structure

- `flappy_bird.py`: The main game implementation
- `train.py`: Script to train the AI using NEAT
- `test.py`: Script to test the trained AI
- `config.txt`: NEAT configuration file
- `imgs/`: Directory containing game images

## How to Use

### Training the AI

1. Make sure you have the required images in the `imgs/` directory:

   - `bird1.png`, `bird2.png`, `bird3.png`: Yellow bird animation frames
   - `bluebird1.png`, `bluebird2.png`, `bluebird3.png`: Blue bird animation frames
   - `redbird1.png`, `redbird2.png`, `redbird3.png`: Red bird animation frames
   - `pipe.png`: Pipe image
   - `base.png`: Ground image
   - `bg.png`: Day background image
   - `bg_night.png`: Night background image
   - `message.png`: Welcome message image

2. Run the training script:

```bash
python train.py
```

This will start the training process. The AI will learn to play Flappy Bird over multiple generations. The best performing neural network will be saved to `winner.pkl`.

### Testing the Trained AI

After training, you can test the AI:

```bash
python test.py
```

This will load the trained model from `winner.pkl` and run a single bird using the trained neural network.

## How It Works

The AI uses a neural network with 3 inputs:

1. Bird's Y position
2. Absolute difference between bird's Y position and pipe's height
3. Absolute difference between bird's Y position and pipe's bottom

The neural network has 1 output: whether to jump or not.

The fitness function rewards birds for:

- Staying alive (+0.1 per frame)
- Passing pipes (+5.0 per pipe)
- And penalizes them for:
- Colliding with pipes (-1.0)
- Hitting the ground or ceiling (-1.0)

## Configuration

You can modify the NEAT parameters in `config.txt` to adjust the training process. Key parameters include:

- `pop_size`: Population size
- `fitness_threshold`: Target fitness to reach
- `num_hidden`: Number of hidden neurons
- `num_inputs`: Number of input neurons (should be 3)
- `num_outputs`: Number of output neurons (should be 1)

## Credits

This project is based on the classic Flappy Bird game and uses the NEAT algorithm for AI training.
