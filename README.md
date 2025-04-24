# Flappy Bird AI Bot

A Python implementation of Flappy Bird with an AI bot that learns to play using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## 🎮 Game Features

- Classic Flappy Bird gameplay
- Multiple bird color variations (yellow, blue, red)
- Day and night background modes
- Score tracking
- Visual representation of the neural network's decision-making process

## 🛠️ Technical Implementation

The AI bot uses a neural network with the following inputs:

- Bird's Y position
- Distance to the next pipe
- Height difference between bird and pipe gap

The neural network has one output: whether to jump or not.

### Fitness Function

The AI is rewarded for:

- Staying alive (+0.1 per frame)
- Successfully passing pipes (+5.0 per pipe)
- And penalized for:
- Colliding with pipes (-1.0)
- Hitting the ground or ceiling (-1.0)

## 📋 Requirements

- Python 3.6+
- Required packages (see requirements.txt):
  - pygame
  - numpy
  - neat-python

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🖼️ Required Assets

Place the following images in the `imgs/` directory:

- Bird animations:
  - `bird1.png`, `bird2.png`, `bird3.png` (yellow bird)
  - `bluebird1.png`, `bluebird2.png`, `bluebird3.png` (blue bird)
  - `redbird1.png`, `redbird2.png`, `redbird3.png` (red bird)
- Environment:
  - `pipe.png` (obstacle)
  - `base.png` (ground)
  - `bg.png` (day background)
  - `bg_night.png` (night background)
  - `message.png` (welcome screen)

## 🚀 How to Run

### Training the AI

```bash
python src/train.py
```

This will:

- Start the NEAT training process
- Show real-time visualization of the learning process
- Save the best performing neural network to `winner.pkl`

### Testing the Trained AI

```bash
python src/test.py
```

This will:

- Load the trained model from `winner.pkl`
- Run a single bird using the trained neural network
- Show the AI's performance in real-time

## 🧠 How the AI Learns

The project uses the NEAT algorithm to evolve neural networks that can play Flappy Bird. The process involves:

1. Creating an initial population of neural networks with random structures
2. Evaluating each network's performance in the game
3. Selecting the best performers for reproduction
4. Creating new networks through mutation and crossover
5. Repeating the process until a satisfactory solution is found

## 📊 Configuration

The NEAT parameters can be adjusted in `config.txt` to optimize the training process:

- Population size
- Fitness threshold
- Number of hidden neurons
- Mutation rates
- Crossover settings

## 🤝 Contributing

Feel free to:

- Report bugs
- Suggest improvements
- Submit pull requests
- Share your trained models

## 📝 License

This project is open source and available under the MIT License.
