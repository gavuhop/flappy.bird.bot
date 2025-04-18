import os
import pygame
import neat
import random
import pickle
from flappy_bird import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, STAT_FONT

# Initialize Pygame
pygame.init()
pygame.font.init()


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Global variables
GEN = 0

# Load images
# the welcome image
WELCOME_IMG = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "message.png"))
)

# the bird image
YELLOW_BIRD_IMGS = [
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "bird" + str(i) + ".png"))
    )
    for i in range(1, 4)
]
BLUE_BIRD_IMGS = [
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "bluebird" + str(i) + ".png"))
    )
    for i in range(1, 4)
]
RED_BIRD_IMGS = [
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "redbird" + str(i) + ".png"))
    )
    for i in range(1, 4)
]

BIRD_IMGS = random.choice((YELLOW_BIRD_IMGS, BLUE_BIRD_IMGS, RED_BIRD_IMGS))

# the pipe image
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))

# the base image
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))

# the background image
BG_DAY = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
BG_NIGHT = pygame.transform.scale2x(
    pygame.image.load(os.path.join("imgs", "bg_night.png"))
)

BG_IMG = random.choice((BG_DAY, BG_NIGHT))

# At the top of the file, after the imports
if not os.path.exists("weights"):
    os.makedirs("weights")


def draw_window(win, birds, pipes, base, score, gen):
    """Draw the game window"""
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        if bird.alive:
            bird.draw(win)

    # Draw score, generation, and number of birds alive
    score_text = STAT_FONT.render(f"Score: {score}", 1, WHITE)
    win.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))

    gen_text = STAT_FONT.render(f"Gen: {gen}", 1, WHITE)
    win.blit(gen_text, (10, 10))

    alive_text = STAT_FONT.render(
        f"Alive: {sum(1 for bird in birds if bird.alive)}", 1, WHITE
    )
    win.blit(alive_text, (10, 60))

    pygame.display.update()


def eval_genomes(genomes, config):
    """Evaluate the fitness of each genome"""
    global GEN
    GEN += 1

    # Create neural networks and birds
    nets = []
    ge = []
    birds = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 0))  # Start birds at the top
        genome.fitness = 0
        ge.append(genome)

    # Initialize game objects
    base = Base(730)  # Base at the bottom
    pipes = [Pipe(600)]

    # Create window
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird AI Training")

    # Initialize clock for controlling frame rate
    clock = pygame.time.Clock()

    # Initialize score
    score = 0

    # Track best bird
    best_bird = None
    best_fitness = 0

    # Main game loop
    run = True
    while run:
        clock.tick(30)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                # Save the best bird's weights before quitting
                if best_bird is not None:
                    if not os.path.exists("weights"):
                        os.makedirs("weights")
                    with open("weights/best_bird.pkl", "wb") as f:
                        pickle.dump(best_bird, f)
                pygame.quit()
                quit()

        # Determine which pipe to focus on
        pipe_ind = 0
        if len(birds) > 0:
            if (
                len(pipes) > 1
                and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width()
            ):
                pipe_ind = 1

        # Process each bird
        for x, bird in enumerate(birds):
            if bird.alive:
                # Increment fitness for staying alive
                ge[x].fitness += 0.1

                # Update best bird if current bird has better fitness
                if ge[x].fitness > best_fitness:
                    best_fitness = ge[x].fitness
                    best_bird = ge[x]

                # Get inputs for neural network
                if len(pipes) > 0:
                    inputs = (
                        bird.y,
                        abs(bird.y - pipes[pipe_ind].height),
                        abs(bird.y - pipes[pipe_ind].bottom),
                    )
                else:
                    inputs = (bird.y, 0, 0)

                # Get output from neural network
                output = nets[x].activate(inputs)

                # Jump if output > 0.5
                if output[0] > 0.5:
                    bird.jump()

                # Move bird
                bird.move()

        # Process pipes
        rem = []
        add_pipe = False

        for pipe in pipes:
            # Move pipe
            pipe.move()

            # Check if pipe is off screen
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            # Check collisions and scoring
            for x, bird in enumerate(birds):
                if bird.alive:
                    # Check if bird passed the pipe
                    if not pipe.passed and pipe.x < bird.x:
                        pipe.passed = True
                        add_pipe = True
                        ge[x].fitness += 5  # Reward for passing a pipe

                    # Check collision
                    if pipe.collide(bird):
                        ge[x].fitness -= 1  # Penalty for collision
                        birds[x].alive = False

        # Remove off-screen pipes
        for r in rem:
            pipes.remove(r)

        # Add new pipe if needed
        if add_pipe:
            score += 1
            pipes.append(Pipe(600))

        # Check if birds hit the ground or ceiling
        for x, bird in enumerate(birds):
            if bird.alive:
                if bird.y + bird.img.get_height() > 730 or bird.y < 0:
                    ge[x].fitness -= 1  # Penalty for hitting ground or ceiling
                    birds[x].alive = False

        # Move base
        base.move()

        # Draw the game
        draw_window(win, birds, pipes, base, score, GEN)

        # Check if all birds are dead
        if all(not bird.alive for bird in birds):
            run = False


def run(config_path):
    """Run the NEAT algorithm to train the Flappy Bird AI"""
    # Load configuration
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create population
    pop = neat.Population(config)

    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    try:
        # Run the algorithm
        winner = pop.run(eval_genomes, 50)

        # Save the winner
        if not os.path.exists("weights"):
            os.makedirs("weights")
        with open("weights/best_bird.pkl", "wb") as f:
            pickle.dump(winner, f)

        print(f"Best fitness: {winner.fitness}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving best bird...")
        # The best bird will be saved in eval_genomes when quitting


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.dirname(local_dir), "config.txt")
    run(config_path)
