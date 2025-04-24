import os
import pygame
import pickle
import random
import sys
from flappy_bird import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, STAT_FONT

# Initialize Pygame
pygame.init()
pygame.font.init()


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

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


def draw_window(win, bird, pipes, base, score):
    """Draw the game window"""
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    bird.draw(win)

    # Draw score
    score_text = STAT_FONT.render(f"Score: {score}", 1, WHITE)
    win.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))

    pygame.display.update()


def test_ai():
    """Test the trained AI model"""
    # Load the trained model
    try:
        with open("weights/best_bird.pkl", "rb") as f:
            winner = pickle.load(f)
    except FileNotFoundError:
        print(
            "Error: No trained model found. Please train the AI first (weights/best_bird.pkl not found)."
        )
        return

    # Create neural network from the winner
    import neat

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.dirname(local_dir), "config.txt")

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Initialize game objects
    bird = Bird(230, 350)
    base = Base(730)  # Base at the bottom
    pipes = [Pipe(600)]

    # Create window
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird AI Test")

    # Initialize clock for controlling frame rate
    clock = pygame.time.Clock()

    # Initialize score
    score = 0

    # Main game loop
    run = True
    while run:
        clock.tick(30)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()

        # Determine which pipe to focus on
        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

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
        output = net.activate(inputs)

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
            # Check if bird passed the pipe
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
                score += 1

            # Check collision
            if pipe.collide(bird):
                run = False

        # Remove off-screen pipes
        for r in rem:
            pipes.remove(r)

        # Add new pipe if needed
        if add_pipe:
            pipes.append(Pipe(600))

        # Check if bird hit the ground or ceiling
        if bird.y + bird.img.get_height() > 730 or bird.y < 0:
            run = False

        # Move base
        base.move()

        # Draw the game
        draw_window(win, bird, pipes, base, score)

    # Game over
    print(f"Game Over! Score: {score}")


if __name__ == "__main__":
    test_ai()
