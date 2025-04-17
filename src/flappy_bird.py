import os
import pygame
import random
import sys
import numpy as np

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
WIN_WIDTH = 500
WIN_HEIGHT = 800
GRAVITY = 0.5
FLAP_STRENGTH = -10.5
PIPE_SPEED = 5
PIPE_GAP = 200
PIPE_FREQUENCY = 1500  # milliseconds
PIPE_SPAWN_DISTANCE = 300  # Distance between pipes

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


# Load images
def load_images():
    # Create imgs directory if it doesn't exist
    if not os.path.exists("imgs"):
        os.makedirs("imgs")
        print("Created 'imgs' directory. Please add bird, pipe, and background images.")
        return None, None, None, None

    # Check if images exist
    if not all(
        os.path.exists(os.path.join("imgs", img))
        for img in ["bird1.png", "pipe.png", "base.png", "bg.png"]
    ):
        print("Missing required images in 'imgs' directory.")
        return None, None, None, None

    # Load images
    BIRD_IMGS = [
        pygame.transform.scale2x(
            pygame.image.load(os.path.join("imgs", f"bird{i}.png"))
        )
        for i in range(1, 4)
    ]
    PIPE_IMG = pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "pipe.png"))
    )
    BASE_IMG = pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "base.png"))
    )
    BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

    return BIRD_IMGS, PIPE_IMG, BASE_IMG, BG_IMG


# Try to load images, use fallback if not available
BIRD_IMGS, PIPE_IMG, BASE_IMG, BG_IMG = load_images()

# If images couldn't be loaded, use simple shapes
if BIRD_IMGS is None:
    BIRD_IMGS = [pygame.Surface((30, 30)) for _ in range(3)]
    for img in BIRD_IMGS:
        img.fill(WHITE)

    PIPE_IMG = pygame.Surface((50, 500))
    PIPE_IMG.fill(GREEN)

    BASE_IMG = pygame.Surface((500, 100))
    BASE_IMG.fill((139, 69, 19))  # Brown color

    BG_IMG = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
    BG_IMG.fill((135, 206, 235))  # Sky blue

# Font
STAT_FONT = pygame.font.SysFont("arial", 40)


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = BIRD_IMGS[0]
        self.alive = True
        self.fitness = 0

    def jump(self):
        self.vel = FLAP_STRENGTH
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        # Calculate displacement
        d = self.vel * self.tick_count + 1.5 * self.tick_count**2

        # Limit maximum downward velocity
        if d >= 16:
            d = 16

        # Apply displacement
        self.y = self.y + d

        # Tilt the bird based on velocity
        if d < 0 or self.y < self.height + 50:
            if self.tilt < 25:
                self.tilt = 25
        else:
            if self.tilt > -90:
                self.tilt -= 5

    def draw(self, win):
        # Animate bird
        self.img_count += 1

        if self.img_count < 5:
            self.img = BIRD_IMGS[0]
        elif self.img_count < 10:
            self.img = BIRD_IMGS[1]
        elif self.img_count < 15:
            self.img = BIRD_IMGS[2]
        elif self.img_count < 20:
            self.img = BIRD_IMGS[1]
        elif self.img_count == 20:
            self.img = BIRD_IMGS[0]
            self.img_count = 0

        # Don't flap wings if bird is falling
        if self.tilt <= -80:
            self.img = BIRD_IMGS[1]
            self.img_count = 10

        # Rotate the image
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center
        )
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randrange(50, 450)
        self.top = self.height - PIPE_IMG.get_height()
        self.bottom = self.height + PIPE_GAP
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    VEL = PIPE_SPEED
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


class Game:
    def __init__(self, num_birds=20):
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Flappy Bird AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.num_birds = num_birds
        self.reset_game()

    def reset_game(self):
        self.birds = [Bird(230, 350) for _ in range(self.num_birds)]
        self.pipes = [Pipe(600)]
        self.base = Base(WIN_HEIGHT - 100)
        self.scores = [0] * len(self.birds)
        self.last_pipe = pygame.time.get_ticks()
        self.game_over = False
        return self.get_state()

    def get_state(self):
        # Find the closest pipe
        closest_pipe = None
        closest_dist = float("inf")

        for pipe in self.pipes:
            if pipe.x + PIPE_IMG.get_width() > self.birds[0].x:
                dist = pipe.x - self.birds[0].x
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pipe = pipe

        # Return states for all birds
        states = []
        for bird in self.birds:
            if bird.alive:
                if closest_pipe:
                    # 3 inputs: bird's y position, distance to pipe,
                    # height difference to pipe center
                    state = np.array(
                        [
                            bird.y / WIN_HEIGHT,  # Bird's Y position (normalized)
                            (closest_pipe.x - bird.x) / WIN_WIDTH,  # Distance to pipe
                            (bird.y - closest_pipe.height) / WIN_HEIGHT,  # Height diff
                        ]
                    )
                else:
                    # Default state if no pipe is found
                    state = np.array([bird.y / WIN_HEIGHT, 1.0, 0.0])
                states.append(state)
            else:
                # Default state for dead birds
                states.append(np.array([0.5, 1.0, 0.0]))

        return states

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset_game()
                    else:
                        for bird in self.birds:
                            if bird.alive:
                                bird.jump()

    def draw_window(self, win, birds, pipes, base, score, gen):
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

    def step(self, actions):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Apply actions to birds
        for i, (bird, action) in enumerate(zip(self.birds, actions)):
            if bird.alive:
                if action == 1:
                    bird.jump()
                bird.move()

        # Add new pipe
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > PIPE_FREQUENCY:
            # Check if the last pipe is far enough to spawn a new one
            if not self.pipes or self.pipes[-1].x < WIN_WIDTH - PIPE_SPAWN_DISTANCE:
                self.pipes.append(Pipe(WIN_WIDTH))
                self.last_pipe = time_now

        # Update pipes and check collisions
        rem = []
        add_pipe = False

        for pipe in self.pipes:
            pipe.move()

            # Check if pipe is off screen
            if pipe.x + PIPE_IMG.get_width() < 0:
                rem.append(pipe)
                continue

            # Check collisions and scoring for each bird
            for i, bird in enumerate(self.birds):
                if bird.alive:
                    # Check if bird passed the pipe
                    if not pipe.passed and pipe.x < bird.x:
                        pipe.passed = True
                        add_pipe = True
                        self.scores[i] += 1

                    # Check collision
                    if pipe.collide(bird):
                        bird.alive = False

        # Remove off-screen pipes
        for r in rem:
            self.pipes.remove(r)

        # Add new pipe if needed
        if add_pipe:
            # Only add a new pipe if the last one is far enough
            if not self.pipes or self.pipes[-1].x < WIN_WIDTH - PIPE_SPAWN_DISTANCE:
                self.pipes.append(Pipe(WIN_WIDTH))

        # Check if birds hit the ground or ceiling
        for i, bird in enumerate(self.birds):
            if bird.alive:
                if bird.y + bird.img.get_height() >= WIN_HEIGHT - 100 or bird.y < 0:
                    bird.alive = False

        # Move base
        self.base.move()

        # Draw the game
        self.draw_window(
            self.screen, self.birds, self.pipes, self.base, max(self.scores), 0
        )

        # Check if all birds are dead
        self.game_over = all(not bird.alive for bird in self.birds)

        # Get states
        states = self.get_state()

        return states, self.game_over, self.scores

    def run(self):
        while True:
            self.handle_events()

            if not self.game_over:
                # Update birds
                for bird in self.birds:
                    if bird.alive:
                        bird.move()

                # Add new pipe
                time_now = pygame.time.get_ticks()
                if time_now - self.last_pipe > PIPE_FREQUENCY:
                    # Check if the last pipe is far enough to spawn a new one
                    if (
                        not self.pipes
                        or self.pipes[-1].x < WIN_WIDTH - PIPE_SPAWN_DISTANCE
                    ):
                        self.pipes.append(Pipe(WIN_WIDTH))
                        self.last_pipe = time_now

                # Update pipes and check collisions
                rem = []
                add_pipe = False

                for pipe in self.pipes:
                    pipe.move()

                    # Check if pipe is off screen
                    if pipe.x + PIPE_IMG.get_width() < 0:
                        rem.append(pipe)
                        continue

                    # Check collisions and scoring for each bird
                    for i, bird in enumerate(self.birds):
                        if bird.alive:
                            # Check if bird passed the pipe
                            if not pipe.passed and pipe.x < bird.x:
                                pipe.passed = True
                                add_pipe = True
                                self.scores[i] += 1

                            # Check collision
                            if pipe.collide(bird):
                                bird.alive = False

                # Remove off-screen pipes
                for r in rem:
                    self.pipes.remove(r)

                # Add new pipe if needed
                if add_pipe:
                    # Only add a new pipe if the last one is far enough
                    if (
                        not self.pipes
                        or self.pipes[-1].x < WIN_WIDTH - PIPE_SPAWN_DISTANCE
                    ):
                        self.pipes.append(Pipe(WIN_WIDTH))

                # Check if birds hit the ground or ceiling
                for i, bird in enumerate(self.birds):
                    if bird.alive:
                        if (
                            bird.y + bird.img.get_height() >= WIN_HEIGHT - 100
                            or bird.y < 0
                        ):
                            bird.alive = False

                # Move base
                self.base.move()

            # Draw the game
            self.draw_window(
                self.screen, self.birds, self.pipes, self.base, max(self.scores), 0
            )

            # Check if all birds are dead
            self.game_over = all(not bird.alive for bird in self.birds)

            # Control frame rate
            self.clock.tick(30)


if __name__ == "__main__":
    game = Game()
    game.run()
