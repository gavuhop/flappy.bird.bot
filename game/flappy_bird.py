import pygame
import random
import numpy as np


class FlappyBird:
    def __init__(self, width=400, height=600):
        # Initialize Pygame
        pygame.init()

        # Constants
        self.WIDTH = width
        self.HEIGHT = height
        self.GRAVITY = 0.5
        self.FLAP_STRENGTH = -8
        self.PIPE_SPEED = 4
        self.PIPE_GAP = 140
        self.PIPE_FREQUENCY = 1000  # milliseconds

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)

        # Game objects
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Flappy Bird AI")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # Bird properties
        self.bird_x = 100
        self.bird_y = self.HEIGHT // 2
        self.bird_velocity = 0
        self.bird_size = 30

        # Pipes
        self.pipes = []
        first_pipe = self._create_pipe()
        first_pipe["x"] = self.WIDTH * 1  # Place first pipe at 60% of screen
        self.pipes.append(first_pipe)

        # Game state
        self.score = 0
        self.game_over = False
        self.last_pipe_time = pygame.time.get_ticks()

        return self._get_state()

    def _create_pipe(self):
        gap_y = random.randint(150, self.HEIGHT - 150)
        return {"x": self.WIDTH, "gap_y": gap_y, "scored": False}

    def _get_state(self):
        # Find closest pipe
        closest_pipe = None
        closest_dist = float("inf")
        for pipe in self.pipes:
            if pipe["x"] + 50 > self.bird_x:  # 50 is pipe width
                dist = pipe["x"] - self.bird_x
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pipe = pipe

        if closest_pipe:
            state = np.array(
                [
                    self.bird_y / self.HEIGHT,  # Normalized bird height
                    self.bird_velocity / 10,  # Normalized bird velocity
                    closest_pipe["gap_y"] / self.HEIGHT,  # Normalized pipe gap position
                    (closest_pipe["x"] - self.bird_x)
                    / self.WIDTH,  # Normalized distance to pipe
                ]
            )
        else:
            state = np.array(
                [
                    self.bird_y / self.HEIGHT,
                    self.bird_velocity / 10,
                    0.5,  # Default gap position
                    1.0,  # Max distance
                ]
            )

        return state

    def step(self, action):
        # Reward shaping: thưởng/phạt dựa trên khoảng cách tới tâm khe ống
        reward = 0.05  # Thưởng nhỏ khi còn sống

        # Handle action
        if action == 1:  # Flap
            self.bird_velocity = self.FLAP_STRENGTH

        # Update bird
        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity

        # Create new pipes
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe_time > self.PIPE_FREQUENCY:
            self.pipes.append(self._create_pipe())
            self.last_pipe_time = time_now

        # Find closest pipe
        closest_pipe = None
        closest_dist = float("inf")
        for pipe in self.pipes:
            if pipe["x"] + 50 > self.bird_x:
                dist = pipe["x"] - self.bird_x
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pipe = pipe

        # Reward shaping: thưởng/phạt dựa trên khoảng cách chim tới tâm khe ống
        if closest_pipe:
            gap_center = closest_pipe["gap_y"]
            bird_center = self.bird_y + self.bird_size / 2
            dist_y = abs(bird_center - gap_center)
            # Chuẩn hóa khoảng cách theo chiều cao màn hình
            norm_dist = dist_y / self.HEIGHT
            # Phạt nhẹ nếu chim xa tâm khe, thưởng nếu gần
            reward += 0.2 * (1 - norm_dist) - 0.1 * norm_dist

        # Update and check pipes
        for pipe in self.pipes[:]:
            pipe["x"] -= self.PIPE_SPEED

            # Remove off-screen pipes
            if pipe["x"] + 50 < 0:  # 50 is pipe width
                self.pipes.remove(pipe)
                continue

            # Score points
            if not pipe["scored"] and pipe["x"] < self.bird_x:
                pipe["scored"] = True
                self.score += 1
                reward = 1  # Giữ nguyên thưởng lớn khi qua ống

            # Check collision
            if self._check_collision(pipe):
                self.game_over = True
                reward = -1
                break

        # Check boundary collision
        if self.bird_y < 0 or self.bird_y + self.bird_size > self.HEIGHT:
            self.game_over = True
            reward = -1

        return self._get_state(), reward, self.game_over, self.score

    def _check_collision(self, pipe):
        # Simple rectangle collision
        if (
            self.bird_x + self.bird_size > pipe["x"] and self.bird_x < pipe["x"] + 50
        ):  # 50 is pipe width
            if (
                self.bird_y < pipe["gap_y"] - self.PIPE_GAP // 2
                or self.bird_y + self.bird_size > pipe["gap_y"] + self.PIPE_GAP // 2
            ):
                return True
        return False

    def render(self):
        # Process Pygame events to prevent "Not Responding" when switching tabs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False

        self.screen.fill(self.BLACK)

        # Draw bird
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            (self.bird_x, self.bird_y, self.bird_size, self.bird_size),
        )

        # Draw pipes
        for pipe in self.pipes:
            # Draw top pipe
            pygame.draw.rect(
                self.screen,
                self.GREEN,
                (pipe["x"], 0, 50, pipe["gap_y"] - self.PIPE_GAP // 2),
            )
            # Draw bottom pipe
            pygame.draw.rect(
                self.screen,
                self.GREEN,
                (
                    pipe["x"],
                    pipe["gap_y"] + self.PIPE_GAP // 2,
                    50,
                    self.HEIGHT - (pipe["gap_y"] + self.PIPE_GAP // 2),
                ),
            )

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        pygame.quit()
