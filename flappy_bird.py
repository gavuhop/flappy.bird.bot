import pygame
import random
import sys

# Khởi tạo Pygame
pygame.init()

# Các hằng số
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.5
FLAP_STRENGTH = -8
PIPE_SPEED = 4
PIPE_GAP = 140
PIPE_FREQUENCY = 1000  # milliseconds

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.size = 30

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.size, self.size))

    def get_mask(self):
        return pygame.mask.Mask((self.size, self.size), True)


class Pipe:
    def __init__(self):
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 150)
        self.x = SCREEN_WIDTH
        self.width = 50
        self.passed = False
        self.scored = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, screen):
        # Vẽ ống trên
        pygame.draw.rect(
            screen, GREEN, (self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        )
        # Vẽ ống dưới
        pygame.draw.rect(
            screen,
            GREEN,
            (
                self.x,
                self.gap_y + PIPE_GAP // 2,
                self.width,
                SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2),
            ),
        )

    def collide(self, bird):
        bird_mask = bird.get_mask()
        
        # Tạo mask cho ống trên
        top_pipe = pygame.Surface((self.width, self.gap_y - PIPE_GAP // 2))
        top_pipe.fill(GREEN)
        top_mask = pygame.mask.from_surface(top_pipe)
        
        # Tạo mask cho ống dưới
        bottom_height = SCREEN_HEIGHT - (self.gap_y + PIPE_GAP // 2)
        bottom_pipe = pygame.Surface((self.width, bottom_height))
        bottom_pipe.fill(GREEN)
        bottom_mask = pygame.mask.from_surface(bottom_pipe)

        # Tính toán offset
        top_offset = (int(self.x - bird.x), int(0 - bird.y))
        bottom_offset = (int(self.x - bird.x), int(self.gap_y + PIPE_GAP // 2 - bird.y))

        # Kiểm tra va chạm
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return b_point or t_point


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset_game()

    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        
        # Thêm ống đầu tiên ngay khi bắt đầu game
        first_pipe = Pipe()
        first_pipe.x = SCREEN_WIDTH * 1  # Đặt ống đầu tiên ở 60% màn hình
        self.pipes.append(first_pipe)
        
        # Cập nhật thời gian cho ống tiếp theo
        self.last_pipe = pygame.time.get_ticks()
        self.game_over = False
        return self.get_state()

    def get_state(self):
        # Tìm ống gần nhất
        closest_pipe = None
        closest_dist = float("inf")
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                dist = pipe.x - self.bird.x
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pipe = pipe

        if closest_pipe:
            return {
                'bird_y': self.bird.y / SCREEN_HEIGHT,
                'bird_velocity': self.bird.velocity / 10,
                'pipe_gap_y': closest_pipe.gap_y / SCREEN_HEIGHT,
                'pipe_distance': (closest_pipe.x - self.bird.x) / SCREEN_WIDTH
            }
        return {
            'bird_y': self.bird.y / SCREEN_HEIGHT,
            'bird_velocity': self.bird.velocity / 10,
            'pipe_gap_y': 0,
            'pipe_distance': 1
        }

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:  # Nếu game over thì reset game
                        self.reset_game()
                    else:  # Nếu đang chơi thì cho chim nhảy
                        self.bird.flap()

    def run(self):
        while True:  # Thay đổi điều kiện vòng lặp để game chạy liên tục
            # Xử lý input từ người chơi
            self.handle_events()
            
            if not self.game_over:  # Chỉ cập nhật game khi chưa game over
                # Cập nhật trạng thái game
                self.bird.update()

                # Tạo ống mới
                time_now = pygame.time.get_ticks()
                if time_now - self.last_pipe > PIPE_FREQUENCY:
                    self.pipes.append(Pipe())
                    self.last_pipe = time_now

                # Cập nhật và kiểm tra va chạm với ống
                for pipe in self.pipes[:]:
                    pipe.update()
                    if pipe.x + pipe.width < 0:
                        self.pipes.remove(pipe)
                    if not pipe.scored and pipe.x < self.bird.x:
                        pipe.scored = True
                        self.score += 1
                    if pipe.collide(self.bird):
                        self.game_over = True

                # Kiểm tra va chạm với đất hoặc trần
                if self.bird.y < 0 or self.bird.y + self.bird.size > SCREEN_HEIGHT:
                    self.game_over = True

            # Vẽ game
            self.draw()
            self.clock.tick(60)

    def draw(self):
        self.screen.fill(BLACK)
        self.bird.draw(self.screen)
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # Hiển thị điểm
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # Hiển thị Game Over nếu kết thúc
        if self.game_over:
            game_over_text = self.font.render("Game Over!", True, WHITE)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()

    def step(self, action):
        # Xử lý sự kiện quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Thực hiện action
        if action == 1:
            self.bird.flap()

        # Cập nhật bird
        self.bird.update()

        # Tạo ống mới
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > PIPE_FREQUENCY:
            self.pipes.append(Pipe())
            self.last_pipe = time_now

        reward = 0.1  # Phần thưởng nhỏ cho việc sống sót
        # Cập nhật và kiểm tra va chạm với ống
        for pipe in self.pipes[:]:
            pipe.update()
            
            # Xóa ống khi đi qua màn hình
            if pipe.x + pipe.width < 0:
                self.pipes.remove(pipe)
                continue
            
            # Tính điểm khi vượt qua ống
            if not pipe.scored and pipe.x < self.bird.x:
                pipe.scored = True
                self.score += 1
                reward = 1
            
            # Kiểm tra va chạm
            if pipe.collide(self.bird):
                self.game_over = True
                reward = -1
                break  # Dừng ngay khi va chạm

        # Kiểm tra va chạm với đất hoặc trần
        if self.bird.y < 0 or self.bird.y + self.bird.size > SCREEN_HEIGHT:
            self.game_over = True
            reward = -1

        # Vẽ trạng thái hiện tại
        self.draw()

        return self.get_state(), reward, self.game_over, self.score
 

if __name__ == "__main__":
    game = Game()
    game.run()
