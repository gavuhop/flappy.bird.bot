# Flappy Bird AI

Một phiên bản Flappy Bird được điều khiển bởi AI sử dụng thuật toán NEAT (NeuroEvolution of Augmenting Topologies).

## Mô tả

Dự án này là một triển khai của trò chơi Flappy Bird với AI tự học. AI sử dụng thuật toán NEAT để học cách chơi trò chơi thông qua quá trình tiến hóa. Mỗi thế hệ, AI sẽ cố gắng bay qua các ống và học từ những lỗi lầm của nó.

## Tính năng

- Trò chơi Flappy Bird cổ điển với đồ họa đơn giản
- AI tự học sử dụng thuật toán NEAT
- Hiển thị số lượng chim còn sống và điểm số
- Hệ thống điểm dựa trên số ống vượt qua
- Cấu hình NEAT có thể tùy chỉnh

## Yêu cầu

- Python 3.8+
- Pygame
- NEAT-Python

## Cài đặt

1. Clone repository:

```bash
git clone https://github.com/yourusername/flappy.bird.bot.git
cd flappy.bird.bot
```

2. Cài đặt dependencies:

```bash
poetry install
```

## Cách sử dụng

1. Chạy trò chơi với AI:

```bash
poetry run python ai_bird.py
```

2. Chạy trò chơi thông thường (điều khiển bằng người):

```bash
poetry run python flappy_bird.py
```

## Cấu trúc dự án

- `flappy_bird.py`: Trò chơi Flappy Bird gốc
- `ai_bird.py`: Phiên bản AI của trò chơi
- `config.txt`: Cấu hình cho thuật toán NEAT
- `poetry.lock` & `pyproject.toml`: Quản lý dependencies

## Cấu hình NEAT

Các thông số chính trong `config.txt`:

- Kích thước quần thể: 50
- Số thế hệ tối đa: 50
- Ngưỡng fitness: 100
- Số input nodes: 4 (vị trí y của chim, vận tốc, vị trí y của ống gần nhất, khoảng cách đến ống)
- Số output nodes: 1 (quyết định nhảy hay không)

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo pull request hoặc mở issue để thảo luận về các thay đổi.

## Giấy phép

MIT License
