import collections
import threading
import time
import random
import sys
import os
import numpy as np
import sounddevice as sd
import tensorflow as tf
import pygame

# Audio / model constants
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000       
INFERENCE_INTERVAL = 0.25   
CONFIDENCE_THRESHOLD = 0.80

# Same sorted order used during training
CHOSEN_COMMANDS = sorted(['down', 'up', 'left', 'right', '_silence_', '_unknown_'])


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.keras')

# Shared state (audio thread → game loop)
_audio_buffer = collections.deque(np.zeros(WINDOW_SAMPLES, dtype=np.float32),
                                   maxlen=WINDOW_SAMPLES)
current_command = '_silence_'
current_confidence = 0.0
_lock = threading.Lock()


# Mel-spectrogram — identical to the training pipeline
def make_spectrogram(audio_np: np.ndarray) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio_np, dtype=tf.float32)
    audio = audio[:WINDOW_SAMPLES]
    pad = tf.maximum(0, WINDOW_SAMPLES - tf.shape(audio)[0])
    audio = tf.pad(audio, [[0, pad]])
    s = tf.signal.stft(audio, frame_length=256, frame_step=128)
    s = tf.abs(s)
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=s.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=8000.0,
    )
    mel = tf.tensordot(s, linear_to_mel, 1)
    mel = tf.math.log(mel + 1e-6)
    return tf.ensure_shape(mel[..., tf.newaxis], [124, 64, 1])


# Sound-device callback: fills the ring buffer
def _audio_callback(indata, frames, time_info, status):
    _audio_buffer.extend(indata[:, 0])


# Inference thread
def _inference_loop(model):
    global current_command, current_confidence
    while True:
        t0 = time.time()
        audio_snap = np.array(_audio_buffer, dtype=np.float32)
        spec = make_spectrogram(audio_snap)[tf.newaxis, ...]
        probs = tf.nn.softmax(model(spec, training=False)).numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        cmd = CHOSEN_COMMANDS[idx] if conf >= CONFIDENCE_THRESHOLD else '_unknown_'
        with _lock:
            current_command = cmd
            current_confidence = conf
        elapsed = time.time() - t0
        time.sleep(max(0.0, INFERENCE_INTERVAL - elapsed))


# Game constants
CELL = 28
COLS, ROWS = 22, 22
WIN_W = CELL * COLS
WIN_H = CELL * ROWS + 80   

BG_COLOR     = (15,  15,  25)
GRID_COLOR   = (30,  30,  45)
SNAKE_HEAD   = (80,  220, 80)
SNAKE_BODY   = (50,  160, 50)
FOOD_COLOR   = (220, 60,  60)
TEXT_COLOR   = (220, 220, 220)
DIM_COLOR    = (100, 100, 120)
BORDER_COLOR = (60,  60,  90)

CMD_COLORS = {
    'up':        (80,  180, 255),
    'down':      (80,  255, 180),
    'left':      (255, 200, 80),
    'right':     (200, 80,  255),
    '_silence_': (100, 100, 120),
    '_unknown_': (180, 100, 100),
}

DIR_MAP = {
    'up':    (0, -1),
    'down':  (0,  1),
    'left':  (-1, 0),
    'right': (1,  0),
}

# Arrow glyphs for HUD (unicode)
CMD_ARROWS = {
    'up':        '↑',
    'down':      '↓',
    'left':      '←',
    'right':     '→',
    '_silence_': '~',
    '_unknown_': '?',
}
# ms between snake moves at start
BASE_SPEED = 500 
# ms between moves at max difficulty  
MIN_SPEED  = 300   


def _new_food(snake: list) -> tuple:
    occupied = set(snake)
    free = [(c, r) for c in range(COLS) for r in range(ROWS) if (c, r) not in occupied]
    return random.choice(free)


def _draw_grid(surf: pygame.Surface):
    for c in range(COLS + 1):
        pygame.draw.line(surf, GRID_COLOR, (c * CELL, 0), (c * CELL, ROWS * CELL))
    for r in range(ROWS + 1):
        pygame.draw.line(surf, GRID_COLOR, (0, r * CELL), (WIN_W, r * CELL))


def _draw_cell(surf: pygame.Surface, col: int, row: int, color: tuple):
    rect = pygame.Rect(col * CELL + 1, row * CELL + 1, CELL - 2, CELL - 2)
    pygame.draw.rect(surf, color, rect, border_radius=5)


def _draw_hud(screen, font_big, font_sm, score, cmd, conf):
    hud_y = ROWS * CELL
    pygame.draw.line(screen, BORDER_COLOR, (0, hud_y), (WIN_W, hud_y), 2)

    score_surf = font_big.render(f"Score: {score}", True, TEXT_COLOR)
    screen.blit(score_surf, (14, hud_y + 8))

    arrow = CMD_ARROWS.get(cmd, '?')
    label = f"{arrow}  {cmd}  ({conf:.0%})"
    cmd_color = CMD_COLORS.get(cmd, DIM_COLOR)
    cmd_surf = font_sm.render(label, True, cmd_color)
    screen.blit(cmd_surf, (14, hud_y + 40))

    hint = font_sm.render("R = restart   ESC = quit", True, DIM_COLOR)
    screen.blit(hint, (WIN_W - hint.get_width() - 14, hud_y + 40))


def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Voice Snake — say up / down / left / right")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont('monospace', 24, bold=True)
    font_sm  = pygame.font.SysFont('monospace', 18)

    def reset():
        snake = [(COLS // 2, ROWS // 2),
                 (COLS // 2 - 1, ROWS // 2),
                 (COLS // 2 - 2, ROWS // 2)]
        direction = (1, 0)
        food = _new_food(snake)
        return snake, direction, food, 0, BASE_SPEED

    snake, direction, food, score, move_interval = reset()
    game_over = False
    move_timer = 0

    while True:
        dt = clock.tick(60)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_r:
                    snake, direction, food, score, move_interval = reset()
                    game_over = False
                    move_timer = 0

        if game_over:
            with _lock:
                cmd, conf = current_command, current_confidence
            screen.fill(BG_COLOR)
            _draw_grid(screen)
            _draw_cell(screen, food[0], food[1], FOOD_COLOR)
            for i, (c, r) in enumerate(snake):
                _draw_cell(screen, c, r, SNAKE_HEAD if i == 0 else SNAKE_BODY)
            overlay = pygame.Surface((WIN_W, ROWS * CELL), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            screen.blit(overlay, (0, 0))
            go1 = font_big.render("GAME OVER", True, (255, 80, 80))
            go2 = font_sm.render(f"Score: {score}   |   Press R to restart", True, TEXT_COLOR)
            cx = WIN_W // 2
            cy = ROWS * CELL // 2
            screen.blit(go1, (cx - go1.get_width() // 2, cy - 30))
            screen.blit(go2, (cx - go2.get_width() // 2, cy + 10))
            _draw_hud(screen, font_big, font_sm, score, cmd, conf)
            pygame.display.flip()
            continue

        # Read latest voice command
        with _lock:
            cmd, conf = current_command, current_confidence

        # Update direction (block 180° reversals)
        if cmd in DIR_MAP:
            new_dir = DIR_MAP[cmd]
            # Sum of opposite unit vectors is (0,0)
            if (new_dir[0] + direction[0], new_dir[1] + direction[1]) != (0, 0):
                direction = new_dir

        # Move snake on timer
        move_timer += dt
        if move_timer >= move_interval:
            move_timer = 0
            head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

            if not (0 <= head[0] < COLS and 0 <= head[1] < ROWS) or head in snake:
                game_over = True
            else:
                snake.insert(0, head)
                if head == food:
                    score += 1
                    food = _new_food(snake)
                    move_interval = max(MIN_SPEED, move_interval - 4)
                else:
                    snake.pop()

        # Draw
        screen.fill(BG_COLOR)
        _draw_grid(screen)
        _draw_cell(screen, food[0], food[1], FOOD_COLOR)
        for i, (c, r) in enumerate(snake):
            _draw_cell(screen, c, r, SNAKE_HEAD if i == 0 else SNAKE_BODY)
        _draw_hud(screen, font_big, font_sm, score, cmd, conf)
        pygame.display.flip()


# Entry point
def main():
    print("Loading model…")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded. Warming up…")
    model(tf.zeros([1, 124, 64, 1]), training=False)
    print("Ready.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=1600,       
        callback=_audio_callback,
    )
    stream.start()
    print("Microphone stream started. Say 'up', 'down', 'left', or 'right' to steer.")

    inf_thread = threading.Thread(target=_inference_loop, args=(model,), daemon=True)
    inf_thread.start()

    try:
        run_game()
    finally:
        stream.stop()
        stream.close()
        pygame.quit()


if __name__ == '__main__':
    main()
