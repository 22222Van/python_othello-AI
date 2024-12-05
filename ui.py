import pygame
import sys
from utils import *

SQUARE_SIZE = 60
WINDOW_SIZE = BOARD_WIDTH * SQUARE_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (249, 214, 91)
FPS = 24

window: Optional[pygame.Surface] = None
clock: Optional[pygame.time.Clock] = None

print()


def init_window() -> None:
    global window, clock
    pygame.init()

    window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('CS181 Project Othello (黑白棋)')
    clock = pygame.time.Clock()


def get_player_action(legal_actions):
    global window, clock
    if clock is None:
        raise RuntimeError("User interface is not initialized")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = get_square_under_mouse()
                if pos in legal_actions:
                    return pos
        clock.tick(FPS)


def draw_board(grid, color, legal_actions):
    global window, clock
    if clock is None or window is None:
        raise RuntimeError("User interface is not initialized")

    window.fill(BACKGROUND_COLOR)
    for y in range(BOARD_WIDTH):
        for x in range(BOARD_WIDTH):
            rect = pygame.Rect(y * SQUARE_SIZE, x *
                               SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(window, BLACK, rect, 1)
            if (x, y) in legal_actions:
                if color == 'B':
                    pygame.draw.circle(
                        window, BLACK, rect.center, SQUARE_SIZE // 8)
                elif color == 'W':
                    pygame.draw.circle(
                        window, WHITE, rect.center, SQUARE_SIZE // 8)
            elif grid[x][y] == 'B':
                pygame.draw.circle(
                    window, BLACK, rect.center, SQUARE_SIZE // 2 - 5)
            elif grid[x][y] == 'W':
                pygame.draw.circle(
                    window, WHITE, rect.center, SQUARE_SIZE // 2 - 5)

    pygame.display.update()

    running = FPS//2  # FIXME: Hardcode
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        clock.tick(FPS)
        running -= 1


def get_square_under_mouse():
    mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
    y, x = [int(v // SQUARE_SIZE) for v in mouse_pos]

    return x, y
