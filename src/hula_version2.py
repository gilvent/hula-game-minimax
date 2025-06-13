import pygame
import math
import sys
import os
import random
import copy
import time
import threading
from collections import deque
import game_ai_minimax

# Set environment variable to ensure the window opens at the center of the screen
os.environ["SDL_VIDEO_CENTERED"] = "1"

# Initialize Pygame
pygame.init()

# Get current screen resolution
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Create a window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define colors and hexagon properties
BG_COLOR = (30, 30, 30)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (188, 188, 188)
DARK_GRAY = (100, 100, 100)
RED = (255, 0, 0)
BRIGHT_YELLOW = (255, 255, 0)
HEX_SIZE = 30
HEX_BORDER = 2
PIECE_RADIUS = int(HEX_SIZE * 0.8)
TURN_TIME_LIMIT_MS = 30_000
TURN_TIME_LIMIT = 30

# Initialize font for text rendering
pygame.font.init()
font = pygame.font.SysFont(None, int(HEX_SIZE * 0.7))
hexagon_board = {}
selected_counts = {}
turn_ended = False
max_selected_counts = {}
initial_counts = {}
game_over = False


# Global event to signal that the current turn is done.
turn_done_event = threading.Event()


def timer_thread(start_tick):
    """
    Countdown timer that runs in a separate thread.
    It prints the remaining time (in seconds) in the terminal once per second.
    The loop stops when either time runs out or turn_done_event is set.
    """
    while not turn_done_event.is_set() and not game_over:
        elapsed_ms = pygame.time.get_ticks() - start_tick
        remaining = TURN_TIME_LIMIT - (elapsed_ms / 1000)
        if remaining <= 0:
            print("Time is up! Auto move triggered.")
            # Optionally, you could also set a global flag here to trigger an auto move in the main loop.
            break
        print(f"Remaining time: {math.ceil(remaining)} seconds")
        time.sleep(1)


def start_turn_timer():
    """
    Starts the countdown timer in a separate thread.
    Returns the start tick of the current turn.
    """
    start_tick = pygame.time.get_ticks()
    turn_done_event.clear()  # Clear the event at the start of the turn
    threading.Thread(target=timer_thread, args=(start_tick,), daemon=True).start()
    return start_tick


def draw_player_turn_button(screen, button_text, message=""):
    """Draws a turn indicator button at the top right of the screen."""
    screen_width, screen_height = screen.get_size()
    font = pygame.font.SysFont(None, 36)  # Set the font size to 36
    button_width, button_height = 150, 50  # Width and height of the button
    button_x = screen_width - button_width - 10  # 10 pixels margin from the right edge
    button_y = 10  # 10 pixels margin from the top edge
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, (0, 0, 255), button_rect)  # Draw a blue button

    text = button_text

    text_surf = font.render(text, True, pygame.Color("white"))
    screen.blit(text_surf, text_surf.get_rect(center=button_rect.center))

    return button_rect


def draw_neutral_stones_button(screen):
    """Draws a turn indicator button at the top right of the screen."""
    screen_width, screen_height = screen.get_size()
    font = pygame.font.SysFont(None, 36)  # Set the font size to 36
    button_width, button_height = 150, 50  # Width and height of the button
    button_x = screen_width - button_width - 10  # 10 pixels margin from the right edge
    button_y = 70  # 10 pixels margin from the top edge
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, (0, 0, 255), button_rect)  # Draw a blue button

    text = "neutral"

    text_surf = font.render(text, True, pygame.Color("white"))
    screen.blit(text_surf, text_surf.get_rect(center=button_rect.center))

    return button_rect


def draw_hexagon(surface, x, y, size, border_color, fill_color, border_thickness=2):
    """Draws a hexagon."""

    if x == WIDTH / 2 and y == HEIGHT / 2:
        fill_color = RED

    angles_deg = [60 * i + 30 for i in range(6)]
    outer_points = [
        (
            x + (size + border_thickness) * math.cos(math.radians(angle)),
            y + (size + border_thickness) * math.sin(math.radians(angle)),
        )
        for angle in angles_deg
    ]
    inner_points = [
        (
            x + size * math.cos(math.radians(angle)),
            y + size * math.sin(math.radians(angle)),
        )
        for angle in angles_deg
    ]

    if fill_color == WHITE:
        border_color = GRAY

    elif fill_color == BLACK:
        border_color = WHITE
    elif fill_color == GRAY:
        border_color = DARK_GRAY

    pygame.draw.polygon(surface, border_color, outer_points)
    pygame.draw.polygon(surface, fill_color, inner_points)


def point_in_hex(x, y, hex_x, hex_y, size):
    """Check if the point (x, y) is inside the hexagon centered at (hex_x, hex_y)."""
    dx = abs(x - hex_x)
    dy = abs(y - hex_y)
    return (
        dx <= size * math.sqrt(3) / 2
        and dy <= size * 3 / 2
        and size * 3 / 2 - dx * math.sqrt(3) / 3 > dy
    )


def draw_hex_shape_grid(surface, center_row, center_col, size):
    """Draws a grid of hexagons on the screen."""
    global hexagon_board
    initial_counts.clear()

    for row in range(-center_row, center_row + 1):
        for col in range(-center_col, center_col + 1):
            dist_from_center = max(abs(row), abs(col), abs(row + col))
            if dist_from_center <= center_row:
                x = WIDTH / 2 + (col + row / 2) * (math.sqrt(3) * (size + HEX_BORDER))
                y = HEIGHT / 2 + row * ((size + HEX_BORDER) * 1.5)
                label = None

                hexagon_board[(row, col)] = {
                    "x": x,
                    "y": y,
                    "selected": False,
                    "color": None,
                }  # , 'disabled': False}
                if row == 0 and col == 0:
                    hexagon_board[(0, 0)]["selected"] = True

                initial_counts[label] = initial_counts.get(label, 0) + 1

                draw_hexagon(surface, x, y, size, (255, 255, 255), (255, 228, 205))


def check_all_hexes_selected():
    """Checks if all hexes on the board have been selected."""
    return all(hex_info["selected"] for hex_info in hexagon_board.values())


def update_selected_hexes(selected_hexes, color):
    """Updates the state and visual representation of selected hexes."""
    global current_turn
    if color == "black":
        fill_color = BLACK
    elif color == "white":
        fill_color = WHITE
    elif color == "gray":
        fill_color = GRAY

    for hex_info in selected_hexes:
        hex_info["selected"] = True
        hex_info["color"] = color

        draw_hexagon(
            screen,
            hex_info["x"],
            hex_info["y"],
            HEX_SIZE,
            (128, 128, 128),
            fill_color=fill_color,
        )
        pygame.display.flip()


def auto_select_remaining_hexes(color):
    """Automatically selects the remaining hexes for a label if the turn timer expires."""
    hexes_by_label = []
    # Collect all unbooked hexes of the current label
    for pos, info in hexagon_board.items():
        if not info.get("selected", False):
            hexes_by_label.append(info)

    selected_hexes = random.sample(hexes_by_label, 1)
    # Process selected hexes
    color = random.choices([color, "gray"], weights=[0.95, 0.05])[0]
    update_selected_hexes(selected_hexes, color=color)
    return selected_hexes


def process_selections(x, y):
    selected_hexes = []
    for (hx, hy), hex_info in hexagon_board.items():
        if point_in_hex(
            x, y, hex_info["x"], hex_info["y"], HEX_SIZE
        ) and not hex_info.get("selected", False):
            selected_hexes.append(hex_info)
            break

    return selected_hexes


def draw_win_border(path):

    border_thickness = 2
    size = HEX_SIZE
    angles_deg = [60 * i + 30 for i in range(6)]
    for x, y in path:
        info = hexagon_board[(x, y)]
        x = info["x"]
        y = info["y"]
        color = info["color"]
        if color == "black":
            fill_color = BLACK
        elif color == "white":
            fill_color = WHITE
        elif color == "gray":
            fill_color = GRAY

        outer_points = [
            (
                x + (size + border_thickness) * math.cos(math.radians(angle)),
                y + (size + border_thickness) * math.sin(math.radians(angle)),
            )
            for angle in angles_deg
        ]
        inner_points = [
            (
                x + size * math.cos(math.radians(angle)),
                y + size * math.sin(math.radians(angle)),
            )
            for angle in angles_deg
        ]

        pygame.draw.polygon(screen, RED, outer_points)
        pygame.draw.polygon(screen, fill_color, inner_points)


################################################################################################################################################


def decide_winner():

    global game_over
    black_form_a_loop, black_best_loop, black_num_neutral_stones, black_loop_len = (
        form_loop("black")
    )
    white_form_a_loop, white_best_loop, white_num_neutral_stones, white_loop_len = (
        form_loop("white")
    )
    if black_form_a_loop == True and white_form_a_loop == True:
        print(
            f"Black player - Neutral stones selected: {black_num_neutral_stones}, Loop length: {black_loop_len}"
        )
        print(
            f"White player - Neutral stones selected: {white_num_neutral_stones}, Loop length: {white_loop_len}"
        )
        if black_loop_len < white_loop_len:
            print("black win!!")
            draw_win_border(black_best_loop)
            game_over = True
            pygame.display.flip()
            pygame.time.delay(5000)
            pygame.quit()
            return True
        elif black_loop_len > white_loop_len:
            print("white win!!")
            draw_win_border(white_best_loop)
            game_over = True
            pygame.display.flip()
            pygame.time.delay(5000)
            pygame.quit()
            return True
        else:
            if black_num_neutral_stones < white_num_neutral_stones:
                print("black win!!")
                draw_win_border(black_best_loop)
                game_over = True
                pygame.display.flip()
                pygame.time.delay(5000)
                pygame.quit()
                return True
            elif black_num_neutral_stones > white_num_neutral_stones:
                print("white win!!")
                draw_win_border(white_best_loop)
                game_over = True
                pygame.display.flip()
                pygame.time.delay(5000)
                pygame.quit()
                return True
            else:
                print("white win!!")
                draw_win_border(white_best_loop)
                game_over = True
                pygame.display.flip()
                pygame.time.delay(5000)
                pygame.quit()
                return True

    elif black_form_a_loop == True:
        print("black win!!")
        print(
            f"Black player - Neutral stones selected: {black_num_neutral_stones}, Loop length: {black_loop_len}"
        )
        draw_win_border(black_best_loop)
        game_over = True
        pygame.display.flip()
        pygame.time.delay(50000)
        pygame.quit()
        return True

    elif white_form_a_loop == True:
        print("white win!!")
        print(
            f"White player - Neutral stones selected: {white_num_neutral_stones}, Loop length: {white_loop_len}"
        )
        draw_win_border(white_best_loop)
        game_over = True
        pygame.display.flip()
        pygame.time.delay(50000)
        pygame.quit()
        return True

    elif (
        black_form_a_loop == False
        and white_form_a_loop == False
        and check_all_hexes_selected() == True
    ):
        bigger_group()
    else:
        return False


def find_shortest_loop(walls, DIR):

    def shortest_cycle(start, walls, DIR):

        queue = deque()
        queue.append((start, [start]))

        while queue:
            current, path = queue.popleft()
            for dx, dy in DIR:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if neighbor == start and len(path) >= 6:

                    return True, path

                if neighbor in walls and neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

        return False, []

    shortest_loop = None
    shortest_length = float("inf")
    shortest_gray_count = None

    for wall in walls:

        loop, path = shortest_cycle(wall, walls, DIR)

        if loop:

            loop_length = len(path)
            gray_count = sum(
                1 for point in path if hexagon_board[point]["color"] == "gray"
            )

            if loop_length < shortest_length:
                shortest_length = loop_length
                shortest_loop = path
                shortest_gray_count = gray_count

            if loop_length == shortest_length:

                if gray_count < shortest_gray_count:
                    shortest_loop = path
                    shortest_gray_count = gray_count

    if shortest_loop:
        return shortest_length, shortest_loop, shortest_gray_count
    else:
        return None, None, None


def form_loop(player_color):

    DIR = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    CENTER = (0, 0)

    visited = set()
    queue = deque()
    queue.append(CENTER)
    visited.add(CENTER)
    walls = set()

    def bfs():

        while queue:
            cur_x, cur_y = queue.popleft()

            for dx, dy in DIR:
                nx, ny = cur_x + dx, cur_y + dy
                neighbor = (nx, ny)

                if neighbor in visited:
                    continue

                if neighbor not in hexagon_board:
                    return False, walls

                color = hexagon_board[neighbor]["color"]
                if color == player_color or color == "gray":
                    walls.add(neighbor)
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

        return True, walls

    result, walls = bfs()

    if result == True:
        shortest_length, shortest_loop, shortest_gray_count = find_shortest_loop(
            walls, DIR
        )
        return True, shortest_loop, shortest_gray_count, shortest_length
    else:
        return False, [], 0, 0


def bigger_group():
    """Neither player forms a loop, the player with the largest connected group of stones wins."""

    global game_over
    black_max_group = 0
    white_max_group = 0
    black_best_group = set()
    white_best_group = set()
    hexagon_board_copy = copy.deepcopy(hexagon_board)

    def dfs(x, y, player_color):
        """DFS to explore connected group and return its size."""
        stack = [(x, y)]
        local_visited = set()
        count = 0

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in local_visited:
                continue

            local_visited.add((cx, cy))
            info = hexagon_board_copy[(cx, cy)]
            if hexagon_board_copy[(cx, cy)]["color"] == "gray":
                count += 2
            else:
                count += 1

            # Only clear black/white stones, not gray
            if info["color"] in ["black", "white"]:
                hexagon_board_copy[(cx, cy)]["color"] = None

            # 6 directions on hex grid
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in hexagon_board_copy or (nx, ny) in local_visited:
                    continue
                neighbor_color = hexagon_board_copy[(nx, ny)]["color"]
                if neighbor_color == player_color or neighbor_color == "gray":
                    stack.append((nx, ny))

        return count, local_visited

    for (x, y), info in list(hexagon_board_copy.items()):
        color = info["color"]
        if color in ["black", "white"]:
            group_size, group_path = dfs(x, y, color)
            if color == "black" and group_size > black_max_group:
                black_max_group = group_size
                black_best_group = group_path
            elif color == "white" and group_size > white_max_group:
                white_max_group = group_size
                white_best_group = group_path

    print(f"Black max group size: {black_max_group}")
    print(f"White max group size: {white_max_group}")
    if black_max_group > white_max_group:
        print("Black wins by larger group!")
        draw_win_border(black_best_group)
        game_over = True
        pygame.display.flip()
    elif white_max_group > black_max_group:
        print("White wins by larger group!")
        draw_win_border(white_best_group)
        game_over = True
        pygame.display.flip()
    else:
        print("White wins!")
        draw_win_border(white_best_group)
        game_over = True
        pygame.display.flip()


def main(black_player, white_player):

    print(f"Player 1: {black_player}, Player 2: {white_player}")

    global game_over
    black_player_type = black_player
    white_player_type = white_player

    current_turn = "black"
    turn_start = start_turn_timer()
    last_printed = -1

    screen.fill(BG_COLOR)
    draw_hex_shape_grid(screen, 5, 5, HEX_SIZE)
    player_stone_rect = draw_player_turn_button(screen, current_turn)
    neutral_stone_rect = draw_neutral_stones_button(screen)

    pygame.display.flip()
    pygame.time.delay(2000)

    position_selected = False
    color_selected = False
    selected_hexes = []
    color = None
    prompt_shown = False
    exit_game = False

    while not check_all_hexes_selected():
        # Check whether the human turn has timed out:
        if (pygame.time.get_ticks() - turn_start) >= TURN_TIME_LIMIT_MS:

            if (current_turn == "black" and black_player_type == "human") or (
                current_turn == "white" and white_player_type == "human"
            ):
                print("Time is up for human turn! Executing auto move.")
                auto_select_remaining_hexes(current_turn)
                # Signal that the turn is over:
                turn_done_event.set()
                pygame.time.delay(1000)
                # Switch turn:
                current_turn = "white" if current_turn == "black" else "black"
                draw_player_turn_button(screen, current_turn)
                pygame.display.flip()
                # Restart the timer for the new turn
                turn_start = start_turn_timer()
                continue

        if (current_turn == "black" and black_player_type == "human") or (
            current_turn == "white" and white_player_type == "human"
        ):

            if not position_selected and not prompt_shown:
                print("Please select a position.")
                prompt_shown = True

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:

                    x, y = pygame.mouse.get_pos()

                    # choose position
                    if not position_selected:
                        selected_hexes = process_selections(x, y)
                        if selected_hexes:
                            position_selected = True
                            print("Please choose a stone type.")

                    # choose weather place neutral_stone
                    elif position_selected and not color_selected:

                        if neutral_stone_rect.collidepoint(event.pos):
                            color = "gray"
                            color_selected = True

                        elif player_stone_rect.collidepoint(event.pos):
                            color = current_turn
                            color_selected = True

                        if color_selected:
                            update_selected_hexes(selected_hexes, color=color)

                            for selected in selected_hexes:
                                x, y = selected["x"], selected["y"]
                                for position, info in hexagon_board.items():
                                    if info["x"] == x and info["y"] == y:
                                        x_pos, y_pos = position
                                        break

                            print(f"Selected: {x_pos, y_pos}, Stone's color: {color}")
                            exit_game = decide_winner()

                    if position_selected and color_selected:
                        turn_done_event.set()
                        pygame.time.delay(1000)
                        current_turn = "white" if current_turn == "black" else "black"
                        draw_player_turn_button(screen, current_turn)
                        pygame.display.flip()

                        position_selected = False
                        color_selected = False
                        selected_hexes = []
                        color = None
                        prompt_shown = False

                        # Restart the timer for the new turn
                        turn_start = start_turn_timer()

        if exit_game:
            game_over = True

            turn_done_event.set()
            break

        if (current_turn == "black" and black_player_type == "random") or (
            current_turn == "white" and white_player_type == "random"
        ):
            start_auto = pygame.time.get_ticks()
            # time.sleep(35)
            selected_hexes = auto_select_remaining_hexes(color=current_turn)
            end_auto = pygame.time.get_ticks()
            if (end_auto - start_auto) >= TURN_TIME_LIMIT_MS:
                print("Timeout! AI took longer than 30 seconds!")
                print("Please modify your AI function!")
                while True:
                    time.sleep(1)

            for selected in selected_hexes:
                x, y = selected["x"], selected["y"]
                for position, info in hexagon_board.items():
                    if info["x"] == x and info["y"] == y:
                        x_pos, y_pos = position
                        break

            if decide_winner():
                break

            turn_done_event.set()
            pygame.time.delay(100)
            current_turn = "white" if current_turn == "black" else "black"
            draw_player_turn_button(screen, current_turn)
            pygame.display.flip()
            turn_start = start_turn_timer()

        if (current_turn == "black" and black_player_type == "AI") or (
            current_turn == "white" and white_player_type == "AI"
        ):
            start_auto = pygame.time.get_ticks()

            """
            pos_x, pos_y, color = your_AI_method()

            
            info = hexagon_board[(pos_x, pos_y)]
            x = info['x']
            y = info['y']
            selected_hexes = process_selections(x, y)
            update_selected_hexes(selected_hexes, color = color)
            """
            pos_x, pos_y, color = game_ai_minimax.make_move(
                hexagon_board=hexagon_board, color=current_turn
            )

            info = hexagon_board[(pos_x, pos_y)]
            x = info["x"]
            y = info["y"]
            selected_hexes = process_selections(x, y)
            update_selected_hexes(selected_hexes, color=color)

            # handle AI longer than 30s
            end_auto = pygame.time.get_ticks()
            if (end_auto - start_auto) >= TURN_TIME_LIMIT_MS:
                print("Timeout! AI took longer than 30 seconds!")
                print("Please modify your AI function!")
                while True:
                    time.sleep(1)
            if decide_winner():
                break

            turn_done_event.set()
            pygame.time.delay(1000)
            current_turn = "white" if current_turn == "black" else "black"
            draw_player_turn_button(screen, current_turn)
            pygame.display.flip()
            turn_start = start_turn_timer()

    turn_done_event.set()
    pygame.time.delay(100000)
    pygame.quit()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python main.py [player1_type] [player2_type]")
        print("player1_type and player2_type should be 'human' or 'random' or 'AI")
        sys.exit(1)  # Exit the script with an error code

    main(sys.argv[1], sys.argv[2])
